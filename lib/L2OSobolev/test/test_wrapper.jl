using Test, L2OSobolev, L2OCore, Random
import DiffOpt, HiGHS

# Self-supervised inner method (objective + squared violations; no labels).
struct ToyPenalty <: AbstractPrimalMethod
    ρ::Float64
end
L2OCore.supports(::ToyPenalty, ::L2OCore.AbstractConvexClass) = true
L2OCore.generate_data(
    ::ToyPenalty,
    pipe;
    sampler,
    n,
    rng = Random.default_rng(),
    kwargs...,
) = L2ODataset(sample_parameters(sampler, rng, n))
function L2OCore.loss(m::ToyPenalty, pipe, P, out; kwargs...)
    ev = pipe.evaluator
    X = L2OCore.complete(m, pipe, P, out).primal
    return (sum(batch_objective(ev, X, P)) + m.ρ * sum(abs2, row_violation(ev, X, P))) /
           size(X, 2)
end

# Inner method whose value labels come from a different solution branch.
struct Shifted <: AbstractPrimalMethod end
L2OCore.supports(::Shifted, ::L2OCore.AbstractProblemClass) = true
function L2OCore.generate_data(::Shifted, pipe; kwargs...)
    ds = L2OCore.generate_data(SupervisedMSE(), pipe; kwargs...)
    return L2ODataset(ds.inputs; primal = ds.primal .+ 1, status = ds.status)
end

struct ToyDual <: AbstractDualMethod end

const QP_SAMPLER = BoxSampler([0.0, 1.0, 0.0], [1.0, 2.0, 1.0])

@testset "construction and forwarding" begin
    w = SobolevRegularized(SupervisedMSE(); weight = 0.5, mask = RandomMask(0.1))
    @test inner_method(w) === SupervisedMSE()
    @test w.weight === 0.5 && w.backend isa CentralDifferences
    @test occursin("SobolevRegularized(SupervisedMSE", sprint(show, w))
    @test supports(w, ConvexQuadratic()) == supports(SupervisedMSE(), ConvexQuadratic())
    @test method_family(w) == PrimalFamily()
    @test_throws ArgumentError SobolevRegularized(SupervisedMSE(); weight = -1)
    @test_throws ArgumentError SobolevRegularized(ToyDual())
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, w)
    @test pipe.decomposition == decompose(prob, SupervisedMSE())
    P = rand(Xoshiro(1), 3, 4)
    out = rand(Xoshiro(2), 2, 4)
    @test complete(pipe, P, out).primal ==
          complete(L2OPipeline(prob, SupervisedMSE()), P, out).primal
end

@testset "generate_data attaches derivative labels ($(nameof(typeof(dirs))))" for dirs in (
    FullJacobian(),
    RandomPullbacks(),
    RandomPushforwards(),
)
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SobolevRegularized(SupervisedMSE(); directions = dirs))
    ds = generate_data(
        pipe;
        sampler = QP_SAMPLER,
        n = 6,
        rng = Xoshiro(3),
        optimizer = HiGHS.Optimizer,
    )
    @test num_samples(ds) == 6 && ds.primal !== nothing
    @test ds.metadata["sobolev_dropped"] == 0
    @test ds.metadata["sobolev_oracle"] == "DiffOptSensitivities"
    J = eq_qp_jacobian()
    if dirs isa FullJacobian
        @test all(ds.derivatives.J[:, :, k] ≈ J for k = 1:6)
    elseif dirs isa RandomPullbacks
        @test ds.derivatives.values ≈ J' * ds.derivatives.seeds atol = 1e-8
    else
        @test ds.derivatives.values ≈ J * ds.derivatives.directions atol = 1e-8
    end
    # Subsetting keeps samples and labels aligned.
    @test num_samples(ds[[1, 3]]) == 2
    # Value labels at the perfect proxy: only the Sobolev term remains.
    exact = P -> reduce(hcat, (eq_qp_solution(P[:, k]) for k in axes(P, 2)))
    @test exact(ds.inputs) ≈ ds.primal atol = 1e-6
    ℓ = loss(pipe, ds.inputs, exact(ds.inputs); targets = ds, proxy = exact)
    @test ℓ < 1e-12                                         # zero at a match
end

@testset "failed and inconsistent samples are dropped" begin
    prob = ParametricProblem(bound_lp())
    pipe = L2OPipeline(prob, SobolevRegularized(SupervisedMSE()))
    sampler = (rng, n) -> [-1.0 2.0 3.0; 1.0 1.0 1.0][:, 1:n]   # first instance infeasible
    ds = generate_data(pipe; sampler = sampler, n = 3, optimizer = HiGHS.Optimizer)
    @test num_samples(ds) == 2 && ds.metadata["sobolev_dropped"] == 1
    @test !any(isnan, ds.derivatives.J)
    keep = generate_data(
        pipe;
        sampler = sampler,
        n = 3,
        optimizer = HiGHS.Optimizer,
        drop_failed = false,
    )
    @test num_samples(keep) == 3 && all(isnan, keep.derivatives.J[:, :, 1])
    shifted = L2OPipeline(prob, SobolevRegularized(Shifted()))
    ds2 = generate_data(shifted; sampler = sampler, n = 3, optimizer = HiGHS.Optimizer)
    @test num_samples(ds2) == 0 && ds2.metadata["sobolev_dropped"] == 3
    @test_throws ArgumentError generate_data(pipe; sampler = sampler, n = 3)
    # The finite-difference oracle can replace DiffOpt.
    ds3 = generate_data(
        pipe;
        sampler = sampler,
        n = 3,
        optimizer = HiGHS.Optimizer,
        oracle = FiniteDifferenceSensitivities(HiGHS.Optimizer),
    )
    @test ds3.derivatives.J[:, :, 1] ≈ bound_lp_jacobian([2.0, 1.0]) atol = 1e-6
end

@testset "loss = inner loss + λ · sobolev_loss ($(nameof(typeof(inner))))" for inner in (
    SupervisedMSE(),
    ToyPenalty(10.0),
)
    prob = ParametricProblem(eq_qp())
    rng = Xoshiro(4)
    w = SobolevRegularized(inner; weight = 3.0)
    pipe = L2OPipeline(prob, w)
    ds = generate_data(
        pipe;
        sampler = QP_SAMPLER,
        n = 5,
        rng = rng,
        optimizer = HiGHS.Optimizer,
    )
    @test ds.derivatives isa JacobianLabels
    @test (ds.primal !== nothing) == (inner isa SupervisedMSE)
    A = randn(rng, 2, 3)
    proxy = P -> A * P .+ 0.1
    P = ds.inputs
    inner_pipe = L2OPipeline(prob, inner)
    ℓ0 = loss(inner_pipe, P, proxy(P); targets = ds)
    ℓd = sobolev_loss(CentralDifferences(), ProxyMap(proxy, pipe), P, ds.derivatives)
    @test ℓd ≈ sum(abs2, A .- eq_qp_jacobian()) / 6 rtol = 1e-6   # affine proxy
    @test loss(pipe, P, proxy(P); targets = ds, proxy = proxy) ≈ ℓ0 + 3.0 * ℓd
    # ReducesToInner: weight 0 needs neither labels nor the proxy.
    w0 = L2OPipeline(prob, SobolevRegularized(inner; weight = 0))
    @test loss(w0, P, proxy(P); targets = ds) == ℓ0
    # Clear errors when the term cannot be evaluated.
    @test_throws ArgumentError loss(pipe, P, proxy(P); targets = ds)
    @test_throws ArgumentError loss(
        pipe,
        P,
        proxy(P);
        targets = with_derivatives(ds, nothing),
        proxy = proxy,
    )
    @test_throws ArgumentError L2OCore.extra_loss(w, pipe, P, proxy(P))
end

@testset "primal-dual methods and completion layers" begin
    prob = ParametricProblem(eq_qp())
    rng = Xoshiro(5)
    J = eq_qp_jacobian()
    labels = JacobianLabels(repeat(J, 1, 1, 4))
    P = rand(rng, 3, 4)
    # Primal-dual: only the primal rows are supervised.
    pd = L2OPipeline(prob, SobolevRegularized(ToyPrimalDual()))
    nd = num_duals(prob)
    proxy = P -> vcat(J * P, randn(Xoshiro(0), nd, 3) * P)
    t = L2ODataset(P; primal = zeros(2, 4), dual = zeros(nd, 4), derivatives = labels)
    @test L2OCore.extra_loss(pd.method, pd, P, proxy(P); targets = t, proxy = proxy) < 1e-12
    # Completion layer (HalfMethod): x₂ = p₃ + 1 - x₁. With x₁ = J[1, :]ᵀp the
    # completed map has Jacobian [J[1, :]ᵀ; -J₁₁ -J₁₂ 1-J₁₃] = J for this QP, so
    # supervising through the completion gives zero loss.
    hm = L2OPipeline(prob, SobolevRegularized(HalfMethod(); through_completion = true))
    h = P -> J[1:1, :] * P
    @test L2OCore.extra_loss(hm.method, hm, P, h(P); targets = t, proxy = h) < 1e-12
    h2 = P -> 2 .* J[1:1, :] * P
    @test L2OCore.extra_loss(hm.method, hm, P, h2(P); targets = t, proxy = h2) ≈
          2 * sum(abs2, J[1, :]) / 6 rtol = 1e-6
    raw = L2OPipeline(prob, SobolevRegularized(HalfMethod()))
    @test_throws ArgumentError L2OCore.extra_loss(
        raw.method,
        raw,
        P,
        h(P);
        targets = t,
        proxy = h,
    )
end

@testset "degenerate samples are filtered with max_label_error" begin
    prob = ParametricProblem(bound_lp())
    pipe = L2OPipeline(prob, SobolevRegularized(SupervisedMSE()))
    # Sample 2 sits on the kink p₁ = p₂ (non-differentiable solution map).
    sampler = (rng, n) -> [3.0 1.0 0.5; 1.0 1.0 1.5][:, 1:n]
    plain = generate_data(pipe; sampler, n = 3, optimizer = HiGHS.Optimizer)
    @test num_samples(plain) == 3 && !haskey(plain.metadata, "sobolev_degenerate")
    ds = generate_data(
        pipe;
        sampler,
        n = 3,
        optimizer = HiGHS.Optimizer,
        max_label_error = 1e-3,
    )
    @test num_samples(ds) == 2
    @test ds.metadata["sobolev_degenerate"] == 1 && ds.metadata["sobolev_dropped"] == 1
    @test ds.inputs == [3.0 0.5; 1.0 1.5]
    @test ds.derivatives.J[:, :, 1] ≈ bound_lp_jacobian([3.0, 1.0]) atol = 1e-8
    @test ds.derivatives.J[:, :, 2] ≈ bound_lp_jacobian([0.5, 1.5]) atol = 1e-8
    @test_throws ArgumentError generate_data(
        pipe;
        sampler,
        n = 3,
        oracle = DiffOptSensitivities(HiGHS.Optimizer),
        max_label_error = 1e-3,
    )
end
