using JuMP, Test, L2OCore, Random, LinearAlgebra
import HiGHS

# Largest violation of the variable bounds over a batch.
function _bound_violation(prob, P, X)
    ev = default_evaluator(prob)
    return maximum(row_violation(ev, X, P)[L2OCore.bound_rows(ev), :])
end

@testset "BoundedOutput" begin
    rng = Random.MersenneTwister(11)
    @test_throws ArgumentError BoundedOutput(ToyPrimal(); mode = :tanh)
    @test_throws ArgumentError L2OPipeline(
        ParametricProblem(lp_problem()),
        BoundedOutput(ToyDual()),
    )

    # Any raw output, however large, completes to
    # a point within the variable bounds (two-sided, one-sided and free bounds).
    for build in (lp_problem, qp_problem, socp_problem), mode in (:smooth, :clamp)
        prob = ParametricProblem(build())
        pipe = L2OPipeline(prob, BoundedOutput(ToyPrimal(); mode))
        n = num_decision_variables(prob)
        @test output_dimension(pipe) == n
        Z = 50 .* randn(rng, n, 16)
        P = randn(rng, num_parameters(prob), 16)
        pred = complete(pipe, P, Z)
        @test _bound_violation(prob, P, pred.primal) <= 0
        @test all(isfinite, pred.primal)
    end

    prob = ParametricProblem(lp_problem())
    d = L2OPipeline(prob, BoundedOutput(ToyPrimal())).decomposition
    l, u = d.lower, d.upper
    @test l == zeros(4) && u == [4, 4, 4, 2.5]

    # Clamp mode is the Euclidean projection onto
    # the box, so it leaves points inside unchanged and is nonexpansive.
    Zin = l .+ (u .- l) .* rand(rng, 4, 8)
    @test bound_transform(d, Zin, :clamp) == Zin
    A, B = 10 .* randn(rng, 4, 8), 10 .* randn(rng, 4, 8)
    CA, CB = bound_transform(d, A, :clamp), bound_transform(d, B, :clamp)
    @test all(norm(CA[:, k] - CB[:, k]) <= norm(A[:, k] - B[:, k]) + 1e-12 for k = 1:8)

    # The smooth map is strictly increasing in
    # each coordinate (no dead gradients) on a range of inputs.
    h = 1e-6
    for z = -5.0:0.5:5.0
        Zp = bound_transform(d, fill(z + h, 4, 1), :smooth)
        Zm = bound_transform(d, fill(z - h, 4, 1), :smooth)
        @test all((Zp .- Zm) ./ 2h .> 0)
    end
    qp = ParametricProblem(qp_problem())   # x ≥ -2 only: softplus branch
    dq = L2OPipeline(qp, BoundedOutput(ToyPrimal())).decomposition
    @test bound_transform(dq, fill(-1000.0, 3, 1), :smooth) ≈ fill(-2.0, 3, 1)
    @test bound_transform(dq, fill(1000.0, 3, 1), :smooth) ≈ fill(1000.0 - 2.0, 3, 1)

    # Losses see the bounded output: with the supervised fallback, the loss of
    # a raw output equals the inner loss of its bounded image.
    pipe = L2OPipeline(prob, BoundedOutput(ToyPrimal(); mode = :clamp))
    inner = L2OPipeline(prob, ToyPrimal())
    t = (primal = ones(4, 8),)
    @test loss(pipe, zeros(2, 8), A; targets = t) ≈
          loss(inner, zeros(2, 8), bound_transform(d, A, :clamp); targets = t)
    ds = generate_data(
        pipe;
        sampler = BoxSampler([0.5, 1.0], [1.5, 3.0]),
        n = 2,
        rng,
        optimizer = HiGHS.Optimizer,
    )
    @test ds.primal !== nothing
end

@testset "ProjectionRepair" begin
    rng = Random.MersenneTwister(12)
    @test !supports(ProjectionRepair(ToyPrimal(), _CLARABEL), SmoothNonlinear())
    @test !supports(ProjectionRepair(ToyPrimal(), _CLARABEL), MixedInteger())
    @test supports(ProjectionRepair(ToyPrimal(), _CLARABEL), LinearConic())

    for (build, sampler) in (
        (lp_problem, BoxSampler([0.5, 1.0], [1.5, 3.0])),
        (qp_problem, BoxSampler([-1.0, -1.0], [1.0, 1.0])),
        (socp_problem, BoxSampler([0.5], [2.0])),
    )
        prob = ParametricProblem(build())
        pipe = L2OPipeline(prob, ProjectionRepair(ToyPrimal(), _CLARABEL))
        P = sample_parameters(sampler, rng, 6)
        ref = solve_instances(prob, P; optimizer = _CLARABEL)
        Xhat = ref.primal .+ randn(rng, size(ref.primal))
        pred = complete(pipe, P, Xhat)
        # The repaired prediction is feasible.
        @test all(evaluate(prob, P, pred).max_violation .<= 1e-6)
        # The projection is nonexpansive: closer (or equal) to the optimum
        @test all(
            norm(pred.primal[:, k] - ref.primal[:, k]) <=
            norm(Xhat[:, k] - ref.primal[:, k]) + 1e-6 for k = 1:6
        )
        # Feasible points are left unchanged, to
        # solver accuracy (an interior-point method stops ~sqrt(μ) ≈ 1e-5 away
        # from active bounds at gap tolerance 1e-10).
        @test complete(pipe, P, ref.primal).primal ≈ ref.primal atol = 1e-4
        # training is unchanged: loss is the inner (supervised) loss
        @test loss(pipe, P, ref.primal; targets = ref) ≈ 0 atol = 1e-12
    end
    # the user's model is not modified
    prob = ParametricProblem(lp_problem())
    obj = objective_function(prob.model)
    complete(
        L2OPipeline(prob, ProjectionRepair(ToyPrimal(), _CLARABEL)),
        [1.0, 2.0],
        zeros(4),
    )
    @test isequal(objective_function(prob.model), obj)
end
