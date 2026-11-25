using JuMP, Test, L2OCore, Random
import HiGHS, Clarabel

struct ToyPrimal <: AbstractPrimalMethod end
struct ToyDual <: AbstractDualMethod end
struct ToyPrimalDual <: AbstractPrimalDualMethod end
struct Unsupported <: AbstractPrimalMethod end
for M in (ToyPrimal, ToyDual, ToyPrimalDual)
    @eval L2OCore.supports(::$M, ::L2OCore.AbstractConvexClass) = true
end

@testset "supports is explicit" begin
    prob = ParametricProblem(lp_problem())
    @test !supports(Unsupported(), LinearConic())
    err = try
        L2OPipeline(prob, Unsupported())
    catch e
        e
    end
    @test err isa UnsupportedProblemError
    msg = sprint(showerror, err)
    @test occursin("Unsupported", msg) && occursin("LinearConic", msg)
    @test guarantees(ToyPrimal()) == (NoGuarantee(),)
    @test method_family(ToyPrimal()) == PrimalFamily()
    @test method_family(ToyDual()) == DualFamily()
    @test method_family(ToyPrimalDual()) == PrimalDualFamily()
end

@testset "family fallbacks: $(nameof(M))" for (M, np, nd) in (
    (ToyPrimal, 4, 0),
    (ToyDual, 0, 11),
    (ToyPrimalDual, 4, 11),
)
    prob = ParametricProblem(lp_problem())
    pipe = L2OPipeline(prob, M())
    @test occursin("L2OPipeline", sprint(show, pipe))
    @test input_dimension(pipe) == 2
    @test output_dimension(pipe) == np + nd
    ds = generate_data(
        pipe;
        sampler = BoxSampler([0.5, 1.0], [1.5, 3.0]),
        n = 4,
        rng = Random.MersenneTwister(4),
        optimizer = HiGHS.Optimizer,
    )
    @test num_samples(ds) == 4
    @test (ds.primal !== nothing) == (np > 0)
    @test (ds.dual !== nothing) == (nd > 0)
    @test occursin("4 samples", sprint(show, ds))
    # A perfect "proxy" that returns the labels:
    out = vcat((x for x in (ds.primal, ds.dual) if x !== nothing)...)
    pred = complete(pipe, ds.inputs, out)
    @test pred.primal == ds.primal
    @test pred.dual == ds.dual
    @test loss(pipe, ds.inputs, out; targets = ds) ≈ 0 atol = 1e-12
    @test loss(pipe, ds.inputs, out .+ 1; targets = ds) > 0
    @test_throws ArgumentError loss(pipe, ds.inputs, out)
    @test_throws DimensionMismatch complete(pipe, ds.inputs, out[1:(end-1), :])
    rep = evaluate(pipe, ds.inputs, pred)
    @test (rep.objective !== nothing) == (np > 0)
    @test (rep.dual_bound !== nothing) == (nd > 0)
    @test predict_batch(x -> 2x, ones(2, 3)) == fill(2.0, 2, 3)
end

@testset "datasets and samplers" begin
    rng = Random.MersenneTwister(5)
    prob = ParametricProblem(lp_problem())
    s = ScaledSampler(prob; low = 0.9, high = 1.1)
    P = sample_parameters(s, rng, 100)
    @test size(P) == (2, 100)
    @test all(0.9 .* [1, 2] .<= minimum(P; dims = 2)[:]) &&
          all(maximum(P; dims = 2)[:] .<= 1.1 .* [1, 2])
    @test size(sample_parameters((r, n) -> zeros(2, n), rng, 3)) == (2, 3)
    @test_throws ArgumentError BoxSampler([1.0], [0.0])
    @test_throws DimensionMismatch BoxSampler([1.0], [0.0, 1.0])
    ds = L2ODataset(
        P;
        primal = rand(rng, 4, 100),
        derivatives = PullbackLabels(rand(rng, 4, 100), rand(rng, 2, 100)),
    )
    sub = ds[1:10]
    @test num_samples(sub) == 10
    @test size(sub.derivatives.seeds) == (4, 10)
    jd = L2ODataset(P; derivatives = JacobianLabels(rand(rng, 4, 2, 100)))
    @test size(jd[[1, 3]].derivatives.J) == (4, 2, 2)
    fd =
        L2ODataset(P; derivatives = PushforwardLabels(rand(rng, 2, 100), rand(rng, 4, 100)))
    @test size(fd[[2, 5, 7]].derivatives.directions) == (2, 3)
    @test size(fd[[2, 5, 7]].derivatives.values) == (4, 3)
    @test_throws DimensionMismatch L2ODataset(P; primal = rand(4, 3))
end

struct AddConstant{M<:AbstractL2OMethod} <: AbstractMethodWrapper
    inner::M
    c::Float64
end
L2OCore.extra_loss(w::AddConstant, pipe, inputs, output; proxy = nothing, kwargs...) =
    proxy === nothing ? w.c : w.c + sum(proxy(inputs))

@testset "method wrappers (hook for add-on loss terms)" begin
    prob = ParametricProblem(lp_problem())
    w = AddConstant(ToyPrimal(), 2.0)
    @test inner_method(w) === w.inner
    @test supports(w, LinearConic()) && method_family(w) == PrimalFamily()
    @test guarantees(w) == guarantees(ToyPrimal())
    pipe = L2OPipeline(prob, w)
    @test pipe.decomposition == decompose(prob, ToyPrimal())
    ds = generate_data(
        pipe;
        sampler = BoxSampler([0.5, 1.0], [1.5, 3.0]),
        n = 3,
        rng = Random.MersenneTwister(6),
        optimizer = HiGHS.Optimizer,
    )
    @test ds.dual === nothing
    @test complete(pipe, ds.inputs, ds.primal).primal == ds.primal
    @test loss(pipe, ds.inputs, ds.primal; targets = ds) ≈ 2.0
    @test loss(
        pipe,
        ds.inputs,
        ds.primal;
        targets = ds,
        proxy = x -> zeros(1, size(x, 2)),
    ) ≈ 2.0
    ds2 = with_derivatives(ds, PullbackLabels(ones(4, 3), ones(2, 3)))
    @test ds2.derivatives isa PullbackLabels && ds2.primal == ds.primal
end

struct NoExtra <: AbstractMethodWrapper
    inner::ToyPrimal
end
@testset "wrapper without extra_loss errors clearly" begin
    pipe = L2OPipeline(ParametricProblem(lp_problem()), NoExtra(ToyPrimal()))
    t = (primal = zeros(4, 1),)
    @test_throws ArgumentError loss(pipe, zeros(2, 1), zeros(4, 1); targets = t)
end

# Method packages typically type only the method argument; this must not be
# ambiguous with the family or wrapper fallbacks.
struct Custom <: AbstractPrimalMethod end
L2OCore.supports(::Custom, ::LinearConic) = true
L2OCore.decompose(problem, ::Custom) = FullSpaceDecomposition(true, false, 4, 0)
L2OCore.complete(::Custom, pipe, inputs, output) = Prediction(primal = 2 .* output)
L2OCore.loss(::Custom, pipe, inputs, output; kwargs...) = sum(output)
L2OCore.generate_data(::Custom, pipe; n, kwargs...) = L2ODataset(zeros(2, n))
@testset "no ambiguities for typical method definitions" begin
    pipe = L2OPipeline(ParametricProblem(lp_problem()), Custom())
    @test complete(pipe, zeros(2, 1), ones(4, 1)).primal == fill(2.0, 4, 1)
    @test loss(pipe, zeros(2, 1), ones(4, 1)) == 4
    @test num_samples(generate_data(pipe; n = 3)) == 3
    w = AddConstant(Custom(), 1.0)
    pw = L2OPipeline(ParametricProblem(lp_problem()), w)
    @test loss(pw, zeros(2, 1), ones(4, 1)) == 5
    @test complete(pw, zeros(2, 1), ones(4, 1)).primal == fill(2.0, 4, 1)
end

@testset "SupervisedMSE (value supervision baseline)" begin
    prob = ParametricProblem(qp_problem())
    m = SupervisedMSE()
    @test supports(m, ConvexQuadratic()) && supports(m, SmoothNonlinear())
    @test guarantees(m) == (NoGuarantee(),)
    pipe = L2OPipeline(prob, m)
    @test output_dimension(pipe) == num_decision_variables(prob)
    ds = generate_data(
        pipe;
        sampler = BoxSampler([0.5, -1.0], [1.5, 0.0]),
        n = 5,
        rng = Random.MersenneTwister(11),
        optimizer = Clarabel.Optimizer,
    )
    @test ds.primal !== nothing && ds.dual === nothing
    @test loss(pipe, ds.inputs, ds.primal; targets = ds) ≈ 0 atol = 1e-14
    # MSE definition: mean over all n × N entries.
    @test loss(pipe, ds.inputs, ds.primal .+ 2; targets = ds) ≈ 4
end
