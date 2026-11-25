using JuMP, Test, L2OCore, Random
import HiGHS

@testset "guarantee checks" begin
    rng = Random.MersenneTwister(3)
    prob = ParametricProblem(lp_problem())
    sampler = BoxSampler([0.5, 1.0], [1.5, 3.0])
    P = sample_parameters(sampler, rng, 8)
    ds = solve_instances(prob, P; optimizer = HiGHS.Optimizer, dual = true)
    opt = Prediction(ds.primal, ds.dual)

    @test check(NoGuarantee(), prob, P, opt).passed
    @test check(Feasible(), prob, P, opt).passed
    @test check(EqualityFeasible(), prob, P, opt).passed
    @test check(BoundFeasible(), prob, P, opt).passed
    r = check(ValidDualBound(), prob, P, opt; reference = ds.objective)
    @test r.passed
    @test occursin("ValidDualBound: passed", sprint(show, r))
    @test_throws ArgumentError check(ValidDualBound(), prob, P, opt)
    @test_throws ArgumentError check(Feasible(), prob, P, Prediction(dual = ds.dual))

    # Weak duality on random instances: in this LP the parameters only enter
    # the right-hand side, so the optimal duals of instance j are dual feasible
    # for every instance k and must give a bound ≤ OPT(k).
    shuffled = Prediction(dual = ds.dual[:, circshift(1:8, 3)])
    r = check(ValidDualBound(), prob, P, shuffled; reference = ds.objective)
    @test r.passed
    bound, _ = dual_bound(default_evaluator(prob), shuffled.dual, P)
    @test all(bound .<= ds.objective .+ 1e-8)

    # Same claim for maximisation (bound ≥ OPT).
    pmax = ParametricProblem(lp_problem(; sense = MOI.MAX_SENSE))
    dsm = solve_instances(pmax, P; optimizer = HiGHS.Optimizer, dual = true)
    bm, _ = dual_bound(default_evaluator(pmax), dsm.dual[:, circshift(1:8, 3)], P)
    @test all(bm .>= dsm.objective .- 1e-8)

    # Random duals are (almost surely) dual infeasible: no bound is claimed.
    Yr = randn(rng, num_duals(prob), 8)
    bound, infeas = dual_bound(default_evaluator(prob), Yr, P)
    @test all(infeas .> 1e-6)
    @test all(==(-Inf), bound)
    @test !check(ValidDualBound(), prob, P, Prediction(dual = Yr); reference = ds.objective).passed

    # Perturbed primal violates equalities.
    Xbad = ds.primal .+ 0.1
    @test !check(EqualityFeasible(), prob, P, Prediction(primal = Xbad)).passed
    @test !check(Feasible(), prob, P, Prediction(primal = Xbad .+ 10)).passed
    @test !check(BoundFeasible(), prob, P, Prediction(primal = Xbad .+ 10)).passed
end
