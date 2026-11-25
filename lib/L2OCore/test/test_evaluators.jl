using JuMP, Test, L2OCore, Random
import HiGHS, Clarabel

@testset "MatrixEvaluator == JuMPEvaluator on random points" begin
    rng = Random.MersenneTwister(1)
    for build in
        (lp_problem, qp_problem, socp_problem, () -> lp_problem(; sense = MOI.MAX_SENSE))
        prob = ParametricProblem(build())
        me = MatrixEvaluator(prob)
        je = JuMPEvaluator(prob)
        @test default_evaluator(prob) isa MatrixEvaluator
        X = randn(rng, num_decision_variables(prob), 5)
        P = randn(rng, num_parameters(prob), 5)
        @test batch_objective(me, X, P) ≈ batch_objective(je, X, P)
        @test constraint_values(me, X, P) ≈ constraint_values(je, X, P)
        @test row_violation(me, X, P) ≈ row_violation(je, X, P)
        @test cone_violation(me, X, P) ≈ cone_violation(je, X, P)
        @test all(>=(0), row_violation(me, X, P))
        @test equality_rows(me) == equality_rows(je)
        @test L2OCore.bound_rows(me) == L2OCore.bound_rows(je)
    end
end

@testset "Float32 evaluation" begin
    prob = ParametricProblem(qp_problem())
    me = MatrixEvaluator(prob; T = Float32)
    X = rand(Float32, num_decision_variables(prob), 3)
    P = rand(Float32, num_parameters(prob), 3)
    @test eltype(batch_objective(me, X, P)) == Float32
    @test eltype(row_violation(me, X, P)) == Float32
    @test batch_objective(me, X, P) ≈
          batch_objective(MatrixEvaluator(prob), Float64.(X), Float64.(P)) rtol = 1e-5
end

@testset "nonlinear problems fall back to JuMPEvaluator" begin
    m = Model()
    @variable(m, p in MOI.Parameter(1.0))
    @variable(m, x)
    @constraint(m, exp(x) <= p + 2)
    @objective(m, Min, (x - 1)^2)
    prob = ParametricProblem(m)
    @test default_evaluator(prob) isa JuMPEvaluator
    @test_throws ArgumentError MatrixEvaluator(prob)
    ev = default_evaluator(prob)
    X = reshape([0.0, 2.0], 1, 2)
    P = reshape([1.0, 1.0], 1, 2)
    @test batch_objective(ev, X, P) ≈ [1.0, 1.0]
    @test vec(row_violation(ev, X, P)) ≈ [0.0, exp(2) - 3]
    @test_throws ArgumentError dual_bound(ev, zeros(1, 2), P)
end

# Guarantee `StrongDualityAtOptimum` (LP/QP, Slater holds): the dual bound of the
# solver's optimal duals equals the solver's optimal value.
@testset "solver optimum: feasibility and strong duality" begin
    rng = Random.MersenneTwister(2)
    cases = (
        (lp_problem, HiGHS.Optimizer, BoxSampler([0.5, 1.0], [1.5, 3.0])),
        (
            () -> lp_problem(; sense = MOI.MAX_SENSE),
            HiGHS.Optimizer,
            BoxSampler([0.5, 1.0], [1.5, 3.0]),
        ),
        (qp_problem, _CLARABEL, BoxSampler([-1.0, -1.0], [1.0, 1.0])),
        (socp_problem, _CLARABEL, BoxSampler([0.5], [2.0])),
        (
            () -> qp_problem(; sense = MOI.MAX_SENSE),
            _CLARABEL,
            BoxSampler([-1.0, -1.0], [1.0, 1.0]),
        ),
        (singular_qp_problem, _CLARABEL, BoxSampler([0.5], [2.5])),
    )
    for (build, opt, sampler) in cases
        prob = ParametricProblem(build())
        P = sample_parameters(sampler, rng, 6)
        ds = solve_instances(prob, P; optimizer = opt, dual = true)
        @test all(==(MOI.OPTIMAL), ds.status)
        ev = default_evaluator(prob)
        @test batch_objective(ev, ds.primal, P) ≈ ds.objective atol = 1e-6
        rep = evaluate(prob, P, Prediction(ds.primal, ds.dual))
        @test maximum(rep.max_violation) <= 1e-6
        @test maximum(rep.dual_infeasibility) <= 1e-6
        @test rep.dual_bound ≈ ds.objective atol = 1e-5
        @test maximum(rep.gap) <= 1e-5
        @test occursin("EvaluationReport", sprint(show, rep))
    end
end
