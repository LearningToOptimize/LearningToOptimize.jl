using JuMP, Test, L2OCore

@testset "detect_problem_class" begin
    @test detect_problem_class(lp_problem()) isa LinearConic
    @test detect_problem_class(lp_problem(; sense = MOI.MAX_SENSE)) isa LinearConic
    @test detect_problem_class(qp_problem()) isa ConvexQuadratic
    @test detect_problem_class(socp_problem()) isa LinearConic

    m = Model()
    @variable(m, x[1:2])
    @objective(m, Min, x[1] * x[2])  # indefinite
    @test detect_problem_class(m) isa SmoothNonlinear
    @objective(m, Max, -x[1]^2 - x[2]^2)  # concave maximisation is convex
    @test detect_problem_class(m) isa ConvexQuadratic

    m = Model()
    @variable(m, x[1:2])
    @constraint(m, sin(x[1]) + x[2] <= 1)
    @test detect_problem_class(m) isa SmoothNonlinear

    m = Model()
    @variable(m, x, Int)
    @test detect_problem_class(m) isa MixedInteger

    # parameter × variable products do not change the class
    m = Model()
    @variable(m, p in MOI.Parameter(1.0))
    @variable(m, x >= 0)
    @constraint(m, p * x >= 1)
    @objective(m, Min, x)
    @test detect_problem_class(m) isa LinearConic
end

@testset "ParametricProblem layouts" begin
    prob = ParametricProblem(lp_problem())
    @test num_parameters(prob) == 2
    @test num_decision_variables(prob) == 4
    # 3 rows + 4 lower bounds + 4 upper bounds
    @test num_duals(prob) == 11
    @test length(dual_layout(prob)) == length(constraint_refs(prob))
    @test reduce(vcat, collect.(dual_layout(prob))) == 1:num_duals(prob)
    @test parameter_values(prob) == [1.0, 2.0]
    @test problem_class(prob) isa LinearConic
    @test occursin("ParametricProblem{LinearConic}", sprint(show, prob))

    soc = ParametricProblem(socp_problem())
    @test num_duals(soc) == 1 + 3

    # explicit class overrides detection
    @test problem_class(ParametricProblem(lp_problem(); class = SmoothNonlinear())) isa SmoothNonlinear
end

@testset "AbstractParametricProblem / objective_sign" begin
    @test ParametricProblem(lp_problem()) isa AbstractParametricProblem
    @test objective_sign(ParametricProblem(lp_problem())) == 1
    @test objective_sign(ParametricProblem(lp_problem(; sense = MOI.MAX_SENSE))) == -1
end
