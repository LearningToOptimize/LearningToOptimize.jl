using JuMP, Test, L2OCore, Random
import HiGHS

# Largest violation per instance over the rows selected by `rows` (all rows
# and cones when `rows === nothing`).
function _maxviol(prob, P, X; rows = nothing)
    ev = default_evaluator(prob)
    rows === nothing && return evaluate(prob, P, Prediction(primal = X)).max_violation
    return vec(maximum(row_violation(ev, X, P)[rows(ev), :]; dims = 1))
end

# Whether every column of `Y` is dual feasible and its dual bound is on the
# correct side of the optimal value `opt` (≤ for minimization, ≥ for
# maximization), up to `atol` relative to max(1, |opt|).
function _bounds_optimum(prob, P, Y, opt; atol = 1e-6)
    bound, infeas = dual_bound(default_evaluator(prob), Y, P; atol)
    s = objective_sign(prob)
    excess = max.(s .* (bound .- opt), 0) ./ max.(1, abs.(opt))
    return all(infeas .<= atol) && all(excess .<= atol)
end

@testset "solver optima are feasible and their duals bound the optimum" begin
    rng = Random.MersenneTwister(3)
    prob = ParametricProblem(lp_problem())
    sampler = BoxSampler([0.5, 1.0], [1.5, 3.0])
    P = sample_parameters(sampler, rng, 8)
    ds = solve_instances(prob, P; optimizer = HiGHS.Optimizer, dual = true)

    @test all(_maxviol(prob, P, ds.primal) .<= 1e-6)
    @test all(_maxviol(prob, P, ds.primal; rows = equality_rows) .<= 1e-6)
    @test all(_maxviol(prob, P, ds.primal; rows = L2OCore.bound_rows) .<= 1e-6)
    @test _bounds_optimum(prob, P, ds.dual, ds.objective)

    # Weak duality on random instances: in this LP the parameters only enter
    # the right-hand side, so the optimal duals of instance j are dual feasible
    # for every instance k and must give a bound ≤ OPT(k).
    shuffled = ds.dual[:, circshift(1:8, 3)]
    @test _bounds_optimum(prob, P, shuffled, ds.objective)
    bound, _ = dual_bound(default_evaluator(prob), shuffled, P)
    @test all(bound .<= ds.objective .+ 1e-8)

    # The same for maximization (bound ≥ OPT).
    pmax = ParametricProblem(lp_problem(; sense = MOI.MAX_SENSE))
    dsm = solve_instances(pmax, P; optimizer = HiGHS.Optimizer, dual = true)
    bm, _ = dual_bound(default_evaluator(pmax), dsm.dual[:, circshift(1:8, 3)], P)
    @test all(bm .>= dsm.objective .- 1e-8)

    # Random duals are (almost surely) dual infeasible: no bound is returned.
    Yr = randn(rng, num_duals(prob), 8)
    bound, infeas = dual_bound(default_evaluator(prob), Yr, P)
    @test all(infeas .> 1e-6)
    @test all(==(-Inf), bound)
    @test !_bounds_optimum(prob, P, Yr, ds.objective)

    # A perturbed primal violates the equalities, then the bounds.
    Xbad = ds.primal .+ 0.1
    @test maximum(_maxviol(prob, P, Xbad; rows = equality_rows)) > 1e-6
    @test maximum(_maxviol(prob, P, Xbad .+ 10)) > 1e-6
    @test maximum(_maxviol(prob, P, Xbad .+ 10; rows = L2OCore.bound_rows)) > 1e-6
end

# Weak duality for a strictly convex QP: with sQ ≻ 0 every y with the
# right signs is dual feasible (r ∈ range(sQ) always), so *random* sign-feasible
# duals must give bounds ≤ OPT. Weak duality, no solver duals involved.
@testset "weak duality on a strictly convex QP with random duals" begin
    rng = Random.MersenneTwister(7)
    for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
        prob = ParametricProblem(qp_problem(; sense))
        ev = default_evaluator(prob)
        P = sample_parameters(BoxSampler([-1.0, -1.0], [1.0, 1.0]), rng, 20)
        ds = solve_instances(prob, P; optimizer = _CLARABEL)
        Y = randn(rng, num_duals(prob), 20)
        # project onto the sign constraints (dual cone of each box row)
        Y = ifelse.(ev.lower .== -Inf, min.(Y, 0), ifelse.(ev.upper .== Inf, max.(Y, 0), Y))
        bound, infeas = dual_bound(ev, Y, P)
        @test all(infeas .<= 1e-8)
        s = objective_sign(prob)
        @test all(s .* bound .<= s .* ds.objective .+ 1e-6)
        @test _bounds_optimum(prob, P, Y, ds.objective)
    end
end

# A non-convex objective mislabeled as ConvexQuadratic must never give a bound
# (inf_x L(x, y) = -∞ for every y).
@testset "no dual bound for a non-convex quadratic objective" begin
    m = Model()
    @variable(m, p in MOI.Parameter(1.0))
    @variable(m, -1 <= x[1:2] <= 1)
    @constraint(m, x[1] + x[2] == p)
    @objective(m, Min, x[1] * x[2])
    prob = ParametricProblem(m; class = ConvexQuadratic())
    bound, infeas =
        dual_bound(default_evaluator(prob), zeros(num_duals(prob), 2), ones(1, 2))
    @test all(==(-Inf), bound)
    @test all(isinf, infeas)
end

# Dual completion for bounded variables, Theorem 1 / Example 1 of Tanneau & Van
# Hentenryck (2024, arXiv:2402.03086). With finite bounds l ≤ x ≤ u, any y for
# the other constraints (in their dual cones) is completed by
# z_l = max(r, 0), z_u = min(r, 0), r = c - Aᵀy (MOI signs), giving a dual
# feasible point whose bound is ≤ OPT; with y = y* the bound equals OPT.
@testset "DLL bounded-variable completion gives valid bounds" begin
    rng = Random.MersenneTwister(8)
    prob = ParametricProblem(lp_problem())
    ev = default_evaluator(prob)
    bnd = L2OCore.bound_rows(ev)
    Ab = ev.A[bnd, :]
    var_of_row = [findfirst(!iszero, Ab[j, :]) for j = 1:size(Ab, 1)]
    is_lower = isfinite.(ev.lower[bnd])
    function complete_dll(y)  # y: duals of the non-bound rows
        Y = zeros(num_duals(prob))
        Y[.!bnd] .= y
        r = ev.c .- ev.A[.!bnd, :]' * y
        Y[findall(bnd)] .= ifelse.(is_lower, max.(r[var_of_row], 0), min.(r[var_of_row], 0))
        return Y
    end
    P = sample_parameters(BoxSampler([0.5, 1.0], [1.5, 3.0]), rng, 10)
    ds = solve_instances(prob, P; optimizer = HiGHS.Optimizer, dual = true)
    nb = count(.!bnd)
    yrand = randn(rng, nb, 10)
    yrand[ev.upper[.!bnd] .== Inf, :] .= abs.(yrand[ev.upper[.!bnd] .== Inf, :])
    yrand[ev.lower[.!bnd] .== -Inf, :] .= .-abs.(yrand[ev.lower[.!bnd] .== -Inf, :])
    Y = reduce(hcat, complete_dll(yrand[:, k]) for k = 1:10)
    @test _bounds_optimum(prob, P, Y, ds.objective)
    # optimal y recovers the optimal value (strongest bound, Theorem 2)
    Ystar = reduce(hcat, complete_dll(ds.dual[.!bnd, k]) for k = 1:10)
    bound, infeas = dual_bound(ev, Ystar, P)
    @test all(infeas .<= 1e-9)
    @test bound ≈ ds.objective atol = 1e-6
end
