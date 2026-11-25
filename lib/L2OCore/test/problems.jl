# Small parametric test problems shared by the test files.

using JuMP
import MathOptInterface as MOI
import Clarabel

const _CLARABEL = optimizer_with_attributes(
    Clarabel.Optimizer,
    "verbose" => false,
    "tol_gap_abs" => 1e-10,
    "tol_gap_rel" => 1e-10,
    "tol_feas" => 1e-10,
)

"""
Parametric LP (min-cost flow on a tiny network): demand parameters `d` enter
the right-hand side, capacities are variable bounds.
"""
function lp_problem(; sense = MOI.MIN_SENSE)
    model = Model()
    @variable(model, d[1:2] in MOI.Parameter.([1.0, 2.0]))
    @variable(model, 0 <= g[1:3] <= 4)
    @variable(model, 0 <= f <= 2.5)
    @constraint(model, bal1, g[1] + g[2] - f == d[1])
    @constraint(model, bal2, g[3] + f == d[2])
    @constraint(model, lim, g[1] + g[3] <= 5)
    cost = [1.0, 3.0, 2.0]
    if sense == MOI.MIN_SENSE
        @objective(model, Min, sum(cost .* g) + 0.1f)
    else
        @objective(model, Max, -sum(cost .* g) - 0.1f)
    end
    return model
end

"""
Parametric strictly convex QP with a parameter-dependent linear cost term
(`p × x` products in the objective) and an interval constraint.
"""
function qp_problem(; sense = MOI.MIN_SENSE)
    model = Model()
    @variable(model, p[1:2] in MOI.Parameter.([1.0, -0.5]))
    @variable(model, x[1:3] >= -2)
    @constraint(model, sum(x) == 1 + p[1])
    @constraint(model, -1 <= x[1] - x[2] <= 1)
    @constraint(model, x[3] <= 3)
    f = x[1]^2 + 2x[2]^2 + x[3]^2 + x[1] * x[2] + p[2] * x[3] - x[1]
    if sense == MOI.MIN_SENSE
        @objective(model, Min, f)
    else
        @objective(model, Max, -f)  # concave maximization: same optimizer
    end
    return model
end

"""
Second-order-cone problem (LinearConic with a non-decomposable cone).
"""
function socp_problem()
    model = Model()
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, t)
    @variable(model, x[1:2])
    @constraint(model, x[1] + x[2] == p)
    @constraint(model, [t; x] in SecondOrderCone())
    @objective(model, Min, t)
    return model
end

"""
Convex QP with a *singular* PSD Hessian (only x[1] is quadratic), so the dual
function needs the pseudo-inverse and the range condition r ∈ range(Q).
"""
function singular_qp_problem()
    model = Model()
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, 0 <= x[1:2] <= 3)
    @constraint(model, x[1] + x[2] >= p)
    @objective(model, Min, x[1]^2 + 2x[2] - x[1])
    return model
end
