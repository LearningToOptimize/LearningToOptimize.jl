# Small parametric problems with known solution maps.

using JuMP
import MathOptInterface as MOI

"""
Strictly convex QP with an equality constraint; bounds inactive near the
nominal parameters. The solution map is affine:
`x*(p) = first 2 rows of K \\ [2p₁; 2p₂; p₃ + 1]` with `K = [Q 1; 1ᵀ 0]`.
"""
function eq_qp(p0 = [1.0, 2.0, 0.5])
    m = Model()
    @variable(m, p[1:3] in MOI.Parameter.(p0))
    @variable(m, x[1:2] >= -10)
    @constraint(m, x[1] + x[2] == p[3] + 1)
    @objective(m, Min, (x[1] - p[1])^2 + (x[2] - p[2])^2 + 0.1 * x[1] * x[2])
    return m
end
const EQ_QP_K = [2.0 0.1 1.0; 0.1 2.0 1.0; 1.0 1.0 0.0]
eq_qp_jacobian() = (EQ_QP_K\[2.0 0 0; 0 2.0 0; 0 0 1.0])[1:2, :]
eq_qp_solution(p) = (EQ_QP_K\[2p[1]; 2p[2]; p[3]+1])[1:2]

"""
LP with a parameter-dependent right-hand side and upper bound:
`min x₁ + 2x₂  s.t.  x₁ + x₂ = p₁, 0 ≤ x₁ ≤ p₂, x₂ ≥ 0`.
For `p₁ > p₂`: `x* = (p₂, p₁ - p₂)`; for `p₁ < p₂`: `x* = (p₁, 0)`;
degenerate (non-differentiable) at `p₁ = p₂`.
"""
function bound_lp(p0 = [2.0, 1.0])
    m = Model()
    @variable(m, p[1:2] in MOI.Parameter.(p0))
    @variable(m, x[1:2] >= 0)
    @constraint(m, x[1] + x[2] == p[1])
    @constraint(m, x[1] <= p[2])
    @objective(m, Min, x[1] + 2x[2])
    return m
end
bound_lp_jacobian(p) = p[1] > p[2] ? [0.0 1.0; 1.0 -1.0] : [1.0 0.0; 0.0 0.0]

"""
Projection onto the unit disk (smooth NLP with an active nonlinear constraint
when `‖p‖ > 1`): `x*(p) = p / ‖p‖`, `Dx* = (I - x xᵀ) / ‖p‖`.
"""
function disk_projection(p0 = [1.5, 0.5])
    m = Model()
    @variable(m, p[1:2] in MOI.Parameter.(p0))
    @variable(m, x[1:2], start = 0.5)
    @constraint(m, x[1]^2 + x[2]^2 <= 1)
    @objective(m, Min, (x[1] - p[1])^2 + (x[2] - p[2])^2)
    return m
end
function disk_jacobian(p)
    r = sqrt(sum(abs2, p))
    x = p ./ r
    return ([1.0 0; 0 1.0] .- x * x') ./ r
end
