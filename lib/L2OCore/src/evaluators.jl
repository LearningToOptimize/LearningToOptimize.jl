"""
    AbstractEvaluator

Batch evaluation backend for a [`ParametricProblem`](@ref). Evaluators turn
batches of parameters `P` (`num_parameters × B`), primal points `X`
(`num_decision_variables × B`) and dual points `Y` (`num_duals × B`) into objective
values, constraint violations and dual bounds.

Built-in backends:

  - [`MatrixEvaluator`](@ref): extracts `A`, `Q`, `c`, … once and evaluates with
    array operations (vectorised, differentiable by array AD, element-type
    generic). Used for [`LinearConic`](@ref) and [`ConvexQuadratic`](@ref)
    problems whose parameters enter affinely.
  - [`JuMPEvaluator`](@ref): per-instance fallback through `JuMP.value`; works
    for any expression JuMP can evaluate (including nonlinear) but is not
    meant to be differentiated.

**Extension point.** Kernel packages (e.g. an ExaModels/BatchNLPKernels
extension) add a backend by subtyping `AbstractEvaluator` and implementing

| function | required | returns |
|:--|:--|:--|
| [`batch_objective`](@ref)`(ev, X, P)` | yes | length-`B` vector |
| [`constraint_values`](@ref)`(ev, X, P)` | yes | `num_duals × B` |
| [`row_violation`](@ref)`(ev, X, P)` | yes | `num_duals × B`, `≥ 0` |
| [`equality_rows`](@ref)`(ev)` | yes | `BitVector` of length `num_duals` |
| [`cone_violation`](@ref)`(ev, X, P)` | if non-decomposable cones | `n_cones × B`, `≥ 0` |
| [`dual_bound`](@ref)`(ev, Y, P)` | for dual methods | `(bound, infeasibility)` vectors |

and, optionally, a method of [`default_evaluator`](@ref) for the problems it
should handle.
"""
abstract type AbstractEvaluator end

"""
    batch_objective(ev::AbstractEvaluator, X, P) -> AbstractVector

Objective value of each column of `X` (with parameters the matching column of
`P`), in the sense of the model (maximisation objectives are not negated).
"""
function batch_objective end

"""
    constraint_values(ev::AbstractEvaluator, X, P) -> AbstractMatrix

Value of every constraint function, stacked in [`dual_layout`](@ref) order
(`num_duals × B`).
"""
function constraint_values end

"""
    row_violation(ev::AbstractEvaluator, X, P) -> AbstractMatrix

Element-wise violation `max(l - f, 0) + max(f - u, 0)` of every row whose set is
a box (`EqualTo`, `GreaterThan`, `LessThan`, `Interval`, `Zeros`,
`Nonnegatives`, `Nonpositives`), and `0` for rows of non-decomposable cones
(see [`cone_violation`](@ref)). Shape `num_duals × B`.
"""
function row_violation end

"""
    cone_violation(ev::AbstractEvaluator, X, P) -> AbstractMatrix

Euclidean distance of each non-decomposable conic constraint (second-order,
PSD, exponential, …) to its set, `n_cones × B`. The fallback returns a `0 × B`
matrix.
"""
cone_violation(::AbstractEvaluator, X::AbstractMatrix, P::AbstractMatrix) =
    similar(X, float(eltype(X)), 0, size(X, 2))

"""
    equality_rows(ev::AbstractEvaluator) -> BitVector

Mask of rows (in [`dual_layout`](@ref) order) that are equality constraints.
"""
function equality_rows end

"""
    bound_rows(ev::AbstractEvaluator) -> BitVector

Mask of rows that are variable bounds (`VariableRef`-in-set constraints).
"""
function bound_rows end

"""
    dual_bound(ev::AbstractEvaluator, Y, P; atol = 1e-6) -> (bound, infeasibility)

Value of the Lagrangian dual function at each column of `Y` (signs as returned
by `JuMP.dual`) and the dual infeasibility of that column.

By weak duality, whenever `infeasibility[k] ≤ atol`, `bound[k]` is a lower
bound (upper bound for maximization) on the optimal value of instance `k`, up
to `infeasibility[k] * ‖x⋆‖₁` when the stationarity residual is nonzero (see
the Theory page). Columns that are not dual feasible get `bound = -Inf`
(`+Inf` for maximization).

Dual signs follow `JuMP.dual` (the MathOptInterface convention). For
quadratic objectives `sQ ⪰ 0` is required and verified; otherwise no bound is
returned.
"""
function dual_bound end

"""
    default_evaluator(problem::ParametricProblem; T = Float64) -> AbstractEvaluator

[`MatrixEvaluator`](@ref) when the problem can be put in matrix form (affine
constraints, at most quadratic objective, parameters entering affinely),
otherwise [`JuMPEvaluator`](@ref). The result is cached in the problem.
"""
function default_evaluator(problem::ParametricProblem; T::Type{<:Real} = Float64)
    key = Symbol(:default_evaluator_, T)
    return get!(problem.cache, key) do
        if problem.class isa Union{LinearConic,ConvexQuadratic} && has_matrix_form(problem)
            MatrixEvaluator(problem; T = T)
        else
            JuMPEvaluator(problem)
        end
    end
end

"""
    EvaluationReport

Per-instance quality metrics returned by [`evaluate`](@ref). Every field is a
length-`B` vector or `nothing` when it does not apply.

  - `objective`: primal objective of the prediction;
  - `max_violation`: largest violation over all constraints;
  - `max_equality_violation`, `max_inequality_violation`;
  - `dual_bound`, `dual_infeasibility`: see [`dual_bound`](@ref);
  - `gap`: `|objective - dual_bound| / max(1, |objective|)` when both a primal
    and a feasible dual prediction are available.
"""
struct EvaluationReport
    objective::Union{Nothing,AbstractVector}
    max_violation::Union{Nothing,AbstractVector}
    max_equality_violation::Union{Nothing,AbstractVector}
    max_inequality_violation::Union{Nothing,AbstractVector}
    dual_bound::Union{Nothing,AbstractVector}
    dual_infeasibility::Union{Nothing,AbstractVector}
    gap::Union{Nothing,AbstractVector}
end

function Base.show(io::IO, r::EvaluationReport)
    println(io, "EvaluationReport")
    for f in fieldnames(EvaluationReport)
        v = getfield(r, f)
        v === nothing && continue
        finite = filter(isfinite, collect(v))
        m = isempty(finite) ? NaN : sum(finite) / length(finite)
        println(
            io,
            "  ",
            rpad(string(f), 26),
            "mean = ",
            m,
            "  (",
            length(v),
            " instances)",
        )
    end
end

_colmax(M::AbstractMatrix) =
    size(M, 1) == 0 ? zeros(eltype(M), size(M, 2)) : vec(maximum(M; dims = 1))
