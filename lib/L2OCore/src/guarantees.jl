"""
    supports(method::AbstractL2OMethod, class::AbstractProblemClass) -> Bool

Whether `method` accepts problems of `class` **with all of its
[`guarantees`](@ref) intact**. The fallback is `false`: a method must opt in
explicitly, class by class, e.g.

```julia
L2OCore.supports(::MyMethod, ::L2OCore.AbstractConvexClass) = true
```

There is deliberately no silent fallback that would run a method on a class
where its guarantees do not hold; see [`check_supports`](@ref).
"""
supports(::AbstractL2OMethod, ::AbstractProblemClass) = false

"""
    UnsupportedProblemError(method, class)

Thrown by [`check_supports`](@ref) when a method is used on a problem class it
does not support.
"""
struct UnsupportedProblemError <: Exception
    method::AbstractL2OMethod
    class::AbstractProblemClass
end

function Base.showerror(io::IO, e::UnsupportedProblemError)
    M = nameof(typeof(e.method))
    print(
        io,
        "UnsupportedProblemError: method `$M` does not support problems of class ",
        "`$(nameof(typeof(e.class)))`. Its guarantees $(map(g -> nameof(typeof(g)), guarantees(e.method))) ",
        "are only established for classes where `supports(::$M, class)` is true. ",
        "If the problem was mis-detected, pass `class = ...` to `ParametricProblem`; ",
        "otherwise choose a method that supports this class.",
    )
end

"""
    check_supports(problem::ParametricProblem, method::AbstractL2OMethod)

Throw an [`UnsupportedProblemError`](@ref) unless
`supports(method, problem_class(problem))`. Called by [`L2OPipeline`](@ref)
before anything else.
"""
function check_supports(problem::ParametricProblem, method::AbstractL2OMethod)
    supports(method, problem_class(problem)) ||
        throw(UnsupportedProblemError(method, problem_class(problem)))
    return nothing
end

"""
    AbstractGuarantee

A named mathematical property of the *completed* predictions of a method
(the output of [`complete`](@ref)). Each guarantee has a numerical
[`check`](@ref), used by the test suites of method packages. The names are
the anchors to which formal proofs can later be attached.

New guarantees are added by subtyping `AbstractGuarantee` and implementing
`check`.
"""
abstract type AbstractGuarantee end

"""
    NoGuarantee()

The method makes no claim about its predictions (e.g. plain penalty methods).
`check` always passes.
"""
struct NoGuarantee <: AbstractGuarantee end

"""
    Feasible()

Every constraint (equalities, inequalities, cones, variable bounds) is
satisfied to tolerance by the primal prediction, for every instance.
"""
struct Feasible <: AbstractGuarantee end

"""
    EqualityFeasible()

Every equality constraint is satisfied to tolerance by the primal prediction
(e.g. DC3 equality completion, solving for dependent variables).
"""
struct EqualityFeasible <: AbstractGuarantee end

"""
    BoundFeasible()

Every variable bound is satisfied by the primal prediction (e.g. sigmoid /
clamping output layers).
"""
struct BoundFeasible <: AbstractGuarantee end

"""
    ValidDualBound()

The dual prediction is dual feasible, so its Lagrangian dual value is a valid
bound on the optimal value: `dual_bound ≤ OPT` for minimisation (`≥` for
maximisation). This is weak duality; it needs no convexity, only dual
feasibility, which methods such as DLL ensure by construction.
"""
struct ValidDualBound <: AbstractGuarantee end

"""
    guarantees(method::AbstractL2OMethod) -> Tuple{Vararg{AbstractGuarantee}}

Guarantees that hold for the output of `complete(method, …)` on every problem
class the method [`supports`](@ref). Fallback: `(NoGuarantee(),)`.
"""
guarantees(::AbstractL2OMethod) = (NoGuarantee(),)

"""
    CheckResult

Outcome of [`check`](@ref): `guarantee`, `passed::Bool`, `worst` (largest
per-instance error), `errors` (per-instance error, `≤ atol` means satisfied).
"""
struct CheckResult{G<:AbstractGuarantee,T}
    guarantee::G
    passed::Bool
    worst::T
    errors::Vector{T}
end

Base.show(io::IO, r::CheckResult) = print(
    io,
    nameof(typeof(r.guarantee)),
    r.passed ? ": passed" : ": FAILED",
    " (worst error ",
    r.worst,
    ", ",
    length(r.errors),
    " instances)",
)

function _result(g, errors::AbstractVector, atol)
    e = collect(float.(errors))
    worst = isempty(e) ? zero(eltype(e)) : maximum(e)
    return CheckResult(g, worst <= atol, worst, e)
end

"""
    check(guarantee, problem, inputs, prediction; atol = 1e-6,
          evaluator = default_evaluator(problem), reference = nothing) -> CheckResult

Numerically verify `guarantee` for a batch of completed predictions.
`inputs` is `num_parameters × B`; `prediction` a [`Prediction`](@ref).
[`ValidDualBound`](@ref) needs `reference`, the optimal objective values of the
instances (e.g. from [`solve_instances`](@ref)).
"""
function check end

function check(g::NoGuarantee, problem::ParametricProblem, inputs, prediction; atol::Real = 1e-6, kwargs...)
    return _result(g, zeros(size(_asbatch(inputs), 2)), atol)
end

function _primal(prediction)
    prediction.primal === nothing &&
        throw(ArgumentError("this guarantee needs a primal prediction"))
    return _asbatch(prediction.primal)
end

function check(
    g::Feasible,
    problem::ParametricProblem,
    inputs,
    prediction;
    atol::Real = 1e-6,
    evaluator::AbstractEvaluator = default_evaluator(problem),
    kwargs...,
)
    X, P = _primal(prediction), _asbatch(inputs)
    e = max.(_colmax(row_violation(evaluator, X, P)), _colmax(cone_violation(evaluator, X, P)::AbstractMatrix))
    return _result(g, e, atol)
end

function check(
    g::EqualityFeasible,
    problem::ParametricProblem,
    inputs,
    prediction;
    atol::Real = 1e-6,
    evaluator::AbstractEvaluator = default_evaluator(problem),
    kwargs...,
)
    X, P = _primal(prediction), _asbatch(inputs)
    V = row_violation(evaluator, X, P)[equality_rows(evaluator), :]
    return _result(g, _colmax(V), atol)
end

function check(
    g::BoundFeasible,
    problem::ParametricProblem,
    inputs,
    prediction;
    atol::Real = 1e-6,
    evaluator::AbstractEvaluator = default_evaluator(problem),
    kwargs...,
)
    X, P = _primal(prediction), _asbatch(inputs)
    V = row_violation(evaluator, X, P)[bound_rows(evaluator), :]
    return _result(g, _colmax(V), atol)
end

function check(
    g::ValidDualBound,
    problem::ParametricProblem,
    inputs,
    prediction;
    atol::Real = 1e-6,
    evaluator::AbstractEvaluator = default_evaluator(problem),
    reference = nothing,
    kwargs...,
)
    prediction.dual === nothing && throw(ArgumentError("ValidDualBound needs a dual prediction"))
    reference === nothing && throw(
        ArgumentError("ValidDualBound needs `reference`: the optimal objective values"),
    )
    Y, P = _asbatch(prediction.dual), _asbatch(inputs)
    bound, infeas = dual_bound(evaluator, Y, P; atol = atol)
    s = JuMP.objective_sense(problem.model) == MOI.MAX_SENSE ? -1 : 1
    # error = dual infeasibility, or bound exceeding the optimum (scaled).
    excess = max.(s .* (bound .- reference), 0) ./ max.(1, abs.(reference))
    e = max.(infeas, excess)
    return _result(g, e, atol)
end
