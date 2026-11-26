# Method-agnostic repair layers, implemented as method wrappers.

function _variable_bounds(problem::ParametricProblem)
    vars = decision_variables(problem)
    l = [
        JuMP.is_fixed(v) ? JuMP.fix_value(v) :
        JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf for v in vars
    ]
    u = [
        JuMP.is_fixed(v) ? JuMP.fix_value(v) :
        JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf for v in vars
    ]
    return float.(l), float.(u)
end

function _primal_rows(d::FullSpaceDecomposition, name)
    d.primal || throw(ArgumentError("$name needs a method that predicts primal variables"))
    return d.nprimal
end
_primal_rows(d::AbstractDecomposition, name) = throw(
    ArgumentError(
        "$name wraps methods whose raw output starts with all primal variables " *
        "(FullSpaceDecomposition); got $(nameof(typeof(d)))",
    ),
)

"""
    BoundedOutput(inner::AbstractL2OMethod; mode = :smooth)

Wrapper that maps the primal part of the raw proxy output into the variable
bounds ``l ≤ x ≤ u`` before the inner method sees it, both in [`loss`](@ref)
and in [`complete`](@ref) — a bound-preserving output layer.

With `mode = :smooth` each coordinate uses

```math
x_j = \\begin{cases}
l_j + (u_j - l_j)\\,σ(z_j) & l_j, u_j \\text{ finite},\\\\
l_j + \\operatorname{softplus}(z_j) & \\text{only } l_j \\text{ finite},\\\\
u_j - \\operatorname{softplus}(-z_j) & \\text{only } u_j \\text{ finite},\\\\
z_j & \\text{free},
\\end{cases}
```

(fixed variables are set to their value), which is smooth and never has a zero
gradient; `mode = :clamp` uses the Euclidean projection onto the box,
``x = \\min(\\max(z, l), u)``, whose gradient vanishes outside the box.

The inner method must use a [`FullSpaceDecomposition`](@ref) with primal
predictions (its raw output starts with all primal variables); the bounds are
those declared on the JuMP variables.

Every completed primal prediction satisfies the variable bounds. Dual outputs
are not modified, so a dual bound of the inner method is unchanged; other
primal properties of the inner method (e.g. satisfied equalities) can be lost,
since changing ``x`` can break them.
"""
struct BoundedOutput{M<:AbstractL2OMethod} <: AbstractMethodWrapper
    inner::M
    mode::Symbol
    function BoundedOutput(inner::M; mode::Symbol = :smooth) where {M<:AbstractL2OMethod}
        mode in (:smooth, :clamp) ||
            throw(ArgumentError("mode must be :smooth or :clamp, got :$mode"))
        return new{M}(inner, mode)
    end
end

"""
    BoundedOutputDecomposition

Decomposition of a [`BoundedOutput`](@ref) pipeline: the inner decomposition
and the variable bounds.
"""
struct BoundedOutputDecomposition{D<:AbstractDecomposition} <: AbstractDecomposition
    inner::D
    lower::Vector{Float64}
    upper::Vector{Float64}
end
output_dimension(d::BoundedOutputDecomposition) = output_dimension(d.inner)

function decompose(problem, w::BoundedOutput)
    d = decompose(problem, inner_method(w))
    _primal_rows(d, "BoundedOutput")
    l, u = _variable_bounds(problem)
    return BoundedOutputDecomposition(d, l, u)
end

_softplus(z) = max(z, zero(z)) + log1p(exp(-abs(z)))
_sigmoid(z) = ifelse(z >= 0, inv(one(z) + exp(-z)), exp(z) / (one(z) + exp(z)))

"""
    bound_transform(d::BoundedOutputDecomposition, Z::AbstractMatrix, mode::Symbol)

Apply the bound-preserving map of [`BoundedOutput`](@ref) to the primal block
`Z` (`num_decision_variables × B`). Non-mutating and differentiable with array
AD.
"""
function bound_transform(d::BoundedOutputDecomposition, Z::AbstractMatrix, mode::Symbol)
    T = eltype(Z)
    l, u = T.(d.lower), T.(d.upper)
    mode === :clamp && return clamp.(Z, l, u)
    lf, uf = isfinite.(l), isfinite.(u)
    lz, uz = _finite.(l), _finite.(u)
    both, lo, up, free = lf .& uf, lf .& .!uf, .!lf .& uf, .!lf .& .!uf
    return both .* (lz .+ (uz .- lz) .* _sigmoid.(Z)) .+ lo .* (lz .+ _softplus.(Z)) .+
           up .* (uz .- _softplus.(.-Z)) .+ free .* Z
end

function _bounded_output(w::BoundedOutput, p, output)
    d = p.decomposition::BoundedOutputDecomposition
    Z = _asbatch(output)
    n = d.inner.nprimal
    return vcat(bound_transform(d, Z[1:n, :], w.mode), Z[(n+1):end, :])
end

# The inner method sees a pipeline whose decomposition is its own.
_inner_pipe(p::L2OPipeline, d) = L2OPipeline(p.problem, p.method, d, p.evaluator)

function complete(w::BoundedOutput, p, inputs, output)
    d = p.decomposition::BoundedOutputDecomposition
    return complete(
        inner_method(w),
        _inner_pipe(p, d.inner),
        inputs,
        _bounded_output(w, p, output),
    )
end

function loss(w::BoundedOutput, p, inputs, output; kwargs...)
    d = p.decomposition::BoundedOutputDecomposition
    return loss(
        inner_method(w),
        _inner_pipe(p, d.inner),
        inputs,
        _bounded_output(w, p, output);
        kwargs...,
    )
end

function generate_data(w::BoundedOutput, p; kwargs...)
    d = p.decomposition::BoundedOutputDecomposition
    return generate_data(inner_method(w), _inner_pipe(p, d.inner); kwargs...)
end

"""
    ProjectionRepair(inner::AbstractL2OMethod, optimizer)

Wrapper that repairs the primal prediction of `inner` at inference time by the
Euclidean projection onto the feasible set ``\\mathcal F(p)`` of each instance,

```math
Π_{\\mathcal F(p)}(\\hat x) = \\arg\\min_{x \\in \\mathcal F(p)} \\; \\|x - \\hat x\\|_2^2,
```

solved with `optimizer` (any JuMP solver for the resulting convex QP or
conic program). Training is unchanged: [`loss`](@ref) is the inner loss.

The problem class must be convex (so ``\\mathcal F(p)`` is closed and convex
and the projection is unique) and the feasible set of each instance non-empty.
The repaired prediction is then feasible to the solver's tolerance, and dual
outputs pass through unchanged. The projection is nonexpansive: for every
feasible ``x``, in particular an optimal ``x^\\star``,
``\\|Π(\\hat x) - x\\| \\le \\|\\hat x - x\\|``. The objective value need
not improve.
"""
struct ProjectionRepair{M<:AbstractL2OMethod,O} <: AbstractMethodWrapper
    inner::M
    optimizer::O
end

supports(w::ProjectionRepair, c::AbstractConvexClass) = supports(inner_method(w), c)
# Projections onto non-convex sets are neither unique nor computable by a
# convex solver: never forward non-convex classes to the inner method.
supports(::ProjectionRepair, ::AbstractProblemClass) = false

"""
    ProjectionDecomposition

Decomposition of a [`ProjectionRepair`](@ref) pipeline: the inner
decomposition and a copy of the JuMP model used for projections.
"""
struct ProjectionDecomposition{D<:AbstractDecomposition} <: AbstractDecomposition
    inner::D
    model::JuMP.Model
    variables::Vector{JuMP.VariableRef}
    parameters::Vector{JuMP.VariableRef}
end
output_dimension(d::ProjectionDecomposition) = output_dimension(d.inner)

function decompose(problem, w::ProjectionRepair)
    d = decompose(problem, inner_method(w))
    model, refmap = JuMP.copy_model(problem.model)
    JuMP.set_optimizer(model, w.optimizer)
    JuMP.set_silent(model)
    vars = [refmap[v] for v in decision_variables(problem)]
    pars = [refmap[v] for v in parameters(problem)]
    return ProjectionDecomposition(d, model, vars, pars)
end

"""
    project_onto_feasible_set(d::ProjectionDecomposition, inputs, X) -> Matrix

Euclidean projection of each column of `X` onto the feasible set of the
matching instance. Columns whose projection problem is not solved to
optimality are returned as `NaN`.
"""
function project_onto_feasible_set(d::ProjectionDecomposition, inputs, X)
    P, Xb = _asbatch(inputs), _asbatch(X)
    out = fill(NaN, size(Xb))
    for k in axes(Xb, 2)
        for (p, v) in zip(d.parameters, view(P, :, k))
            JuMP.set_parameter_value(p, v)
        end
        JuMP.@objective(d.model, Min, sum((d.variables .- view(Xb, :, k)) .^ 2))
        JuMP.optimize!(d.model)
        if JuMP.termination_status(d.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) &&
           JuMP.primal_status(d.model) == MOI.FEASIBLE_POINT
            out[:, k] .= JuMP.value.(d.variables)
        end
    end
    return out
end

generate_data(w::ProjectionRepair, p; kwargs...) = generate_data(
    inner_method(w),
    _inner_pipe(p, (p.decomposition::ProjectionDecomposition).inner);
    kwargs...,
)

function loss(w::ProjectionRepair, p, inputs, output; kwargs...)
    d = p.decomposition::ProjectionDecomposition
    return loss(inner_method(w), _inner_pipe(p, d.inner), inputs, output; kwargs...)
end

function complete(w::ProjectionRepair, p, inputs, output)
    d = p.decomposition::ProjectionDecomposition
    pred = complete(inner_method(w), _inner_pipe(p, d.inner), inputs, output)
    pred.primal === nothing && throw(
        ArgumentError("ProjectionRepair needs a method that predicts primal variables"),
    )
    return Prediction(project_onto_feasible_set(d, inputs, pred.primal), pred.dual)
end
