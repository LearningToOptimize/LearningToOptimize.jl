"""
    AbstractProblemClass

Trait describing the mathematical class of a parametric problem *for fixed
parameters*. Methods declare which classes they accept with [`supports`](@ref).

The built-in classes are ordered from most to least structured:
[`LinearConic`](@ref) ⊂ [`ConvexQuadratic`](@ref) (as objective classes),
[`SmoothNonlinear`](@ref) and [`MixedInteger`](@ref). Packages may add classes
by subtyping `AbstractProblemClass` (or [`AbstractConvexClass`](@ref)) and
passing the class explicitly to [`ParametricProblem`](@ref).
"""
abstract type AbstractProblemClass end

"""
    AbstractConvexClass <: AbstractProblemClass

Problem classes for which every instance (for fixed parameters) is a convex
optimization problem, so that weak *and* strong duality arguments
(under a constraint qualification) apply.
"""
abstract type AbstractConvexClass <: AbstractProblemClass end

"""
    LinearConic()

Affine objective, affine constraint functions in (products of) closed convex
cones: LPs, SOCPs, SDPs, exponential-cone programs, …

Parameters may appear affinely in the objective and in the constraint constants.
"""
struct LinearConic <: AbstractConvexClass end

"""
    ConvexQuadratic()

Convex quadratic objective (`½xᵀQx + cᵀx` with `Q ⪰ 0` for minimisation,
`Q ⪯ 0` for maximisation) and affine conic constraints.
"""
struct ConvexQuadratic <: AbstractConvexClass end

"""
    SmoothNonlinear()

Continuous problems with nonlinear (or nonconvex quadratic) objective or
constraints, e.g. AC optimal power flow. Only first-order stationarity is
meaningful; weak duality still holds but certificates require extra work.
"""
struct SmoothNonlinear <: AbstractProblemClass end

"""
    MixedInteger()

Problems with integrality restrictions (`Integer`, `ZeroOne`, semi-continuous or
SOS constraints).
"""
struct MixedInteger <: AbstractProblemClass end

const _INTEGRALITY_SETS = Union{
    MOI.Integer,
    MOI.ZeroOne,
    MOI.Semicontinuous,
    MOI.Semiinteger,
    MOI.SOS1,
    MOI.SOS2,
}

_is_nonlinear_function(::Type) = false
_is_nonlinear_function(::Type{<:JuMP.GenericNonlinearExpr}) = true
_is_nonlinear_function(::Type{<:AbstractVector{F}}) where {F} = _is_nonlinear_function(F)

_is_quadratic_function(::Type) = false
_is_quadratic_function(::Type{<:JuMP.GenericQuadExpr}) = true
_is_quadratic_function(::Type{<:AbstractVector{F}}) where {F} = _is_quadratic_function(F)

# Does the quadratic expression contain a product of two *decision* variables
# (as opposed to parameter × variable or parameter × parameter)?
function _has_variable_product(q::JuMP.GenericQuadExpr)
    for (_, a, b) in JuMP.quad_terms(q)
        if !JuMP.is_parameter(a) && !JuMP.is_parameter(b)
            return true
        end
    end
    return false
end
_has_variable_product(v::AbstractVector) = any(_has_variable_product, v)
_has_variable_product(::Any) = false

"""
    detect_problem_class(model::JuMP.Model; atol = 1e-9) -> AbstractProblemClass

Inspect a JuMP model and return the most structured built-in class that
describes it:

 1. any integrality/SOS/semi-continuous constraint → [`MixedInteger`](@ref);
 2. any nonlinear expression, legacy `@NL` block or quadratic constraint with a
    product of two decision variables → [`SmoothNonlinear`](@ref);
 3. quadratic objective with a product of decision variables → convexity is
    checked numerically (`eigmin(sQ) ≥ -atol`, `s = ±1` for the sense):
    [`ConvexQuadratic`](@ref) if convex, [`SmoothNonlinear`](@ref) otherwise;
 4. otherwise [`LinearConic`](@ref).

Products `parameter × variable` do not change the class (for fixed parameters
they are affine in the decision variables).
"""
function detect_problem_class(model::JuMP.Model; atol::Real = 1e-9)
    smooth = false
    if JuMP.nonlinear_model(model) !== nothing
        smooth = true
    end
    for (F, S) in JuMP.list_of_constraint_types(model)
        if S <: _INTEGRALITY_SETS
            return MixedInteger()
        end
        if _is_nonlinear_function(F)
            smooth = true
        elseif _is_quadratic_function(F)
            for c in JuMP.all_constraints(model, F, S)
                if _has_variable_product(JuMP.constraint_object(c).func)
                    smooth = true
                    break
                end
            end
        end
    end
    smooth && return SmoothNonlinear()
    obj = JuMP.objective_function(model)
    if obj isa JuMP.GenericNonlinearExpr
        return SmoothNonlinear()
    elseif obj isa JuMP.GenericQuadExpr && _has_variable_product(obj)
        vars = filter(!JuMP.is_parameter, JuMP.all_variables(model))
        Q = _quadratic_matrix(obj, _index_map(vars), length(vars), Float64)
        s = JuMP.objective_sense(model) == MOI.MAX_SENSE ? -1.0 : 1.0
        λ = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(Matrix(s .* Q)))
        return λ >= -atol ? ConvexQuadratic() : SmoothNonlinear()
    end
    return LinearConic()
end

_index_map(vars) = Dict(v => i for (i, v) in enumerate(vars))

# Symmetric Q such that the quadratic part of `q` equals ½ xᵀ Q x (decision
# variables only; terms involving parameters are skipped).
function _quadratic_matrix(q::JuMP.GenericQuadExpr, idx::Dict, n::Int, ::Type{T}) where {T}
    I, J, V = Int[], Int[], T[]
    for (coef, a, b) in JuMP.quad_terms(q)
        (JuMP.is_parameter(a) || JuMP.is_parameter(b)) && continue
        i, j = idx[a], idx[b]
        if i == j
            push!(I, i); push!(J, i); push!(V, 2 * coef)
        else
            push!(I, i); push!(J, j); push!(V, coef)
            push!(I, j); push!(J, i); push!(V, coef)
        end
    end
    return SparseArrays.sparse(I, J, V, n, n)
end
