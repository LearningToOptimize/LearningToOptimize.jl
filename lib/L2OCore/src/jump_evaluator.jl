"""
    JuMPEvaluator(problem::ParametricProblem)

Reference evaluator that loops over the batch and evaluates every expression
with `JuMP.value(var_value, expr)`. It supports anything JuMP can evaluate
(affine, quadratic and `GenericNonlinearExpr` functions, any MOI set with
`MOI.Utilities.distance_to_set`) and serves as the ground truth against which
faster backends are tested. It is not meant for training loops.

[`dual_bound`](@ref) is not available for this backend.
"""
struct JuMPEvaluator <: AbstractEvaluator
    problem::ParametricProblem
    eq::BitVector
    bnd::BitVector
    box::BitVector
end

function JuMPEvaluator(problem::ParametricProblem)
    m = num_duals(problem)
    eq, bnd, box = falses(m), falses(m), falses(m)
    for (c, rows) in zip(problem.constraints, problem.layout)
        obj = JuMP.constraint_object(c)
        set = JuMP.moi_set(obj)
        if set isa Union{_BOX_SCALAR_SETS,_BOX_VECTOR_SETS}
            l, u = _box(set)
            box[rows] .= true
            eq[rows] .= l == u
        end
        bnd[rows] .=
            obj.func isa JuMP.AbstractVariableRef ||
            obj.func isa AbstractVector{<:JuMP.AbstractVariableRef}
    end
    return JuMPEvaluator(problem, eq, bnd, box)
end

function _var_value(problem::ParametricProblem, x::AbstractVector, p::AbstractVector)
    d = Dict{JuMP.VariableRef,eltype(x)}()
    for (v, xv) in zip(problem.variables, x)
        d[v] = xv
    end
    for (v, pv) in zip(problem.parameters, p)
        d[v] = pv
    end
    return v -> d[v]
end

_value_vec(f, expr) = [JuMP.value(f, expr)]
_value_vec(f, expr::AbstractVector) = [JuMP.value(f, e) for e in expr]

function batch_objective(ev::JuMPEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    obj = JuMP.objective_function(ev.problem.model)
    return [
        JuMP.value(_var_value(ev.problem, view(X, :, k), view(P, :, k)), obj) for
        k in axes(X, 2)
    ]
end

function constraint_values(ev::JuMPEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    F = zeros(Float64, num_duals(ev.problem), size(X, 2))::Matrix{Float64}
    for k in axes(X, 2)
        vv = _var_value(ev.problem, view(X, :, k), view(P, :, k))
        for (c, rows) in zip(ev.problem.constraints, ev.problem.layout)
            obj = JuMP.constraint_object(c)
            F[rows, k] .= _value_vec(vv, obj.func)
        end
    end
    return F
end

function row_violation(ev::JuMPEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    F = constraint_values(ev, X, P)
    V = zero(F)
    for (c, rows) in zip(ev.problem.constraints, ev.problem.layout)
        set = JuMP.moi_set(JuMP.constraint_object(c))
        set isa Union{_BOX_SCALAR_SETS,_BOX_VECTOR_SETS} || continue
        l, u = _box(set)
        V[rows, :] .= max.(l .- F[rows, :], 0) .+ max.(F[rows, :] .- u, 0)
    end
    return V
end

function cone_violation(ev::JuMPEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    F = constraint_values(ev, X, P)
    blocks = [
        (rows, JuMP.moi_set(JuMP.constraint_object(c))) for
        (c, rows) in zip(ev.problem.constraints, ev.problem.layout) if
        !(JuMP.moi_set(JuMP.constraint_object(c)) isa Union{_BOX_SCALAR_SETS,_BOX_VECTOR_SETS})
    ]
    return _cone_distances(F, blocks)
end

equality_rows(ev::JuMPEvaluator) = ev.eq
bound_rows(ev::JuMPEvaluator) = ev.bnd

dual_bound(::JuMPEvaluator, Y, P; atol::Real = 1e-6) = throw(
    ArgumentError("dual_bound is not implemented for JuMPEvaluator; use MatrixEvaluator"),
)
