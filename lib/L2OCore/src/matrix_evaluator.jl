const _BOX_SCALAR_SETS = Union{MOI.EqualTo,MOI.GreaterThan,MOI.LessThan,MOI.Interval}
const _BOX_VECTOR_SETS = Union{MOI.Zeros,MOI.Nonnegatives,MOI.Nonpositives}

_box(s::MOI.EqualTo) = (s.value, s.value)
_box(s::MOI.GreaterThan{T}) where {T} = (s.lower, typemax(T))
_box(s::MOI.LessThan{T}) where {T} = (typemin(T), s.upper)
_box(s::MOI.Interval) = (s.lower, s.upper)
_box(::MOI.Zeros) = (0.0, 0.0)
_box(::MOI.Nonnegatives) = (0.0, Inf)
_box(::MOI.Nonpositives) = (-Inf, 0.0)

_affine_rows(f::JuMP.AbstractVariableRef) = [JuMP.AffExpr(0.0, f => 1.0)]
_affine_rows(f::JuMP.GenericAffExpr) = [f]
_affine_rows(f::AbstractVector) = reduce(vcat, map(_affine_rows, f); init = JuMP.AffExpr[])
_affine_rows(f) = throw(ArgumentError("function of type $(typeof(f)) has no matrix form"))

_affine_function(::JuMP.AbstractVariableRef) = true
_affine_function(::JuMP.GenericAffExpr) = true
_affine_function(f::AbstractVector) = all(_affine_function, f)
_affine_function(::Any) = false

"""
    has_matrix_form(problem::ParametricProblem) -> Bool

Whether [`MatrixEvaluator`](@ref) can represent `problem`: affine constraint
functions (parameters may appear in the constants), at most quadratic
objective, no legacy `@NL` block.
"""
function has_matrix_form(problem::ParametricProblem)
    model = problem.model
    JuMP.nonlinear_model(model) === nothing || return false
    obj = JuMP.objective_function(model)
    obj isa Union{JuMP.AbstractVariableRef,JuMP.GenericAffExpr,JuMP.GenericQuadExpr} ||
        return false
    for c in problem.constraints
        _affine_function(JuMP.constraint_object(c).func) || return false
    end
    return true
end

"""
    MatrixEvaluator(problem::ParametricProblem; T = Float64)

Matrix-form evaluator for problems of the form (for each instance `p`)

```math
\\min_x \\; \\tfrac12 x^\\top Q x + c^\\top x + x^\\top H p + c_p^\\top p + \\tfrac12 p^\\top Q_{pp} p + c_0
\\quad \\text{s.t.} \\quad A x + B p + b \\in \\mathcal{K}
```

where `𝒦` is the product of the constraint sets in [`dual_layout`](@ref)
order (`max` is handled through the sense). Box-like sets become row bounds
`l ≤ Ax + Bp + b ≤ u`; other cones are kept as blocks.

All batch operations are matrix products and broadcasts, so they are
vectorised, allocation-light and differentiable with array AD (Zygote,
Enzyme, …). [`cone_violation`](@ref) and the cone part of
[`dual_bound`](@ref) evaluate `MOI.Utilities.distance_to_set` per column
(differentiable with Zygote for the second-order cone; not GPU-friendly).
"""
struct MatrixEvaluator{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractEvaluator
    sense::T            # +1 for min (or feasibility), -1 for max
    A::M
    B::M
    b::V
    lower::V
    upper::V
    Q::M
    c::V
    H::M
    cp::V
    Qpp::M
    c0::T
    box::BitVector
    eq::BitVector
    bnd::BitVector
    cones::Vector{Tuple{UnitRange{Int},MOI.AbstractVectorSet}}
    Qpinv::Base.RefValue{Any}
end

function MatrixEvaluator(problem::ParametricProblem; T::Type{<:Real} = Float64)
    has_matrix_form(problem) || throw(
        ArgumentError(
            "problem has no matrix form (non-affine constraints or non-quadratic " *
            "objective); use JuMPEvaluator or a kernel-based evaluator",
        ),
    )
    model = problem.model
    vidx = _index_map(problem.variables)
    pidx = _index_map(problem.parameters)
    n, np, m = num_decision_variables(problem), num_parameters(problem), num_duals(problem)
    AI, AJ, AV = Int[], Int[], T[]
    BI, BJ, BV = Int[], Int[], T[]
    b = zeros(T, m)
    lower = fill(T(-Inf), m)
    upper = fill(T(Inf), m)
    box = falses(m)
    eq = falses(m)
    bnd = falses(m)
    cones = Tuple{UnitRange{Int},MOI.AbstractVectorSet}[]
    for (c, rows) in zip(problem.constraints, problem.layout)
        obj = JuMP.constraint_object(c)
        set = JuMP.moi_set(obj)
        isbound =
            obj.func isa JuMP.AbstractVariableRef ||
            obj.func isa AbstractVector{<:JuMP.AbstractVariableRef}
        for (r, aff) in zip(rows, _affine_rows(obj.func))
            for (coef, v) in JuMP.linear_terms(aff)
                if JuMP.is_parameter(v)
                    push!(BI, r);
                    push!(BJ, pidx[v]);
                    push!(BV, coef)
                else
                    push!(AI, r);
                    push!(AJ, vidx[v]);
                    push!(AV, coef)
                end
            end
            b[r] = JuMP.constant(aff)
            bnd[r] = isbound
        end
        if set isa Union{_BOX_SCALAR_SETS,_BOX_VECTOR_SETS}
            l, u = _box(set)
            lower[rows] .= l
            upper[rows] .= u
            box[rows] .= true
            eq[rows] .= l == u
        else
            set isa MOI.AbstractVectorSet ||
                throw(ArgumentError("unsupported scalar set $(typeof(set))"))
            push!(cones, (rows, set))
        end
    end
    A = SparseArrays.sparse(AI, AJ, AV, m, n)
    B = SparseArrays.sparse(BI, BJ, BV, m, np)
    # Objective
    f = JuMP.objective_function(model)
    q = convert(
        JuMP.QuadExpr,
        f isa JuMP.AbstractVariableRef ? JuMP.AffExpr(0.0, f => 1.0) : f,
    )
    Q = _quadratic_matrix(q, vidx, n, T)
    c = zeros(T, n)
    cp = zeros(T, np)
    for (coef, v) in JuMP.linear_terms(q)
        JuMP.is_parameter(v) ? (cp[pidx[v]] += coef) : (c[vidx[v]] += coef)
    end
    HI, HJ, HV = Int[], Int[], T[]
    PI, PJ, PV = Int[], Int[], T[]
    for (coef, a, bb) in JuMP.quad_terms(q)
        pa, pb = JuMP.is_parameter(a), JuMP.is_parameter(bb)
        if pa && pb
            i, j = pidx[a], pidx[bb]
            if i == j
                push!(PI, i);
                push!(PJ, i);
                push!(PV, 2coef)
            else
                push!(PI, i, j);
                push!(PJ, j, i);
                push!(PV, coef, coef)
            end
        elseif pa
            push!(HI, vidx[bb]);
            push!(HJ, pidx[a]);
            push!(HV, coef)
        elseif pb
            push!(HI, vidx[a]);
            push!(HJ, pidx[bb]);
            push!(HV, coef)
        end
    end
    H = SparseArrays.sparse(HI, HJ, HV, n, np)
    Qpp = SparseArrays.sparse(PI, PJ, PV, np, np)
    s = JuMP.objective_sense(model) == MOI.MAX_SENSE ? -one(T) : one(T)
    return MatrixEvaluator(
        s,
        A,
        B,
        b,
        lower,
        upper,
        Q,
        c,
        H,
        cp,
        Qpp,
        T(JuMP.constant(q)),
        box,
        eq,
        bnd,
        cones,
        Ref{Any}(nothing),
    )
end

_quadform(Q, X) = vec(sum(X .* (Q * X); dims = 1))

function batch_objective(ev::MatrixEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    v = vec(ev.c' * X) .+ vec(ev.cp' * P) .+ ev.c0
    if SparseArrays.nnz(ev.Q) > 0
        v = v .+ _quadform(ev.Q, X) ./ 2
    end
    if SparseArrays.nnz(ev.H) > 0
        v = v .+ vec(sum(X .* (ev.H * P); dims = 1))
    end
    if SparseArrays.nnz(ev.Qpp) > 0
        v = v .+ _quadform(ev.Qpp, P) ./ 2
    end
    return v
end

constraint_values(ev::MatrixEvaluator, X::AbstractMatrix, P::AbstractMatrix) =
    ev.A * X .+ ev.B * P .+ ev.b

function row_violation(ev::MatrixEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    F = constraint_values(ev, X, P)
    return max.(ev.lower .- F, 0) .+ max.(F .- ev.upper, 0)
end

function cone_violation(ev::MatrixEvaluator, X::AbstractMatrix, P::AbstractMatrix)
    isempty(ev.cones) && return zeros(eltype(X), 0, size(X, 2))
    F = constraint_values(ev, X, P)
    return _cone_distances(F, ev.cones)
end

# Non-mutating so that array AD (Zygote) can differentiate conic penalties.
function _cone_distances(F::AbstractMatrix, blocks)
    isempty(blocks) && return zeros(eltype(F), 0, size(F, 2))
    D = [
        MOI.Utilities.distance_to_set(F[rows, k], set) for (rows, set) in blocks,
        k in axes(F, 2)
    ]
    return D::AbstractMatrix
end

equality_rows(ev::MatrixEvaluator) = ev.eq
bound_rows(ev::MatrixEvaluator) = ev.bnd

# Pseudo-inverse of sQ (minimisation form) and whether sQ ⪰ 0, computed once.
# The quadratic dual function below is only valid for sQ ⪰ 0 (otherwise
# inf_x L(x, y) = -∞ for every y), so non-convex objectives never claim a bound.
function _qpinv(ev::MatrixEvaluator{T}) where {T}
    if ev.Qpinv[] === nothing
        Qm = LinearAlgebra.Symmetric(Matrix(ev.sense .* ev.Q))
        λmin = LinearAlgebra.eigmin(Qm)
        tol = sqrt(eps(T)) * max(one(T), LinearAlgebra.opnorm(Qm, 1))
        ev.Qpinv[] = (LinearAlgebra.pinv(Matrix(Qm)), λmin >= -tol)
    end
    return ev.Qpinv[]::Tuple{Matrix{T},Bool}
end

_finite(x) = ifelse(isfinite(x), x, zero(x))

function dual_bound(
    ev::MatrixEvaluator,
    Y::AbstractMatrix,
    P::AbstractMatrix;
    atol::Real = 1e-6,
)
    s = ev.sense
    # Stationarity residual of the Lagrangian of the (minimisation-form) problem.
    R = s .* (ev.c .+ ev.H * P) .- ev.A' * Y
    lf = _finite.(ev.lower)
    uf = _finite.(ev.upper)
    boxterm = ifelse.(Y .> 0, Y .* lf, Y .* uf) .* ev.box
    g =
        s .* (vec(ev.cp' * P) .+ ev.c0) .- vec(sum(Y .* (ev.B * P .+ ev.b); dims = 1)) .+
        vec(sum(boxterm; dims = 1))
    if SparseArrays.nnz(ev.Qpp) > 0
        g = g .+ s .* _quadform(ev.Qpp, P) ./ 2
    end
    if SparseArrays.nnz(ev.Q) > 0
        Qi, convex = _qpinv(ev)
        Z = Qi * R
        g = g .- vec(sum(R .* Z; dims = 1)) ./ 2
        # r must lie in range(sQ); otherwise inf_x L = -∞.
        stat = _colmax(abs.((ev.sense .* ev.Q) * Z .- R))
        convex || (stat = fill(convert(eltype(stat), Inf), length(stat)))
    else
        stat = _colmax(abs.(R))
    end
    # Dual-cone membership: box rows with an infinite side are sign-constrained.
    sign_viol =
        max.(Y, 0) .* (ev.box .& (ev.lower .== -Inf)) .+
        max.(-Y, 0) .* (ev.box .& (ev.upper .== Inf))
    infeas = max.(stat, _colmax(sign_viol))
    for (rows, set) in ev.cones
        dset = MOI.dual_set(set)
        for k in axes(Y, 2)
            infeas[k] =
                max(infeas[k], MOI.Utilities.distance_to_set(collect(Y[rows, k]), dset))
        end
    end
    bound = ifelse.(infeas .<= atol, s .* g, s .* typemin(eltype(g)))
    return bound, infeas
end
