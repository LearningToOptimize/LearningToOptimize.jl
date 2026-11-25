"""
    AbstractParameterSampler

Distribution of problem parameters. Implement
[`sample_parameters`](@ref)`(sampler, rng, n)` returning a
`num_parameters × n` matrix. Any function `f(rng, n)` can be used directly as
a sampler.
"""
abstract type AbstractParameterSampler end

"""
    sample_parameters(sampler, rng::Random.AbstractRNG, n::Integer) -> AbstractMatrix

Draw `n` parameter vectors, one per column.
"""
sample_parameters(f::Function, rng::Random.AbstractRNG, n::Integer) = f(rng, n)

"""
    BoxSampler(lower, upper)

Independent uniform parameters, `p[i] ~ U(lower[i], upper[i])`.
"""
struct BoxSampler{T<:Real} <: AbstractParameterSampler
    lower::Vector{T}
    upper::Vector{T}
    function BoxSampler(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T<:Real}
        length(lower) == length(upper) ||
            throw(DimensionMismatch("lower and upper differ in length"))
        all(lower .<= upper) || throw(ArgumentError("need lower ≤ upper"))
        return new{T}(collect(lower), collect(upper))
    end
end

function sample_parameters(s::BoxSampler{T}, rng::Random.AbstractRNG, n::Integer) where {T}
    return s.lower .+ (s.upper .- s.lower) .* rand(rng, T, length(s.lower), n)
end

"""
    ScaledSampler(nominal; low = 0.8, high = 1.2)
    ScaledSampler(problem::ParametricProblem; low = 0.8, high = 1.2)

Multiplicative perturbation of a nominal parameter vector,
`p[i] = nominal[i] * U(low, high)`. With a problem, the nominal values are the
current parameter values of the JuMP model.
"""
struct ScaledSampler{T<:Real} <: AbstractParameterSampler
    nominal::Vector{T}
    low::T
    high::T
end
ScaledSampler(nominal::AbstractVector{<:Real}; low::Real = 0.8, high::Real = 1.2) =
    ScaledSampler(collect(float.(nominal)), float(low), float(high))
ScaledSampler(problem::ParametricProblem; kwargs...) =
    ScaledSampler(parameter_values(problem); kwargs...)

function sample_parameters(
    s::ScaledSampler{T},
    rng::Random.AbstractRNG,
    n::Integer,
) where {T}
    return s.nominal .* (s.low .+ (s.high - s.low) .* rand(rng, T, length(s.nominal), n))
end

"""
    AbstractDerivativeLabels

Optional derivative (sensitivity) labels attached to a dataset, e.g. computed
by differentiating through the solver. Datasets never require them; methods
that use them (Sobolev-type training) check for their presence.
"""
abstract type AbstractDerivativeLabels end

"""
    JacobianLabels(J)

Full Jacobians of the primal solution with respect to the parameters:
`J` is `num_decision_variables × num_parameters × N`.
"""
struct JacobianLabels{A<:AbstractArray{<:Real,3}} <: AbstractDerivativeLabels
    J::A
end

"""
    PullbackLabels(seeds, values)

Vector–Jacobian products: for each sample `k`, `values[:, k] = J_kᵀ seeds[:, k]`
where `J_k = ∂x*/∂p`; `seeds` is `num_decision_variables × N`, `values` is
`num_parameters × N`. Cheaper than full Jacobians and sufficient for
pullback-based Sobolev losses.
"""
struct PullbackLabels{M<:AbstractMatrix{<:Real}} <: AbstractDerivativeLabels
    seeds::M
    values::M
end

"""
    L2ODataset

Columns-as-samples dataset (`N` samples):

  - `inputs`: `num_parameters × N` (always present);
  - `primal`: `num_decision_variables × N` or `nothing`;
  - `dual`: `num_duals × N` or `nothing`;
  - `objective`: length `N` or `nothing`;
  - `status`: termination status per sample or `nothing` (unsolved data, e.g.
    for self-supervised methods);
  - `derivatives`: an [`AbstractDerivativeLabels`](@ref) or `nothing`;
  - `metadata`: free-form `Dict{String,Any}` (problem id, sampler, solver,
    tolerances…), mirrored by the dataset-card conventions in the design docs.

Samples whose solve failed hold `NaN` labels; filter them with
`ds[findall(ok)]` where `ok` comes from `status`.
"""
struct L2ODataset{T<:Real}
    inputs::Matrix{T}
    primal::Union{Nothing,Matrix{T}}
    dual::Union{Nothing,Matrix{T}}
    objective::Union{Nothing,Vector{T}}
    status::Union{Nothing,Vector{MOI.TerminationStatusCode}}
    derivatives::Union{Nothing,AbstractDerivativeLabels}
    metadata::Dict{String,Any}
end

function L2ODataset(
    inputs::AbstractMatrix{T};
    primal = nothing,
    dual = nothing,
    objective = nothing,
    status = nothing,
    derivatives = nothing,
    metadata = Dict{String,Any}(),
) where {T<:Real}
    N = size(inputs, 2)
    for (name, x) in ((:primal, primal), (:dual, dual))
        x === nothing ||
            size(x, 2) == N ||
            throw(DimensionMismatch("$name has $(size(x, 2)) columns, expected $N"))
    end
    return L2ODataset{T}(
        Matrix(inputs),
        primal,
        dual,
        objective,
        status,
        derivatives,
        metadata,
    )
end

"""
    num_samples(ds::L2ODataset) -> Int
"""
num_samples(ds::L2ODataset) = size(ds.inputs, 2)

_cols(::Nothing, idx) = nothing
_cols(x::AbstractMatrix, idx) = x[:, idx]
_cols(x::AbstractVector, idx) = x[idx]
_cols(d::AbstractDerivativeLabels, idx) = nothing

function Base.getindex(ds::L2ODataset, idx::AbstractVector{<:Integer})
    return L2ODataset(
        ds.inputs[:, idx];
        primal = _cols(ds.primal, idx),
        dual = _cols(ds.dual, idx),
        objective = _cols(ds.objective, idx),
        status = _cols(ds.status, idx),
        derivatives = _subset_derivatives(ds.derivatives, idx),
        metadata = copy(ds.metadata),
    )
end
_subset_derivatives(::Nothing, idx) = nothing
_subset_derivatives(d::JacobianLabels, idx) = JacobianLabels(d.J[:, :, idx])
_subset_derivatives(d::PullbackLabels, idx) =
    PullbackLabels(d.seeds[:, idx], d.values[:, idx])

Base.show(io::IO, ds::L2ODataset{T}) where {T} = print(
    io,
    "L2ODataset{$T}: ",
    num_samples(ds),
    " samples; labels: ",
    join(
        [
            string(f) for f in (:primal, :dual, :objective, :status, :derivatives) if
            getfield(ds, f) !== nothing
        ],
        ", ",
    ),
)

"""
    with_derivatives(ds::L2ODataset, labels::Union{Nothing,AbstractDerivativeLabels}) -> L2ODataset

Copy of `ds` with its derivative labels replaced (datasets are immutable).
"""
with_derivatives(ds::L2ODataset, labels) = L2ODataset(
    ds.inputs;
    primal = ds.primal,
    dual = ds.dual,
    objective = ds.objective,
    status = ds.status,
    derivatives = labels,
    metadata = ds.metadata,
)

_flat_dual(c) = vcat(JuMP.dual(c)...)

"""
    solve_instances(problem, inputs; optimizer = nothing, primal = true,
                    dual = false, silent = true) -> L2ODataset

Solve one instance per column of `inputs` by setting the parameter values of the
JuMP model and calling `optimize!`. Records the objective value, termination
status and, as requested, the primal solution ([`decision_variables`](@ref)
order) and the dual solution ([`dual_layout`](@ref) order, `JuMP.dual` signs).

The JuMP model is modified in place (parameter values, optimizer).
"""
function solve_instances(
    problem::ParametricProblem,
    inputs::AbstractMatrix;
    optimizer = nothing,
    primal::Bool = true,
    dual::Bool = false,
    silent::Bool = true,
)
    model = problem.model
    optimizer === nothing || JuMP.set_optimizer(model, optimizer)
    silent && JuMP.set_silent(model)
    N = size(inputs, 2)
    T = Float64
    X = primal ? fill(T(NaN), num_decision_variables(problem), N) : nothing
    Y = dual ? fill(T(NaN), num_duals(problem), N) : nothing
    obj = fill(T(NaN), N)
    status = Vector{MOI.TerminationStatusCode}(undef, N)
    for k = 1:N
        for (p, v) in zip(problem.parameters, view(inputs, :, k))
            JuMP.set_parameter_value(p, v)
        end
        JuMP.optimize!(model)
        status[k] = JuMP.termination_status(model)
        if JuMP.has_values(model)
            obj[k] = JuMP.objective_value(model)
            primal && (X[:, k] .= JuMP.value.(problem.variables))
        end
        if dual && JuMP.has_duals(model)
            for (c, rows) in zip(problem.constraints, problem.layout)
                Y[rows, k] .= _flat_dual(c)
            end
        end
    end
    meta = Dict{String,Any}("solver" => JuMP.solver_name(model))
    return L2ODataset(
        Matrix{T}(inputs);
        primal = X,
        dual = Y,
        objective = obj,
        status = status,
        metadata = meta,
    )
end
