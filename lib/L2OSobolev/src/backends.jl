"""
    ProxyMap(proxy, pipe; through_completion = false)

The map ``P \\mapsto \\hat x_θ(P)`` whose input derivatives are supervised:
the primal prediction of `proxy` on a batch of parameters `P`
(`num_parameters × B` → `num_decision_variables × B`).

  - `through_completion = false`: the primal rows of the raw proxy output
    (requires the [`FullSpaceDecomposition`](@ref L2OCore.FullSpaceDecomposition)
    of the pipeline's method, where these rows *are* the primal prediction
    before any repair); this is ``D\\hat g_θ`` of the paper.
  - `through_completion = true`: the primal part of
    [`complete`](@ref L2OCore.complete)`(pipe, P, proxy(P))`, i.e. derivatives
    taken through a completion / repair / projection layer (the paper's
    "through the feasibility module" variant).

**Assumption:** the map acts column by column (sample `k` of the output
depends only on column `k` of the input), as for feed-forward networks without
batch statistics. Batched input-derivative computations rely on it.
"""
struct ProxyMap{M,L}
    proxy::M
    pipe::L
    through_completion::Bool
end
ProxyMap(proxy, pipe; through_completion::Bool = false) =
    ProxyMap(proxy, pipe, through_completion)

function (f::ProxyMap)(P::AbstractMatrix)
    out = L2OCore.predict_batch(f.proxy, P)
    f.through_completion && return L2OCore.complete(f.pipe, P, out).primal
    return _primal_rows(f.pipe.decomposition, out)
end

_primal_rows(d::L2OCore.FullSpaceDecomposition, out) = d.dual ? out[1:d.nprimal, :] : out
_primal_rows(d::L2OCore.AbstractDecomposition, out) = throw(
    ArgumentError(
        "the raw output of a $(nameof(typeof(d))) is not the primal prediction; " *
        "use `through_completion = true` with a backend that differentiates " *
        "arbitrary callables (e.g. `CentralDifferences()`)",
    ),
)

"""
    AbstractSobolevBackend

How input derivatives of the proxy map are computed *inside* the training loss
(which is itself differentiated with respect to the proxy weights: nested
differentiation). A backend implements

  - [`proxy_jacobian`](@ref)`(backend, f, P)` → `n × d × B`;
  - [`proxy_pushforward`](@ref)`(backend, f, P, V)` → `n × B` (Jacobian–vector products);
  - [`proxy_pullback`](@ref)`(backend, f, P, W)` → `d × B` (vector–Jacobian products),

for a column-wise map `f` (usually a [`ProxyMap`](@ref)). Each result must be
differentiable by the outer AD used for training.
"""
abstract type AbstractSobolevBackend end

"""
    CentralDifferences(; step = nothing)

Framework-agnostic backend: input derivatives by central differences of the
map, ``J v ≈ (f(P + h v) - f(P - h v)) / 2h``, evaluated as *batched forward
passes*. The outer AD only differentiates forward passes, so it works with any
callable proxy (Flux, Lux, plain functions), any outer AD and on GPUs, and also
through completion layers. All `d` directions of a Jacobian are evaluated in a
single forward pass of width `2 d B`.

The truncation error is ``O(h^2 \\|D^3 f\\|)`` and the rounding error
``O(ε (\\|f\\| + \\|D f\\|\\,\\|P\\|_∞) / h)`` (rounding of both the outputs
and the perturbed inputs). The default step balances the two,
``h = ε^{1/3} \\max(1, \\|P\\|_∞)`` of the input element type (≈ 6e-6 for
`Float64`, ≈ 5e-3 for `Float32`, times the input scale); `step = h` fixes an
absolute step instead. Directions should be of unit scale. The approximation
assumes the map is ``C^3`` near the inputs; for piecewise-linear activations
(ReLU) the differences straddle kinks with small probability and then return
a one-sided average. For exact derivatives use [`LuxNestedAD`](@ref) with Lux
models, or [`ZygoteNestedAD`](@ref) with Flux models and other
Zygote-differentiable callables.
"""
struct CentralDifferences{T} <: AbstractSobolevBackend
    step::T
end
CentralDifferences(; step::Union{Nothing,Real} = nothing) = CentralDifferences(step)

function _step(b::CentralDifferences{Nothing}, P)
    T = float(eltype(P))
    return cbrt(eps(T)) * max(one(T), T(maximum(abs, P)))
end
_step(b::CentralDifferences{<:Real}, P) = convert(float(eltype(P)), b.step)

"""
    LuxNestedAD()

Exact input derivatives of a Lux model with Lux's nested-AD helpers
(`Lux.batched_jacobian` and `Lux.jacobian_vector_product` with
`AutoForwardDiff`, `Lux.vector_jacobian_product` with `AutoZygote`), which have
dedicated rules for being differentiated again by Zygote with respect to the
weights. Requires `using Lux` (package extension) and `Zygote` for
pullbacks and training.

The proxy must be a `Lux.StatefulLuxLayer` and the map a
[`ProxyMap`](@ref) with `through_completion = false` (Lux's rules apply to the
layer itself, not to closures around it; measured, see the design page). For
derivatives through a completion layer use [`CentralDifferences`](@ref).
"""
struct LuxNestedAD <: AbstractSobolevBackend end

"""
    ZygoteNestedAD()

Exact input derivatives of any callable that Zygote can differentiate twice:
a Flux `Chain`, a closure over Flux parameters, or a plain Julia function. The
input derivatives are Zygote pullbacks, and an outer Zygote gradient with
respect to the weights (`Zygote.gradient`, `Flux.gradient`) differentiates
through them (Zygote over Zygote; measured correct for Flux, see the design
page). Requires `using Zygote` (package extension).

  - [`proxy_pullback`](@ref): one reverse pass of width `B`;
  - [`proxy_jacobian`](@ref): one reverse pass of width `n B` (the `n` output
    rows are seeded at once on a repeated batch);
  - [`proxy_pushforward`](@ref): the Jacobian contracted with the directions
    (reverse mode only, so it costs as much as `proxy_jacobian`).

For a [`ProxyMap`](@ref) the backend differentiates the proxy itself and keeps
the primal rows through the seeds, so the map must use
`through_completion = false` with a
[`FullSpaceDecomposition`](@ref L2OCore.FullSpaceDecomposition). Zygote cannot
differentiate completion layers twice in general (keyword constructors,
mutation, branches); use [`CentralDifferences`](@ref) for derivatives through a
completion layer. Any other callable `f` is differentiated as given, so its
body must be differentiable twice by Zygote (no mutation; a closure such as
`P -> re(θ)(P)` around a Flux model is fine). For Lux models use
[`LuxNestedAD`](@ref): nesting Zygote pullbacks around a `StatefulLuxLayer`
fails (design page).
"""
struct ZygoteNestedAD <: AbstractSobolevBackend end

"""
    proxy_jacobian(backend, f, P) -> AbstractArray{T,3}

Input Jacobians of the column-wise map `f` at every column of `P`:
`J[:, :, k] = ∂f(P)[:, k] / ∂P[:, k]` (`n × d × B`, the layout of
[`JacobianLabels`](@ref L2OCore.JacobianLabels)).
"""
function proxy_jacobian end

"""
    proxy_pushforward(backend, f, P, V) -> AbstractMatrix

Jacobian–vector products `J_k * V[:, k]` for every column (`n × B`).
"""
function proxy_pushforward end

"""
    proxy_pullback(backend, f, P, W) -> AbstractMatrix

Vector–Jacobian products `J_kᵀ * W[:, k]` for every column (`d × B`).
"""
function proxy_pullback end

for fn in (:proxy_jacobian, :proxy_pushforward, :proxy_pullback)
    @eval function $fn(b::AbstractSobolevBackend, f, args...)
        throw(
            ArgumentError(
                "$(nameof(typeof(b))) does not implement `$($(string(fn)))` for " *
                "$(nameof(typeof(f)))" *
                (
                    b isa LuxNestedAD ?
                    "; load Lux (`using Lux`) and pass a " *
                    "`ProxyMap` around a `StatefulLuxLayer`" :
                    b isa ZygoteNestedAD ? "; load Zygote (`using Zygote`)" : ""
                ),
            ),
        )
    end
end

# Copy a host array to the array type/device of `like` (CPU or GPU) without
# scalar indexing.
function _to_like(like::AbstractArray, A::AbstractArray)
    D = similar(like, eltype(like), size(A))
    copyto!(D, A)
    return D
end

# m × (k·B) matrix whose j-th block of B columns is the unit vector e_{idx[j]}
# (on the device of `like`).
function _unit_blocks(
    like::AbstractArray,
    m::Integer,
    idx::AbstractVector{<:Integer},
    B::Integer,
)
    T = eltype(like)
    E = zeros(T, m, length(idx) * B)
    for (j, i) in enumerate(idx)
        E[i, ((j-1)*B+1):(j*B)] .= one(T)
    end
    return _to_like(like, E)
end
_unit_blocks(P::AbstractMatrix, B::Integer) = _unit_blocks(P, size(P, 1), 1:size(P, 1), B)

# m × (k·B) seeds e_1, …, e_k ∈ ℝᵐ (first k output rows), as used by the
# Zygote backend.
_seed_blocks(like::AbstractArray, m::Integer, k::Integer, B::Integer) =
    _unit_blocks(like, m, 1:k, B)

# Number of output rows of the column-wise map `f` (one forward pass on one column).
_output_rows(f, P::AbstractMatrix) = size(f(P[:, 1:1]), 1)

function proxy_pushforward(b::CentralDifferences, f, P::AbstractMatrix, V::AbstractMatrix)
    size(V) == size(P) ||
        throw(DimensionMismatch("directions have size $(size(V)), inputs $(size(P))"))
    h = _step(b, P)
    Vd = _to_like(P, V)
    return (f(P .+ h .* Vd) .- f(P .- h .* Vd)) ./ (2h)
end

function proxy_jacobian(b::CentralDifferences, f, P::AbstractMatrix)
    d, B = size(P)
    h = _step(b, P)
    E = _unit_blocks(P, B)
    Pr = repeat(P, 1, d)
    # One forward pass of width 2dB: block j holds J_k e_j for every sample k.
    Y = (f(Pr .+ h .* E) .- f(Pr .- h .* E)) ./ (2h)
    n = size(Y, 1)
    return permutedims(reshape(Y, n, B, d), (1, 3, 2))
end

function proxy_pullback(b::CentralDifferences, f, P::AbstractMatrix, W::AbstractMatrix)
    J = proxy_jacobian(b, f, P)
    n, d, B = size(J)
    size(W) == (n, B) ||
        throw(DimensionMismatch("seeds have size $(size(W)), expected $((n, B))"))
    Wd = _to_like(J, W)
    return reshape(sum(reshape(Wd, n, 1, B) .* J; dims = 1), d, B)
end
