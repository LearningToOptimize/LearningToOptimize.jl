"""
    sobolev_loss(backend, f, P, labels; mask = DenseMask(), rng = Random.default_rng()) -> Real

Masked derivative-matching loss of a batch, eq. (2)–(3) of the masked Sobolev
training paper without the weight ``λ``:

```math
\\frac{1}{B} \\sum_{k=1}^{B}
  \\frac{\\big\\| \\mathcal{M} \\odot (\\hat D_k - D_k) \\big\\|_2^2}{\\|\\mathcal{M}\\|_0},
```

where ``\\hat D_k`` are input derivatives of the column-wise map `f` at
`P[:, k]` (computed by `backend`) and ``D_k`` the matching derivative labels:

| `labels` | ``\\hat D_k`` | ``D_k`` | mask shape |
|:--|:--|:--|:--|
| [`JacobianLabels`](@ref L2OCore.JacobianLabels) | ``D\\hat g_θ(p_k)`` | `J[:, :, k]` | `n × d` |
| [`PullbackLabels`](@ref L2OCore.PullbackLabels) | ``D\\hat g_θ(p_k)^⊤ w_k`` | `values[:, k]` | `d` |
| [`PushforwardLabels`](@ref L2OCore.PushforwardLabels) | ``D\\hat g_θ(p_k)\\, v_k`` | `values[:, k]` | `n` |

A single mask is drawn per call (shared by the batch) with
[`sample_mask`](@ref); [`DenseMask`](@ref) gives the mean squared error over
all entries.

The loss is `0` when the proxy derivatives equal the labels, and with
[`RandomMask`](@ref) its expectation over masks is the dense loss (eq. (4) of
the paper).
"""
function sobolev_loss(
    backend::AbstractSobolevBackend,
    f,
    P::AbstractMatrix,
    labels::L2OCore.JacobianLabels;
    mask::AbstractDerivativeMask = DenseMask(),
    rng::Random.AbstractRNG = Random.default_rng(),
)
    n, d, B = size(labels.J)
    _check_batch(P, d, B)
    Ĵ = proxy_jacobian(backend, f, P)
    size(Ĵ) == (n, d, B) ||
        throw(DimensionMismatch("proxy Jacobian has size $(size(Ĵ)), labels $((n, d, B))"))
    return _masked_mse(Ĵ .- _to_like(Ĵ, labels.J), sample_mask(mask, rng, (n, d)))
end

function sobolev_loss(
    backend::AbstractSobolevBackend,
    f,
    P::AbstractMatrix,
    labels::L2OCore.PullbackLabels;
    mask::AbstractDerivativeMask = DenseMask(),
    rng::Random.AbstractRNG = Random.default_rng(),
)
    d, B = size(labels.values)
    _check_batch(P, d, B)
    Ŵ = proxy_pullback(backend, f, P, labels.seeds)
    size(Ŵ) == (d, B) ||
        throw(DimensionMismatch("proxy pullback has size $(size(Ŵ)), labels $((d, B))"))
    return _masked_mse(Ŵ .- _to_like(Ŵ, labels.values), sample_mask(mask, rng, (d,)))
end

function sobolev_loss(
    backend::AbstractSobolevBackend,
    f,
    P::AbstractMatrix,
    labels::L2OCore.PushforwardLabels;
    mask::AbstractDerivativeMask = DenseMask(),
    rng::Random.AbstractRNG = Random.default_rng(),
)
    n, B = size(labels.values)
    _check_batch(P, size(labels.directions, 1), B)
    V̂ = proxy_pushforward(backend, f, P, labels.directions)
    size(V̂) == (n, B) ||
        throw(DimensionMismatch("proxy pushforward has size $(size(V̂)), labels $((n, B))"))
    return _masked_mse(V̂ .- _to_like(V̂, labels.values), sample_mask(mask, rng, (n,)))
end

function _check_batch(P, d, B)
    size(P) == (d, B) || throw(
        DimensionMismatch(
            "inputs have size $(size(P)) but the derivative labels are for " *
            "$d parameters and $B samples",
        ),
    )
    return nothing
end

# Eq. (3): ‖M ⊙ Δ_k‖² / ‖M‖₀ averaged over the batch (last dimension of Δ).
_masked_mse(Δ::AbstractArray, ::Nothing) = sum(abs2, Δ) / length(Δ)
function _masked_mse(Δ::AbstractArray, M::AbstractArray{Bool})
    B = size(Δ, ndims(Δ))
    Md = _to_like(Δ, M)
    return sum(abs2, Δ .* Md) / (count(M) * B)
end
