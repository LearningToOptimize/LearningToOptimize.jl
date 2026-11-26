"""
    AbstractDerivativeMask

Which derivative entries enter the Sobolev term at one loss evaluation: the
binary mask ``\\mathcal{M}`` of eq. (2)–(3) of the masked Sobolev training
paper. Implement [`sample_mask`](@ref) for a new mask type.
"""
abstract type AbstractDerivativeMask end

"""
    DenseMask()

Every entry is supervised (``\\mathcal{M} = \\mathbf{1}``): dense derivative
matching, ``\\ell_d = \\|D\\hat g_θ - Dg\\|_F^2 / (n d)``.
"""
struct DenseMask <: AbstractDerivativeMask end

"""
    RandomMask(density)

Uniformly random subset of a **fixed number** of entries,
`k = clamp(round(Int, density * m), 1, m)` out of the `m` supervised entries,
redrawn at every loss evaluation and shared by the samples of the batch (the
same ``\\mathcal{M}`` in every term of eq. (2) of the paper).

With the normalization by ``\\|\\mathcal{M}\\|_0 = k``, the masked loss is an
unbiased
estimator of the dense loss, ``\\mathbb{E}_{\\mathcal{M}}[\\ell_d] =
\\|D\\hat g_θ - Dg\\|_F^2 / (n d)``, because every entry is retained with
probability ``k/m``.

The paper reports that keeping 5–10% of the Jacobian entries (`density =
0.05`–`0.1`) gave lower infeasibility than dense matching on AC-OPF.
"""
struct RandomMask{T<:Real} <: AbstractDerivativeMask
    density::T
    function RandomMask(density::T) where {T<:Real}
        0 < density <= 1 || throw(ArgumentError("mask density must be in (0, 1]"))
        return new{T}(density)
    end
end

"""
    FixedMask(mask::AbstractArray{Bool})

A user-chosen, deterministic mask (for example structural sparsity of the
solution map). Its shape must match the supervised quantity: `n × d` for
[`JacobianLabels`](@ref L2OCore.JacobianLabels), `d` for
[`PullbackLabels`](@ref L2OCore.PullbackLabels), `n` for
[`PushforwardLabels`](@ref L2OCore.PushforwardLabels).
"""
struct FixedMask{A<:AbstractArray{Bool}} <: AbstractDerivativeMask
    mask::A
    function FixedMask(mask::A) where {A<:AbstractArray{Bool}}
        any(mask) || throw(ArgumentError("a fixed mask must keep at least one entry"))
        return new{A}(mask)
    end
end

"""
    sample_mask(mask::AbstractDerivativeMask, rng, dims::Dims) -> Union{Nothing,Array{Bool}}

Draw the mask for one loss evaluation over entries of shape `dims`.
`nothing` means "all entries" (no masking). Not differentiated.
"""
sample_mask(::DenseMask, rng::Random.AbstractRNG, dims::Dims) = nothing

function sample_mask(m::RandomMask, rng::Random.AbstractRNG, dims::Dims)
    total = prod(dims)
    k = clamp(round(Int, m.density * total), 1, total)
    M = falses(dims)
    for i in Iterators.take(Random.randperm(rng, total), k)
        M[i] = true
    end
    return Array{Bool}(M)
end

function sample_mask(m::FixedMask, rng::Random.AbstractRNG, dims::Dims)
    size(m.mask) == dims || throw(
        DimensionMismatch("fixed mask has size $(size(m.mask)), the labels need $dims"),
    )
    return Array{Bool}(m.mask)
end
