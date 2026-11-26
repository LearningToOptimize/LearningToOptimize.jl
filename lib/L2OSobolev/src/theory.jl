"""
    covering_radius(train::AbstractMatrix, query::AbstractMatrix) -> Real

Empirical covering radius of the training parameters (columns of `train`)
over the parameters in `query`:

```math
δ = \\max_{q ∈ \\text{query}} \\min_{p_i ∈ \\text{train}} \\|q - p_i\\|_2 ,
```

the quantity ``δ`` of Theorems 1–3 of the masked Sobolev training paper
(evaluated on a finite set of query points instead of the supremum over the
whole parameter space; a dense query set gives a lower estimate of ``δ``).
Under exact interpolation, value supervision bounds the proxy error by
``(L_g + L_{\\hat g}) δ`` and joint value-and-Jacobian supervision by
``\\tfrac12 (M_g + M_{\\hat g}) δ^2``; `covering_radius` tells how far a test
set is from the training data in these terms.
"""
function covering_radius(train::AbstractMatrix, query::AbstractMatrix)
    size(train, 1) == size(query, 1) ||
        throw(DimensionMismatch("train and query have different numbers of rows"))
    size(train, 2) > 0 || throw(ArgumentError("empty training set"))
    δ = zero(float(promote_type(eltype(train), eltype(query))))
    for q in eachcol(query)
        dmin = minimum(LinearAlgebra.norm(q .- p) for p in eachcol(train))
        δ = max(δ, dmin)
    end
    return δ
end
