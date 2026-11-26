module L2OSobolevZygoteExt

import L2OCore
import L2OSobolev
import Zygote

# The raw proxy as a branch-free callable. Zygote differentiates this callable
# twice (inner pullback, then the outer gradient through it); the body of
# `ProxyMap` itself has control flow that Zygote's second-order compilation
# does not handle, so the primal rows are selected through the seeds instead.
struct RawProxy{M}
    proxy::M
end
(r::RawProxy)(P) = L2OCore.predict_batch(r.proxy, P)

# The callable to differentiate and the number of leading output rows that form
# the map (`nothing`: all rows).
_inner(f) = (f, nothing)
function _inner(f::L2OSobolev.ProxyMap)
    f.through_completion && throw(
        ArgumentError(
            "ZygoteNestedAD differentiates the proxy itself; derivatives through a " *
            "completion layer need `CentralDifferences()` (Zygote cannot differentiate " *
            "completion layers twice in general)",
        ),
    )
    dec = f.pipe.decomposition
    dec isa L2OCore.FullSpaceDecomposition || throw(
        ArgumentError(
            "ZygoteNestedAD needs a FullSpaceDecomposition (raw primal rows); " *
            "use `CentralDifferences()` with `through_completion = true`",
        ),
    )
    return RawProxy(f.proxy), dec.nprimal
end

_rows(nout, ::Nothing) = nout
_rows(nout, nrows::Integer) = nrows

function L2OSobolev.proxy_jacobian(::L2OSobolev.ZygoteNestedAD, f, P::AbstractMatrix)
    g, nrows = _inner(f)
    d, B = size(P)
    nout = L2OSobolev._output_rows(g, P)
    n = _rows(nout, nrows)
    # One reverse pass of width nB: block i of the repeated batch is seeded
    # with e_i, so block i of the input cotangent holds row i of every J_k.
    _, back = Zygote.pullback(g, repeat(P, 1, n))
    G = back(L2OSobolev._seed_blocks(P, nout, n, B))[1]
    return permutedims(reshape(G, d, B, n), (3, 1, 2))
end

function L2OSobolev.proxy_pushforward(
    b::L2OSobolev.ZygoteNestedAD,
    f,
    P::AbstractMatrix,
    V::AbstractMatrix,
)
    size(V) == size(P) ||
        throw(DimensionMismatch("directions have size $(size(V)), inputs $(size(P))"))
    J = L2OSobolev.proxy_jacobian(b, f, P)
    n, d, B = size(J)
    Vd = L2OSobolev._to_like(J, V)
    return reshape(sum(J .* reshape(Vd, 1, d, B); dims = 2), n, B)
end

function L2OSobolev.proxy_pullback(
    ::L2OSobolev.ZygoteNestedAD,
    f,
    P::AbstractMatrix,
    W::AbstractMatrix,
)
    g, nrows = _inner(f)
    B = size(P, 2)
    nout = L2OSobolev._output_rows(g, P)
    n = _rows(nout, nrows)
    size(W) == (n, B) ||
        throw(DimensionMismatch("seeds have size $(size(W)), expected $((n, B))"))
    Wd = L2OSobolev._to_like(P, W)
    if nout > n
        # Zero seeds on the remaining (dual) rows: Jᵀ [w; 0] = J_primalᵀ w.
        Wd = vcat(Wd, L2OSobolev._to_like(P, zeros(nout - n, B)))
    end
    _, back = Zygote.pullback(g, P)
    return back(Wd)[1]
end

end # module L2OSobolevZygoteExt
