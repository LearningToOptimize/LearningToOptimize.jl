module L2OSobolevLuxExt

import L2OCore
import L2OSobolev
import Lux

const LuxMap = L2OSobolev.ProxyMap{<:Lux.StatefulLuxLayer}

function _check(f::L2OSobolev.ProxyMap)
    f.through_completion && throw(
        ArgumentError(
            "LuxNestedAD differentiates the Lux layer itself; derivatives through a " *
            "completion layer need `CentralDifferences()` (Lux's nested-AD rules do " *
            "not apply to closures around the layer)",
        ),
    )
    f.pipe.decomposition isa L2OCore.FullSpaceDecomposition || throw(
        ArgumentError(
            "LuxNestedAD needs a FullSpaceDecomposition (raw primal rows); " *
            "use `CentralDifferences()` with `through_completion = true`",
        ),
    )
    return f.pipe.decomposition
end

# Rows of the raw output that hold the primal prediction.
_nprimal(d::L2OCore.FullSpaceDecomposition) = d.nprimal
_hasdual(d::L2OCore.FullSpaceDecomposition) = d.dual

function L2OSobolev.proxy_jacobian(::L2OSobolev.LuxNestedAD, f::LuxMap, P::AbstractMatrix)
    dec = _check(f)
    J = Lux.batched_jacobian(f.proxy, Lux.AutoForwardDiff(), P)
    return _hasdual(dec) ? J[1:_nprimal(dec), :, :] : J
end

function L2OSobolev.proxy_pushforward(
    ::L2OSobolev.LuxNestedAD,
    f::LuxMap,
    P::AbstractMatrix,
    V::AbstractMatrix,
)
    dec = _check(f)
    Vd = L2OSobolev._to_like(P, V)
    Y = Lux.jacobian_vector_product(f.proxy, Lux.AutoForwardDiff(), P, Vd)
    return _hasdual(dec) ? Y[1:_nprimal(dec), :] : Y
end

function L2OSobolev.proxy_pullback(
    ::L2OSobolev.LuxNestedAD,
    f::LuxMap,
    P::AbstractMatrix,
    W::AbstractMatrix,
)
    dec = _check(f)
    Wd = L2OSobolev._to_like(P, W)
    if _hasdual(dec)
        # Zero seeds on the dual rows: Jᵀ [w; 0] = J_primalᵀ w.
        Wd = vcat(Wd, L2OSobolev._to_like(P, zeros(dec.ndual, size(W, 2))))
    end
    return Lux.vector_jacobian_product(f.proxy, Lux.AutoZygote(), P, Wd)
end

end # module L2OSobolevLuxExt
