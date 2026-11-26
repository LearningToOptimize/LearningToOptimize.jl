"""
    L2OSobolev

Method-agnostic **Sobolev (derivative) supervision** for learning-to-optimize
proxies, following *Masked Sobolev Training for Feasibility-Reliable
Optimization Proxies* (Rosemberg, Dias Garcia, Bent and Van Hentenryck,
NeurIPS 2026).

[`SobolevRegularized`](@ref) wraps any `L2OCore` method whose proxy predicts
primal variables and adds the masked derivative-matching term of eq. (2)–(3)
of the paper to its loss:

```math
\\mathcal{L}(θ) = \\mathcal{L}_{\\text{inner}}(θ)
  + \\frac{λ}{N} \\sum_{i=1}^{N}
    \\frac{\\big\\| \\mathcal{M} \\odot \\big(D\\hat g_θ(p_i) - D g(p_i)\\big) \\big\\|_F^2}{\\|\\mathcal{M}\\|_0},
```

where ``Dg(p_i) = ∂x^\\star/∂p`` are solver sensitivities stored as optional
derivative labels of the dataset ([`sensitivity_labels`](@ref)) and
``D\\hat g_θ`` is computed by an [`AbstractSobolevBackend`](@ref).

Package extensions:

  - `DiffOpt` (+ `JuMP`): [`DiffOptSensitivities`](@ref), derivative labels
    from implicit differentiation of the KKT conditions;
  - `Lux`: [`LuxNestedAD`](@ref), exact proxy Jacobians with Lux's nested-AD
    helpers;
  - `Zygote`: [`ZygoteNestedAD`](@ref), exact proxy Jacobians of Flux models
    and other Zygote-differentiable callables (Zygote over Zygote).
"""
module L2OSobolev

import ChainRulesCore
import L2OCore
import LinearAlgebra
import MathOptInterface as MOI
import Random

include("masks.jl")
include("backends.jl")
include("loss.jl")
include("labels.jl")
include("wrapper.jl")
include("theory.jl")
include("ad_rules.jl")

export AbstractDerivativeMask, DenseMask, RandomMask, FixedMask, sample_mask
export AbstractSobolevBackend,
    CentralDifferences,
    LuxNestedAD,
    ZygoteNestedAD,
    ProxyMap,
    proxy_jacobian,
    proxy_pushforward,
    proxy_pullback
export sobolev_loss
export AbstractDerivativeDirections,
    FullJacobian,
    RandomPullbacks,
    RandomPushforwards,
    AbstractSensitivityOracle,
    DiffOptSensitivities,
    FiniteDifferenceSensitivities,
    sensitivity_labels,
    label_errors
export SobolevRegularized
export covering_radius

end # module L2OSobolev
