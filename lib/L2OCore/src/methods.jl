"""
    AbstractL2OMethod

Supertype of every Learning-to-Optimize method.

A method is a (usually small, immutable) struct holding its hyper-parameters.
Concrete methods subtype one of the three *families*
[`AbstractPrimalMethod`](@ref), [`AbstractDualMethod`](@ref) or
[`AbstractPrimalDualMethod`](@ref); the family selects the fallback behavior of
[`decompose`](@ref), [`generate_data`](@ref), [`loss`](@ref) and
[`complete`](@ref).

To add a method, a package defines

```julia
struct MyMethod <: L2OCore.AbstractPrimalMethod
    ρ::Float64
end
L2OCore.supports(::MyMethod, ::L2OCore.LinearConic) = true
```

and overrides only the pipeline steps that differ from the family fallback.
"""
abstract type AbstractL2OMethod end

"""
    AbstractPrimalMethod <: AbstractL2OMethod

Methods whose proxy predicts (a subset of) the primal decision variables, e.g.
penalty methods, DC3, LDF.
"""
abstract type AbstractPrimalMethod <: AbstractL2OMethod end

"""
    AbstractDualMethod <: AbstractL2OMethod

Methods whose proxy predicts (a subset of) the dual variables, e.g. Dual
Lagrangian Learning (DLL). A dual feasible prediction gives a valid bound on
the optimal value through [`dual_bound`](@ref).
"""
abstract type AbstractDualMethod <: AbstractL2OMethod end

"""
    AbstractPrimalDualMethod <: AbstractL2OMethod

Methods that predict both primal and dual variables, e.g. primal-dual learning
with an augmented Lagrangian (ALM/PDL).
"""
abstract type AbstractPrimalDualMethod <: AbstractL2OMethod end

"""
    PrimalFamily()

Value of [`method_family`](@ref) for [`AbstractPrimalMethod`](@ref)s. Family
values are convenient for code that branches on *what is predicted* without
dispatching on the method type itself (for example, data recorders).
"""
struct PrimalFamily end

"""
    DualFamily()

Value of [`method_family`](@ref) for [`AbstractDualMethod`](@ref)s.
"""
struct DualFamily end

"""
    PrimalDualFamily()

Value of [`method_family`](@ref) for [`AbstractPrimalDualMethod`](@ref)s.
"""
struct PrimalDualFamily end

"""
    method_family(method::AbstractL2OMethod)

Return `PrimalFamily()`, `DualFamily()` or `PrimalDualFamily()`.
"""
method_family(::AbstractPrimalMethod) = PrimalFamily()
method_family(::AbstractDualMethod) = DualFamily()
method_family(::AbstractPrimalDualMethod) = PrimalDualFamily()

predicts_primal(::PrimalFamily) = true
predicts_primal(::DualFamily) = false
predicts_primal(::PrimalDualFamily) = true
predicts_dual(::PrimalFamily) = false
predicts_dual(::DualFamily) = true
predicts_dual(::PrimalDualFamily) = true

"""
    AbstractProxyModel

Optional supertype for proxy models (the learned map from parameters to raw
outputs).

`L2OCore` is framework-agnostic: **any callable** `f(inputs::AbstractMatrix)`
returning an `output_dimension × B` array can be used as a proxy (a
`Flux.Chain`, a closure over a `Lux` model and its state, a plain function).
Subtyping `AbstractProxyModel` is only needed to attach extra behavior, for
example a custom [`predict_batch`](@ref) or trainable-parameter handling.
"""
abstract type AbstractProxyModel end

"""
    predict_batch(proxy, inputs::AbstractMatrix)

Apply a proxy to a batch of inputs (`input_dimension × B`) and return the raw
output (`output_dimension × B`). The fallback calls `proxy(inputs)`; extend it
for proxies that are not directly callable.
"""
predict_batch(proxy, inputs::AbstractMatrix) = proxy(inputs)
