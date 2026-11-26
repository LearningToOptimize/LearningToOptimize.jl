# L2OSobolev API

```@meta
CurrentModule = L2OSobolev
```

`L2OSobolev` lives in `lib/L2OSobolev` of this repository. The DiffOpt, Lux
and Zygote features are package extensions: load `DiffOpt` (and `JuMP`) for
[`DiffOptSensitivities`](@ref), `Lux` for [`LuxNestedAD`](@ref), `Zygote` for
[`ZygoteNestedAD`](@ref).

```@docs
L2OSobolev
```

## Wrapper

```@docs
SobolevRegularized
L2OCore.generate_data(::SobolevRegularized, ::Any)
```

## Loss and masks

```@docs
sobolev_loss
AbstractDerivativeMask
DenseMask
RandomMask
FixedMask
sample_mask
```

## Proxy derivatives

```@docs
ProxyMap
AbstractSobolevBackend
CentralDifferences
LuxNestedAD
ZygoteNestedAD
proxy_jacobian
proxy_pushforward
proxy_pullback
```

## Derivative labels

```@docs
AbstractDerivativeDirections
FullJacobian
RandomPullbacks
RandomPushforwards
AbstractSensitivityOracle
DiffOptSensitivities
FiniteDifferenceSensitivities
sensitivity_labels
label_errors
```

## Approximation diagnostics

```@docs
covering_radius
```
