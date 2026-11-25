# Developer scripts

Scripts for maintainers. They are not part of the test suite and are not run by CI.

## `nested_ad_matrix.jl`

Reproduces the nested-AD support matrix on the design page: the gradient with respect to
the weights of a loss that contains an input derivative, for Lux and Flux with Zygote,
ForwardDiff and Enzyme, compared with finite differences.

```sh
julia dev/nested_ad_matrix.jl
```
