# Developer scripts

Scripts for maintainers. They are not part of the test suite and are not run by CI.

## `nested_ad_matrix.jl`

Reproduces the nested-AD support matrix on the design page: the gradient with respect to
the weights of a loss that contains an input derivative, for Lux and Flux with Zygote,
ForwardDiff and Enzyme, compared with finite differences.

```sh
julia dev/nested_ad_matrix.jl
```

## `gpu_check.jl`: GPU check for the open pull requests

CI runs on CPUs only. This script checks the GPU code paths of the open pull requests
on a real NVIDIA GPU with a single command:

```sh
julia dev/gpu_check.jl                      # CUDA (default)
julia dev/gpu_check.jl --only=dll,nested_ad # a subset of the checks
julia dev/gpu_check.jl --device=jlarray     # dry run without a GPU (JLArrays)
julia dev/gpu_check.jl --activations=tanh   # nested_ad with one activation only
```

The script installs the branches under review into a temporary environment, together with
CUDA.jl and LuxCUDA (or JLArrays for a dry run). It prints the package versions and the
GPU, then runs every check on the device and on the CPU and compares the results:

| check | what runs on the device | compared with the CPU |
|:--|:--|:--|
| `dll` | L2ODLL `batched_layer` (LP, SOCP, QP), dense and sparse, moved with `Adapt.adapt` | dual bounds, Zygote gradient, `batched_completion` |
| `dc3` | L2ODC3 affine QP decomposition moved with `adapt(CuArray{Float32}, ·)` | `predict`, `training_loss`, gradient with respect to the weights of a Lux proxy |
| `alm` | L2OALM `ALMFormulation` of an ExaModel on `CUDABackend()` (BatchNLPKernels), Lux weights on the device | primal loss, its gradient, the violation measure; a few `train!` iterations must lower the loss |
| `nested_ad` | gradients with respect to the weights of a loss containing an input VJP, with tanh, relu, sigmoid and leakyrelu: Lux (`Lux.vector_jacobian_product` and hand-nested Zygote), Flux (Zygote over Zygote), and the full L2OSobolev loss with each backend | the same gradient on the CPU in Float64 |
| `core` | L2OCore `MatrixEvaluator` on device arrays | reported as not yet supported until issue #39 is done |

Each case reports `pass`, `FAIL` (with the relative error), `ERROR` (with a one-line
message) or `returns nothing` (a gradient that silently came back as `nothing`). The final
table and a summary line are printed and also written to `dev/gpu_check_results.md`,
ready to paste into an issue. The L2OSobolev cases need the L2OCore copy on their own
branch, so they run in a child process with a separate environment.

With `--device=jlarray`, arrays are `JLArray`s (CPU-backed, scalar indexing disabled),
and the ExaModels check uses the KernelAbstractions `CPU()` backend. This exercises the
same code as the GPU run, so it catches scalar indexing and host/device mix-ups before
the script is run on a GPU.

Excluding the first package install, a run takes a few minutes on one GPU.
