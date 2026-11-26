# L2OCore

[![CI](https://github.com/LearningToOptimize/LearningToOptimize.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/LearningToOptimize.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LearningToOptimize.github.io/LearningToOptimize.jl/dev/api_l2ocore/)
[![Codecov](https://codecov.io/gh/LearningToOptimize/LearningToOptimize.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LearningToOptimize/LearningToOptimize.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

The interface shared by the learning-to-optimize methods of the
[LearningToOptimize](https://github.com/LearningToOptimize) ecosystem: method families, problem classes, the
`ParametricProblem` wrapper with fixed layouts, batched evaluators (objective, violations, Lagrangian dual bound),
datasets and the method-dispatched pipeline. It depends only on JuMP, MathOptInterface and standard libraries, so
method packages can depend on it without pulling in a machine-learning framework.

Install it from this repository (it is not registered yet):

```julia
using Pkg
Pkg.add(url = "https://github.com/LearningToOptimize/LearningToOptimize.jl", subdir = "lib/L2OCore")
```

Most users install [LearningToOptimize.jl](../../README.md), which re-exports L2OCore. The
[design page](https://LearningToOptimize.github.io/LearningToOptimize.jl/dev/design/) explains the interface and
the [adding a method guide](https://LearningToOptimize.github.io/LearningToOptimize.jl/dev/guide/extending/) shows
what a method package implements.
