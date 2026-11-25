# Contributing

Thanks for helping! Issues and pull requests are welcome.

- **Adding a method?** Read the [design page](https://LearningToOptimize.github.io/LearningToOptimize.jl/dev/design/)
  and the "Adding a method" guide. New methods usually live in their own package depending on `lib/L2OCore`;
  changes to the interface itself go in this repository.
- **Tests:** `julia --project=lib/L2OCore -e 'using Pkg; Pkg.test()'` for the core and
  `julia --project -e 'using Pkg; Pkg.develop(path="lib/L2OCore"); Pkg.test()'` for the main package. Keep tests
  small, seeded and fast. Every mathematical claim a method makes should be a named guarantee with a numerical check.
- **Format:** run `using JuliaFormatter; format(".")` before pushing (CI checks it).
- **Docs:** every exported symbol needs a docstring; build locally with
  `julia --project=docs docs/make.jl` (after `Pkg.develop` of the package and `lib/L2OCore` in `docs`).
- Keep pull requests focused: one concern per commit makes review and reverts easy.
