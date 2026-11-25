# Adding a method

A method package depends only on `L2OCore`. It defines a method type in one of
the three families, declares the problem classes it supports, and overrides
the pipeline steps that differ from the family fallback. Everything else
(data generation, batched evaluation, samplers) comes from `L2OCore`.

```julia
module L2OMyMethod

import L2OCore
using L2OCore: AbstractPrimalMethod, AbstractConvexClass, Prediction

struct MyMethod <: AbstractPrimalMethod
    ρ::Float64
end

# 1. Problem classes the method applies to (default: none).
L2OCore.supports(::MyMethod, ::AbstractConvexClass) = true

# 2. Structure computed once per problem (default: predict everything).
struct MyDecomposition <: L2OCore.AbstractDecomposition
    independent::Vector{Int}
    # ... factorisations, index maps, ...
end
L2OCore.output_dimension(d::MyDecomposition) = length(d.independent)
L2OCore.decompose(problem, ::MyMethod) = MyDecomposition(...)

# 3. Completion: raw proxy output -> prediction (e.g. solve the equalities).
function L2OCore.complete(::MyMethod, pipe, inputs, output)
    X = ...                     # array operations only (AD- and GPU-friendly)
    return Prediction(primal = X)
end

# 4. Training loss (default: supervised MSE against `targets`).
function L2OCore.loss(m::MyMethod, pipe, inputs, output; kwargs...)
    X = L2OCore.complete(m, pipe, inputs, output).primal
    ev = pipe.evaluator
    B = size(X, 2)
    return sum(L2OCore.batch_objective(ev, X, inputs)) / B +
           m.ρ * sum(abs2, L2OCore.row_violation(ev, X, inputs)) / B
end

end
```

Rules of thumb:

- Type only the method argument of `complete`, `loss`, `decompose` and
  `generate_data`; the fallbacks leave the other arguments untyped so your
  methods are never ambiguous with them.
- Accept `kwargs...` in `loss`: training loops pass `targets` and `proxy`.
- Write completions and losses with array operations on batches (batch = last
  dimension) so they can be differentiated and moved to a GPU.
- Test what the method's mathematics says on random instances, using
  `evaluate` and solver optima. For a completion that solves the equality
  constraints:

```julia
pred = complete(pipe, P, proxy(P))
report = evaluate(pipe, P, pred)
@test maximum(report.max_equality_violation) <= 1e-8
gap = abs.(report.objective .- data.objective) ./ max.(1, abs.(data.objective))
```

## Wrapping a method

Method-agnostic additions (for example derivative supervision) subtype
`AbstractMethodWrapper`. The wrapper forwards everything to `inner_method(w)`
and adds `extra_loss(w, pipe, inputs, output; kwargs...)` to the inner loss;
it can attach derivative labels to datasets with `with_derivatives`.

## Adding an evaluator backend

Subtype `AbstractEvaluator` and implement `batch_objective`,
`constraint_values`, `row_violation`, `equality_rows`, `L2OCore.bound_rows`,
and, where relevant, `cone_violation` and `dual_bound`. Pass it to
`L2OPipeline(problem, method; evaluator = MyEvaluator(problem))`; the tests of
`L2OCore` compare backends against `JuMPEvaluator` on random points, and a new
backend should do the same.

## Adding a problem class

Subtype `AbstractProblemClass` (or `AbstractConvexClass`) and pass it as
`ParametricProblem(model; class = MyClass())`.
