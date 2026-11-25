"""
    L2OCore

Lightweight interfaces shared by every Learning-to-Optimize (L2O) method in the
LearningToOptimize ecosystem.

`L2OCore` defines *what* an L2O method is (a type in a small hierarchy), *which
problems* it accepts ([`supports`](@ref)), *what it promises* about its
predictions ([`guarantees`](@ref)) and the method-dispatched pipeline

```
ParametricProblem ─ decompose ─ generate_data ─ (train proxy) ─ complete ─ evaluate/check
```

with sane fallbacks for each method family. It depends only on JuMP,
MathOptInterface and standard libraries, so method packages can depend on it
without pulling in a machine-learning framework.

Conventions:

  - **Batch-first:** the batch is always the *last* dimension. Parameters are
    `num_parameters × B`, primal predictions `num_decision_variables × B`, dual
    predictions `num_duals × B`.
  - **Element-type generic:** evaluators are parametrised by the element type.
  - **Minimal scalar indexing** in vectorised evaluation code so arrays can be moved to
    accelerators by a future kernels extension.
"""
module L2OCore

import JuMP
import LinearAlgebra
import MathOptInterface as MOI
import Random
import SparseArrays

include("methods.jl")
include("classes.jl")
include("problem.jl")
include("evaluators.jl")
include("matrix_evaluator.jl")
include("jump_evaluator.jl")
include("guarantees.jl")
include("data.jl")
include("pipeline.jl")

# Method hierarchy and proxies
export AbstractL2OMethod,
    AbstractPrimalMethod,
    AbstractDualMethod,
    AbstractPrimalDualMethod,
    AbstractProxyModel,
    predict_batch,
    method_family,
    PrimalFamily,
    DualFamily,
    PrimalDualFamily
# Problem classes and the problem wrapper
export AbstractProblemClass,
    AbstractConvexClass,
    LinearConic,
    ConvexQuadratic,
    SmoothNonlinear,
    MixedInteger,
    detect_problem_class,
    AbstractParametricProblem,
    ParametricProblem,
    objective_sign,
    problem_class,
    parameters,
    decision_variables,
    constraint_refs,
    num_parameters,
    num_decision_variables,
    num_duals,
    dual_layout,
    parameter_values
# Evaluation
export AbstractEvaluator,
    MatrixEvaluator,
    JuMPEvaluator,
    default_evaluator,
    batch_objective,
    constraint_values,
    row_violation,
    cone_violation,
    equality_rows,
    dual_bound,
    EvaluationReport,
    evaluate
# Assumptions and guarantees
export AbstractGuarantee,
    NoGuarantee,
    Feasible,
    EqualityFeasible,
    BoundFeasible,
    ValidDualBound,
    CheckResult,
    supports,
    check_supports,
    UnsupportedProblemError,
    guarantees,
    check
# Data
export AbstractParameterSampler,
    BoxSampler,
    ScaledSampler,
    sample_parameters,
    L2ODataset,
    AbstractDerivativeLabels,
    JacobianLabels,
    PullbackLabels,
    num_samples,
    with_derivatives,
    solve_instances
# Pipeline
export AbstractDecomposition,
    FullSpaceDecomposition,
    Prediction,
    L2OPipeline,
    decompose,
    input_dimension,
    output_dimension,
    generate_data,
    loss,
    complete,
    train_proxy!,
    AbstractMethodWrapper,
    inner_method,
    extra_loss

end # module L2OCore
