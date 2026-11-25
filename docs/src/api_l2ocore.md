# L2OCore API

```@meta
CurrentModule = L2OCore
```

`L2OCore` lives in `lib/L2OCore` of this repository. All of its exported
names are re-exported by `LearningToOptimize`.

```@docs
L2OCore
```

## Methods and proxies

```@docs
AbstractL2OMethod
AbstractPrimalMethod
AbstractDualMethod
AbstractPrimalDualMethod
AbstractMethodWrapper
inner_method
extra_loss
method_family
PrimalFamily
DualFamily
PrimalDualFamily
AbstractProxyModel
predict_batch
```

## Problems and classes

```@docs
AbstractProblemClass
AbstractConvexClass
LinearConic
ConvexQuadratic
SmoothNonlinear
MixedInteger
detect_problem_class
AbstractParametricProblem
ParametricProblem
problem_class
parameters
decision_variables
constraint_refs
num_parameters
num_decision_variables
num_duals
dual_layout
parameter_values
objective_sign
supports
check_supports
UnsupportedProblemError
```

## Evaluation

```@docs
AbstractEvaluator
MatrixEvaluator
JuMPEvaluator
default_evaluator
L2OCore.has_matrix_form
batch_objective
constraint_values
row_violation
cone_violation
equality_rows
L2OCore.bound_rows
dual_bound
EvaluationReport
evaluate
```

## Data

```@docs
AbstractParameterSampler
sample_parameters
BoxSampler
ScaledSampler
L2ODataset
num_samples
with_derivatives
AbstractDerivativeLabels
JacobianLabels
PullbackLabels
solve_instances
```

## Pipeline

```@docs
Prediction
AbstractDecomposition
FullSpaceDecomposition
L2OPipeline
decompose
input_dimension
output_dimension
generate_data
complete
loss
train_proxy!
```

## Other exported names

```@docs
guarantees
AbstractGuarantee
NoGuarantee
Feasible
EqualityFeasible
BoundFeasible
ValidDualBound
check
CheckResult
```
