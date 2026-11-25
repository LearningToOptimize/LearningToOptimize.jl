# # Getting started: a penalty proxy for a parametric LP
#
# This example trains a small neural network to predict solutions of a
# parametric economic-dispatch LP, using the self-supervised
# [`PenaltyMethod`](@ref) (no solver needed for training), and evaluates it
# against solver optima. It runs on a CPU in a few seconds.

using LearningToOptimize
using JuMP, HiGHS, Flux, Random

# ## The parametric problem
#
# Three generators with costs 1, 3 and 2 serve two demands `d` (parameters);
# capacities are variable bounds.

model = Model()
@variable(model, d[1:2] in MOI.Parameter.([1.0, 2.0]))
@variable(model, 0 <= g[1:3] <= 4)
@constraint(model, g[1] + g[2] == d[1])
@constraint(model, g[3] == d[2])
@objective(model, Min, g[1] + 3g[2] + 2g[3])

problem = ParametricProblem(model)

# The problem class is detected from the model and the layouts are fixed:

problem_class(problem),
num_parameters(problem),
num_decision_variables(problem),
num_duals(problem)

# ## Method and pipeline
#
# `L2OPipeline` checks that the method supports the problem class and
# prepares the evaluator used by the loss.

method = PenaltyMethod(; ρ_eq = 20.0, ρ_ineq = 20.0)
pipe = L2OPipeline(problem, method)

# ## Data
#
# Training data are only parameter samples; a smaller test set is solved
# with HiGHS for reference.

rng = Random.MersenneTwister(1)
sampler = BoxSampler([0.5, 1.0], [1.5, 2.0])
train = generate_data(pipe; sampler, n = 128, rng)
test = generate_data(pipe; sampler, n = 32, rng, optimizer = HiGHS.Optimizer)

# ## Training

proxy = Flux.f64(Chain(Dense(2 => 32, relu), Dense(32 => 3)))
history = train_flux!(
    pipe,
    proxy,
    train;
    epochs = 200,
    batchsize = 32,
    rng,
    optimiser = Flux.Optimisers.Adam(1e-2),
)
history[[1, 50, 100, 200]]

# ## Evaluation

prediction = complete(pipe, test.inputs, proxy(test.inputs))
report = evaluate(pipe, test.inputs, prediction)

# Mean absolute optimality gap and worst constraint violation:

sum(abs.(report.objective .- test.objective)) / 32, maximum(report.max_violation)

# The penalty method makes no feasibility promise:

guarantees(method)

# Feasibility can still be measured with the named checks. With a tolerance
# of `1e-2` the check typically fails: the proxy is close to, but not exactly
# on, the feasible set. Methods with a completion step (DC3, DLL, …) are the
# ones that turn such checks into guarantees.

check(Feasible(), problem, test.inputs, prediction; atol = 1e-2)

# ## Adding guarantees with repair layers
#
# Repair layers wrap any primal method. `BoundedOutput` maps the network
# output into the variable bounds during training and inference, so
# `BoundFeasible` holds by construction:

bounded = BoundedOutput(method)
bpipe = L2OPipeline(problem, bounded)
bproxy = Flux.f64(Chain(Dense(2 => 32, relu), Dense(32 => 3)))
train_flux!(
    bpipe,
    bproxy,
    train;
    epochs = 200,
    batchsize = 32,
    rng,
    optimiser = Flux.Optimisers.Adam(1e-2),
)
bpred = complete(bpipe, test.inputs, bproxy(test.inputs))
guarantees(bounded), check(BoundFeasible(), problem, test.inputs, bpred; atol = 0.0)

# `ProjectionRepair` projects the prediction onto the feasible set of each
# instance at inference time (a small QP per instance), which gives
# `Feasible` and never moves the prediction away from the optimum:

repaired = L2OPipeline(problem, ProjectionRepair(bounded, HiGHS.Optimizer))
rpred = complete(repaired, test.inputs, bproxy(test.inputs))
check(Feasible(), problem, test.inputs, rpred; atol = 1e-6)

# Mean optimality gap before and after the repair:

gap(pred) = sum(abs.(evaluate(pipe, test.inputs, pred).objective .- test.objective)) / 32
gap(bpred), gap(rpred)
