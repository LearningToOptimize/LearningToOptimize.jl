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

# The penalty loss only discourages violations, so the proxy lands close to,
# but not exactly on, the feasible set. The report separates equality and
# inequality (here: capacity bound) violations:

maximum(report.max_equality_violation), maximum(report.max_inequality_violation)

# Fraction of test instances whose worst violation is below `1e-2` (often
# none at this training budget). Methods with a completion step (DC3, DLL, …)
# are the ones that restore feasibility or dual feasibility exactly.

count(<=(1e-2), report.max_violation) / 32

# ## Repair layers
#
# Repair layers wrap any primal method. `BoundedOutput` maps the network
# output into the variable bounds during training and inference, so the
# bounds hold by construction. Here the only inequalities are the capacity
# bounds, so their largest violation is exactly zero:

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
maximum(evaluate(pipe, test.inputs, bpred).max_inequality_violation)

# `ProjectionRepair` projects the prediction onto the feasible set of each
# instance at inference time (a small QP per instance). The result is feasible
# up to the solver tolerance, and since the projection onto a convex set is
# nonexpansive it is never farther from the optimal solution than the raw
# prediction:

repaired = L2OPipeline(problem, ProjectionRepair(bounded, HiGHS.Optimizer))
rpred = complete(repaired, test.inputs, bproxy(test.inputs))
maximum(evaluate(pipe, test.inputs, rpred).max_violation)

# Mean optimality gap before and after the repair:

gap(pred) = sum(abs.(evaluate(pipe, test.inputs, pred).objective .- test.objective)) / 32
gap(bpred), gap(rpred)
