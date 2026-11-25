# # Sobolev-regularized proxy for a parametric QP
#
# This example trains two proxies for the solution map of a small parametric
# quadratic program from only **8 solved instances**:
#
# - a plain value-supervised proxy ([`SupervisedMSE`](@ref L2OCore.SupervisedMSE));
# - the same method wrapped in [`SobolevRegularized`](@ref L2OSobolev.SobolevRegularized),
#   which also matches the solver sensitivities ``∂x^\star/∂p`` (eq. (2) of the
#   masked Sobolev training paper; see the [Sobolev theory page](@ref "Theory: Sobolev (derivative) supervision")).
#
# Everything runs on the CPU in well under a minute.

using L2OCore, L2OSobolev
using JuMP, HiGHS, DiffOpt                   # problem, solver, sensitivity labels
using Lux, Zygote, Optimisers, ComponentArrays
using Random, Statistics

# ## The parametric problem
#
# Four decision variables, three parameters: two enter the linear cost, one the
# right-hand side of a budget constraint. The lower bounds are never active for
# the sampled parameters, so the solution map is smooth (affine) on the whole
# sampling box: the regular regime in which derivative supervision is expected
# to help (Theorem 3 of the paper).

model = Model()
@variable(model, p[1:3] in MOI.Parameter.(zeros(3)))
@variable(model, x[1:4] >= -5)
@constraint(model, sum(x) == 1 + p[3])
@objective(
    model,
    Min,
    sum((i + 1) * x[i]^2 for i = 1:4) + x[1] * x[2] - p[1] * x[1] - p[2] * x[2] +
    0.5 * p[3] * x[4]
)
problem = ParametricProblem(model)

# ## Methods and data
#
# `SobolevRegularized` keeps everything of the wrapped method (supported
# problem classes, decomposition, completion) and adds the derivative term. `LuxNestedAD()`
# computes the proxy Jacobian exactly with Lux's nested-AD helpers.

plain = SupervisedMSE()
sobolev = SobolevRegularized(plain; weight = 1.0, backend = LuxNestedAD())

# `generate_data` solves every sampled instance (value labels) and
# differentiates the KKT conditions with DiffOpt (derivative labels, here the
# full ``4 × 3`` Jacobian per sample).

sampler = BoxSampler(-ones(3), ones(3))
pipe = L2OPipeline(problem, sobolev)
train = generate_data(pipe; sampler, n = 8, rng = Xoshiro(2), optimizer = HiGHS.Optimizer)
test = generate_data(pipe; sampler, n = 200, rng = Xoshiro(3), optimizer = HiGHS.Optimizer)
train

# The labels can be compared with finite differences of the solver map:
# small errors mean the samples are regular points (no active-set change).

maximum(label_errors(train.derivatives, problem, train.inputs; optimizer = HiGHS.Optimizer))

# ## Training loop (Lux)
#
# Any training loop works as long as it passes the proxy to `loss`
# (`proxy = sm`), because the Sobolev term differentiates it with respect to
# its inputs.

function train_proxy(method, data; epochs = 800)
    pipe = L2OPipeline(problem, method)
    net = Lux.Chain(Lux.Dense(3 => 32, tanh), Lux.Dense(32 => 4))
    ps, st = Lux.setup(Xoshiro(1), net)
    ps = ComponentArray(ps)
    opt = Optimisers.setup(Optimisers.Adam(5e-3), ps)
    X = Float32.(data.inputs)
    for _ = 1:epochs
        g = Zygote.gradient(ps) do θ
            sm = StatefulLuxLayer{true}(net, θ, st)
            loss(pipe, X, sm(X); targets = data, proxy = sm)
        end[1]
        opt, ps = Optimisers.update!(opt, ps, g)
    end
    return StatefulLuxLayer{true}(net, ps, st)
end

proxies =
    Dict("plain" => train_proxy(plain, train), "Sobolev" => train_proxy(sobolev, train));
nothing #hide

# ## Generalisation from 8 samples
#
# Test value error and test Jacobian error of both proxies:

Xt = Float32.(test.inputs)
for (name, sm) in sort(collect(proxies); by = first)
    J = proxy_jacobian(LuxNestedAD(), ProxyMap(sm, pipe), Xt)
    println(
        rpad(name, 8),
        " value MSE = ",
        round(mean(abs2, sm(Xt) .- test.primal); sigdigits = 3),
        "   Jacobian MSE = ",
        round(mean(abs2, J .- test.derivatives.J); sigdigits = 3),
    )
end

# Each sample now carries ``4 × 3 = 12`` derivative conditions in addition to
# its 4 values, and the Sobolev proxy generalises markedly better.
#
# ## Variations
#
# - **Masking** (the paper's algorithmic control): match a random 25% of the
#   Jacobian entries per update with `mask = RandomMask(0.25)`. The masked loss
#   is normalised so it is an unbiased estimate of the dense one.
# - **Cheaper labels**: `directions = RandomPullbacks()` records one random row
#   of the Jacobian per sample (one reverse sensitivity solve) instead of all of
#   it.
# - **Any proxy / any framework**: `backend = CentralDifferences()` (the
#   default) approximates the proxy Jacobian with batched forward passes and
#   works with Flux models or plain functions, and through completion layers
#   (`through_completion = true`).
# - **Self-supervised methods**: wrap a penalty or augmented-Lagrangian method
#   instead of `SupervisedMSE`; the dataset then carries derivative labels but
#   the value term needs no labels.

masked = SobolevRegularized(
    plain;
    weight = 1.0,
    mask = RandomMask(0.25),
    backend = LuxNestedAD(),
)
sm = train_proxy(masked, train)
mean(abs2, sm(Xt) .- test.primal)
