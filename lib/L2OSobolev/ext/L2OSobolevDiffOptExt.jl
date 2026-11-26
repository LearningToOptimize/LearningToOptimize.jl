module L2OSobolevDiffOptExt

import DiffOpt
import JuMP
import L2OCore
import L2OSobolev
import Random

const MOI = JuMP.MOI

function _model_constructor(o::L2OSobolev.DiffOptSensitivities, problem)
    o.model_constructor === nothing || return o.model_constructor
    L2OCore.problem_class(problem) isa L2OCore.SmoothNonlinear &&
        return DiffOpt.NonLinearProgram.Model
    return nothing
end

# Differentiate a copy of the JuMP model: DiffOpt's input sensitivities are
# stored as model attributes and would otherwise leak into later solves of the
# user's model with a non-differentiable optimizer.
function _prepare(o::L2OSobolev.DiffOptSensitivities, problem)
    model, refmap = JuMP.copy_model(problem.model)
    JuMP.set_optimizer(model, () -> DiffOpt.diff_optimizer(o.optimizer))
    o.silent && JuMP.set_silent(model)
    mc = _model_constructor(o, problem)
    mc === nothing || JuMP.set_attribute(model, DiffOpt.ModelConstructor(), mc)
    ps = [refmap[p] for p in L2OCore.parameters(problem)]
    xs = [refmap[x] for x in L2OCore.decision_variables(problem)]
    return model, ps, xs
end

_ok(model) =
    JuMP.termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) &&
    JuMP.primal_status(model) == MOI.FEASIBLE_POINT

# Forward mode (eq. (1) applied to a parameter direction v): J v.
function _pushforward!(model, ps, xs, v::AbstractVector)
    for (p, vj) in zip(ps, v)
        MOI.set(
            model,
            DiffOpt.ForwardConstraintSet(),
            JuMP.ParameterRef(p),
            MOI.Parameter(Float64(vj)),
        )
    end
    DiffOpt.forward_differentiate!(model)
    return [MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) for x in xs]
end

# Reverse mode: Jᵀ w.
function _pullback!(model, ps, xs, w::AbstractVector)
    for (x, wi) in zip(xs, w)
        MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, Float64(wi))
    end
    DiffOpt.reverse_differentiate!(model)
    return [
        MOI.get(model, DiffOpt.ReverseConstraintSet(), JuMP.ParameterRef(p)).value for
        p in ps
    ]
end

function _jacobian!(model, ps, xs, mode::Symbol)
    n, d = length(xs), length(ps)
    J = zeros(n, d)
    if mode === :forward
        for j = 1:d
            J[:, j] .= _pushforward!(model, ps, xs, [i == j ? 1.0 : 0.0 for i = 1:d])
        end
    else
        for i = 1:n
            J[i, :] .= _pullback!(model, ps, xs, [k == i ? 1.0 : 0.0 for k = 1:n])
        end
    end
    return J
end

# Solve every instance and call `record!(model, k)` on the solved ones.
function _foreach_solved(record!, o, problem, P::AbstractMatrix)
    model, ps, xs = _prepare(o, problem)
    N = size(P, 2)
    X = fill(NaN, length(xs), N)
    ok = falses(N)
    for k = 1:N
        JuMP.set_parameter_value.(ps, P[:, k])
        JuMP.optimize!(model)
        _ok(model) || continue
        X[:, k] .= JuMP.value.(xs)
        ok[k] = try
            record!(model, ps, xs, k)
            true
        catch err
            err isa InterruptException && rethrow()
            @debug "DiffOpt differentiation failed for sample $k" exception = err
            false
        end
    end
    return X, ok
end

function L2OSobolev.sensitivity_labels(
    o::L2OSobolev.DiffOptSensitivities,
    problem::L2OCore.ParametricProblem,
    P::AbstractMatrix,
    ::L2OSobolev.FullJacobian;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    J = fill(NaN, L2OCore.num_decision_variables(problem), size(P)...)
    X, ok = _foreach_solved(o, problem, P) do model, ps, xs, k
        return J[:, :, k] .= _jacobian!(model, ps, xs, o.mode)
    end
    J[:, :, .!ok] .= NaN
    return (labels = L2OCore.JacobianLabels(J), primal = X, ok = ok)
end

function L2OSobolev.sensitivity_labels(
    o::L2OSobolev.DiffOptSensitivities,
    problem::L2OCore.ParametricProblem,
    P::AbstractMatrix,
    dirs::L2OSobolev.RandomPullbacks;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    n, (d, N) = L2OCore.num_decision_variables(problem), size(P)
    W = L2OSobolev._draw(dirs.distribution, rng, n, N)
    vals = fill(NaN, d, N)
    X, ok = _foreach_solved(o, problem, P) do model, ps, xs, k
        vals[:, k] .=
            o.mode === :reverse ? _pullback!(model, ps, xs, W[:, k]) :
            _jacobian!(model, ps, xs, :forward)' * W[:, k]
        return nothing
    end
    vals[:, .!ok] .= NaN
    return (labels = L2OCore.PullbackLabels(W, vals), primal = X, ok = ok)
end

function L2OSobolev.sensitivity_labels(
    o::L2OSobolev.DiffOptSensitivities,
    problem::L2OCore.ParametricProblem,
    P::AbstractMatrix,
    dirs::L2OSobolev.RandomPushforwards;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    n, (d, N) = L2OCore.num_decision_variables(problem), size(P)
    V = L2OSobolev._draw(dirs.distribution, rng, d, N)
    vals = fill(NaN, n, N)
    X, ok = _foreach_solved(o, problem, P) do model, ps, xs, k
        vals[:, k] .=
            o.mode === :forward ? _pushforward!(model, ps, xs, V[:, k]) :
            _jacobian!(model, ps, xs, :reverse) * V[:, k]
        return nothing
    end
    vals[:, .!ok] .= NaN
    return (labels = L2OCore.PushforwardLabels(V, vals), primal = X, ok = ok)
end

end # module L2OSobolevDiffOptExt
