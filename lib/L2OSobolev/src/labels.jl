"""
    AbstractDerivativeDirections

Which derivative information of the solution map ``g(p) = x^\\star(p)`` is
recorded per sample:

  - [`FullJacobian`](@ref) → [`JacobianLabels`](@ref L2OCore.JacobianLabels);
  - [`RandomPullbacks`](@ref) → [`PullbackLabels`](@ref L2OCore.PullbackLabels);
  - [`RandomPushforwards`](@ref) → [`PushforwardLabels`](@ref L2OCore.PushforwardLabels).
"""
abstract type AbstractDerivativeDirections end

"""
    FullJacobian()

Record the full sensitivity matrix ``Dg(p_k) = ∂x^\\star/∂p`` (`n × d`) per
sample. Costs `min(n, d)` sensitivity solves per sample with
[`DiffOptSensitivities`](@ref).
"""
struct FullJacobian <: AbstractDerivativeDirections end

const _DISTRIBUTIONS = (:onehot, :gaussian)

"""
    RandomPullbacks(; distribution = :onehot)

Record one vector–Jacobian product ``Dg(p_k)^⊤ w_k`` per sample (one reverse
sensitivity solve). Seeds ``w_k ∈ ℝ^n`` are drawn per sample: `:onehot` picks a
uniformly random decision variable (the label is a random *row* of the
Jacobian, i.e. a sparse, stochastic subset of its entries), `:gaussian` a
uniformly random unit direction.
"""
struct RandomPullbacks <: AbstractDerivativeDirections
    distribution::Symbol
    function RandomPullbacks(; distribution::Symbol = :onehot)
        distribution in _DISTRIBUTIONS ||
            throw(ArgumentError("distribution must be one of $_DISTRIBUTIONS"))
        return new(distribution)
    end
end

"""
    RandomPushforwards(; distribution = :onehot)

Record one Jacobian–vector product ``Dg(p_k)\\, v_k`` per sample (one forward
sensitivity solve). Directions ``v_k ∈ ℝ^d``: `:onehot` picks a random
parameter (a random *column* of the Jacobian), `:gaussian` a random unit
direction.
"""
struct RandomPushforwards <: AbstractDerivativeDirections
    distribution::Symbol
    function RandomPushforwards(; distribution::Symbol = :onehot)
        distribution in _DISTRIBUTIONS ||
            throw(ArgumentError("distribution must be one of $_DISTRIBUTIONS"))
        return new(distribution)
    end
end

function _draw(distribution::Symbol, rng::Random.AbstractRNG, m::Integer, N::Integer)
    if distribution === :onehot
        S = zeros(m, N)
        for k = 1:N
            S[rand(rng, 1:m), k] = 1
        end
        return S
    end
    S = randn(rng, m, N)
    return S ./ sqrt.(sum(abs2, S; dims = 1))
end

"""
    AbstractSensitivityOracle

Source of derivative labels ``Dg(p) = ∂x^\\star/∂p`` of a parametric problem.
Implement [`sensitivity_labels`](@ref) for a new oracle (for example a
batched GPU solver).
"""
abstract type AbstractSensitivityOracle end

"""
    DiffOptSensitivities(optimizer; mode = :reverse, model_constructor = nothing,
                         silent = true)

Derivative labels by implicit differentiation of the KKT conditions with
[DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl): at a solution
``s^\\star = (x^\\star, λ^\\star)`` of ``F(s, p) = 0`` (stationarity, primal
feasibility, complementarity),

```math
∇_p s^\\star = -\\big(∇_s F(s^\\star, p)\\big)^{-1} ∇_p F(s^\\star, p)
```

(eq. (1) of the masked Sobolev training paper). Reverse mode
(`mode = :reverse`) gives vector–Jacobian products, forward mode
(`mode = :forward`) Jacobian–vector products; both reuse one solve per sample.
Labels of the other kind are assembled from full Jacobians (`n` reverse or `d`
forward sensitivity solves). The default is reverse mode: with DiffOpt 0.6 we
measured wrong forward-mode sensitivities for parameters that enter inequality
constraints, while reverse mode agreed with finite differences on every test
problem (DiffOpt 0.5 was correct in both modes).

`optimizer` is the solver constructor (wrapped with `DiffOpt.diff_optimizer`).
`model_constructor` selects DiffOpt's differentiation model
(`DiffOpt.QuadraticProgram.Model`, `DiffOpt.ConicProgram.Model`,
`DiffOpt.NonLinearProgram.Model`); `nothing` picks `NonLinearProgram` for
[`SmoothNonlinear`](@ref L2OCore.SmoothNonlinear) problems and DiffOpt's
automatic choice otherwise.

The labels are the derivatives of the solution map under LICQ, second-order
sufficient conditions and strict complementarity at ``x^\\star(p)``
(Fiacco 1976), so that the solution map is locally
differentiable and ``∇_s F`` is invertible. At degenerate points (weakly active
constraints, active-set changes) the result is one element of a generalized
derivative and can disagree with finite differences; see
[`label_errors`](@ref).

Requires `using DiffOpt` (package extension). Labels are computed on a copy
of the problem's JuMP model (`JuMP.copy_model`), so the user's model is left
unchanged.
"""
struct DiffOptSensitivities{O,C} <: AbstractSensitivityOracle
    optimizer::O
    mode::Symbol
    model_constructor::C
    silent::Bool
    function DiffOptSensitivities(
        optimizer::O,
        mode::Symbol,
        mc::C,
        silent::Bool,
    ) where {O,C}
        mode in (:reverse, :forward) ||
            throw(ArgumentError("mode must be :reverse or :forward"))
        return new{O,C}(optimizer, mode, mc, silent)
    end
end
DiffOptSensitivities(
    optimizer;
    mode::Symbol = :reverse,
    model_constructor = nothing,
    silent::Bool = true,
) = DiffOptSensitivities(optimizer, mode, model_constructor, silent)

"""
    FiniteDifferenceSensitivities(optimizer; step = 1e-5)

Reference oracle by central differences of the solver's solution map,
``Dg(p) v ≈ (x^\\star(p + h v) - x^\\star(p - h v)) / 2h``, using only
[`solve_instances`](@ref L2OCore.solve_instances) (any JuMP solver, no
differentiation layer). Costs `2d` solves per sample for Jacobians and
pullbacks and `2` per sample for pushforwards; used to verify other oracles
([`label_errors`](@ref)) and as a fallback when no differentiable solver is
available.

The differences are accurate when the solution is unique, the active set does
not change within ``[p - h v, p + h v]`` and the solver tolerance is well
below `step`.
"""
struct FiniteDifferenceSensitivities{O,T<:Real} <: AbstractSensitivityOracle
    optimizer::O
    step::T
end
FiniteDifferenceSensitivities(optimizer; step::Real = 1e-5) =
    FiniteDifferenceSensitivities(optimizer, float(step))

"""
    sensitivity_labels(oracle, problem, inputs, directions;
                       rng = Random.default_rng()) -> (; labels, primal, ok)

Solve the instances in the columns of `inputs` (`num_parameters × N`) and
record derivative labels of the requested `directions`
([`AbstractDerivativeDirections`](@ref)). Returns a named tuple with the
labels (an [`AbstractDerivativeLabels`](@ref L2OCore.AbstractDerivativeLabels)
with `NaN` columns for failed samples), the primal solutions
(`num_decision_variables × N`) and a `BitVector` `ok` of samples whose solves
(and differentiation) succeeded.
"""
function sensitivity_labels end

function sensitivity_labels(
    o::AbstractSensitivityOracle,
    problem,
    inputs::AbstractMatrix,
    directions::AbstractDerivativeDirections;
    kwargs...,
)
    throw(
        ArgumentError(
            "$(nameof(typeof(o))) has no `sensitivity_labels` method for " *
            "$(nameof(typeof(problem)))" *
            (o isa DiffOptSensitivities ? "; load DiffOpt (`using DiffOpt`)" : ""),
        ),
    )
end

_solved(s::MOI.TerminationStatusCode) = s == MOI.OPTIMAL || s == MOI.LOCALLY_SOLVED

# Solve the instances P with `optimizer`; returns primal matrix and ok flags.
function _solve(problem, P::AbstractMatrix, optimizer)
    ds = L2OCore.solve_instances(problem, P; optimizer = optimizer, primal = true)
    ok = _solved.(ds.status) .& vec(all(isfinite, ds.primal; dims = 1))
    return ds.primal, ok
end

# Finite-difference Jacobians, n × d × N, and ok flags.
function _fd_jacobians(o::FiniteDifferenceSensitivities, problem, P::AbstractMatrix)
    d, N = size(P)
    h = o.step
    # Columns: N base points, then for every sample k the 2d perturbed points.
    E = Matrix{Float64}(LinearAlgebra.I, d, d)
    Pall = hcat(P, reduce(hcat, (hcat(P[:, k] .+ h .* E, P[:, k] .- h .* E) for k = 1:N)))
    X, okall = _solve(problem, Pall, o.optimizer)
    n = size(X, 1)
    J = fill(NaN, n, d, N)
    ok = falses(N)
    for k = 1:N
        cols = N .+ (((k-1)*2d+1):(k*2d))
        okall[k] && all(okall[cols]) || continue
        Xk = X[:, cols]
        J[:, :, k] .= (Xk[:, 1:d] .- Xk[:, (d+1):2d]) ./ (2h)
        ok[k] = true
    end
    return J, X[:, 1:N], ok
end

function sensitivity_labels(
    o::FiniteDifferenceSensitivities,
    problem::L2OCore.ParametricProblem,
    P::AbstractMatrix,
    ::FullJacobian;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    J, X, ok = _fd_jacobians(o, problem, P)
    return (labels = L2OCore.JacobianLabels(J), primal = X, ok = ok)
end

function sensitivity_labels(
    o::FiniteDifferenceSensitivities,
    problem::L2OCore.ParametricProblem,
    P::AbstractMatrix,
    dirs::RandomPullbacks;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    J, X, ok = _fd_jacobians(o, problem, P)
    W = _draw(dirs.distribution, rng, size(J, 1), size(P, 2))
    return (labels = L2OCore.PullbackLabels(W, _pullbacks(J, W)), primal = X, ok = ok)
end

function sensitivity_labels(
    o::FiniteDifferenceSensitivities,
    problem::L2OCore.ParametricProblem,
    P::AbstractMatrix,
    dirs::RandomPushforwards;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    V = _draw(dirs.distribution, rng, size(P, 1), size(P, 2))
    return _fd_pushforwards(o, problem, P, V)
end

function _fd_pushforwards(o::FiniteDifferenceSensitivities, problem, P, V)
    N = size(P, 2)
    h = o.step
    X, okall = _solve(problem, hcat(P, P .+ h .* V, P .- h .* V), o.optimizer)
    ok = okall[1:N] .& okall[(N+1):2N] .& okall[(2N+1):3N]
    vals = (X[:, (N+1):2N] .- X[:, (2N+1):3N]) ./ (2h)
    vals[:, .!ok] .= NaN
    return (labels = L2OCore.PushforwardLabels(V, vals), primal = X[:, 1:N], ok = ok)
end

function _pullbacks(J::AbstractArray{<:Real,3}, W::AbstractMatrix)
    vals = similar(W, size(J, 2), size(J, 3))
    for k in axes(J, 3)
        vals[:, k] .= J[:, :, k]' * W[:, k]
    end
    return vals
end

"""
    label_errors(labels, problem, inputs; optimizer, step = 1e-5) -> Vector{Float64}

Check derivative labels against central finite differences of the solver's
solution map (under LICQ, SOSC and strict complementarity the KKT
sensitivities of eq. (1) of the paper *are* the derivatives of
``p \\mapsto x^\\star(p)``). Returns, per sample, the relative
error ``\\|D - D_{\\mathrm{fd}}\\| / \\max(1, \\|D_{\\mathrm{fd}}\\|)`` (`NaN`
if a perturbed solve failed). Large errors flag degenerate samples where the
solution map is not differentiable (active-set changes within `± step`).
"""
function label_errors(
    labels::L2OCore.AbstractDerivativeLabels,
    problem::L2OCore.ParametricProblem,
    inputs::AbstractMatrix;
    optimizer,
    step::Real = 1e-5,
)
    o = FiniteDifferenceSensitivities(optimizer; step = step)
    ref, fd = _fd_reference(o, labels, problem, inputs)
    return [
        ok ? LinearAlgebra.norm(a .- b) / max(1, LinearAlgebra.norm(b)) : NaN for
        (a, b, ok) in zip(_columns(labels), _columns(ref), fd)
    ]
end

function _fd_reference(o, labels::L2OCore.JacobianLabels, problem, P)
    J, _, ok = _fd_jacobians(o, problem, P)
    return L2OCore.JacobianLabels(J), ok
end
function _fd_reference(o, labels::L2OCore.PullbackLabels, problem, P)
    J, _, ok = _fd_jacobians(o, problem, P)
    return L2OCore.PullbackLabels(labels.seeds, _pullbacks(J, labels.seeds)), ok
end
function _fd_reference(o, labels::L2OCore.PushforwardLabels, problem, P)
    r = _fd_pushforwards(o, problem, P, labels.directions)
    return r.labels, r.ok
end

_columns(l::L2OCore.JacobianLabels) = eachslice(l.J; dims = 3)
_columns(l::L2OCore.PullbackLabels) = eachcol(l.values)
_columns(l::L2OCore.PushforwardLabels) = eachcol(l.values)
