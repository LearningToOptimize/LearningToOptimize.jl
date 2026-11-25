"""
    PenaltyMethod(; ρ_eq = 10.0, ρ_ineq = 10.0, squared = true)

Self-supervised **penalty method**: the reference primal method of the
LearningToOptimize ecosystem, implemented only through the `L2OCore`
interface.

The proxy ``x_θ(p)`` predicts all decision variables and is trained to
minimize, over a distribution of parameters,

```math
\\mathcal{L}(θ) = \\mathbb{E}_p\\Big[\\, s\\, f\\big(x_θ(p); p\\big)
  + ρ_{\\mathrm{eq}}\\, \\big\\|ν_{\\mathrm{eq}}\\big(x_θ(p); p\\big)\\big\\|
  + ρ_{\\mathrm{ineq}}\\, \\big\\|ν_{\\mathrm{ineq}}\\big(x_θ(p); p\\big)\\big\\| \\Big]
```

where ``s = ±1`` turns maximization into minimization, ``ν`` are the
element-wise constraint violations returned by
[`L2OCore.row_violation`](@ref) (equality rows and all other rows,
respectively, plus distances to non-decomposable cones) and ``\\|·\\|`` is the squared 2-norm when
`squared = true` and the 1-norm otherwise. With `squared = true` this is the
soft loss of eq. (2) in Donti, Rolnick & Kolter, "DC3: A learning method for
optimization with hard constraints" (ICLR 2021, arXiv:2104.12225), with
``λ_h = ρ_eq`` and ``λ_g = ρ_ineq``; box rows (two-sided constraints, variable
bounds) are penalized through ``\\max(l - f, 0) + \\max(f - u, 0)``, which is
the ``\\mathrm{ReLU}(g)`` of eq. (2) for each side.

The method supports the problem classes [`LinearConic`](@ref) and
[`ConvexQuadratic`](@ref), so the default `MatrixEvaluator` is used and the
loss is differentiable with array AD. No solver is needed for training.

Penalty proxies are in general *infeasible*; pair them with a completion or
repair method when feasibility is required, and measure their violations with
[`evaluate`](@ref).
"""
struct PenaltyMethod{T<:Real} <: L2OCore.AbstractPrimalMethod
    ρ_eq::T
    ρ_ineq::T
    squared::Bool
end

function PenaltyMethod(; ρ_eq::Real = 10.0, ρ_ineq::Real = 10.0, squared::Bool = true)
    (ρ_eq >= 0 && ρ_ineq >= 0) || throw(ArgumentError("penalty weights must be ≥ 0"))
    T = float(promote_type(typeof(ρ_eq), typeof(ρ_ineq)))
    return PenaltyMethod{T}(T(ρ_eq), T(ρ_ineq), squared)
end

L2OCore.supports(::PenaltyMethod, ::L2OCore.LinearConic) = true
L2OCore.supports(::PenaltyMethod, ::L2OCore.ConvexQuadratic) = true

# The loss is differentiated through the matrix-form evaluator; problems the
# matrix form cannot represent (e.g. parameter × variable products in
# constraints) are rejected when the pipeline is built, with a clear message,
# instead of failing inside automatic differentiation.
function L2OCore.decompose(problem, m::PenaltyMethod)
    L2OCore.has_matrix_form(problem) || throw(
        ArgumentError(
            "PenaltyMethod differentiates its loss through MatrixEvaluator, which needs " *
            "affine constraints with parameters only in the constants and an at most " *
            "quadratic objective. This problem has no such form; a kernel-based " *
            "evaluator is needed (planned).",
        ),
    )
    return invoke(L2OCore.decompose, Tuple{Any,L2OCore.AbstractL2OMethod}, problem, m)
end

"""
    generate_data(::PenaltyMethod, pipe; sampler, n, rng, optimizer = nothing)

Penalty training is self-supervised: only parameters are sampled. When an
`optimizer` is given, instances are also solved so that the dataset carries
reference optima for evaluation (objective gaps, dual bounds).
"""
function L2OCore.generate_data(
    ::PenaltyMethod,
    pipe;
    sampler,
    n::Integer,
    rng::Random.AbstractRNG = Random.default_rng(),
    optimizer = nothing,
    kwargs...,
)
    P = L2OCore.sample_parameters(sampler, rng, n)
    optimizer === nothing && return L2OCore.L2ODataset(Matrix{Float64}(P))
    return L2OCore.solve_instances(pipe.problem, P; optimizer = optimizer, kwargs...)
end

_penalty_norm(V, squared::Bool) = squared ? sum(abs2, V) : sum(V)

# Soft loss, eq. (2) of DC3 (arXiv:2104.12225): f + λ_h‖h‖² + λ_g‖ReLU(g)‖².
function L2OCore.loss(m::PenaltyMethod, pipe, inputs, output; kwargs...)
    ev = pipe.evaluator
    X = L2OCore.complete(m, pipe, inputs, output).primal
    s = Zygote.ignore(() -> L2OCore.objective_sign(pipe.problem))
    eq = Zygote.ignore(() -> L2OCore.equality_rows(ev))
    B = size(X, 2)
    V = L2OCore.row_violation(ev, X, inputs)
    obj = s * sum(L2OCore.batch_objective(ev, X, inputs))
    pen =
        m.ρ_eq * _penalty_norm(V .* eq, m.squared) +
        m.ρ_ineq * _penalty_norm(V .* .!eq, m.squared)
    has_cones = Zygote.ignore(() -> !isempty(ev.cones))
    if has_cones
        # Distances to non-decomposable cones (e.g. second-order cones) are
        # inequality constraints g(x) ∈ K: penalized with ρ_ineq like ReLU(g).
        pen += m.ρ_ineq * _penalty_norm(L2OCore.cone_violation(ev, X, inputs), m.squared)
    end
    return (obj + pen) / B
end

"""
    train_flux!(pipe::L2OPipeline, model, data::L2ODataset; epochs = 100,
                batchsize = 32, optimiser = Optimisers.Adam(1e-3),
                rng = Random.default_rng(), callback = nothing) -> Vector{Float64}

Train a Flux model in place on `pipe` by minimizing
`loss(pipe, X, model(X); targets, proxy = model)` with mini-batches of columns
of `data`. Works for every method whose loss is differentiable with Zygote.
Returns the mean training loss of each epoch. `callback(epoch, loss)` is
called after every epoch.
"""
function train_flux!(
    pipe::L2OCore.L2OPipeline,
    model,
    data::L2OCore.L2ODataset;
    epochs::Integer = 100,
    batchsize::Integer = 32,
    optimiser = Optimisers.Adam(1e-3),
    rng::Random.AbstractRNG = Random.default_rng(),
    callback = nothing,
)
    state = Flux.setup(optimiser, model)
    N = L2OCore.num_samples(data)
    history = Float64[]
    for epoch = 1:epochs
        perm = Random.randperm(rng, N)
        total = 0.0
        for start = 1:batchsize:N
            idx = perm[start:min(start+batchsize-1, N)]
            batch = data[idx]
            X = batch.inputs
            ℓ, grads = Flux.withgradient(model) do mdl
                L2OCore.loss(pipe, X, mdl(X); targets = batch, proxy = mdl)
            end
            Flux.update!(state, model, grads[1])
            total += ℓ * length(idx)
        end
        push!(history, total / N)
        callback === nothing || callback(epoch, history[end])
    end
    return history
end
