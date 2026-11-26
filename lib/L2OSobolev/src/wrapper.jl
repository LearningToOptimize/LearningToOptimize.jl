"""
    SobolevRegularized(method; weight = 1.0, mask = DenseMask(),
                       directions = FullJacobian(), backend = CentralDifferences(),
                       through_completion = false, rng = Random.Xoshiro(0))

Add masked **Sobolev (derivative) supervision** to any `L2OCore` method whose
proxy predicts primal variables (primal and primal-dual families). The
training loss becomes

```math
\\mathcal{L}(θ) = \\mathcal{L}_{\\texttt{method}}(θ)
  + \\frac{λ}{N} \\sum_{i=1}^{N} \\ell_d\\big(\\mathcal{M} \\odot D\\hat g_θ(p_i),\\,
      \\mathcal{M} \\odot Dg(p_i)\\big),
```

eq. (2) of the masked Sobolev training paper with the value term replaced by
the wrapped method's own loss: [`SupervisedMSE`](@ref L2OCore.SupervisedMSE)
gives exactly eq. (2); a self-supervised method (penalty, augmented
Lagrangian) gives the paper's semi-self-supervised variant; a method with a
completion layer can be supervised before or through that layer
(`through_completion`). ``\\ell_d`` is the normalized masked loss of eq. (3)
([`sobolev_loss`](@ref)).

Arguments:

  - `weight` (``λ ≥ 0``): strength of the derivative term;
  - `mask` ([`AbstractDerivativeMask`](@ref)): which entries are matched at each
    loss evaluation, e.g. `RandomMask(0.05)`;
  - `directions` ([`AbstractDerivativeDirections`](@ref)): which derivative
    labels [`generate_data`](@ref L2OCore.generate_data) records;
  - `backend` ([`AbstractSobolevBackend`](@ref)): how proxy input derivatives
    are computed inside the loss ([`CentralDifferences`](@ref) for any proxy,
    [`LuxNestedAD`](@ref) for exact derivatives of Lux models,
    [`ZygoteNestedAD`](@ref) for exact derivatives of Flux models and other
    Zygote-differentiable callables);
  - `through_completion`: differentiate the completed primal prediction
    instead of the raw primal output (see [`ProxyMap`](@ref));
  - `rng`: random stream of the masks.

Pipeline behavior:

  - `supports`, `decompose`, `complete` are the wrapped method's (the
    regularizer changes training only, not how predictions are completed);
  - `generate_data(pipe; sampler, n, optimizer, oracle, ...)` builds the
    wrapped method's dataset and attaches derivative labels from `oracle`
    (default [`DiffOptSensitivities`](@ref)`(optimizer)`); see
    [`L2OCore.generate_data`](@ref);
  - `loss(pipe, P, proxy(P); targets, proxy)` needs `targets` with derivative
    labels and the `proxy` itself (training loops pass `proxy = model`).

The labels are meaningful where the solution map is differentiable at the
training samples (LICQ, SOSC, strict complementarity; Remark 1 of the paper),
and the proxy map must act column by column. The approximation
benefit (Theorem 3 of the paper: uniform error ``≤ \\tfrac12 (M_g + M_{\\hat
g}) δ^2`` under exact value and Jacobian interpolation, versus ``(L_g +
L_{\\hat g}) δ`` for values only) holds in regions where the solution map is
``C^1`` with Lipschitz Jacobian; it is not a statement about the convergence
of training.
"""
struct SobolevRegularized{
    M<:L2OCore.AbstractL2OMethod,
    T<:Real,
    K<:AbstractDerivativeMask,
    D<:AbstractDerivativeDirections,
    B<:AbstractSobolevBackend,
    R<:Random.AbstractRNG,
} <: L2OCore.AbstractMethodWrapper
    inner::M
    weight::T
    mask::K
    directions::D
    backend::B
    through_completion::Bool
    rng::R
end

function SobolevRegularized(
    method::L2OCore.AbstractL2OMethod;
    weight::Real = 1.0,
    mask::AbstractDerivativeMask = DenseMask(),
    directions::AbstractDerivativeDirections = FullJacobian(),
    backend::AbstractSobolevBackend = CentralDifferences(),
    through_completion::Bool = false,
    rng::Random.AbstractRNG = Random.Xoshiro(0),
)
    weight >= 0 || throw(ArgumentError("the Sobolev weight must be ≥ 0"))
    L2OCore.method_family(method) isa L2OCore.DualFamily && throw(
        ArgumentError(
            "SobolevRegularized supervises primal predictions; " *
            "$(nameof(typeof(method))) is a dual method",
        ),
    )
    return SobolevRegularized(
        method,
        float(weight),
        mask,
        directions,
        backend,
        through_completion,
        rng,
    )
end

function Base.show(io::IO, w::SobolevRegularized)
    print(
        io,
        "SobolevRegularized(",
        nameof(typeof(w.inner)),
        "; weight = ",
        w.weight,
        ", mask = ",
        w.mask,
        ", directions = ",
        w.directions,
        ", backend = ",
        nameof(typeof(w.backend)),
        ")",
    )
end

"""
    generate_data(::SobolevRegularized, pipe; sampler, n, rng = Random.default_rng(),
                  optimizer = nothing, oracle = DiffOptSensitivities(optimizer),
                  drop_failed = true, consistency_atol = 1e-4,
                  consistency_rtol = 1e-4, max_label_error = Inf,
                  fd_step = 1e-5, kwargs...)

Dataset of the wrapped method (same `sampler`, `n`, `rng`, `optimizer`,
`kwargs`) with derivative labels of `w.directions` from `oracle` attached as
`ds.derivatives`. Samples whose label solve failed are dropped when
`drop_failed`; so are samples whose primal labels (if the wrapped method
records them) differ from the oracle's solution by more than
`consistency_atol + consistency_rtol * ‖x‖_∞` in the max norm (a different
local optimum would make value and derivative labels describe different
solution branches). The number of dropped samples is stored in `ds.metadata["sobolev_dropped"]`.

With a finite `max_label_error`, the labels are also verified against central
finite differences of the solver map ([`label_errors`](@ref) with `step =
fd_step`, `2d` extra solves per sample) and samples whose relative error
exceeds it are dropped: these are the **degenerate samples** (weakly active
constraints, active-set changes within `± fd_step`) where the solution map is
not differentiable and the KKT sensitivities of eq. (1) of the paper are not
its derivative. Their number is stored in `ds.metadata["sobolev_degenerate"]`.
"""
function L2OCore.generate_data(
    w::SobolevRegularized,
    pipe;
    sampler,
    n::Integer,
    rng::Random.AbstractRNG = Random.default_rng(),
    optimizer = nothing,
    oracle = nothing,
    drop_failed::Bool = true,
    consistency_atol::Real = 1e-4,
    consistency_rtol::Real = 1e-4,
    max_label_error::Real = Inf,
    fd_step::Real = 1e-5,
    kwargs...,
)
    ds = L2OCore.generate_data(
        L2OCore.inner_method(w),
        pipe;
        sampler = sampler,
        n = n,
        rng = rng,
        optimizer = optimizer,
        kwargs...,
    )
    if oracle === nothing
        optimizer === nothing && throw(
            ArgumentError("derivative labels need a solver: pass `optimizer` or `oracle`"),
        )
        oracle = DiffOptSensitivities(optimizer)
    end
    r = sensitivity_labels(oracle, pipe.problem, ds.inputs, w.directions; rng = rng)
    ok = copy(r.ok)
    if ds.primal !== nothing
        gap = vec(maximum(abs.(ds.primal .- r.primal); dims = 1))
        scale = vec(maximum(abs.(ds.primal); dims = 1))
        ok .&= gap .<= consistency_atol .+ consistency_rtol .* scale
    end
    degenerate = falses(length(ok))
    if isfinite(max_label_error)
        optimizer === nothing && throw(
            ArgumentError(
                "`max_label_error` needs `optimizer` for the finite-difference check",
            ),
        )
        errs = label_errors(r.labels, pipe.problem, ds.inputs; optimizer, step = fd_step)
        degenerate .= ok .& .!(errs .<= max_label_error)        # NaN errors count as degenerate
        ok .&= .!degenerate
    end
    out = L2OCore.with_derivatives(ds, r.labels)
    isfinite(max_label_error) && (out.metadata["sobolev_degenerate"] = count(degenerate))
    out.metadata["sobolev_directions"] = string(w.directions)
    out.metadata["sobolev_oracle"] = string(nameof(typeof(oracle)))
    out.metadata["sobolev_dropped"] = drop_failed ? count(!, ok) : 0
    drop_failed && !all(ok) && return out[findall(ok)]
    return out
end

function L2OCore.extra_loss(
    w::SobolevRegularized,
    pipe,
    inputs,
    output;
    targets = nothing,
    proxy = nothing,
    kwargs...,
)
    iszero(w.weight) && return zero(eltype(output))
    labels = targets === nothing ? nothing : targets.derivatives
    labels === nothing && throw(
        ArgumentError(
            "SobolevRegularized needs `targets` with derivative labels " *
            "(see `generate_data` and `with_derivatives`)",
        ),
    )
    proxy === nothing && throw(
        ArgumentError(
            "SobolevRegularized differentiates the proxy: pass `proxy = model` to `loss`",
        ),
    )
    f = ProxyMap(proxy, pipe, w.through_completion)
    return w.weight * sobolev_loss(w.backend, f, inputs, labels; mask = w.mask, rng = w.rng)
end
