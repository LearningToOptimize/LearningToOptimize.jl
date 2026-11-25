"""
    Prediction(; primal = nothing, dual = nothing)

A batch of (completed) predictions: `primal` is `num_decision_variables × B` in
[`decision_variables`](@ref) order, `dual` is `num_duals × B` in
[`dual_layout`](@ref) order. Either may be `nothing`.
"""
struct Prediction{X,Y}
    primal::X
    dual::Y
end
Prediction(; primal = nothing, dual = nothing) = Prediction(primal, dual)

"""
    AbstractDecomposition

Method-specific structure computed once per problem by [`decompose`](@ref):
which quantities the proxy predicts and how the rest is recovered by
[`complete`](@ref). Examples: DLL's split of the duals into predicted `y` and
completed `z`; DC3's independent/dependent variables. A decomposition must
implement [`output_dimension`](@ref).
"""
abstract type AbstractDecomposition end

"""
    FullSpaceDecomposition(primal::Bool, dual::Bool)

The family fallback: the proxy predicts *all* primal variables (if `primal`)
followed by *all* duals (if `dual`), stacked vertically, and completion is the
identity.
"""
struct FullSpaceDecomposition <: AbstractDecomposition
    primal::Bool
    dual::Bool
    nprimal::Int
    ndual::Int
end

"""
    output_dimension(decomposition::AbstractDecomposition) -> Int
    output_dimension(pipe::L2OPipeline) -> Int

Number of rows the proxy must output.
"""
output_dimension(d::FullSpaceDecomposition) = d.primal * d.nprimal + d.dual * d.ndual

"""
    decompose(problem::ParametricProblem, method::AbstractL2OMethod) -> AbstractDecomposition

Compute the method-specific structure of `problem`. Called once by
[`L2OPipeline`](@ref) after [`check_supports`](@ref).

Fallback by family ([`FullSpaceDecomposition`](@ref)): primal methods predict
all primal variables, dual methods all duals, primal-dual methods both.
"""
function decompose(problem, method::AbstractL2OMethod)
    fam = method_family(method)
    return FullSpaceDecomposition(
        predicts_primal(fam),
        predicts_dual(fam),
        num_decision_variables(problem),
        num_duals(problem),
    )
end

"""
    L2OPipeline(problem::ParametricProblem, method::AbstractL2OMethod;
                evaluator = default_evaluator(problem))

Bind a method to a problem: checks [`supports`](@ref) (throwing
[`UnsupportedProblemError`](@ref)), computes [`decompose`](@ref) once and keeps
an [`AbstractEvaluator`](@ref). All pipeline steps take the pipeline:

```julia
pipe = L2OPipeline(problem, method)
data = generate_data(pipe; sampler, n = 100, optimizer)
ℓ    = loss(pipe, data.inputs, proxy(data.inputs))
pred = complete(pipe, inputs, proxy(inputs))
evaluate(pipe, inputs, pred)
```
"""
struct L2OPipeline{
    P<:AbstractParametricProblem,
    M<:AbstractL2OMethod,
    D<:AbstractDecomposition,
    E<:AbstractEvaluator,
}
    problem::P
    method::M
    decomposition::D
    evaluator::E
end

function L2OPipeline(
    problem::AbstractParametricProblem,
    method::AbstractL2OMethod;
    evaluator::AbstractEvaluator = default_evaluator(problem),
)
    check_supports(problem, method)
    return L2OPipeline(problem, method, decompose(problem, method), evaluator)
end

Base.show(io::IO, p::L2OPipeline) = print(
    io,
    "L2OPipeline(",
    nameof(typeof(p.method)),
    " on ",
    p.problem,
    "; ",
    nameof(typeof(p.decomposition)),
    ", ",
    nameof(typeof(p.evaluator)),
    ")",
)

output_dimension(p::L2OPipeline) = output_dimension(p.decomposition)

"""
    input_dimension(pipe::L2OPipeline) -> Int

Number of rows of the proxy input; the fallback is `num_parameters`.
"""
input_dimension(p::L2OPipeline) = num_parameters(p.problem)

"""
    generate_data(pipe::L2OPipeline; sampler, n, rng = Random.default_rng(), kwargs...)
    generate_data(method, pipe; sampler, n, rng, kwargs...)  # extension point

Build an [`L2ODataset`](@ref) for training `pipe.method`.

Family fallback: sample `n` parameter vectors from `sampler` and solve every
instance with [`solve_instances`](@ref) (`kwargs` such as `optimizer` are
forwarded), recording primal labels for primal methods, dual labels for dual
methods and both for primal-dual methods. Methods override the second form,
e.g. self-supervised methods that need only inputs, or DC3 storing only
independent variables.
"""
generate_data(p::L2OPipeline; kwargs...) = generate_data(p.method, p; kwargs...)

function generate_data(
    method::AbstractL2OMethod,
    p;
    sampler,
    n::Integer,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    P = sample_parameters(sampler, rng, n)
    fam = method_family(method)
    return solve_instances(
        p.problem,
        P;
        primal = predicts_primal(fam),
        dual = predicts_dual(fam),
        kwargs...,
    )
end

"""
    complete(pipe::L2OPipeline, inputs, output) -> Prediction
    complete(method, pipe, inputs, output)  # extension point

Turn the raw proxy `output` (`output_dimension × B`) into a
[`Prediction`](@ref) in the method's own sense (projection, equality
completion, dual completion, repair…). The
`FullSpaceDecomposition` fallback only splits the output into primal and dual
blocks.

Implementations should be written with array operations so they can be
differentiated and run on accelerators when the loss needs them.
"""
complete(p::L2OPipeline, inputs, output) = complete(p.method, p, inputs, output)

function complete(::AbstractL2OMethod, p, inputs, output)
    d = p.decomposition
    d isa FullSpaceDecomposition || throw(
        ArgumentError(
            "method $(nameof(typeof(p.method))) uses $(nameof(typeof(d))) and must " *
            "implement `L2OCore.complete`",
        ),
    )
    Z = _asbatch(output)
    size(Z, 1) == output_dimension(d) || throw(
        DimensionMismatch("output has $(size(Z, 1)) rows, expected $(output_dimension(d))"),
    )
    np = d.primal ? d.nprimal : 0
    X = d.primal ? Z[1:np, :] : nothing
    Y = d.dual ? Z[(np+1):end, :] : nothing
    return Prediction(X, Y)
end

"""
    loss(pipe::L2OPipeline, inputs, output; targets = nothing) -> Real
    loss(method, pipe, inputs, output; targets)  # extension point

Training loss of a batch of raw proxy outputs. Training loops should also pass
`proxy = <the proxy>` so that loss terms which differentiate the proxy with
respect to its inputs (Sobolev / derivative supervision) can use it; methods
that do not need it ignore it. Fallback: supervised mean
squared error between the completed prediction and the labels in `targets`
(an [`L2ODataset`](@ref) restricted to the batch, or a `Prediction`).
Self-supervised methods override it (see the penalty method in
LearningToOptimize.jl).
"""
loss(p::L2OPipeline, inputs, output; kwargs...) =
    loss(p.method, p, inputs, output; kwargs...)

function loss(method::AbstractL2OMethod, p, inputs, output; targets = nothing, kwargs...)
    targets === nothing && throw(
        ArgumentError(
            "the fallback loss of $(nameof(typeof(method))) is supervised; pass `targets`",
        ),
    )
    pred = complete(method, p, inputs, output)
    ℓ = zero(eltype(output))
    if pred.primal !== nothing
        ℓ += _mse(pred.primal, targets.primal)
    end
    if pred.dual !== nothing
        ℓ += _mse(pred.dual, targets.dual)
    end
    return ℓ
end

_mse(a, b) = sum(abs2, a .- b) / length(a)

"""
    train_proxy!(method, pipe::L2OPipeline, proxy, data; kwargs...)

Train `proxy` in place for `method`. `L2OCore` defines no method for it, since
training depends on the ML framework; framework-specific packages (e.g.
LearningToOptimize.jl for Flux) and method packages add methods.
"""
function train_proxy! end

"""
    evaluate(pipe::L2OPipeline, inputs, prediction; atol = 1e-6)
    evaluate(problem::ParametricProblem, inputs, prediction;
             evaluator = default_evaluator(problem), atol = 1e-6)

Quality metrics of a batch of completed predictions as an
[`EvaluationReport`](@ref): objective and violations of the primal part, dual
bound and dual infeasibility of the dual part, and the gap when both exist.
"""
evaluate(p::L2OPipeline, inputs, prediction::Prediction; kwargs...) =
    evaluate(p.problem, inputs, prediction; evaluator = p.evaluator, kwargs...)

function evaluate(
    problem::AbstractParametricProblem,
    inputs,
    prediction::Prediction;
    evaluator::AbstractEvaluator = default_evaluator(problem),
    atol::Real = 1e-6,
)
    P = _asbatch(inputs)
    obj = maxv = eqv = ineqv = db = dinf = gap = nothing
    if prediction.primal !== nothing
        X = _asbatch(prediction.primal)
        obj = batch_objective(evaluator, X, P)
        V = row_violation(evaluator, X, P)
        C = cone_violation(evaluator, X, P)::AbstractMatrix
        eq = equality_rows(evaluator)
        eqv = _colmax(V[eq, :])
        ineqv = max.(_colmax(V[.!eq, :]), _colmax(C))
        maxv = max.(eqv, ineqv)
    end
    if prediction.dual !== nothing
        db, dinf = dual_bound(evaluator, _asbatch(prediction.dual), P; atol = atol)
    end
    if obj !== nothing && db !== nothing
        gap = abs.(obj .- db) ./ max.(1, abs.(obj))
    end
    return EvaluationReport(obj, maxv, eqv, ineqv, db, dinf, gap)
end

"""
    AbstractMethodWrapper <: AbstractL2OMethod

A method that wraps another method and changes only part of its behaviour —
the hook for method-agnostic add-ons such as derivative (Sobolev) supervision
or extra regularisers. A wrapper must implement

  - `L2OCore.inner_method(w)` → the wrapped method;
  - `L2OCore.extra_loss(w, pipe, inputs, output; targets, proxy, kwargs...)` →
    the additional loss term (a scalar).

Leave the `pipe`, `inputs` and `output` arguments untyped (or typed no more
narrowly than the fallbacks) to avoid method ambiguities. Everything else
(family, [`supports`](@ref),
[`decompose`](@ref), [`generate_data`](@ref), [`complete`](@ref)) is forwarded
to the inner method, and [`loss`](@ref) is `loss(inner) + extra_loss(wrapper)`.
A wrapper may override any of them, e.g. `generate_data` to attach derivative
labels with [`with_derivatives`](@ref).
"""
abstract type AbstractMethodWrapper <: AbstractL2OMethod end

"""
    inner_method(w::AbstractMethodWrapper) -> AbstractL2OMethod

The method wrapped by `w`. Fallback: the field `w.inner`.
"""
inner_method(w::AbstractMethodWrapper) = w.inner

"""
    extra_loss(w::AbstractMethodWrapper, pipe, inputs, output; kwargs...) -> Real

Additional loss term contributed by a wrapper; see
[`AbstractMethodWrapper`](@ref).
"""
function extra_loss(w::AbstractMethodWrapper, p, inputs, output; kwargs...)
    throw(ArgumentError("wrapper $(nameof(typeof(w))) must implement `L2OCore.extra_loss`"))
end

method_family(w::AbstractMethodWrapper) = method_family(inner_method(w))
supports(w::AbstractMethodWrapper, c::AbstractProblemClass) = supports(inner_method(w), c)
decompose(problem, w::AbstractMethodWrapper) = decompose(problem, inner_method(w))
generate_data(w::AbstractMethodWrapper, p; kwargs...) =
    generate_data(inner_method(w), p; kwargs...)
complete(w::AbstractMethodWrapper, p, inputs, output) =
    complete(inner_method(w), p, inputs, output)
function loss(w::AbstractMethodWrapper, p, inputs, output; kwargs...)
    return loss(inner_method(w), p, inputs, output; kwargs...) +
           extra_loss(w, p, inputs, output; kwargs...)
end
