"""
    proxy_metrics(pipe, inputs, primal; reference = nothing, include_bounds = true)
        -> (; mse, gap, inf, max_violation, violations)

Per-sample evaluation metrics of the masked Sobolev training paper (AC-OPF
evaluation appendix) for a batch of primal predictions `primal`
(`num_decision_variables × B`) at parameters `inputs`:

```math
\\mathrm{MSE}_i = \\tfrac1n \\|\\tilde x_i - x^\\star_i\\|_2^2, \\qquad
\\mathrm{GAP}_i = \\frac{|f(\\tilde x_i) - f(x^\\star_i)|}{|f(x^\\star_i)|}, \\qquad
\\mathrm{INF}_i = \\frac{\\sum_{c ∈ \\mathcal E} |c(\\tilde x_i)| + \\sum_{g ∈ \\mathcal I} \\max\\{g(\\tilde x_i), 0\\}}{|\\mathcal E| + |\\mathcal I|},
```

and the largest violation ``\\max_j \\mathrm{Infeas}_{ij}``. Violations come
from the pipeline's evaluator ([`row_violation`](@ref L2OCore.row_violation)
plus [`cone_violation`](@ref L2OCore.cone_violation)); `violations` is the
`m × B` matrix ``\\mathrm{Infeas}_{ij}``. Variable bounds count as inequalities
unless `include_bounds = false`. `mse` and `gap` need a `reference` dataset
with `primal` and `objective` labels for the same inputs (`nothing`
otherwise). Use [`max_infeasibility_reduction`](@ref) to compare two proxies.
"""
function proxy_metrics(
    pipe::L2OCore.L2OPipeline,
    inputs::AbstractMatrix,
    primal::AbstractMatrix;
    reference = nothing,
    include_bounds::Bool = true,
)
    ev = pipe.evaluator
    V = L2OCore.row_violation(ev, primal, inputs)
    keep = include_bounds ? trues(size(V, 1)) : .!L2OCore.bound_rows(ev)
    Vall = vcat(V[keep, :], L2OCore.cone_violation(ev, primal, inputs))
    m, B = size(Vall)
    inf = m == 0 ? zeros(B) : vec(sum(Vall; dims = 1)) ./ m
    maxv = m == 0 ? zeros(B) : vec(maximum(Vall; dims = 1))
    mse = gap = nothing
    if reference !== nothing && reference.primal !== nothing
        mse = vec(sum(abs2, primal .- reference.primal; dims = 1)) ./ size(primal, 1)
    end
    if reference !== nothing && reference.objective !== nothing
        z = L2OCore.batch_objective(ev, primal, inputs)
        gap = abs.(z .- reference.objective) ./ abs.(reference.objective)
    end
    return (mse = mse, gap = gap, inf = inf, max_violation = maxv, violations = Vall)
end

"""
    max_infeasibility_reduction(benchmark, candidate) -> Vector{Float64}

Relative reduction in maximum infeasibility (RMI, in percent) of a
`candidate` proxy over a `benchmark` proxy on the same instances, as defined
in the evaluation appendix of the masked Sobolev training paper:

```math
\\mathrm{RMI}_i = 100\\% × \\frac{\\max_j \\mathrm{Infeas}^{\\text{bench}}_{ij}
  - \\max_j \\mathrm{Infeas}^{\\text{cand}}_{ij}}{\\max_{i,j} \\mathrm{Infeas}^{\\text{cand}}_{ij}} .
```

Positive values mean the candidate reduced the worst violation of instance
`i`. Both arguments are results of [`proxy_metrics`](@ref). The normalization
by the single largest candidate violation avoids division by zero on
instances without violation; if the candidate has no violation at all, the
denominator is `eps()`.
"""
function max_infeasibility_reduction(benchmark::NamedTuple, candidate::NamedTuple)
    length(benchmark.max_violation) == length(candidate.max_violation) ||
        throw(DimensionMismatch("the two evaluations have different numbers of instances"))
    denom = max(maximum(candidate.max_violation; init = 0.0), eps())
    return 100 .* (benchmark.max_violation .- candidate.max_violation) ./ denom
end
