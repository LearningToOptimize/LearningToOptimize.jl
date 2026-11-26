# Helpers that are constant for automatic differentiation: masks, copies of
# host data to the device of the inputs, shape checks. Explicit `rrule`s (no
# tangent) keep reverse-mode AD from tracing their mutating bodies.
_no_tangents(n::Int) = _ -> ntuple(_ -> ChainRulesCore.NoTangent(), n)

for f in (
    :sample_mask,
    :_to_like,
    :_unit_blocks,
    :_seed_blocks,
    :_output_rows,
    :_check_batch,
    :_step,
)
    @eval ChainRulesCore.rrule(::typeof($f), args...) =
        ($f(args...), _no_tangents(length(args) + 1))
end
