"""
    SupervisedMSE()

Plain **value supervision**: the proxy predicts all decision variables and is
trained on solver labels with the mean squared error

```math
\\mathcal{L}(θ) = \\frac{1}{N n} \\sum_{i=1}^{N} \\big\\| \\hat g_θ(p_i) - x^\\star(p_i) \\big\\|_2^2 ,
```

the empirical-risk-minimization baseline of the optimization-proxy literature
(the value term of supervised Sobolev training). It is the primal family
fallback made concrete: [`generate_data`](@ref) solves every sampled instance
for primal labels, [`complete`](@ref) is the identity and [`loss`](@ref) is
the supervised MSE against `targets`.

It needs a solver that returns (locally) optimal primal solutions for the
sampled parameters and accepts any problem class, since labels are only data.
A regression proxy is in general infeasible; measure its violations with
[`evaluate`](@ref).
"""
struct SupervisedMSE <: AbstractPrimalMethod end

supports(::SupervisedMSE, ::AbstractProblemClass) = true
