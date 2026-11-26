# Reproduces the nested-AD support matrix in docs/src/design.md.
#
# The loss contains a vector-Jacobian product of the proxy with respect to its
# *inputs* (as in derivative/Sobolev supervision); we need its gradient with
# respect to the proxy *weights*. Each combination is compared with central
# finite differences. Run with: julia dev/nested_ad_matrix.jl
using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Lux", "Flux", "Zygote", "ForwardDiff", "Enzyme", "ComponentArrays"]; io = devnull)
using Lux, Flux, Zygote, ForwardDiff, Enzyme, ComponentArrays, Random, LinearAlgebra

rng = Random.Xoshiro(0)
X, V, T = randn(rng, 3, 4), randn(rng, 2, 4), randn(rng, 3, 4)

function fd_ref(f, p)
    g = zero(p)
    h = 1e-6
    for i in eachindex(p)
        e = zero(p)
        e[i] = h
        g[i] = (f(p .+ e) - f(p .- e)) / 2h
    end
    return g
end
verdict(g, gref) =
    g === nothing ? "returns nothing (silently wrong)" :
    norm(collect(g) .- collect(gref)) / norm(gref) < 1e-5 ? "correct" : "WRONG"
function report(name, f, gref)
    r = try
        verdict(f(), gref)
    catch e
        "error: " * first(sprint(showerror, e), 100)
    end
    return println(rpad(name, 60), r)
end

# Lux (Float64 parameters; Float32 parameters make finite differences unreliable)
lux = Lux.Chain(Lux.Dense(3 => 8, tanh), Lux.Dense(8 => 2))
ps, st = Lux.setup(rng, lux)
ps = ComponentArray(Lux.f64(ps))
sm(p) = StatefulLuxLayer{true}(lux, p, st)
lz(p) = sum(abs2, Zygote.gradient(x -> sum(sm(p)(x) .* V), X)[1] .- T)
lf(p) = sum(abs2, ForwardDiff.gradient(x -> sum(sm(p)(x) .* V), X) .- T)
lv(p) = sum(abs2, Lux.vector_jacobian_product(sm(p), AutoZygote(), X, V) .- T)
g = fd_ref(lz, ps)
report("Lux: Zygote over Zygote", () -> Zygote.gradient(lz, ps)[1], g)
report("Lux: Zygote over ForwardDiff", () -> Zygote.gradient(lf, ps)[1], g)
report("Lux: Zygote over Lux.vector_jacobian_product", () -> Zygote.gradient(lv, ps)[1], g)
report("Lux: ForwardDiff over Zygote", () -> ForwardDiff.gradient(lz, ps), g)
report(
    "Lux: Enzyme over Enzyme",
    () -> begin
        inner(p) = (
            dX = zero(X);
            Enzyme.autodiff(
                Enzyme.Reverse,
                x -> sum(sm(p)(x) .* V),
                Enzyme.Active,
                Enzyme.Duplicated(copy(X), dX),
            );
            sum(abs2, dX .- T)
        )
        dp = zero(ps)
        Enzyme.autodiff(
            Enzyme.Reverse,
            inner,
            Enzyme.Active,
            Enzyme.Duplicated(copy(ps), dp),
        )
        dp
    end,
    g,
)

# Flux
θ, re = Flux.destructure(Flux.f64(Flux.Chain(Flux.Dense(3 => 8, tanh), Flux.Dense(8 => 2))))
fz(t) = sum(abs2, Zygote.gradient(x -> sum(re(t)(x) .* V), X)[1] .- T)
ff(t) = sum(abs2, ForwardDiff.gradient(x -> sum(re(t)(x) .* V), X) .- T)
g = fd_ref(fz, θ)
report("Flux: Zygote over Zygote", () -> Zygote.gradient(fz, θ)[1], g)
report("Flux: Zygote over ForwardDiff", () -> Zygote.gradient(ff, θ)[1], g)
report("Flux: ForwardDiff over Zygote", () -> ForwardDiff.gradient(fz, θ), g)
