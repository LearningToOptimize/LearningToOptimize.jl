using Test, L2OSobolev, L2OCore, Random, LinearAlgebra
import Lux, ComponentArrays, Zygote, Flux

# Smooth column-wise map ℝ² → ℝ³ with a known Jacobian.
smooth(P) = vcat(sin.(P[1:1, :] .* P[2:2, :]), P[1:1, :] .^ 2, exp.(P[2:2, :]))
smooth_jac(p) = [
    cos(p[1] * p[2])*p[2] cos(p[1] * p[2])*p[1]
    2p[1] 0
    0 exp(p[2])
]

struct ToyPrimalDual <: AbstractPrimalDualMethod end
L2OCore.supports(::ToyPrimalDual, ::L2OCore.AbstractProblemClass) = true

@testset "CentralDifferences matches analytic derivatives" begin
    rng = Xoshiro(4)
    P = randn(rng, 2, 5)
    V, W = randn(rng, 2, 5), randn(rng, 3, 5)
    be = CentralDifferences()
    J = proxy_jacobian(be, smooth, P)
    @test size(J) == (3, 2, 5)
    for k = 1:5
        @test J[:, :, k] ≈ smooth_jac(P[:, k]) atol = 1e-8
        @test proxy_pushforward(be, smooth, P, V)[:, k] ≈ smooth_jac(P[:, k]) * V[:, k] atol =
            1e-8
        @test proxy_pullback(be, smooth, P, W)[:, k] ≈ smooth_jac(P[:, k])' * W[:, k] atol =
            1e-8
    end
    # Float32 inputs use a larger default step and stay accurate to ~1e-3.
    J32 = proxy_jacobian(be, smooth, Float32.(P))
    @test eltype(J32) == Float32 && maximum(abs.(J32 .- J)) < 1e-2
    @test proxy_jacobian(CentralDifferences(; step = 1e-4), smooth, P) ≈ J atol = 1e-6
    @test_throws DimensionMismatch proxy_pushforward(be, smooth, P, V[:, 1:2])
    @test_throws DimensionMismatch proxy_pullback(be, smooth, P, W[1:2, :])
end

@testset "pullback and pushforward are adjoint: wᵀ(J v) = (Jᵀ w)ᵀ v" begin
    rng = Xoshiro(5)
    P, V, W = randn(rng, 2, 4), randn(rng, 2, 4), randn(rng, 3, 4)
    be = CentralDifferences()
    jv = proxy_pushforward(be, smooth, P, V)
    jtw = proxy_pullback(be, smooth, P, W)
    @test vec(sum(W .* jv; dims = 1)) ≈ vec(sum(jtw .* V; dims = 1)) rtol = 1e-7
end

function lux_proxy(rng, din, dout)
    model = Lux.Chain(Lux.Dense(din => 8, tanh), Lux.Dense(8 => dout))
    ps, st = Lux.setup(rng, model)
    ps = ComponentArrays.ComponentArray(Lux.f64(ps))
    return model, ps, st
end

@testset "LuxNestedAD agrees with central differences" begin
    rng = Xoshiro(6)
    prob = ParametricProblem(eq_qp())
    for (method, dout) in ((SupervisedMSE(), 2), (ToyPrimalDual(), 2 + num_duals(prob)))
        pipe = L2OPipeline(prob, method)
        model, ps, st = lux_proxy(rng, 3, dout)
        sm = Lux.StatefulLuxLayer{true}(model, ps, st)
        f = ProxyMap(sm, pipe)
        P, V, W = randn(rng, 3, 4), randn(rng, 3, 4), randn(rng, 2, 4)
        @test size(f(P)) == (2, 4)
        Jl = proxy_jacobian(LuxNestedAD(), f, P)
        Jc = proxy_jacobian(CentralDifferences(), f, P)
        @test size(Jl) == (2, 3, 4)
        @test Jl ≈ Jc atol = 1e-7
        @test proxy_pushforward(LuxNestedAD(), f, P, V) ≈
              proxy_pushforward(CentralDifferences(), f, P, V) atol = 1e-7
        @test proxy_pullback(LuxNestedAD(), f, P, W) ≈
              proxy_pullback(CentralDifferences(), f, P, W) atol = 1e-7
        # Adjoint consistency of the exact backend.
        @test vec(sum(W .* proxy_pushforward(LuxNestedAD(), f, P, V); dims = 1)) ≈
              vec(sum(proxy_pullback(LuxNestedAD(), f, P, W) .* V; dims = 1))
    end
end

function flux_proxy(rng, din, dout)
    Random.seed!(rng)
    return Flux.f64(Flux.Chain(Flux.Dense(din => 8, tanh), Flux.Dense(8 => dout)))
end

@testset "ZygoteNestedAD agrees with analytic and central differences" begin
    # Plain callable with a known Jacobian.
    rng = Xoshiro(4)
    P, V, W = randn(rng, 2, 5), randn(rng, 2, 5), randn(rng, 3, 5)
    J = proxy_jacobian(ZygoteNestedAD(), smooth, P)
    @test size(J) == (3, 2, 5)
    for k = 1:5
        @test J[:, :, k] ≈ smooth_jac(P[:, k])
        @test proxy_pushforward(ZygoteNestedAD(), smooth, P, V)[:, k] ≈
              smooth_jac(P[:, k]) * V[:, k]
        @test proxy_pullback(ZygoteNestedAD(), smooth, P, W)[:, k] ≈
              smooth_jac(P[:, k])' * W[:, k]
    end
    @test_throws DimensionMismatch proxy_pushforward(ZygoteNestedAD(), smooth, P, V[:, 1:2])
    @test_throws DimensionMismatch proxy_pullback(ZygoteNestedAD(), smooth, P, W[1:2, :])

    # Flux proxies, through a ProxyMap (with and without dual rows).
    rng = Xoshiro(6)
    prob = ParametricProblem(eq_qp())
    for (method, dout) in ((SupervisedMSE(), 2), (ToyPrimalDual(), 2 + num_duals(prob)))
        pipe = L2OPipeline(prob, method)
        f = ProxyMap(flux_proxy(rng, 3, dout), pipe)
        P, V, W = randn(rng, 3, 4), randn(rng, 3, 4), randn(rng, 2, 4)
        Jz = proxy_jacobian(ZygoteNestedAD(), f, P)
        @test size(Jz) == (2, 3, 4)
        @test Jz ≈ proxy_jacobian(CentralDifferences(), f, P) atol = 1e-7
        @test proxy_pushforward(ZygoteNestedAD(), f, P, V) ≈
              proxy_pushforward(CentralDifferences(), f, P, V) atol = 1e-7
        @test proxy_pullback(ZygoteNestedAD(), f, P, W) ≈
              proxy_pullback(CentralDifferences(), f, P, W) atol = 1e-7
        @test vec(sum(W .* proxy_pushforward(ZygoteNestedAD(), f, P, V); dims = 1)) ≈
              vec(sum(proxy_pullback(ZygoteNestedAD(), f, P, W) .* V; dims = 1))
        @test_throws DimensionMismatch proxy_pullback(ZygoteNestedAD(), f, P, W[:, 1:2])
    end
end

@testset "ProxyMap and backend errors" begin
    rng = Xoshiro(7)
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    model, ps, st = lux_proxy(rng, 3, 2)
    sm = Lux.StatefulLuxLayer{true}(model, ps, st)
    P = randn(rng, 3, 2)
    # Through the (identity) completion: same map.
    @test ProxyMap(sm, pipe; through_completion = true)(P) ≈ ProxyMap(sm, pipe)(P)
    @test_throws ArgumentError proxy_jacobian(
        LuxNestedAD(),
        ProxyMap(sm, pipe; through_completion = true),
        P,
    )
    # Lux helpers do not apply to arbitrary callables.
    err = try
        proxy_jacobian(LuxNestedAD(), ProxyMap(x -> 2x[1:2, :], pipe), P)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("StatefulLuxLayer", sprint(showerror, err))
    # ZygoteNestedAD differentiates the raw proxy, not completion layers.
    m = flux_proxy(rng, 3, 2)
    @test_throws ArgumentError proxy_jacobian(
        ZygoteNestedAD(),
        ProxyMap(m, pipe; through_completion = true),
        P,
    )
    @test_throws ArgumentError proxy_pullback(
        ZygoteNestedAD(),
        ProxyMap(m, pipe; through_completion = true),
        P,
        randn(rng, 2, 2),
    )
end

# A method with a completion layer: predicts x₁, completes x₂ = p₃ + 1 - x₁
# (satisfies the equality of eq_qp exactly).
struct HalfDecomposition <: L2OCore.AbstractDecomposition end
L2OCore.output_dimension(::HalfDecomposition) = 1
struct HalfMethod <: AbstractPrimalMethod end
L2OCore.supports(::HalfMethod, ::L2OCore.AbstractProblemClass) = true
L2OCore.decompose(problem, ::HalfMethod) = HalfDecomposition()
L2OCore.complete(::HalfMethod, pipe, P, out) =
    Prediction(primal = vcat(out, P[3:3, :] .+ 1 .- out))

@testset "custom decompositions need through_completion" begin
    pipe = L2OPipeline(ParametricProblem(eq_qp()), HalfMethod())
    proxy = P -> P[1:1, :] .^ 2
    P = [1.0 2.0; 0.0 1.0; 0.5 0.5]
    @test_throws ArgumentError ProxyMap(proxy, pipe)(P)
    @test_throws ArgumentError proxy_jacobian(ZygoteNestedAD(), ProxyMap(proxy, pipe), P)
    f = ProxyMap(proxy, pipe; through_completion = true)
    J = proxy_jacobian(CentralDifferences(), f, P)
    @test J[:, :, 2] ≈ [4.0 0 0; -4.0 0 1.0] atol = 1e-8
end

@testset "CentralDifferences step scales with the inputs" begin
    # Quadratic map: central differences have no truncation error, so the
    # error is pure rounding, which an absolute step makes large for
    # large-magnitude Float32 inputs (here ≈ 1e3).
    f = P -> P .^ 2 ./ 1000
    P = Float32[1000.5 1500.25 800.125]
    J = proxy_jacobian(CentralDifferences(), f, P)
    @test maximum(abs.(vec(J) .- 2 .* vec(P) ./ 1000)) < 1e-4
    Jabs = proxy_jacobian(CentralDifferences(; step = cbrt(eps(Float32))), f, P)
    @test maximum(abs.(vec(Jabs) .- 2 .* vec(P) ./ 1000)) > 1e-3
end
