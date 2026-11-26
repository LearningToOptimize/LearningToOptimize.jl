using Test, L2OSobolev, L2OCore, Random
import Lux

@testset "structured masks: sampling" begin
    rng = Xoshiro(20)
    M = sample_mask(RandomColumnMask(2), rng, (3, 5))
    @test all(M[:, j] == fill(M[1, j], 3) for j = 1:5) && count(M[1, :]) == 2
    R = sample_mask(RandomRowMask(2), rng, (3, 5))
    @test all(R[i, :] == fill(R[i, 1], 5) for i = 1:3) && count(R[:, 1]) == 2
    @test count(sample_mask(RandomColumnMask(9), rng, (3, 5))) == 15     # clamped
    @test count(sample_mask(RandomRowMask(2), rng, (7,))) == 2           # directional labels
    @test_throws ArgumentError RandomColumnMask(0)
    @test_throws ArgumentError RandomRowMask(0)
end

@testset "structured masks: cheap path == generic masked loss ($(nameof(typeof(be))), $(nameof(typeof(mk))))" for be in
                                                                                                                  (
        CentralDifferences(),
        LuxNestedAD(),
    ),
    mk in (RandomColumnMask(2), RandomRowMask(1))

    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    model, ps, st = lux_proxy(Xoshiro(21), 3, 2)
    sm = Lux.StatefulLuxLayer{true}(model, ps, st)
    f = ProxyMap(sm, pipe)
    P = randn(Xoshiro(22), 3, 5)
    lab = JacobianLabels(randn(Xoshiro(23), 2, 3, 5))
    for seed = 1:4
        r = Xoshiro(seed)
        M = sample_mask(mk, copy(r), (2, 3))
        cheap = sobolev_loss(be, f, P, lab; mask = mk, rng = copy(r))
        generic = sobolev_loss(be, f, P, lab; mask = FixedMask(M))
        @test cheap ≈ generic rtol = 1e-10
    end
end

@testset "structured masks: unbiased (exact enumeration)" begin
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    f = ProxyMap(P -> [1.0 2.0 0.0; 0.5 -1.0 3.0] * P .^ 2, pipe)
    P = randn(Xoshiro(24), 3, 4)
    lab = JacobianLabels(randn(Xoshiro(25), 2, 3, 4))
    be = CentralDifferences()
    dense = sobolev_loss(be, f, P, lab)
    cols = [
        sobolev_loss(be, f, P, lab; mask = FixedMask(repeat([j == c for c = 1:3]', 2, 1))) for j = 1:3
    ]
    rows = [
        sobolev_loss(be, f, P, lab; mask = FixedMask(repeat([i == r for r = 1:2], 1, 3))) for i = 1:2
    ]
    @test sum(cols) / 3 ≈ dense rtol = 1e-10     # all 1-column masks equally likely
    @test sum(rows) / 2 ≈ dense rtol = 1e-10
end

@testset "structured masks: cost (forward-pass width with CentralDifferences)" begin
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    widths = Int[]
    g = P -> (push!(widths, size(P, 2)); [1.0 2.0 0.0; 0.5 -1.0 3.0] * P .^ 2)
    f = ProxyMap(g, pipe)
    P, lab = randn(Xoshiro(26), 3, 4), JacobianLabels(randn(Xoshiro(27), 2, 3, 4))
    sobolev_loss(CentralDifferences(), f, P, lab)
    @test widths == [3 * 4, 3 * 4]                      # full Jacobian: d·B per side
    empty!(widths)
    sobolev_loss(CentralDifferences(), f, P, lab; mask = RandomColumnMask(1))
    @test widths == [4, 4]                              # one column: B per side
end

import Zygote
@testset "weight gradient is correct with structured masks ($(nameof(typeof(be))), $(nameof(typeof(mk))))" for be in
                                                                                                               (
        LuxNestedAD(),
        CentralDifferences(),
    ),
    mk in (RandomColumnMask(2), RandomRowMask(1))

    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    model, ps, st = lux_proxy(Xoshiro(28), 3, 2)
    P = randn(Xoshiro(29), 3, 4)
    lab = JacobianLabels(randn(Xoshiro(30), 2, 3, 4))
    ℓ(θ) = sobolev_loss(
        be,
        ProxyMap(Lux.StatefulLuxLayer{true}(model, θ, st), pipe),
        P,
        lab;
        mask = mk,
        rng = Xoshiro(31),                      # same mask at every evaluation
    )
    g = Zygote.gradient(ℓ, ps)[1]
    @test relerr(g, fd_gradient(ℓ, ps)) < (be isa LuxNestedAD ? 1e-6 : 1e-4)
end

import Flux
@testset "structured masks with ZygoteNestedAD and a Flux proxy ($(nameof(typeof(mk))))" for mk in
                                                                                             (
    RandomColumnMask(2),
    RandomRowMask(1),
)
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    θ, re = Flux.destructure(flux_proxy(Xoshiro(32), 3, 2))
    P = randn(Xoshiro(33), 3, 5)
    lab = JacobianLabels(randn(Xoshiro(34), 2, 3, 5))
    f = ProxyMap(re(θ), pipe)
    for seed = 1:4
        r = Xoshiro(seed)
        M = sample_mask(mk, copy(r), (2, 3))
        cheap = sobolev_loss(ZygoteNestedAD(), f, P, lab; mask = mk, rng = copy(r))
        generic = sobolev_loss(ZygoteNestedAD(), f, P, lab; mask = FixedMask(M))
        @test cheap ≈ generic rtol = 1e-10
    end
    ℓ(t) = sobolev_loss(
        ZygoteNestedAD(),
        ProxyMap(re(t), pipe),
        P,
        lab;
        mask = mk,
        rng = Xoshiro(35),
    )
    @test relerr(Zygote.gradient(ℓ, θ)[1], fd_gradient(ℓ, θ)) < 1e-6
end
