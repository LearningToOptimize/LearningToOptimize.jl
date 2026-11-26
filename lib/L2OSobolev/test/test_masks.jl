using Test, L2OSobolev, L2OCore, Random

# A third-party backend added by dispatch: returns a fixed Jacobian batch.
struct FixedJacobian{A} <: AbstractSobolevBackend
    J::A
end
L2OSobolev.proxy_jacobian(b::FixedJacobian, f, P) = b.J
L2OSobolev.proxy_pullback(b::FixedJacobian, f, P, W) =
    reduce(hcat, (b.J[:, :, k]' * W[:, k] for k in axes(W, 2)))
L2OSobolev.proxy_pushforward(b::FixedJacobian, f, P, V) =
    reduce(hcat, (b.J[:, :, k] * V[:, k] for k in axes(V, 2)))

@testset "mask sampling" begin
    rng = Xoshiro(1)
    @test sample_mask(DenseMask(), rng, (3, 4)) === nothing
    M = sample_mask(RandomMask(0.25), rng, (4, 5))
    @test size(M) == (4, 5) && count(M) == 5
    @test count(sample_mask(RandomMask(1e-6), rng, (7,))) == 1     # at least one entry
    @test count(sample_mask(RandomMask(1.0), rng, (3, 3))) == 9
    @test_throws ArgumentError RandomMask(0.0)
    @test_throws ArgumentError RandomMask(1.5)
    F = FixedMask(Bool[1 0; 0 1])
    @test sample_mask(F, rng, (2, 2)) == Bool[1 0; 0 1]
    @test_throws DimensionMismatch sample_mask(F, rng, (3,))
    @test_throws ArgumentError FixedMask(falses(2))
    # Uniformity: every entry is retained with probability k / m.
    freq = zeros(3, 4)
    for _ = 1:20_000
        freq .+= sample_mask(RandomMask(0.25), rng, (3, 4))
    end
    @test maximum(abs.(freq ./ 20_000 .- 3 / 12)) < 0.02
end

@testset "the masked loss is an unbiased estimate of the dense loss (eq. (4))" begin
    rng = Xoshiro(2)
    n, d, B = 2, 2, 3
    Ĵ, J = randn(rng, n, d, B), randn(rng, n, d, B)
    P = zeros(d, B)
    labels = JacobianLabels(J)
    be = FixedJacobian(Ĵ)
    dense = sobolev_loss(be, identity, P, labels)
    @test dense ≈ sum(abs2, Ĵ .- J) / (n * d * B)
    # Exact expectation: average over all C(4, 2) masks with k = 2 entries.
    subsets = [(i, j) for i = 1:4 for j = (i+1):4]
    vals = map(subsets) do (i, j)
        M = falses(n, d)
        M[i] = M[j] = true
        sobolev_loss(be, identity, P, labels; mask = FixedMask(M))
    end
    @test sum(vals) / length(vals) ≈ dense rtol = 1e-12
    # Monte Carlo with RandomMask(0.5) (k = 2 of 4 entries).
    mc = sum(
        sobolev_loss(be, identity, P, labels; mask = RandomMask(0.5), rng = rng) for
        _ = 1:20_000
    )
    @test mc / 20_000 ≈ dense rtol = 0.03
end

@testset "the loss is zero at a match; normalization by ‖M‖₀ (eq. (3))" begin
    rng = Xoshiro(3)
    J = randn(rng, 3, 2, 4)
    be = FixedJacobian(copy(J))
    @test sobolev_loss(be, identity, zeros(2, 4), JacobianLabels(J)) == 0
    Δ = zeros(3, 2, 4)
    Δ[1, 1, :] .= 2.0                       # error only in entry (1, 1)
    be2 = FixedJacobian(J .+ Δ)
    M = falses(3, 2)
    M[1, 1] = true
    M[2, 2] = true
    # ‖M ⊙ Δ‖² / ‖M‖₀ = 4 / 2 per sample.
    @test sobolev_loss(be2, identity, zeros(2, 4), JacobianLabels(J); mask = FixedMask(M)) ≈
          2.0
    @test_throws DimensionMismatch sobolev_loss(
        be,
        identity,
        zeros(3, 4),
        JacobianLabels(J),
    )
end

@testset "directional labels are unbiased (one-hot seeds and directions)" begin
    rng = Xoshiro(9)
    n, d, B = 3, 2, 4
    Ĵ, J = randn(rng, n, d, B), randn(rng, n, d, B)
    be = FixedJacobian(Ĵ)
    P = zeros(d, B)
    dense = sobolev_loss(be, identity, P, JacobianLabels(J))
    onehot(m, i) = repeat([j == i ? 1.0 : 0.0 for j = 1:m], 1, B)
    pull(W) = PullbackLabels(W, reduce(hcat, (J[:, :, k]' * W[:, k] for k = 1:B)))
    push(V) = PushforwardLabels(V, reduce(hcat, (J[:, :, k] * V[:, k] for k = 1:B)))
    # Average over the n (resp. d) equally likely one-hot seeds = dense loss.
    @test sum(sobolev_loss(be, identity, P, pull(onehot(n, i))) for i = 1:n) / n ≈ dense
    @test sum(sobolev_loss(be, identity, P, push(onehot(d, j))) for j = 1:d) / d ≈ dense
    @test sobolev_loss(be, identity, P, pull(onehot(n, 1)); mask = FixedMask(trues(d))) ≈
          sobolev_loss(be, identity, P, pull(onehot(n, 1)))
end

@testset "Gaussian unit seeds are unbiased too (E[wwᵀ] = I/n)" begin
    rng = Xoshiro(10)
    n, d, B = 3, 2, 2
    Ĵ, J = randn(rng, n, d, B), randn(rng, n, d, B)
    be = FixedJacobian(Ĵ)
    P = zeros(d, B)
    dense = sobolev_loss(be, identity, P, JacobianLabels(J))
    unit(m) = (S = randn(rng, m, B); S ./ sqrt.(sum(abs2, S; dims = 1)))
    mc_pull = sum(1:20_000) do _
        W = unit(n)
        lab = PullbackLabels(W, reduce(hcat, (J[:, :, k]' * W[:, k] for k = 1:B)))
        sobolev_loss(be, identity, P, lab)
    end
    mc_push = sum(1:20_000) do _
        V = unit(d)
        lab = PushforwardLabels(V, reduce(hcat, (J[:, :, k] * V[:, k] for k = 1:B)))
        sobolev_loss(be, identity, P, lab)
    end
    @test mc_pull / 20_000 ≈ dense rtol = 0.05
    @test mc_push / 20_000 ≈ dense rtol = 0.05
end

@testset "the normalized loss does not scale with the mask density" begin
    rng = Xoshiro(11)
    Ĵ, J = randn(rng, 4, 5, 3), randn(rng, 4, 5, 3)
    be = FixedJacobian(Ĵ)
    P = zeros(5, 3)
    dense = sobolev_loss(be, identity, P, JacobianLabels(J))
    for density in (0.05, 0.25, 0.75)
        mc = sum(
            sobolev_loss(
                be,
                identity,
                P,
                JacobianLabels(J);
                mask = RandomMask(density),
                rng = rng,
            ) for _ = 1:20_000
        )
        @test mc / 20_000 ≈ dense rtol = 0.05
    end
end
