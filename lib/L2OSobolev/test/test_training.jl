using Test, L2OSobolev, L2OCore, Random, LinearAlgebra
import Lux, ComponentArrays, Zygote, Optimisers, HiGHS, DiffOpt, JuMP, Flux

function fd_gradient(f, θ; h = 1e-6)
    g = zero(θ)
    for i in eachindex(θ)
        e = zero(θ)
        e[i] = h
        g[i] = (f(θ .+ e) - f(θ .- e)) / 2h
    end
    return g
end
relerr(a, b) = norm(collect(a) .- collect(b)) / max(norm(collect(b)), 1e-12)

@testset "weight gradient of the Sobolev loss is correct: $(nameof(typeof(be))), $(nameof(typeof(lab))), $(nameof(typeof(method)))" for be in
                                                                                                                                        (
        LuxNestedAD(),
        CentralDifferences(),
    ),
    lab in (
        JacobianLabels(randn(Xoshiro(1), 2, 3, 4)),
        PullbackLabels(randn(Xoshiro(2), 2, 4), randn(Xoshiro(3), 3, 4)),
        PushforwardLabels(randn(Xoshiro(4), 3, 4), randn(Xoshiro(5), 2, 4)),
    ),
    method in (SupervisedMSE(), ToyPrimalDual())

    prob = ParametricProblem(eq_qp())
    rng = Xoshiro(6)
    dout = method isa SupervisedMSE ? 2 : 2 + num_duals(prob)
    model, ps, st = lux_proxy(rng, 3, dout)
    P = randn(rng, 3, 4)
    pipe = L2OPipeline(prob, method)
    # Fixed mask so that the finite-difference reference sees the same loss.
    msize = lab isa JacobianLabels ? (2, 3) : lab isa PullbackLabels ? (3,) : (2,)
    mask =
        FixedMask(rand(Xoshiro(7), Bool, msize...) .| (reshape(1:prod(msize), msize) .== 1))
    function ℓ(θ)
        sm = Lux.StatefulLuxLayer{true}(model, θ, st)
        return sobolev_loss(be, ProxyMap(sm, pipe), P, lab; mask = mask)
    end
    g = Zygote.gradient(ℓ, ps)[1]
    @test g !== nothing
    # CentralDifferences is checked against finite differences of the same
    # approximate loss; looser tolerance for its O(h²) inner approximation.
    @test relerr(g, fd_gradient(ℓ, ps)) < (be isa LuxNestedAD ? 1e-6 : 1e-4)
end

@testset "CentralDifferences weight gradient ≈ exact nested-AD gradient" begin
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    model, ps, st = lux_proxy(Xoshiro(12), 3, 2)
    P = randn(Xoshiro(13), 3, 6)
    lab = JacobianLabels(randn(Xoshiro(14), 2, 3, 6))
    grad(be) = Zygote.gradient(ps) do θ
        sm = Lux.StatefulLuxLayer{true}(model, θ, st)
        sobolev_loss(be, ProxyMap(sm, pipe), P, lab)
    end[1]
    @test relerr(grad(CentralDifferences()), grad(LuxNestedAD())) < 1e-6
end

@testset "weight gradient through loss(pipe, …) with a Float32 Lux proxy" begin
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(
        prob,
        SobolevRegularized(SupervisedMSE(); weight = 2.0, backend = LuxNestedAD()),
    )
    rng = Xoshiro(8)
    model = Lux.Chain(Lux.Dense(3 => 8, tanh), Lux.Dense(8 => 2))
    ps, st = Lux.setup(rng, model)
    ps = ComponentArrays.ComponentArray(ps)
    P = rand(rng, Float32, 3, 5)
    t = L2ODataset(
        P;
        primal = rand(rng, Float32, 2, 5),
        derivatives = JacobianLabels(repeat(Float32.(eq_qp_jacobian()), 1, 1, 5)),
    )
    total(θ, X) = (
        sm = Lux.StatefulLuxLayer{true}(model, θ, st);
        loss(pipe, X, sm(X); targets = t, proxy = sm)
    )
    g = Zygote.gradient(θ -> total(θ, P), ps)[1]
    @test eltype(g) == Float32 && all(isfinite, g)
    # Same gradient as the Float64 computation (checked against finite differences).
    ps64, P64 = Float64.(ps), Float64.(P)
    g64 = Zygote.gradient(θ -> total(θ, P64), ps64)[1]
    @test relerr(g64, fd_gradient(θ -> total(θ, P64), ps64)) < 1e-6
    @test relerr(g, g64) < 1e-4
end

@testset "ZygoteNestedAD weight gradient = finite differences: $(nameof(typeof(lab))), $(nameof(typeof(method)))" for lab in
                                                                                                                      (
        JacobianLabels(randn(Xoshiro(1), 2, 3, 4)),
        PullbackLabels(randn(Xoshiro(2), 2, 4), randn(Xoshiro(3), 3, 4)),
        PushforwardLabels(randn(Xoshiro(4), 3, 4), randn(Xoshiro(5), 2, 4)),
    ),
    method in (SupervisedMSE(), ToyPrimalDual())

    prob = ParametricProblem(eq_qp())
    dout = method isa SupervisedMSE ? 2 : 2 + num_duals(prob)
    θ, re = Flux.destructure(flux_proxy(Xoshiro(6), 3, dout))
    P = randn(Xoshiro(8), 3, 4)
    pipe = L2OPipeline(prob, method)
    msize = lab isa JacobianLabels ? (2, 3) : lab isa PullbackLabels ? (3,) : (2,)
    mask =
        FixedMask(rand(Xoshiro(7), Bool, msize...) .| (reshape(1:prod(msize), msize) .== 1))
    ℓ(t) = sobolev_loss(ZygoteNestedAD(), ProxyMap(re(t), pipe), P, lab; mask = mask)
    g = Zygote.gradient(ℓ, θ)[1]
    @test g !== nothing
    @test relerr(g, fd_gradient(ℓ, θ)) < 1e-6
    # Same gradient as the CentralDifferences approximation, up to its O(h²) error.
    ℓcd(t) = sobolev_loss(CentralDifferences(), ProxyMap(re(t), pipe), P, lab; mask = mask)
    @test relerr(Zygote.gradient(ℓcd, θ)[1], g) < 1e-4
end

@testset "ZygoteNestedAD with a closure over Flux parameters" begin
    θ, re = Flux.destructure(flux_proxy(Xoshiro(9), 3, 2))
    P = randn(Xoshiro(10), 3, 5)
    lab = JacobianLabels(randn(Xoshiro(11), 2, 3, 5))
    # The closure captures the model rebuilt from the parameters.
    ℓ(t) = (m = re(t); sobolev_loss(ZygoteNestedAD(), X -> 2 .* m(X), P, lab))
    @test relerr(Zygote.gradient(ℓ, θ)[1], fd_gradient(ℓ, θ)) < 1e-6
end

@testset "full Sobolev loss through loss(pipe, …) with a Flux proxy: gradient = finite differences" begin
    prob = ParametricProblem(eq_qp())
    for method in (SupervisedMSE(), ToyPrimalDual())
        pipe = L2OPipeline(
            prob,
            SobolevRegularized(method; weight = 2.0, backend = ZygoteNestedAD()),
        )
        dout = method isa SupervisedMSE ? 2 : 2 + num_duals(prob)
        θ, re = Flux.destructure(flux_proxy(Xoshiro(12), 3, dout))
        rng = Xoshiro(13)
        P = rand(rng, 3, 5)
        t = L2ODataset(
            P;
            primal = rand(rng, 2, 5),
            dual = method isa SupervisedMSE ? nothing : rand(rng, num_duals(prob), 5),
            derivatives = JacobianLabels(repeat(eq_qp_jacobian(), 1, 1, 5)),
        )
        total(θ) = (m = re(θ); loss(pipe, P, m(P); targets = t, proxy = m))
        g = Zygote.gradient(total, θ)[1]
        @test relerr(g, fd_gradient(total, θ)) < 1e-6
    end
end

# End-to-end behavior of the method in its regular regime (smooth solution map,
# few samples): value + derivative supervision generalizes better than value
# supervision alone (the approximation argument of Theorem 3 of the paper).
function four_var_qp()
    m = JuMP.Model()
    JuMP.@variable(m, p[1:3] in JuMP.MOI.Parameter.(zeros(3)))
    JuMP.@variable(m, x[1:4] >= -5)
    JuMP.@constraint(m, sum(x) == 1 + p[3])
    JuMP.@objective(
        m,
        Min,
        sum((i + 1) * x[i]^2 for i = 1:4) + x[1] * x[2] - p[1] * x[1] - p[2] * x[2] +
        0.5 * p[3] * x[4]
    )
    return m
end

function train_lux(pipe, data; epochs, seed = 1)
    model = Lux.Chain(Lux.Dense(3 => 32, tanh), Lux.Dense(32 => 4))
    ps, st = Lux.setup(Xoshiro(seed), model)
    ps = ComponentArrays.ComponentArray(ps)
    opt = Optimisers.setup(Optimisers.Adam(5e-3), ps)
    X = Float32.(data.inputs)
    for _ = 1:epochs
        g = Zygote.gradient(ps) do θ
            sm = Lux.StatefulLuxLayer{true}(model, θ, st)
            loss(pipe, X, sm(X); targets = data, proxy = sm)
        end[1]
        opt, ps = Optimisers.update!(opt, ps, g)
    end
    return Lux.StatefulLuxLayer{true}(model, ps, st)
end

@testset "Sobolev training lowers the test error (smooth QP, 8 samples)" begin
    prob = ParametricProblem(four_var_qp())
    sampler = BoxSampler(-ones(3), ones(3))
    sob = SobolevRegularized(SupervisedMSE(); backend = LuxNestedAD())
    pipe_sob = L2OPipeline(prob, sob)
    train = generate_data(
        pipe_sob;
        sampler,
        n = 8,
        rng = Xoshiro(2),
        optimizer = HiGHS.Optimizer,
    )
    test = generate_data(
        pipe_sob;
        sampler,
        n = 100,
        rng = Xoshiro(3),
        optimizer = HiGHS.Optimizer,
    )
    errs = map((L2OPipeline(prob, SupervisedMSE()), pipe_sob)) do pipe
        sm = train_lux(pipe, train; epochs = 800)
        Xt = Float32.(test.inputs)
        J = proxy_jacobian(LuxNestedAD(), ProxyMap(sm, pipe), Xt)
        (
            value = sum(abs2, sm(Xt) .- test.primal) / length(test.primal),
            jac = sum(abs2, J .- test.derivatives.J) / length(J),
        )
    end
    plain, sobolev = errs
    @test sobolev.value < plain.value / 2
    @test sobolev.jac < plain.jac / 2
end

@testset "Flux Chain training with ZygoteNestedAD decreases the Sobolev loss" begin
    prob = ParametricProblem(four_var_qp())
    pipe =
        L2OPipeline(prob, SobolevRegularized(SupervisedMSE(); backend = ZygoteNestedAD()))
    data = generate_data(
        pipe;
        sampler = BoxSampler(-ones(3), ones(3)),
        n = 8,
        rng = Xoshiro(2),
        optimizer = HiGHS.Optimizer,
    )
    X = data.inputs
    Random.seed!(1)
    model = Flux.f64(Flux.Chain(Flux.Dense(3 => 32, tanh), Flux.Dense(32 => 4)))
    total(m) = loss(pipe, X, m(X); targets = data, proxy = m)
    jacerr(m) = sum(
        abs2,
        proxy_jacobian(ZygoteNestedAD(), ProxyMap(m, pipe), X) .- data.derivatives.J,
    )
    opt = Flux.setup(Flux.Adam(5e-3), model)
    history = Float64[total(model)]
    jac0 = jacerr(model)
    for _ = 1:30
        ℓ, g = Flux.withgradient(total, model)
        Flux.update!(opt, model, g[1])
        push!(history, ℓ)
    end
    push!(history, total(model))
    @test all(isfinite, history)
    @test history[end] < history[1] / 2
    @test jacerr(model) < jac0
end
