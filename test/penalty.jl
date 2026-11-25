using Test, JuMP, Random, Flux, HiGHS
using LearningToOptimize
import LearningToOptimize: L2OCore

function penalty_test_lp()
    model = Model()
    @variable(model, d[1:2] in MOI.Parameter.([1.0, 2.0]))
    @variable(model, 0 <= g[1:3] <= 4)
    @constraint(model, g[1] + g[2] == d[1])
    @constraint(model, g[3] == d[2])
    @objective(model, Min, g[1] + 3g[2] + 2g[3])
    return model
end

function test_penalty_method()
    @testset "PenaltyMethod" begin
        rng = Random.MersenneTwister(42)
        prob = ParametricProblem(penalty_test_lp())
        m = PenaltyMethod(; ρ_eq = 20.0, ρ_ineq = 20.0)
        @test_throws ArgumentError PenaltyMethod(; ρ_eq = -1.0)
        @test guarantees(m) == (NoGuarantee(),)
        @test supports(m, LinearConic()) && supports(m, ConvexQuadratic())
        @test !supports(m, SmoothNonlinear())
        pipe = L2OPipeline(prob, m)
        sampler = BoxSampler([0.5, 1.0], [1.5, 2.0])
        data = generate_data(pipe; sampler, n = 64, rng)
        @test data.primal === nothing && data.status === nothing   # self-supervised
        ref = generate_data(pipe; sampler, n = 16, rng, optimizer = HiGHS.Optimizer)
        @test all(==(MOI.OPTIMAL), ref.status)

        # Claim `PenaltyLossAtOptimum`: at a solver optimum the penalty term
        # vanishes and the loss equals the mean optimal objective.
        @test loss(pipe, ref.inputs, ref.primal) ≈ sum(ref.objective) / 16 atol = 1e-6

        # Claim `PenaltyLossGradient`: Zygote gradient == central finite differences.
        X0 = ref.primal .+ 0.3 .* randn(rng, size(ref.primal))
        P0 = ref.inputs
        f(X) = loss(pipe, P0, X)
        G = Flux.gradient(f, X0)[1]
        h = 1e-6
        Gfd = similar(X0)
        for i in eachindex(X0)
            e = zero(X0); e[i] = h
            Gfd[i] = (f(X0 .+ e) - f(X0 .- e)) / 2h
        end
        @test G ≈ Gfd rtol = 1e-4 atol = 1e-6

        # End-to-end: training reduces the loss and gives a small optimality gap.
        model = Flux.f64(Chain(Dense(2 => 32, relu), Dense(32 => 3)))
        before = loss(pipe, ref.inputs, model(ref.inputs))
        hist = train_flux!(pipe, model, data; epochs = 200, batchsize = 16, rng,
            optimiser = Flux.Optimisers.Adam(1e-2))
        after = loss(pipe, ref.inputs, model(ref.inputs))
        @test after < before
        @test hist[end] < hist[1]
        pred = complete(pipe, ref.inputs, model(ref.inputs))
        rep = evaluate(pipe, ref.inputs, pred)
        @test sum(abs.(rep.objective .- ref.objective)) / 16 < 0.5
        @test check(NoGuarantee(), prob, ref.inputs, pred).passed
    end
end
