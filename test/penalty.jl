# Own module: the main test file loads MLJ, whose exports (`Model`,
# `evaluate`, ...) clash with JuMP and L2OCore names.
module PenaltyTests

using Test, JuMP, Random, Flux, HiGHS
using LearningToOptimize

export test_penalty_method, test_penalty_loss_definition

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
        @test supports(m, LinearConic()) && supports(m, ConvexQuadratic())
        @test !supports(m, SmoothNonlinear())
        pipe = L2OPipeline(prob, m)
        sampler = BoxSampler([0.5, 1.0], [1.5, 2.0])
        data = generate_data(pipe; sampler, n = 64, rng)
        @test data.primal === nothing && data.status === nothing   # self-supervised
        ref = generate_data(pipe; sampler, n = 16, rng, optimizer = HiGHS.Optimizer)
        @test all(==(MOI.OPTIMAL), ref.status)

        # At a solver optimum the penalty term
        # vanishes and the loss equals the mean optimal objective.
        @test loss(pipe, ref.inputs, ref.primal) ≈ sum(ref.objective) / 16 atol = 1e-6

        # The Zygote gradient matches central finite differences.
        X0 = ref.primal .+ 0.3 .* randn(rng, size(ref.primal))
        P0 = ref.inputs
        f(X) = loss(pipe, P0, X)
        G = Flux.gradient(f, X0)[1]
        h = 1e-6
        Gfd = similar(X0)
        for i in eachindex(X0)
            e = zero(X0)
            e[i] = h
            Gfd[i] = (f(X0 .+ e) - f(X0 .- e)) / 2h
        end
        @test G ≈ Gfd rtol = 1e-4 atol = 1e-6

        # End-to-end: training reduces the loss and gives a small optimality gap.
        model = Flux.f64(Chain(Dense(2 => 32, relu), Dense(32 => 3)))
        before = loss(pipe, ref.inputs, model(ref.inputs))
        hist = train_flux!(
            pipe,
            model,
            data;
            epochs = 200,
            batchsize = 16,
            rng,
            optimiser = Flux.Optimisers.Adam(1e-2),
        )
        after = loss(pipe, ref.inputs, model(ref.inputs))
        @test after < before
        @test hist[end] < hist[1]
        # The generic L2OCore entry point dispatches to train_flux!.
        small = Flux.f64(Chain(Dense(2 => 8, relu), Dense(8 => 3)))
        @test length(train_proxy!(m, pipe, small, data; epochs = 3, rng)) == 3
        pred = complete(pipe, ref.inputs, model(ref.inputs))
        rep = evaluate(pipe, ref.inputs, pred)
        @test sum(abs.(rep.objective .- ref.objective)) / 16 < 0.5
    end
end


function test_penalty_loss_definition()
    @testset "PenaltyMethod: loss definition and gradients" begin
        rng = Random.MersenneTwister(43)
        prob = ParametricProblem(penalty_test_lp())
        λh, λg = 3.0, 5.0
        pipe = L2OPipeline(prob, PenaltyMethod(; ρ_eq = λh, ρ_ineq = λg))
        # The loss equals eq. (2) of DC3,
        # f + λ_h‖h‖² + λ_g‖ReLU(g)‖², averaged over the batch, written out by
        # hand for this LP (h: two balances, g: 0 ≤ x ≤ 4 as -x ≤ 0, x - 4 ≤ 0).
        X = 6 .* rand(rng, 3, 5) .- 1
        P = rand(rng, 2, 5) .+ 0.5
        relu(v) = max.(v, 0)
        manual = sum(1:5) do k
            x, d = X[:, k], P[:, k]
            f = x[1] + 3x[2] + 2x[3]
            h = [x[1] + x[2] - d[1], x[3] - d[2]]
            g = vcat(-x, x .- 4)
            f + λh * sum(abs2, h) + λg * sum(abs2, relu(g))
        end / 5
        @test loss(pipe, P, X) ≈ manual

        # Maximization: the loss minimizes -f.
        mmax = Model()
        @variable(mmax, d in MOI.Parameter(1.0))
        @variable(mmax, 0 <= x <= 2)
        @constraint(mmax, x <= d)
        @objective(mmax, Max, 2x)
        pmax = L2OPipeline(ParametricProblem(mmax), PenaltyMethod())
        Pm = reshape([0.5, 1.5], 1, 2)
        @test loss(pmax, Pm, Pm) ≈ -sum(2 .* Pm) / 2   # x* = d is feasible

        # The gradient check with a second-order cone (conic distances are
        # differentiated too).
        msoc = Model()
        @variable(msoc, p in MOI.Parameter(1.0))
        @variable(msoc, t)
        @variable(msoc, y[1:2])
        @constraint(msoc, y[1] + y[2] == p)
        @constraint(msoc, [t; y] in SecondOrderCone())
        @objective(msoc, Min, t)
        psoc = L2OPipeline(ParametricProblem(msoc), PenaltyMethod())
        X0 = [0.1 0.2 -0.3; 1.0 0.4 0.9; -0.5 0.8 0.3]   # infeasible: cone active
        P0 = [1.0 0.5 2.0]
        fs(Z) = loss(psoc, P0, Z)
        G = Flux.gradient(fs, X0)[1]
        h = 1e-6
        Gfd = map(CartesianIndices(X0)) do I
            E = zero(X0)
            E[I] = h
            (fs(X0 .+ E) - fs(X0 .- E)) / 2h
        end
        @test G ≈ Gfd rtol = 1e-5 atol = 1e-7

        # Problems without a matrix form are rejected when the pipeline is built.
        mbad = Model()
        @variable(mbad, q in MOI.Parameter(1.0))
        @variable(mbad, z >= 0)
        @constraint(mbad, q * z >= 1)
        @objective(mbad, Min, z)
        @test_throws ArgumentError L2OPipeline(ParametricProblem(mbad), PenaltyMethod())
    end
end

end # module
