using Test, L2OSobolev, L2OCore, Random
import HiGHS, Ipopt, JuMP

@testset "proxy_metrics (paper's MSE, GAP, INF) and RMI" begin
    prob = ParametricProblem(eq_qp())
    pipe = L2OPipeline(prob, SupervisedMSE())
    ref = generate_data(
        pipe;
        sampler = BoxSampler([0.0, 1.0, 0.0], [1.0, 2.0, 1.0]),
        n = 4,
        rng = Xoshiro(40),
        optimizer = HiGHS.Optimizer,
    )
    P = ref.inputs
    exact = proxy_metrics(pipe, P, ref.primal; reference = ref)
    @test maximum(exact.mse) < 1e-14 && maximum(exact.gap) < 1e-6
    @test maximum(exact.inf) < 1e-7 && maximum(exact.max_violation) < 1e-7
    # Shift x₁ by δ: only the equality row (of 3 rows: 1 equality, 2 bounds) is violated.
    δ = 0.3
    shift(a) = ref.primal .+ [a; 0.0]
    bench = proxy_metrics(pipe, P, shift(δ); reference = ref)
    @test bench.mse ≈ fill(δ^2 / 2, 4)
    @test bench.inf ≈ fill(δ / 3, 4) && bench.max_violation ≈ fill(δ, 4)
    @test size(bench.violations) == (3, 4)
    @test proxy_metrics(pipe, P, shift(δ); include_bounds = false).inf ≈ fill(δ, 4)
    @test proxy_metrics(pipe, P, shift(δ)).mse === nothing
    cand = proxy_metrics(pipe, P, shift(δ / 2); reference = ref)
    @test max_infeasibility_reduction(bench, cand) ≈ fill(100.0, 4)
    @test max_infeasibility_reduction(cand, bench) ≈ fill(-50.0, 4)
    @test all(isfinite, max_infeasibility_reduction(bench, exact))
    @test_throws DimensionMismatch max_infeasibility_reduction(
        bench,
        proxy_metrics(pipe, P[:, 1:2], ref.primal[:, 1:2]),
    )
    # Nonlinear problem (JuMPEvaluator): a point outside the unit disk.
    disk = L2OPipeline(ParametricProblem(disk_projection()), SupervisedMSE())
    m = proxy_metrics(disk, [1.5; 0.5;;], [2.0; 0.0;;])
    @test m.max_violation ≈ [3.0] && m.inf ≈ [3.0]
end
