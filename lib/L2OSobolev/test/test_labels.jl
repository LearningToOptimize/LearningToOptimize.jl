using Test, L2OSobolev, L2OCore, Random, LinearAlgebra
import DiffOpt, HiGHS, Ipopt, JuMP

const IPOPT = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-10)

# Problem, solver, parameter samples (inside one differentiable region), analytic Jacobian.
const CASES = [
    (
        "convex QP",
        eq_qp,
        HiGHS.Optimizer,
        [0.5 1.5 1.0; 1.0 2.5 -0.5; 0.2 0.8 1.0],
        p -> eq_qp_jacobian(),
    ),
    (
        "LP",
        bound_lp,
        HiGHS.Optimizer,
        [2.0 3.0 1.0 0.5; 1.0 1.5 2.0 1.5],
        bound_lp_jacobian,
    ),
    (
        "NLP (active nonlinear constraint)",
        disk_projection,
        IPOPT,
        [1.5 -1.2 0.3; 0.5 1.0 -2.0],
        disk_jacobian,
    ),
]

# DiffOpt 0.6 returns wrong forward-mode sensitivities when a parameter enters
# an inequality constraint (bound_lp: x₁ ≤ p₂); reverse mode and DiffOpt 0.5
# are correct. `@test_broken` records this, so the test reports an unexpected
# pass once it is fixed upstream.
forward_broken(name, mode) =
    name == "LP" && mode === :forward && pkgversion(DiffOpt) >= v"0.6"

@testset "labels match finite differences of the solution map: $name, $mode mode" for (
        name,
        build,
        opt,
        P,
        jac,
    ) in CASES,
    mode in (:reverse, :forward)

    prob = ParametricProblem(build())
    N = size(P, 2)
    oracle = DiffOptSensitivities(opt; mode = mode)
    # DiffOpt (implicit differentiation of the KKT system, eq. (1)).
    r = sensitivity_labels(oracle, prob, P, FullJacobian())
    @test all(r.ok)
    exact = all(isapprox(r.labels.J[:, :, k], jac(P[:, k]); atol = 1e-6) for k = 1:N)
    if forward_broken(name, mode)
        @test_broken exact
        continue
    end
    @test exact
    @test maximum(label_errors(r.labels, prob, P; optimizer = opt)) < 1e-4
    # The finite-difference oracle agrees with the analytic Jacobian too.
    fd = sensitivity_labels(FiniteDifferenceSensitivities(opt), prob, P, FullJacobian())
    @test all(fd.ok)
    @test fd.labels.J ≈ r.labels.J atol = 1e-4
    @test fd.primal ≈ r.primal atol = 1e-6
    # Directional labels are projections of the same Jacobians.
    rng = Xoshiro(8)
    pb = sensitivity_labels(oracle, prob, P, RandomPullbacks(); rng = rng)
    pf = sensitivity_labels(
        oracle,
        prob,
        P,
        RandomPushforwards(; distribution = :gaussian);
        rng = rng,
    )
    @test all(pb.ok) && all(pf.ok)
    @test all(sum(pb.labels.seeds; dims = 1) .== 1)            # one-hot seeds
    @test all(sum(abs2, pf.labels.directions; dims = 1) .≈ 1)  # unit directions
    for k = 1:N
        Jk = jac(P[:, k])
        @test pb.labels.values[:, k] ≈ Jk' * pb.labels.seeds[:, k] atol = 1e-6
        @test pf.labels.values[:, k] ≈ Jk * pf.labels.directions[:, k] atol = 1e-6
    end
    @test maximum(label_errors(pb.labels, prob, P; optimizer = opt)) < 1e-4
    @test maximum(label_errors(pf.labels, prob, P; optimizer = opt)) < 1e-4
    fdpf = sensitivity_labels(
        FiniteDifferenceSensitivities(opt),
        prob,
        P,
        RandomPushforwards();
        rng = Xoshiro(1),
    )
    fdpb = sensitivity_labels(
        FiniteDifferenceSensitivities(opt),
        prob,
        P,
        RandomPullbacks();
        rng = Xoshiro(1),
    )
    for k = 1:N
        @test fdpf.labels.values[:, k] ≈ jac(P[:, k]) * fdpf.labels.directions[:, k] atol =
            1e-4
        @test fdpb.labels.values[:, k] ≈ jac(P[:, k])' * fdpb.labels.seeds[:, k] atol = 1e-4
    end
end

@testset "failure modes: degenerate point and infeasible instance" begin
    prob = ParametricProblem(bound_lp())
    # At p₁ = p₂ the bound x₁ ≤ p₂ is weakly active: the solution map has a kink,
    # LICQ/strict complementarity fail and no derivative exists. The KKT system
    # returns one one-sided derivative; central differences average the two.
    P = [1.0 2.0; 1.0 1.0]
    r = sensitivity_labels(DiffOptSensitivities(HiGHS.Optimizer), prob, P, FullJacobian())
    err = label_errors(r.labels, prob, P; optimizer = HiGHS.Optimizer)
    @test err[1] > 0.1      # degenerate sample is flagged
    @test err[2] < 1e-4     # regular sample is fine
    # Infeasible instance (x ≥ 0 and x₁ + x₂ = p₁ < 0): NaN labels, ok = false.
    P2 = [-1.0 2.0; 1.0 1.0]
    for oracle in (
        DiffOptSensitivities(HiGHS.Optimizer),
        FiniteDifferenceSensitivities(HiGHS.Optimizer),
    )
        r2 = sensitivity_labels(oracle, prob, P2, FullJacobian())
        @test r2.ok == [false, true]
        @test all(isnan, r2.labels.J[:, :, 1]) && !any(isnan, r2.labels.J[:, :, 2])
        r3 = sensitivity_labels(oracle, prob, P2, RandomPushforwards())
        @test r3.ok == [false, true] && all(isnan, r3.labels.values[:, 1])
    end
    @test isnan(
        label_errors(
            PushforwardLabels(ones(2, 1), ones(2, 1)),
            prob,
            [-1.0; 1.0;;];
            optimizer = HiGHS.Optimizer,
        )[1],
    )
end

@testset "single-sample labels keep matrix shapes" begin
    prob = ParametricProblem(eq_qp())
    P = reshape([0.5, 1.0, 0.2], 3, 1)
    fd = FiniteDifferenceSensitivities(HiGHS.Optimizer)
    r = sensitivity_labels(fd, prob, P, RandomPullbacks(); rng = Xoshiro(1))
    @test size(r.labels.values) == (3, 1) && size(r.labels.seeds) == (2, 1)
    @test size(label_errors(r.labels, prob, P; optimizer = HiGHS.Optimizer)) == (1,)
end

struct NoOracle <: AbstractSensitivityOracle end
@testset "oracle errors" begin
    prob = ParametricProblem(eq_qp())
    @test_throws ArgumentError sensitivity_labels(
        NoOracle(),
        prob,
        ones(3, 1),
        FullJacobian(),
    )
    @test_throws ArgumentError RandomPullbacks(; distribution = :uniform)
    @test_throws ArgumentError RandomPushforwards(; distribution = :uniform)
    @test_throws ArgumentError DiffOptSensitivities(HiGHS.Optimizer; mode = :both)
end
