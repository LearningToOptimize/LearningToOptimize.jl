# Numerical checks of the approximation theorems of the masked Sobolev
# training paper (Theorems 1–3) in their idealized setting: exact value
# (and Jacobian) interpolation on a training grid of a C² map g on [0, 1].
using Test, L2OSobolev, LinearAlgebra

g_ref(x) = sin(3x)
dg_ref(x) = 3cos(3x)
const L_G = 3.0          # sup |g'|
const M_G = 9.0          # sup |g''|  (Lipschitz constant of g')

# Piecewise-linear interpolant (values only) and piecewise-cubic Hermite
# interpolant (values and derivatives) on the nodes `t`.
function interval(t, x)
    i = clamp(searchsortedlast(t, x), 1, length(t) - 1)
    return i, t[i+1] - t[i], (x - t[i]) / (t[i+1] - t[i])
end
function linear_interp(t, x)
    i, h, s = interval(t, x)
    return (1 - s) * g_ref(t[i]) + s * g_ref(t[i+1])
end
function hermite(t, x)
    i, h, s = interval(t, x)
    h00, h10, h01, h11 = 2s^3 - 3s^2 + 1, s^3 - 2s^2 + s, -2s^3 + 3s^2, s^3 - s^2
    return h00 * g_ref(t[i]) +
           h10 * h * dg_ref(t[i]) +
           h01 * g_ref(t[i+1]) +
           h11 * h * dg_ref(t[i+1])
end
function hermite_d(t, x)       # derivative of the Hermite interpolant
    i, h, s = interval(t, x)
    d00, d10, d01, d11 = 6s^2 - 6s, 3s^2 - 4s + 1, -6s^2 + 6s, 3s^2 - 2s
    return (d00 * g_ref(t[i]) + d01 * g_ref(t[i+1])) / h +
           d10 * dg_ref(t[i]) +
           d11 * dg_ref(t[i+1])
end
function hermite_dd(t, x)      # second derivative (piecewise linear in s)
    i, h, s = interval(t, x)
    e00, e10, e01, e11 = 12s - 6, 6s - 4, -12s + 6, 6s - 2
    return (e00 * g_ref(t[i]) + e01 * g_ref(t[i+1])) / h^2 +
           (e10 * dg_ref(t[i]) + e11 * dg_ref(t[i+1])) / h
end

@testset "covering_radius" begin
    t = collect(range(0, 1; length = 6))
    xs = collect(range(0, 1; length = 2001))
    @test covering_radius(reshape(t, 1, :), reshape(xs, 1, :)) ≈ 0.1
    @test covering_radius([0.0 1.0; 0.0 0.0], [0.5; 0.5;;]) ≈ sqrt(0.5)
    @test_throws DimensionMismatch covering_radius(zeros(2, 3), zeros(1, 3))
    @test_throws ArgumentError covering_radius(zeros(1, 0), zeros(1, 3))
end

@testset "approximation bounds of Theorems 1–3, N = $N nodes" for N in (4, 8, 16)
    t = collect(range(0, 1; length = N))
    xs = collect(range(0, 1; length = 4001))
    δ = covering_radius(reshape(t, 1, :), reshape(xs, 1, :))
    # Theorem 1: value interpolation, ‖ĝ - g‖ ≤ (L_g + L_ĝ) δ.
    L_hat = maximum(abs(g_ref(t[i+1]) - g_ref(t[i])) / (t[i+1] - t[i]) for i = 1:(N-1))
    err_value = maximum(abs(linear_interp(t, x) - g_ref(x)) for x in xs)
    @test err_value <= (L_G + L_hat) * δ
    # Theorem 2: Jacobian interpolation, ‖Dĝ - Dg‖ ≤ (M_g + M_ĝ) δ.
    M_hat = maximum(abs(hermite_dd(t, x)) for x in xs)
    err_jac = maximum(abs(hermite_d(t, x) - dg_ref(x)) for x in xs)
    @test err_jac <= (M_G + M_hat) * δ
    # Theorem 3: value and Jacobian interpolation, ‖ĝ - g‖ ≤ ½ (M_g + M_ĝ) δ².
    err_sobolev = maximum(abs(hermite(t, x) - g_ref(x)) for x in xs)
    @test err_sobolev <= (M_G + M_hat) * δ^2 / 2
    # The quadratic bound beats the linear one once δ is small enough.
    N >= 8 && @test (M_G + M_hat) * δ^2 / 2 < (L_G + L_hat) * δ
    @test err_sobolev < err_value
end
