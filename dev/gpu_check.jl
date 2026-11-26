# One-command GPU check for the GPU code paths of the open pull requests.
#
#     julia dev/gpu_check.jl [--device=cuda|jlarray] [--only=name,...] [--activations=tanh,...]
#
# Installs the branches below into a temporary environment, runs every check on
# the device and on the CPU, and compares the results. The checks are dll, dc3,
# alm, nested_ad and core. `--device=jlarray` runs the same code with JLArrays (a
# CPU-backed array type that forbids scalar indexing), without a GPU. ExaModels
# then uses the KernelAbstractions `CPU()` backend. The final table is printed
# and also written to `gpu_check_results.md` next to this script.
# `--activations` restricts the nested_ad check to some of tanh, relu,
# sigmoid and leakyrelu (default: all four).
#
# L2OSobolev needs the L2OCore copy on its own branch, so its cases run in a
# child process with its own environment (`--sobolev-worker=<file>`).

const CHECKS = ["dll", "dc3", "alm", "nested_ad", "core"]

function _arg(name, default)
    for a in ARGS
        startswith(a, "--$name=") && return split(a, '='; limit = 2)[2]
    end
    return default
end
const DEVICE = _arg("device", "cuda")
DEVICE in ("cuda", "jlarray") || error("--device must be cuda or jlarray, got $DEVICE")
const ONLY = let o = _arg("only", "")
    isempty(o) ? CHECKS : String.(split(o, ','))
end
for c in ONLY
    c in CHECKS || error("unknown check $c; available: $(join(CHECKS, ", "))")
end
const WORKER = _arg("sobolev-worker", "")
const IS_WORKER = !isempty(WORKER)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
using Pkg
Pkg.activate(; temp = true)
const ORG = "https://github.com/LearningToOptimize/"
const MONOREPO = ORG * "LearningToOptimize.jl"
const DEVICE_PKGS = DEVICE == "cuda" ? ["CUDA", "LuxCUDA"] : ["JLArrays"]
const SPECS = if IS_WORKER
    [
        PackageSpec(; url = MONOREPO, subdir = "lib/L2OCore", rev = "main"),
        PackageSpec(; url = MONOREPO, subdir = "lib/L2OSobolev", rev = "main"),
    ]
else
    [
        PackageSpec(; url = MONOREPO, subdir = "lib/L2OCore", rev = "main"),
        PackageSpec(; url = ORG * "L2ODLL.jl", rev = "main"),
        PackageSpec(; url = ORG * "L2ODC3.jl", rev = "main"),
        PackageSpec(; url = ORG * "L2OALM.jl", rev = "main"),
        PackageSpec(; url = ORG * "BatchNLPKernels.jl", rev = "main"),
    ]
end
const REGISTERED = if IS_WORKER
    ["JuMP", "Lux", "Flux", "Zygote", "ComponentArrays"]
else
    [
        "Adapt",
        "ComponentArrays",
        "ExaModels",
        "Flux",
        "JuMP",
        "KernelAbstractions",
        "Lux",
        "Optimisers",
        "Zygote",
    ]
end
IS_WORKER || println("Installing packages (first run downloads and precompiles)...")
Pkg.add([SPECS; PackageSpec.(REGISTERED); PackageSpec.(DEVICE_PKGS)]; io = devnull)

using LinearAlgebra, Printf, Random
using JuMP, Lux, Flux, Zygote, ComponentArrays
const MOI = JuMP.MOI
if IS_WORKER
    using L2OCore, L2OSobolev
else
    using Adapt, ExaModels, KernelAbstractions, Optimisers
    using L2OCore, L2ODLL, L2ODC3, L2OALM
    using BatchNLPKernels: BatchModel, BatchModelConfig
end
if DEVICE == "cuda"
    using CUDA, LuxCUDA
    CUDA.allowscalar(false)
    const DevArray = CUDA.CuArray
else
    using JLArrays
    JLArrays.allowscalar(false)
    const DevArray = JLArrays.JLArray
end

todev(x::AbstractArray) = DevArray(x)
todev(T::Type, x::AbstractArray) = DevArray{T}(x)
todev(p::ComponentArray) = ComponentArray(todev(getdata(p)), getaxes(p))
host(x) = x === nothing ? nothing : Array(x)

# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------
const RESULTS = NamedTuple{(:check, :case, :status, :result),NTuple{4,String}}[]

oneline(err) = first(replace(sprint(showerror, err), r"\s+" => " "), 160)
fmt(x) = @sprintf("%.1e", x)

"Relative error of `a` against `b`, or `nothing` if `a` is `nothing`."
function relerr(a, b)
    a === nothing && return nothing
    A, Bv = collect(host(a)), collect(host(b))
    size(A) == size(Bv) || error("size $(size(A)) differs from reference size $(size(Bv))")
    return norm(A .- Bv) / max(norm(Bv), floatmin(Float64))
end
relerr(a::Number, b::Number) = abs(a - b) / max(abs(b), floatmin(Float64))
"Largest of several relative errors (`nothing` if any is `nothing`)."
maxerr(es...) = any(isnothing, es) ? nothing : maximum(es)

struct NotSupported
    msg::String
end

"""
Run `f` and record one row. `f` returns a relative error (compared with `tol`),
`nothing` (a gradient that silently returns nothing), a `(passed, message)`
tuple, or a `NotSupported` message (recorded without failing).
"""
function run_case(f, check, case; tol)
    t = time()
    status, result = try
        r = f()
        if r === nothing
            ("fail", "returns nothing")
        elseif r isa NotSupported
            ("skip", r.msg)
        elseif r isa Tuple
            r[1] ? ("pass", "pass ($(r[2]))") : ("fail", "FAIL ($(r[2]))")
        elseif r <= tol
            ("pass", "pass (rel err $(fmt(r)))")
        else
            ("fail", "FAIL (rel err $(fmt(r)), tol $(fmt(tol)))")
        end
    catch err
        ("error", "ERROR: " * oneline(err))
    end
    push!(RESULTS, (; check, case, status, result))
    @printf("  %-10s %-52s %s  [%.1fs]\n", check, case, result, time() - t)
    flush(stdout)
    return nothing
end

const ALL_ACTIVATIONS =
    (("tanh", tanh), ("relu", relu), ("sigmoid", sigmoid), ("leakyrelu", leakyrelu))
const ACTIVATIONS = let names = _arg("activations", "")
    sel = isempty(names) ? first.(ALL_ACTIVATIONS) : split(names, ',')
    for s in sel
        s in first.(ALL_ACTIVATIONS) || error("unknown activation $s")
    end
    Tuple(a for a in ALL_ACTIVATIONS if a[1] in sel)
end

# ===========================================================================
# Worker: full L2OSobolev loss gradients
# ===========================================================================
if IS_WORKER
    # Strictly convex QP with one equality (lib/L2OSobolev/test/problems.jl).
    function eq_qp(p0 = [1.0, 2.0, 0.5])
        m = Model()
        @variable(m, p[1:3] in MOI.Parameter.(p0))
        @variable(m, x[1:2] >= -10)
        @constraint(m, x[1] + x[2] == p[3] + 1)
        @objective(m, Min, (x[1] - p[1])^2 + (x[2] - p[2])^2 + 0.1 * x[1] * x[2])
        return m
    end

    function sobolev_cases()
        rng = Random.Xoshiro(3)
        pipe = L2OPipeline(ParametricProblem(eq_qp()), SupervisedMSE())
        d, n, B = 3, 2, 16
        P = randn(rng, d, B)
        jac = JacobianLabels(randn(rng, n, d, B))
        pb = PullbackLabels(randn(rng, n, B), randn(rng, d, B))
        devlabels(l::JacobianLabels) = JacobianLabels(todev(l.J))
        devlabels(l::PullbackLabels) = PullbackLabels(todev(l.seeds), todev(l.values))
        Pd = todev(P)
        # Pullback labels with every activation; Jacobian labels (the other
        # code path of each backend) with the first activation only, to bound
        # compile time.
        for (aname, act) in ACTIVATIONS
            # Lux proxy: LuxNestedAD and CentralDifferences.
            lux = Lux.Chain(Lux.Dense(d => 16, act), Lux.Dense(16 => n))
            ps, st = Lux.setup(rng, lux)
            ps = ComponentArray(Lux.f64(ps))
            psd = todev(ps)
            luxmap(p) = ProxyMap(StatefulLuxLayer{true}(lux, p, st), pipe)
            # Flux proxy: ZygoteNestedAD.
            θ, re = Flux.destructure(
                Flux.f64(Flux.Chain(Flux.Dense(d => 16, act), Flux.Dense(16 => n))),
            )
            θd = todev(θ)
            fluxmap(t) = ProxyMap(re(t), pipe)
            for (lname, lab) in (
                aname == first(ACTIVATIONS)[1] ?
                (("pullback labels", pb), ("Jacobian labels", jac)) :
                (("pullback labels", pb),)
            )
                labd = devlabels(lab)
                for (bname, be, mk, w, wd) in (
                    ("Lux, LuxNestedAD", LuxNestedAD(), luxmap, ps, psd),
                    ("Lux, CentralDifferences", CentralDifferences(), luxmap, ps, psd),
                    ("Flux, ZygoteNestedAD", ZygoteNestedAD(), fluxmap, θ, θd),
                )
                    tol = be isa CentralDifferences ? 1e-5 : 1e-8
                    run_case("nested_ad", "L2OSobolev $bname, $lname, $aname"; tol) do
                        ref = Zygote.gradient(q -> sobolev_loss(be, mk(q), P, lab), w)[1]
                        g = Zygote.gradient(q -> sobolev_loss(be, mk(q), Pd, labd), wd)[1]
                        return relerr(g, ref)
                    end
                end
            end
        end
    end

    sobolev_cases()
    open(WORKER, "w") do io
        for r in RESULTS
            println(io, join((r.check, r.case, r.status, r.result), '\t'))
        end
    end
    exit(0)
end

# ===========================================================================
# Main process
# ===========================================================================
println("\n== Environment")
println("Julia ", VERSION, ", device mode: ", DEVICE)
const REVS = let deps = Pkg.dependencies()
    [
        string(
            d.name,
            " ",
            d.version,
            d.git_revision === nothing ? "" : " @ $(d.git_revision)",
        ) for d in values(deps) if d.is_direct_dep
    ] |> sort
end
foreach(r -> println("  ", r), REVS)
const GPU_NAME = if DEVICE == "cuda"
    vi = sprint(io -> CUDA.versioninfo(io))
    println(join(first(split(vi, '\n'), 12), '\n'))
    CUDA.name(CUDA.device())
else
    "none (JLArrays dry run)"
end
println("GPU: ", GPU_NAME)

# Backend for ExaModels: CUDA on a GPU, the KernelAbstractions CPU backend in
# the dry run (JLArrays has no ExaModels backend).
exa_backend() = DEVICE == "cuda" ? CUDA.CUDABackend() : KernelAbstractions.CPU()
exa_dev(x) = DEVICE == "cuda" ? CUDA.CuArray(x) : copy(x)
exa_dev(p::ComponentArray) = DEVICE == "cuda" ? todev(p) : copy(p)

# ---------------------------------------------------------------------------
# 1. dll: batched closed-form DLL layers (L2ODLL test/instances.jl)
# ---------------------------------------------------------------------------
function dll_bounded_lp(rng; n = 5, m = 3)
    A, E = randn(rng, m, n), randn(rng, 1, n)
    x0 = rand(rng, n) .- 0.5
    b, e = A * x0 .- rand(rng, m), E * x0
    c, C = randn(rng, n), 0.5 .* randn(rng, n, m)
    model = Model()
    @variable(model, -2 <= x[1:n] <= 2)
    @variable(model, p[1:m] in MOI.Parameter.(zeros(m)))
    @constraint(model, [i = 1:m], sum(A[i, j] * x[j] for j = 1:n) >= b[i] + p[i])
    @constraint(model, sum(E[1, j] * x[j] for j = 1:n) == e[1] + 0.1 * p[1])
    @objective(model, Min, sum((c[j] + sum(C[j, i] * p[i] for i = 1:m)) * x[j] for j = 1:n))
    return model, rng -> 0.2 .* (2 .* rand(rng, m) .- 1)
end

function dll_bounded_socp(rng; n = 4)
    F = 0.1 .* randn(rng, n, n)
    model = Model()
    @variable(model, 0 <= x[1:n] <= 1)
    @variable(model, μ[1:n] in MOI.Parameter.(zeros(n)))
    @variable(model, γ in MOI.Parameter(1.0))
    @constraint(model, sum(x) == 1)
    @constraint(model, [γ; F * x] in SecondOrderCone())
    @objective(model, Max, dot(μ, x))
    γmin = opnorm(F)
    return model, rng -> [randn(rng, n); γmin * (1 + rand(rng))]
end

function dll_convex_qp(rng; n = 4, m = 2)
    M = randn(rng, n, n)
    Q = M * M' + I
    A, E = randn(rng, m, n), randn(rng, 1, n)
    x0 = rand(rng, n) .- 0.5
    b, e = A * x0 .- rand(rng, m), E * x0
    c, C = randn(rng, n), 0.5 .* randn(rng, n, m)
    model = Model()
    @variable(model, x[1:n] >= -3)
    @variable(model, p[1:m] in MOI.Parameter.(zeros(m)))
    @constraint(model, [i = 1:m], sum(A[i, j] * x[j] for j = 1:n) >= b[i] + p[i])
    @constraint(model, sum(E[1, j] * x[j] for j = 1:n) == e[1])
    lin = sum((c[j] + sum(C[j, i] * p[i] for i = 1:m)) * x[j] for j = 1:n)
    @objective(model, Min, 0.5 * sum(Q[i, j] * x[i] * x[j] for i = 1:n, j = 1:n) + lin)
    return model, rng -> 0.2 .* (2 .* rand(rng, m) .- 1)
end

function check_dll()
    for (name, make) in (
        ("bounded LP", dll_bounded_lp),
        ("bounded SOCP", dll_bounded_socp),
        ("convex QP", dll_convex_qp),
    )
        rng = Random.Xoshiro(11)
        model, sampler = make(rng)
        L2ODLL.decompose!(model)
        layer = batched_layer(model)
        B = 16
        P = reduce(hcat, [sampler(rng) for _ = 1:B])
        Y = randn(rng, sum(y_shape(model)), B)
        Pd, Yd = todev(P), todev(Y)
        for (lname, l) in (("dense", L2ODLL.densify(layer)), ("sparse", layer))
            dev = Ref{Any}(nothing)
            run_case("dll", "$name ($lname): dual bounds"; tol = 1e-10) do
                dev[] = Adapt.adapt(DevArray, l)
                return relerr(dev[](Yd, Pd), layer(Y, P))
            end
            dev[] === nothing && continue
            run_case("dll", "$name ($lname): Zygote gradient"; tol = 1e-10) do
                ref = Zygote.gradient(y -> sum(layer(y, P)), Y)[1]
                return relerr(Zygote.gradient(y -> sum(dev[](y, Pd)), Yd)[1], ref)
            end
            run_case("dll", "$name ($lname): batched_completion"; tol = 1e-10) do
                c, cd = batched_completion(layer, Y, P), batched_completion(dev[], Yd, Pd)
                return maxerr(
                    relerr(cd.y, c.y),
                    isempty(c.z) ? 0.0 : relerr(cd.z, c.z),
                    isempty(c.w) ? 0.0 : relerr(cd.w, c.w),
                )
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 2. dc3: affine QP of Donti et al. Section 4.1 (L2ODC3 test/problems.jl)
# ---------------------------------------------------------------------------
function dc3_qp(rng; n = 8, neq = 3, nin = 4)
    Q, c = Diagonal(rand(rng, n)), rand(rng, n)
    A, G = randn(rng, neq, n), randn(rng, nin, n)
    h = vec(sum(abs.(G * pinv(A)); dims = 2))
    model = Model()
    @variable(model, x[1:neq] in MOI.Parameter.(zeros(neq)))
    @variable(model, y[1:n])
    @constraint(model, A * y .== x)
    @constraint(model, G * y .<= h)
    @objective(model, Min, 0.5 * y' * Q * y + c' * y)
    return model
end

function check_dc3()
    rng = Random.Xoshiro(31)
    n, neq, B = 8, 3, 32
    method = DC3(; train_steps = 5, test_steps = 10)
    pipe = L2OPipeline(ParametricProblem(dc3_qp(rng; n, neq)), method)
    d = pipe.decomposition
    P = 2 .* rand(rng, neq, B) .- 1
    Z = randn(rng, n - neq, B)
    P32, Z32 = todev(Float32, P), todev(Float32, Z)
    d32 = Ref{Any}(nothing)
    run_case("dc3", "adapt(DevArray{Float32}, decomposition)"; tol = 0) do
        d32[] = Adapt.adapt(DevArray{Float32}, d)
        ok = d32[].form.A isa DevArray && eltype(d32[].form.A) == Float32
        return (ok, "A is $(typeof(d32[].form.A).name.name){$(eltype(d32[].form.A))}")
    end
    d32[] === nothing && return
    run_case("dc3", "predict (Float32 device vs Float64 CPU)"; tol = 1e-3) do
        return relerr(predict(method, d32[], P32, Z32), predict(method, d, P, Z))
    end
    run_case("dc3", "training_loss (Float32 device vs Float64 CPU)"; tol = 1e-4) do
        return relerr(
            training_loss(method, d32[], P32, Z32),
            training_loss(method, d, P, Z),
        )
    end
    # Gradient of the loss with respect to the weights of a Lux proxy P ↦ Z.
    lux = Lux.Chain(Lux.Dense(neq => 16, tanh), Lux.Dense(16 => n - neq))
    ps, st = Lux.setup(rng, lux)
    ps64 = ComponentArray(Lux.f64(ps))
    psd = todev(ComponentArray(ps))
    loss(p, dec, Pm) = training_loss(method, dec, Pm, first(lux(Pm, p, st)))
    run_case("dc3", "Zygote gradient wrt Lux proxy weights"; tol = 1e-3) do
        ref = Zygote.gradient(p -> loss(p, d, P), ps64)[1]
        return relerr(Zygote.gradient(p -> loss(p, d32[], P32), psd)[1], ref)
    end
end

# ---------------------------------------------------------------------------
# 3. alm: ExaModels + BatchNLPKernels evaluator (L2OALM test/problems.jl)
# ---------------------------------------------------------------------------
# min ‖x‖² + x₁ - 3x₃ s.t. x₁ + x₂ + x₃ = θ₁, x₁ - x₂ ≤ θ₂, x₂ + 2x₃ ≥ -1, -2 ≤ x ≤ 2.
# BatchNLPKernels needs at least one augmented constraint (`constraint!`).
function alm_qp_exa(backend, batch_size)
    c = ExaCore(; backend)
    x = variable(c, 3; lvar = -2.0, uvar = 2.0)
    θ = parameter(c, [1.0, 1.0])
    s = constraint(c, x[1] + x[2] - θ[1] for i = 1:1)
    constraint!(c, s, i => x[3] for i = 1:1)
    constraint(c, x[1] - x[2] - θ[2] for i = 1:1; lcon = -Inf, ucon = 0.0)
    constraint(c, x[2] + 2x[3] for i = 1:1; lcon = -1.0, ucon = Inf)
    objective(c, x[i]^2 for i = 1:3)
    objective(c, x[1] - 3x[3] for i = 1:1)
    return BatchModel(
        ExaModel(c; prod = true),
        batch_size;
        config = BatchModelConfig(:viol_grad),
    )
end

function check_alm()
    rng = Random.Xoshiro(14)
    B = 32
    method = ALMMethod(; α = 2.0, ρmax = 1e3)
    forms = Ref{Any}(nothing)
    run_case(
        "alm",
        "ExaModel on $(nameof(typeof(exa_backend()))) → ALMFormulation";
        tol = 0,
    ) do
        forms[] = (
            ALMFormulation(alm_qp_exa(KernelAbstractions.CPU(), B)),
            ALMFormulation(alm_qp_exa(exa_backend(), B)),
        )
        return (num_rows(forms[][2]) == 6, "$(num_rows(forms[][2])) rows")
    end
    forms[] === nothing && return
    fcpu, fdev = forms[]
    nr = num_rows(fcpu)
    primal = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 3))
    dual = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => nr))
    ps, st = Lux.setup(rng, primal)
    pd, sd = Lux.setup(rng, dual)
    ps, pd = ComponentArray(Lux.f64(ps)), ComponentArray(Lux.f64(pd))
    P = 2 .* rand(rng, 2, B) .- 1
    Λ = abs.(randn(rng, nr, B))
    Pd, Λd, psd = exa_dev(P), exa_dev(Λ), exa_dev(ps)
    function loss(p, form, Pm, Λm)
        X = map_primal(method, form, first(primal(Pm, p, st)))
        return primal_loss(method, form, Pm, X, Λm; ρ = 3.0)
    end
    run_case("alm", "primal_loss (device vs CPU)"; tol = 1e-10) do
        return relerr(loss(psd, fdev, Pd, Λd), loss(ps, fcpu, P, Λ))
    end
    run_case("alm", "Zygote gradient wrt Lux weights"; tol = 1e-8) do
        ref = Zygote.gradient(p -> loss(p, fcpu, P, Λ), ps)[1]
        return relerr(Zygote.gradient(p -> loss(p, fdev, Pd, Λd), psd)[1], ref)
    end
    run_case("alm", "violation_measure (device vs CPU)"; tol = 1e-10) do
        X, Xd = first(primal(P, ps, st)), first(primal(Pd, psd, st))
        return relerr(
            violation_measure(method, fdev, Pd, Xd, Λd; ρ = 3.0),
            violation_measure(method, fcpu, P, X, Λ; ρ = 3.0),
        )
    end
    # A few PDL outer iterations on the device: the penalty objective
    # (Λ = 0, ρ = 1) on the training inputs and the violation should decrease.
    run_case("alm", "train! (4 outer iterations): loss decreases"; tol = 0) do
        Ptr = exa_dev(2 .* rand(rng, 2, 2B) .- 1)
        tr = ALMTrainer(
            Lux.Training.TrainState(primal, exa_dev(ps), st, Optimisers.Adam(1e-2)),
            Lux.Training.TrainState(dual, exa_dev(zero(pd)), sd, Optimisers.Adam(1e-2));
            ρ = 1.0,
        )
        Λ0 = exa_dev(zeros(nr, B))
        pen(p) = sum(loss(p, fdev, Ptr[:, ((k-1)*B+1):(k*B)], Λ0) for k = 1:2) / 2
        ℓ0 = pen(tr.primal.parameters)
        L2OALM.train!(
            tr,
            method,
            fdev,
            Ptr;
            outer_iterations = 4,
            primal_epochs = 10,
            dual_epochs = 5,
            batchsize = B,
            rng,
        )
        ℓ1 = pen(tr.primal.parameters)
        v0, v1 = tr.history[1].violation, tr.history[end].violation
        msg = "penalty loss $(fmt(ℓ0)) → $(fmt(ℓ1)), violation $(fmt(v0)) → $(fmt(v1))"
        return (ℓ1 < ℓ0 && v1 <= v0, msg)
    end
end

# ---------------------------------------------------------------------------
# 4. nested_ad: gradient wrt weights of a loss containing an input VJP
# ---------------------------------------------------------------------------
function check_nested_ad()
    rng = Random.Xoshiro(0)
    nin, nout, B = 6, 4, 32
    X, V, T = randn(rng, nin, B), randn(rng, nout, B), randn(rng, nin, B)
    Xd, Vd, Td = todev(X), todev(V), todev(T)
    cpu_failure = Ref{Union{Nothing,String}}(nothing)
    # One hidden layer: nested Zygote compiles a new pullback for every layer
    # type and activation, and compilation dominates the run time.
    for (aname, act) in ACTIVATIONS
        lux = Lux.Chain(Lux.Dense(nin => 32, act), Lux.Dense(32 => nout))
        ps, st = Lux.setup(rng, lux)
        ps = ComponentArray(Lux.f64(ps))
        psd = todev(ps)
        sm(p) = StatefulLuxLayer{true}(lux, p, st)
        lv(p, x, v, t) =
            sum(abs2, Lux.vector_jacobian_product(sm(p), AutoZygote(), x, v) .- t)
        lz(p, x, v, t) = sum(abs2, Zygote.gradient(y -> sum(sm(p)(y) .* v), x)[1] .- t)
        ref = Zygote.gradient(p -> lv(p, X, V, T), ps)[1]
        run_case(
            "nested_ad",
            "Lux, Zygote over Lux.vector_jacobian_product, $aname";
            tol = 1e-8,
        ) do
            return relerr(Zygote.gradient(p -> lv(p, Xd, Vd, Td), psd)[1], ref)
        end
        run_case("nested_ad", "Lux, hand-nested Zygote, $aname"; tol = 1e-8) do
            # Run on the CPU first, so that a path that is broken everywhere is
            # reported as such (and not compiled a second time for the device).
            # The failure does not depend on the activation, so after a CPU
            # failure the other activations are not recompiled.
            cpu_failure[] === nothing || throw(
                ErrorException(
                    "fails on the CPU as well (as for $(first(ACTIVATIONS)[1]); " *
                    "not retried): " *
                    cpu_failure[],
                ),
            )
            g = try
                Zygote.gradient(p -> lz(p, X, V, T), ps)[1]
            catch err
                cpu_failure[] = oneline(err)
                throw(ErrorException("fails on the CPU as well: " * cpu_failure[]))
            end
            g === nothing && return nothing
            return relerr(Zygote.gradient(p -> lz(p, Xd, Vd, Td), psd)[1], ref)
        end
        fm = Flux.f64(Flux.Chain(Flux.Dense(nin => 32, act), Flux.Dense(32 => nout)))
        θ, re = Flux.destructure(fm)
        fz(t, x, v, tt) = sum(abs2, Zygote.gradient(y -> sum(re(t)(y) .* v), x)[1] .- tt)
        run_case("nested_ad", "Flux, Zygote over Zygote, $aname"; tol = 1e-8) do
            refθ = Zygote.gradient(t -> fz(t, X, V, T), θ)[1]
            return relerr(Zygote.gradient(t -> fz(t, Xd, Vd, Td), todev(θ))[1], refθ)
        end
    end
end

# Full L2OSobolev loss gradients, in a child process (see the header). On a GPU
# the child runs in parallel with the other checks (compilation of the nested
# Zygote calls dominates both); with `--device=jlarray` or `--serial` it runs
# after them.
const PARALLEL = DEVICE == "cuda" && !("--serial" in ARGS)
function start_worker()
    out, log = tempname(), tempname()
    acts = join(first.(ACTIVATIONS), ',')
    cmd = `$(Base.julia_cmd()) --startup-file=no $(@__FILE__) --device=$DEVICE --activations=$acts --sobolev-worker=$out`
    println("Started the L2OSobolev child process (log: $log)")
    return (; proc = run(pipeline(cmd; stdout = log, stderr = log); wait = false), out, log)
end
function collect_worker(w)
    println("\n== nested_ad (L2OSobolev child process)")
    wait(w.proc)
    isfile(w.log) && print(read(w.log, String))
    if isfile(w.out)
        for line in eachline(w.out)
            c, case, status, result = String.(split(line, '\t'; limit = 4))
            push!(RESULTS, (; check = c, case, status, result))
        end
    end
    success(w.proc) || push!(
        RESULTS,
        (;
            check = "nested_ad",
            case = "L2OSobolev child process",
            status = "error",
            result = "ERROR: child process failed (log: $(w.log))",
        ),
    )
    return nothing
end

# ---------------------------------------------------------------------------
# 5. core: L2OCore MatrixEvaluator on device arrays
# ---------------------------------------------------------------------------
function check_core()
    rng = Random.Xoshiro(5)
    model, sampler = dll_bounded_lp(rng)
    ev = L2OCore.MatrixEvaluator(ParametricProblem(model))
    B = 8
    P = reduce(hcat, [sampler(rng) for _ = 1:B])
    X = randn(rng, num_decision_variables(ParametricProblem(model)), B)
    run_case("core", "MatrixEvaluator: batch_objective, row_violation"; tol = 1e-10) do
        evd = Adapt.adapt(DevArray, ev)
        evd.A isa DevArray || return NotSupported("not yet supported (issue #39)")
        Pd, Xd = todev(P), todev(X)
        return maxerr(
            relerr(L2OCore.batch_objective(evd, Xd, Pd), L2OCore.batch_objective(ev, X, P)),
            relerr(L2OCore.row_violation(evd, Xd, Pd), L2OCore.row_violation(ev, X, P)),
        )
    end
end

# ---------------------------------------------------------------------------
# Run and report
# ---------------------------------------------------------------------------
const RUNNERS = Dict(
    "dll" => check_dll,
    "dc3" => check_dc3,
    "alm" => check_alm,
    "nested_ad" => check_nested_ad,
    "core" => check_core,
)
const T0 = time()
const WORKER_RUN = "nested_ad" in ONLY && PARALLEL ? start_worker() : nothing
for name in ONLY
    println("\n== ", name)
    try
        RUNNERS[name]()
    catch err
        # A failure outside a case (for example while building an instance).
        push!(
            RESULTS,
            (;
                check = name,
                case = "(setup)",
                status = "error",
                result = "ERROR: " * oneline(err),
            ),
        )
    end
end
if "nested_ad" in ONLY
    collect_worker(WORKER_RUN === nothing ? start_worker() : WORKER_RUN)
end
# Group the rows by check, in the order of `--only`.
sort!(RESULTS; by = r -> findfirst(==(r.check), ONLY))

count_status(s) = count(r -> r.status == s, RESULTS)
summary = string(
    count_status("pass"),
    " pass, ",
    count_status("fail"),
    " fail, ",
    count_status("error"),
    " error",
    count_status("skip") > 0 ? ", $(count_status("skip")) not yet supported" : "",
)
mdcell(s) = replace(s, '|' => "\\|")
table = join(
    [
        "| check | case | result |",
        "|:--|:--|:--|",
        ["| $(r.check) | $(mdcell(r.case)) | $(mdcell(r.result)) |" for r in RESULTS]...,
    ],
    '\n',
)
report = """
# GPU check results

- device mode: `$DEVICE`; GPU: $GPU_NAME
- Julia $VERSION; run time $(round(Int, time() - T0)) s
- packages: $(join(REVS, "; "))

$table

**$summary**
"""
println("\n", table, "\n\n", summary)
const OUTFILE = joinpath(@__DIR__, "gpu_check_results.md")
write(OUTFILE, report)
println("Wrote ", OUTFILE)
