# Benchmarks for DSGE nonlinear solvers
# Run: julia --project=. benchmarks/bench_nonlinear.jl
# Optional: julia --project=. -t 4 benchmarks/bench_nonlinear.jl  (for threading)

using MacroEconometricModels
using Random
using Printf

println("Julia threads: ", Threads.nthreads())
println()

# ─── Model Setup ───

# Simple AR(1)
spec_ar1 = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec_ar1 = compute_steady_state(spec_ar1)

# Neoclassical growth model (2 variables, 1 shock)
spec_growth = @dsge begin
    parameters: α = 0.36, β = 0.99, δ = 0.025, γ = 2.0, σ_e = 0.01
    endogenous: k, c
    exogenous: ε
    c[t]^(-γ) - β * c[t+1]^(-γ) * (α * k[t]^(α - 1) + 1 - δ) = 0
    k[t] - k[t-1]^α - (1 - δ) * k[t-1] + c[t] - σ_e * ε[t] = 0
    steady_state = begin
        k_ss = (α / (1/β - 1 + δ))^(1 / (1 - α))
        c_ss = k_ss^α - δ * k_ss
        [k_ss, c_ss]
    end
end
spec_growth = compute_steady_state(spec_growth)

# ─── Benchmarks ───

function bench(f, name; n_runs=3)
    # Warmup
    f()
    # Timed runs
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t)
    end
    median_t = sort(times)[div(n_runs, 2) + 1]
    @printf("  %-45s %8.3f ms (median of %d)\n", name, median_t * 1000, n_runs)
    return median_t
end

println("═══ AR(1) model (1 var, 1 shock) ═══")
bench("Projection (degree=5)") do
    solve(spec_ar1; method=:projection, degree=5, verbose=false)
end
bench("PFI (degree=5)") do
    solve(spec_ar1; method=:pfi, degree=5, verbose=false)
end
bench("VFI (degree=5)") do
    solve(spec_ar1; method=:vfi, degree=5, verbose=false)
end
bench("VFI + Howard(5)") do
    solve(spec_ar1; method=:vfi, degree=5, howard_steps=5, verbose=false)
end
bench("VFI + Anderson(3)") do
    solve(spec_ar1; method=:vfi, degree=5, anderson_m=3, verbose=false)
end
bench("VFI + Howard(5) + Anderson(3)") do
    solve(spec_ar1; method=:vfi, degree=5, howard_steps=5, anderson_m=3, verbose=false)
end

println()
println("═══ Growth model (2 vars, 1 shock) ═══")
bench("Projection (degree=5, tol=1e-3)") do
    solve(spec_growth; method=:projection, degree=5, verbose=false, tol=1e-3)
end
bench("PFI (degree=5, tol=1e-3)") do
    solve(spec_growth; method=:pfi, degree=5, verbose=false, tol=1e-3)
end
bench("VFI (degree=5, tol=1e-3)") do
    solve(spec_growth; method=:vfi, degree=5, verbose=false, tol=1e-3)
end
bench("VFI + Howard(10, tol=1e-3)") do
    solve(spec_growth; method=:vfi, degree=5, howard_steps=10, verbose=false, tol=1e-3)
end

if Threads.nthreads() > 1
    println()
    println("═══ Threading comparison (growth model) ═══")
    bench("PFI sequential") do
        solve(spec_growth; method=:pfi, degree=5, threaded=false, verbose=false, tol=1e-3)
    end
    bench("PFI threaded") do
        solve(spec_growth; method=:pfi, degree=5, threaded=true, verbose=false, tol=1e-3)
    end
    bench("VFI sequential") do
        solve(spec_growth; method=:vfi, degree=5, threaded=false, verbose=false, tol=1e-3)
    end
    bench("VFI threaded") do
        solve(spec_growth; method=:vfi, degree=5, threaded=true, verbose=false, tol=1e-3)
    end
end

println()
println("Done.")
