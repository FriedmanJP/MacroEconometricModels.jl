# Benchmarks for Lewis TVV-ID and BB SV-SVAR (v0.9.6, #827)
# Run: julia --project=. benchmarks/bench_tvv_sv.jl
#
# T×n timing grid on SV-DGP data (persistent AR(1) log-volatility, the DGP
# both estimators are designed for). Seeds fixed; EM/GMM run to convergence
# (no maxiter caps), so cells measure full-estimation cost.

using MacroEconometricModels
using Random
using LinearAlgebra
using Printf

const MEM = MacroEconometricModels

function sv_data(n, Tobs; seed=827)
    rng = Xoshiro(seed)
    rhos = n == 2 ? [0.97, 0.85] : [0.98, 0.96, 0.94]
    sigs = n == 2 ? [0.25, 0.20] : [0.22, 0.32, 0.38]
    h = zeros(Tobs, n)
    for t in 2:Tobs, i in 1:n
        h[t, i] = rhos[i] * h[t - 1, i] + sigs[i] * randn(rng)
    end
    E = randn(rng, Tobs, n) .* exp.(h ./ 2)
    B0 = n == 2 ? [1.0 0.4; -0.2 1.0] : [1.0 0.3 0.1; 0.2 1.0 0.2; 0.1 0.3 1.0]
    A = [0.5 * Matrix{Float64}(I, n, n)]
    Y = similar(E)
    Y[1, :] = (B0 * E[1, :])
    for t in 2:Tobs
        Y[t, :] = A[1] * Y[t - 1, :] + B0 * E[t, :]
    end
    return Y
end

function bench(f, name; n_runs=3)
    f()  # warmup (compile + settles BLAS)
    times = [(@elapsed f()) for _ in 1:n_runs]
    med = sort(times)[div(n_runs, 2) + 1]
    @printf("%-28s median %8.1fs  (runs: %s)\n", name, med,
            join(round.(times; digits=1), ", "))
    return med
end

println("Julia threads: ", Threads.nthreads())
println()

for (n, Tobs) in ((2, 2000), (2, 20000), (3, 20000))
    Y = sv_data(n, Tobs)
    m = estimate_var(Y, 1)
    bench(() -> identify_lewis_tvv(m; rng=Xoshiro(827)), "lewis_tvv n=$n T=$Tobs")
end
println()
for (n, Tobs, runs) in ((2, 1000, 2), (2, 2000, 2), (3, 3000, 1))
    Y = sv_data(n, Tobs)
    bench(() -> identify_sv_svar(Y, 1; rng=Xoshiro(827)), "sv_em n=$n T=$Tobs"; n_runs=runs)
end
println()
println("done")
