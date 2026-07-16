# EV-11 Monte-Carlo critical-value generator (reproducible provenance for PO_ZA_CV / HANSEN_LC_CV).
#
# Phillips–Ouliaris Ẑ_α and Hansen Lc have no closed-form MacKinnon response surface, so their
# critical values are Monte-Carlo null quantiles of THIS PACKAGE's own estimators (self-consistent
# oracle). Run:
#   julia --project=. scripts/ev11_mc_cv.jl [REPS] [T]
# and paste the printed literals into src/teststat/critical_values.jl.
#
# PO_ZA_CV[:case][N]  = (q01,q05,q10) LOWER quantiles of Ẑ_α  (reject if Ẑ_α ≤ cv), N = k+1.
# HANSEN_LC_CV[:case][k] = (q90,q95,q99) UPPER quantiles of Lc (reject if Lc ≥ cv).

using MacroEconometricModels, Random, Statistics
const MEM = MacroEconometricModels

reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
Tn   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
seed = 20260716
cases = [:none, :constant, :trend]

rw(rng, T) = cumsum(randn(rng, T))                       # driftless I(1)

# --- Phillips–Ouliaris Ẑ_α under H0: independent random walks (no cointegration) ---
function po_za_table(reps, Tn, seed)
    tbl = Dict(c => Vector{NTuple{3,Float64}}(undef, 6) for c in cases)
    for c in cases
        for k in 1:5
            rng = MersenneTwister(seed + 1000*Int(hash(c)%1000) + k)
            za = Vector{Float64}(undef, reps)
            for r in 1:reps
                y = rw(rng, Tn)
                X = hcat((rw(rng, Tn) for _ in 1:k)...)
                resid = MEM._coint_levels_resid(y, X, c)
                _, zav, _ = MEM._po_pp_stats(resid)
                za[r] = zav
            end
            q = quantile(za, [0.01, 0.05, 0.10])
            tbl[c][k+1] = (round(q[1], digits=3), round(q[2], digits=3), round(q[3], digits=3))
        end
        tbl[c][1] = tbl[c][2]                            # N=1 unused (k≥1); safe fallback
    end
    return tbl
end

# --- Hansen Lc under H0: stable cointegration (β=1, I(0) iid errors) ---
function hansen_lc_table(reps, Tn, seed)
    tbl = Dict(c => Vector{NTuple{3,Float64}}(undef, 5) for c in cases)
    trendmap = Dict(:none => :none, :constant => :const, :trend => :linear)
    for c in cases
        for k in 1:5
            rng = MersenneTwister(seed + 7000*Int(hash(c)%1000) + k)
            lc = Vector{Float64}(undef, reps)
            for r in 1:reps
                X = hcat((rw(rng, Tn) for _ in 1:k)...)
                u = randn(rng, Tn)
                y = vec(sum(X, dims=2)) .+ u             # β = 1 on each regressor
                m = estimate_cointreg(y, X; method=:fmols, trend=trendmap[c])
                lc[r] = hansen_instability_test(m).statistic
            end
            q = quantile(lc, [0.90, 0.95, 0.99])
            tbl[c][k] = (round(q[1], digits=3), round(q[2], digits=3), round(q[3], digits=3))
        end
    end
    return tbl
end

function emit(name, tbl, cases, nrow)
    println("const $name = Dict(")
    for c in cases
        entries = join([string(tbl[c][i]) for i in 1:nrow], ", ")
        println("    :$c => [$entries],")
    end
    println(")")
end

println("# reps=$reps  T=$Tn  seed=$seed")
@time po = po_za_table(reps, Tn, seed)
emit("PO_ZA_CV", po, cases, 6)
@time lc = hansen_lc_table(reps, Tn, seed)
emit("HANSEN_LC_CV", lc, cases, 5)
