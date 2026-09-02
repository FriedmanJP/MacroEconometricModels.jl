# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Regenerates test/fixtures/serialization/v1/*.jld2 (RSER family). Run only
# when SERIALIZATION_FORMAT_VERSION bumps. DSER RBC/HA fixtures are written
# by test/dsge/gen_serialization_v1_fixtures.jl and are not overwritten here.

using MacroEconometricModels, Random, LinearAlgebra, DataFrames, Statistics

const _DIR = joinpath(@__DIR__, "fixtures", "serialization", "v1")
mkpath(_DIR)

# ── VAR + IRF bundle ─────────────────────────────────────────────────────────
Y = randn(MersenneTwister(787), 80, 2)
var_m = estimate_var(Y, 2)
ir = irf(var_m, 4)
save_model(Dict("var" => var_m, "irf" => ir), joinpath(_DIR, "var_irf.jld2");
           note="v1 VAR+IRF")

# ── BVAR posterior ───────────────────────────────────────────────────────────
post = estimate_bvar(Y, 2; n_draws=30, seed=787)
save_model(post, joinpath(_DIR, "bvar.jld2"))

# ── ARIMA ────────────────────────────────────────────────────────────────────
ya = randn(MersenneTwister(7871), 80)
ar = estimate_arima(ya, 1, 0, 1; method=:css)
save_model(ar, joinpath(_DIR, "arima.jld2"))

# ── GARCH ────────────────────────────────────────────────────────────────────
yg = randn(MersenneTwister(7872), 200)
garch = estimate_garch(yg, 1, 1)
save_model(garch, joinpath(_DIR, "garch.jld2"))

# ── factor ───────────────────────────────────────────────────────────────────
X = randn(MersenneTwister(7873), 60, 8)
fm = estimate_factors(X, 2)
save_model(fm, joinpath(_DIR, "factor.jld2"))

# ── nowcast DFM ──────────────────────────────────────────────────────────────
function _v1_nowcast_data(; T_obs=60, nM=4, nQ=1, r=1, seed=778)
    rng = MersenneTwister(seed)
    F = randn(rng, T_obs, r)
    for t in 2:T_obs
        F[t, :] = 0.7 * F[t-1, :] + 0.3 * randn(rng, r)
    end
    X_M = F * randn(rng, nM, r)' + 0.2 * randn(rng, T_obs, nM)
    X_Q = F * randn(rng, nQ, r)' + 0.2 * randn(rng, T_obs, nQ)
    for t in 1:T_obs
        if mod(t, 3) != 0
            X_Q[t, :] .= NaN
        end
    end
    return hcat(X_M, X_Q)
end
dfm = nowcast_dfm(_v1_nowcast_data(), 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)
save_model(dfm, joinpath(_DIR, "nowcast_dfm.jld2"))

# ── DiD event study ──────────────────────────────────────────────────────────
function _v1_did_panel(; n_units=36, n_periods=16, seed=780)
    rng = MersenneTwister(seed)
    n_cohorts = 2
    units_per = n_units ÷ (n_cohorts + 1)
    treat_times = zeros(Int, n_units)
    for c in 1:n_cohorts
        t0 = 5 + 3 * (c - 1)
        for u in ((c - 1) * units_per + 1):(c * units_per)
            treat_times[u] = t0
        end
    end
    N = n_units * n_periods
    id = Vector{Int}(undef, N)
    t = Vector{Int}(undef, N)
    y = Vector{Float64}(undef, N)
    treat_time = Vector{Float64}(undef, N)
    cohort = Vector{Int}(undef, N)
    row = 1
    for i in 1:n_units
        a = randn(rng)
        for tt in 1:n_periods
            g = treat_times[i]
            te = (g > 0 && tt >= g) ? 1.5 * (1 + 0.1 * (tt - g)) : 0.0
            y[row] = a + 0.1 * tt + te + 0.4 * randn(rng)
            id[row] = i
            t[row] = tt
            treat_time[row] = Float64(g)
            cohort[row] = g
            row += 1
        end
    end
    df = DataFrame(id=id, t=t, y=y, treat_time=treat_time, cohort=cohort)
    xtset(df, :id, :t; cohort=:cohort, frequency=Quarterly)
end
es = estimate_event_study_lp(_v1_did_panel(), :y, :treat_time, 3; leads=2, lags=1)
save_model(es, joinpath(_DIR, "did_event_study.jld2"))

# ── OPP result ───────────────────────────────────────────────────────────────
H = 6
rng_opp = MersenneTwister(783)
ce = PolicyCausalEffects(outcomes=[:u, :infl], instruments=[:rate],
                         Theta_x=[randn(rng_opp, H, 2), 0.5 .* randn(rng_opp, H, 2)],
                         Theta_z=[randn(rng_opp, H, 2)], source=:var)
loss = policy_loss([:u, :infl], H; lambda=[2.0, 1.0], beta=0.97)
fc = PolicyForecast{Float64}([:u, :infl],
                             [randn(rng_opp, H), randn(rng_opp, H)],
                             nothing, H, "2021Q2")
opp_pt = opp(fc, ce, loss)
save_model(opp_pt, joinpath(_DIR, "opp.jld2"))

# ── teststat bundle ──────────────────────────────────────────────────────────
adf = adf_test(ya; lags=1)
kpss = kpss_test(ya)
save_model(Dict("adf" => adf, "kpss" => kpss), joinpath(_DIR, "teststat.jld2");
           note="v1 teststat")

println("wrote RSER v1 fixtures in ", _DIR)
for f in ("var_irf.jld2", "bvar.jld2", "arima.jld2", "garch.jld2", "factor.jld2",
          "nowcast_dfm.jld2", "did_event_study.jld2", "opp.jld2", "teststat.jld2")
    p = joinpath(_DIR, f)
    println("  ", f, "  ", filesize(p), " bytes")
end
