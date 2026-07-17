# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# =============================================================================
# EViews-parity fixture generator (provenance / regeneration record).
#
# Julia's `MersenneTwister` stream is NOT stable across Julia minor versions
# (e.g. `rand(MersenneTwister(0))` == 0.8236475079774124 on Julia 1.10 but
# 0.44373084494754944 on Julia 1.12). The EV-series oracle tests were built by
# generating their DGP data on the author's Julia 1.12 toolchain, feeding the
# resulting CSVs to external engines (R / statsmodels / linearmodels / cointReg),
# and hard-coding the returned numbers. Regenerating that data with the same
# seed on CI's `lts` (1.10.x) yields a DIFFERENT stream, so the external oracles
# no longer apply and the tests fail.
#
# The fix: capture the exact Julia-1.12 DGP data ONCE and commit it as CSV
# fixtures that the tests load verbatim, making them independent of the running
# Julia RNG. This script is that capture. Each recipe is copied byte-for-byte
# from the corresponding test so the fixture equals what the external oracle saw.
#
# AUTHORITATIVE ARTIFACT = the committed CSVs, not this script. Re-running this
# on a Julia version whose MersenneTwister stream differs from 1.12 would
# regenerate different data and invalidate the external oracles — do not do so
# unless you also re-derive every external oracle constant. (Generated once on
# Julia 1.12.6.)
# =============================================================================

using Random, LinearAlgebra, DelimitedFiles

const TESTDIR = @__DIR__
_datadir(sub) = (d = joinpath(TESTDIR, sub, "data"); mkpath(d); d)
_write(sub, name, M) = (writedlm(joinpath(_datadir(sub), name), M, ','); println("wrote $sub/data/$name  size=", size(M)))

# --- reg/test_reg_diagnostics.jl : _diag_oracle_dgp() -----------------------
let
    rng = MersenneTwister(20260716); n = 200
    x1 = randn(rng, n); x2 = randn(rng, n)
    u = (0.5 .+ 0.8 .* abs.(x1)) .* randn(rng, n)
    y = 1.0 .+ 2.0 .* x1 .- 1.0 .* x2 .+ u
    _write("reg", "reg_diag_oracle.csv", hcat(y, x1, x2))
end

# --- reg/test_stability.jl : _stab_oracle_dgp() -----------------------------
let
    rng = MersenneTwister(20260717); n = 60
    x1 = randn(rng, n); x2 = randn(rng, n); u = randn(rng, n)
    y = 1.0 .+ 0.8 .* x1 .- 0.5 .* x2 .+ u
    _write("reg", "reg_stab_oracle.csv", hcat(y, x1, x2))
end

# --- reg/test_ivkclass.jl : _ev36_dgp() -------------------------------------
let
    rng = MersenneTwister(20240736); n = 500
    z1 = randn(rng, n); z2 = randn(rng, n); z3 = randn(rng, n)
    w = randn(rng, n); u = randn(rng, n); v = randn(rng, n)
    xend = 0.6 .* z1 .+ 0.4 .* z2 .+ 0.3 .* z3 .+ 0.5 .* w .+ 0.7 .* u .+ v
    y = 1.0 .+ 2.0 .* xend .+ 0.5 .* w .+ u
    _write("reg", "reg_ev36.csv", hcat(y, w, xend, z1, z2, z3))
end

# --- reg/test_selection.jl : _sel_dgp_sparse() (default seed 42) ------------
let
    rng = MersenneTwister(42); n = 200; k = 20; active = [2, 5, 8, 11, 14]; b = 1.5
    X = hcat(ones(n), randn(rng, n, k))
    beta = zeros(k + 1); beta[1] = 0.5
    for a in active; beta[a] = b; end
    y = X * beta + randn(rng, n)
    _write("reg", "reg_sel_sparse42.csv", hcat(y, X[:, 2:end]))   # y + 20 candidate cols
end

# --- teststat/test_dumitrescu_hurlin.jl : make_dh_panel() -------------------
let
    rng = MersenneTwister(20240717); N = 8; T = 30
    ys = Float64[]; xs = Float64[]
    for i in 1:N
        x = randn(rng, T)
        y = zeros(T)
        b = 0.3 + 0.1 * (i - 1) / N
        for t in 3:T
            y[t] = 0.4 * y[t-1] - 0.1 * y[t-2] + b * x[t-1] + 0.2 * x[t-2] + randn(rng)
        end
        append!(ys, y); append!(xs, x)
    end
    _write("teststat", "dh_panel.csv", hcat(ys, xs))              # 240×2, id/time reconstructed
end

# --- teststat/test_cointegration_resid.jl : coint_pair / indep_pair ---------
_coint_pair(seed, T; beta = 2.0, rho = 0.5) = begin
    rng = MersenneTwister(seed)
    x = cumsum(randn(rng, T)); e = zeros(T)
    for t in 2:T; e[t] = rho * e[t-1] + randn(rng); end
    y = 1.0 .+ beta .* x .+ e
    (y, x)
end
_indep_pair(seed, T) = begin
    rng = MersenneTwister(seed)
    x = cumsum(randn(rng, T)); y = cumsum(randn(rng, T))
    (y, x)
end
let (y, x) = _coint_pair(12345, 200); _write("teststat", "coint_pair_12345_200.csv", hcat(y, x)) end
let (y, x) = _indep_pair(999, 200);   _write("teststat", "indep_pair_999_200.csv",   hcat(y, x)) end
let (y, x) = _coint_pair(2024, 250);  _write("teststat", "coint_pair_2024_250.csv",  hcat(y, x)) end

# --- teststat/test_hegy.jl : _seasonal_rw(360, 12; seed=44) -----------------
let
    rng = MersenneTwister(44); n = 360; s = 12
    y = zeros(n); e = randn(rng, n)
    for t in 1:n; y[t] = (t <= s ? 0.0 : y[t-s]) + e[t]; end
    _write("teststat", "hegy_srw_360_12_44.csv", reshape(y, :, 1))
end

# --- teststat/test_panel_unitroot_firstgen.jl : degenerate testset ----------
let
    rng = MersenneTwister(7)
    Xrw = cumsum(randn(rng, 60, 20); dims = 1)   # _rw_panel(rng)
    Xst = randn(rng, 60, 20)                     # _stat_panel(rng)
    _write("teststat", "firstgen_deg_rw.csv", Xrw)
    _write("teststat", "firstgen_deg_st.csv", Xst)
end

# --- teststat/test_panel_cointegration.jl : _nocoint_dgp(MT(42), 60, 20) ----
let
    rng = MersenneTwister(42); Tobs = 60; N = 20; k = 1
    Y = zeros(Tobs, N); X = zeros(Tobs, N, k)
    for i in 1:N
        Y[:, i] = cumsum(randn(rng, Tobs))
        for j in 1:k; X[:, i, j] = cumsum(randn(rng, Tobs)); end
    end
    _write("teststat", "pcoint_nocoint_Y.csv", Y)
    _write("teststat", "pcoint_nocoint_X.csv", X[:, :, 1])
end

# --- teststat/test_bubble.jl : y_rw and yb ----------------------------------
let
    rng = MersenneTwister(12345)
    y_rw = cumsum(randn(rng, 100))
    _write("teststat", "bubble_yrw.csv", reshape(y_rw, :, 1))
end
let
    r = MersenneTwister(2024); T = 120; yy = zeros(T)
    for t in 2:T; yy[t] = (50 <= t <= 75 ? 1.05 : 1.0) * yy[t-1] + randn(r); end
    _write("teststat", "bubble_yb.csv", reshape(yy, :, 1))
end

# --- cointreg/test_cointreg.jl : coint_dgp(; endog=true) and multi-regressor -
let
    rng = MersenneTwister(20260716); T = 200
    v = randn(rng, T); e = randn(rng, T); x = cumsum(v)
    u = zeros(T)
    for t in 1:T
        ulag = t == 1 ? 0.0 : u[t-1]
        u[t] = 0.4 * ulag + e[t] + 0.6 * v[t]
    end
    y = 2.0 .+ 1.5 .* x .+ u
    _write("cointreg", "coint_dgp_endog.csv", hcat(y, x))
end
let
    rng = MersenneTwister(11); Tn = 220
    x1 = cumsum(randn(rng, Tn)); x2 = cumsum(randn(rng, Tn))
    uu = zeros(Tn)
    for t in 2:Tn; uu[t] = 0.3uu[t-1] + randn(rng); end
    _write("cointreg", "cointreg_multireg.csv", hcat(x1, x2, uu))
end

# --- ardl/test_nardl.jl : _nardl_dgp(987654321, 250; θp=1.2, θn=-0.7) -------
let
    seed = 987654321; N = 250; θp = 1.2; θn = -0.7; φ = 0.35; ψ = 0.25; σ = 0.5
    rng = MersenneTwister(seed)
    x = zeros(N)
    for t in 2:N; x[t] = x[t-1] + randn(rng); end
    dx = diff(x)
    xp = zeros(N); xn = zeros(N)
    for t in 2:N
        xp[t] = xp[t-1] + max(dx[t-1], 0.0)
        xn[t] = xn[t-1] + min(dx[t-1], 0.0)
    end
    y = zeros(N)
    for t in 2:N
        ec = y[t-1] - (θp * xp[t-1] + θn * xn[t-1])
        y[t] = y[t-1] - φ * ec + ψ * dx[t-1] + σ * randn(rng)
    end
    _write("ardl", "nardl_987654321_250.csv", hcat(y, x))
end

# --- nonlinear/test_star.jl : _sim_lstar(n=1200, gamma=15, c=0, seed=...) ----
let
    rng = MersenneTwister(20240716); n = 1200; gamma = 15.0; c = 0.0
    phi1 = (0.5, 0.6); phi2 = (-0.4, -0.3); sigma = 0.3
    y = zeros(n)
    for t in 2:n
        s = y[t-1]
        G = 1 / (1 + exp(-gamma * (s - c)))
        y[t] = (phi1[1] + phi1[2] * y[t-1]) * (1 - G) +
               (phi2[1] + phi2[2] * y[t-1]) * G + sigma * randn(rng)
    end
    _write("nonlinear", "star_lstar_1200.csv", reshape(y, :, 1))
end

println("\nAll EV fixtures generated.")
