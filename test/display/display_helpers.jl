# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Display regression harness — shared helpers (T176 / #275)
# =============================================================================
#
# The goldens/invariants suite renders every fixture through ONE code path
# (`_render`) and asserts two complementary things:
#
#   * `test_display_invariants.jl` — RAW-string semantic invariants that lock in
#     the Stage-11 display fixes (S1 crop-off, S2 fixed-decimal / no `-0.0`, S6
#     dust/reference-row dashes, S7 no empty-header band, S8 Yes/No booleans).
#     These are regex/`occursin` checks, so they are cross-platform robust.
#
#   * `test_display_goldens.jl` — a CANONICALIZED snapshot of the full rendered
#     table (`_check_golden`). Canonicalization masks the volatile numeric cells
#     (which drift in the last ulp across BLAS/OS) and collapses column-padding
#     whitespace, so the golden locks the STRUCTURE — column headers, row labels,
#     notes, dialect markers, `%`/integer labels, stars, `—`/`(ref)` markers,
#     section order — without failing on a platform-dependent final digit. It is
#     regenerated wholesale with `MACRO_UPDATE_GOLDENS=1` whenever any sibling
#     display issue reformats output (churn is intended, per the plan).
#
# Loaded (idempotently) by both test files; guarded so re-`include` in the same
# process is a no-op.

using MacroEconometricModels
using Random, LinearAlgebra, Statistics

const _GOLDEN_DIR = joinpath(@__DIR__, "goldens")
const _UPDATE_GOLDENS = get(ENV, "MACRO_UPDATE_GOLDENS", "") == "1"

# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

"""
    _render(x) -> String

Render `x` exactly as a non-interactive consumer (a file, a pipe, Documenter, a
CI log) would see it: the `:text` backend, no color, a narrow 24×80 display so the
S1 crop defect would surface if it regressed. Prefers `report(io, x)` (the full
publication body — this is what exercises the three BESPOKE VAR/VECM/DSGEEstimation
report renderers that hardcode a `_pretty_table` rather than routing through `show`),
falling back to `show(io, MIME("text/plain"), x)` for the types whose `report`
convenience form only wraps `show`.

`redirect_stdout` is deliberately NOT used — it is thread-unsafe under the spawned
parallel runner and broken for `IOBuffer` on Julia 1.12. Capturing via an explicit
`IOContext` is why the bespoke types needed their `report(io, ...)` io-plumbing.
"""
function _render(x)
    with_display_backend(:text) do
        buf = IOBuffer()
        ioc = IOContext(buf, :displaysize => (24, 80), :color => false)
        try
            report(ioc, x)
        catch e
            e isa MethodError || rethrow()
            show(ioc, MIME("text/plain"), x)
        end
        return String(take!(buf))
    end
end

# -----------------------------------------------------------------------------
# Golden comparison
# -----------------------------------------------------------------------------

"""
    _canonicalize(s) -> String

Reduce a rendered table to its platform-stable structural skeleton. Everything
that is DERIVED from floating-point arithmetic — and therefore drifts in the last
ulp across BLAS/LAPACK/OS/Julia versions — is masked away, because a display
golden must survive Julia 1.10 (LTS) vs 1.12 CI without churn. In order:

  * floating-point cells (`-?\\d+\\.\\d+`, optional exponent) → `N`;
  * p-value threshold prefixes `<N`/`>N` (a value straddling 0.001/0.999 flips the
    `<0.001`/`>0.999` rendering) → `N`;
  * remaining integers → `N` — sample sizes are stable, but iteration counts,
    auto-selected lag orders, and loading ranks are algorithm-path-dependent and
    differ across environments, and the two cannot be told apart generically;
  * significance stars (`*`) stripped — a coefficient whose p-value straddles a
    0.01/0.05/0.10 boundary flips its stars across environments. The invariants
    suite checks star PRESENCE on the raw string; the golden locks structure only;
  * runs of ≥2 spaces → one space (column padding tracks the masked-away widths);
  * trailing whitespace and leading/trailing blank lines trimmed.

What survives — and what the golden therefore LOCKS — is the environment-invariant
text skeleton: titles, column headers, row labels, notes, legends, `%` symbols,
`—`/`(ref)` markers, `Yes`/`No`, dialect markers, section order, row counts, and
the overall line structure. The numeric VALUES and significance are locked instead
by `test_display_invariants.jl` and the per-issue unit tests in
`test/core/test_display_backends.jl`.
"""
function _canonicalize(s::AbstractString)
    out = String[]
    for raw in split(String(s), '\n')
        ln = replace(String(raw), r"-?\d+\.\d+(?:[eE][-+]?\d+)?" => "N")  # decimals/scientific
        ln = replace(ln, r"[<>]N" => "N")                                 # <0.001 / >0.999 threshold prefix
        ln = replace(ln, r"\d+" => "N")                                   # remaining (algorithm-path) integers
        ln = replace(ln, "*" => "")                                       # significance stars (p-value-derived)
        ln = replace(ln, r"[ \t]{2,}" => " ")                             # column padding
        push!(out, rstrip(ln))
    end
    while !isempty(out) && isempty(out[1]);   popfirst!(out); end
    while !isempty(out) && isempty(out[end]); pop!(out);      end
    return join(out, '\n')
end

"""
    _check_golden(name, s) -> Bool

Compare the canonicalized render of `s` against `goldens/<name>.txt`. With
`MACRO_UPDATE_GOLDENS=1` set, (over)writes the golden and returns `true` instead —
run once after the whole Stage-11 batch, eyeball the diff, and commit.
"""
function _check_golden(name::AbstractString, s::AbstractString)
    path = joinpath(_GOLDEN_DIR, name * ".txt")
    canon = _canonicalize(s)
    if _UPDATE_GOLDENS
        mkpath(_GOLDEN_DIR)
        write(path, canon * "\n")
        return true
    end
    isfile(path) || error("golden missing: $path — regenerate with MACRO_UPDATE_GOLDENS=1")
    expected = rstrip(read(path, String), '\n')
    ok = canon == expected
    if !ok
        # Surface the first divergence to make CI logs actionable.
        el = split(expected, '\n'); gl = split(canon, '\n')
        for i in 1:max(length(el), length(gl))
            e = i <= length(el) ? el[i] : "«missing»"
            g = i <= length(gl) ? gl[i] : "«missing»"
            if e != g
                @warn "golden mismatch: $name line $i" expected=e got=g
                break
            end
        end
    end
    return ok
end

# -----------------------------------------------------------------------------
# Deterministic fixtures
# -----------------------------------------------------------------------------
# Each entry: (name, obj, stars, ref). `stars=true` → the render must contain a
# significance `*` (high signal-to-noise DGPs keep this robust cross-platform).
# `ref=true` → the render must contain a dashed reference row (`—`) with no `***`
# on it. All fixtures are checked against the universal invariants regardless.
#
# Sizes are reduced and every RNG is an explicit MersenneTwister so the goldens
# are reproducible. NO Krusell-Smith / full HA solve (too heavy, per the plan).

# strongly-significant OLS design: y = X β + small noise
function _reg_fixture()
    rng = MersenneTwister(11)
    X = randn(rng, 120, 3)
    y = X * [1.5, -1.2, 0.8] .+ 0.15 .* randn(rng, 120)
    estimate_reg(y, X)
end

function _binary_fixtures()
    rng = MersenneTwister(21)
    X = randn(rng, 300, 2)
    η = X * [1.4, -1.0]
    p = @. 1 / (1 + exp(-η))
    y = Float64.(rand(rng, 300) .< p)
    estimate_logit(y, X), estimate_probit(y, X)
end

function _ologit_fixture()
    rng = MersenneTwister(31)
    n = 800
    β = [1.2, -0.8]
    X = randn(rng, n, 2)
    xb = X * β
    cut = (-0.5, 1.0)
    u = rand(rng, n)
    logistic(z) = 1 / (1 + exp(-z))
    y = Vector{Int}(undef, n)
    for i in 1:n
        y[i] = u[i] < logistic(cut[1] - xb[i]) ? 1 :
               u[i] < logistic(cut[2] - xb[i]) ? 2 : 3
    end
    estimate_ologit(y, X; varnames = ["x1", "x2"])
end

function _mlogit_fixture()
    rng = MersenneTwister(41)
    n = 800
    β = [0.5 -0.3; 1.2 -0.6; -0.7 0.9]   # K=3 (incl. intercept) × (J-1)=2
    X = [ones(n) randn(rng, n, 2)]
    V = X * β
    y = Vector{Int}(undef, n)
    for i in 1:n
        vmax = max(0.0, maximum(@view V[i, :]))
        denom = exp(-vmax) + sum(exp.(@view(V[i, :]) .- vmax))
        probs = (exp(-vmax) / denom, (exp.(@view(V[i, :]) .- vmax) ./ denom)...)
        u = rand(rng); acc = 0.0; y[i] = 3
        for j in 1:3
            acc += probs[j]
            if u < acc; y[i] = j; break; end
        end
    end
    estimate_mlogit(y, X; varnames = ["const", "x1", "x2"])
end

function _panel_fixture()
    rng = MersenneTwister(51)
    N_g = 8; T_p = 12; n = N_g * T_p
    gid = repeat(1:N_g, inner = T_p)
    tid = repeat(1:T_p, N_g)
    gmean = repeat(randn(rng, N_g), inner = T_p)        # genuine between variation
    x1 = 0.6 .* gmean .+ randn(rng, n)
    x2 = randn(rng, n)
    alpha = repeat(2 .* randn(rng, N_g), inner = T_p)
    y = alpha .+ 1.5 .* x1 .- 0.8 .* x2 .+ 0.3 .* randn(rng, n)
    pd = PanelData{Float64}(hcat(y, x1, x2), ["y", "x1", "x2"], Quarterly, [1, 1, 1],
                            gid, tid, nothing, ["g$i" for i in 1:N_g],
                            N_g, 3, n, true, ["panel fixture"],
                            Dict{String,String}(), Symbol[])
    estimate_xtreg(pd, :y, [:x1, :x2])
end

function _did_fixture()
    rng = MersenneTwister(61)
    n_units = 30; n_periods = 16; te = 2.0
    units_per_group = n_units ÷ 3
    treat_times = zeros(Int, n_units)
    for c in 1:2, u in ((c - 1) * units_per_group + 1):(c * units_per_group)
        treat_times[u] = 5 + 3 * (c - 1)
    end
    N_obs = n_units * n_periods
    data = Matrix{Float64}(undef, N_obs, 3)
    gid = Vector{Int}(undef, N_obs); tid = Vector{Int}(undef, N_obs)
    row = 1
    for i in 1:n_units
        ai = randn(rng)
        for t in 1:n_periods
            eff = (treat_times[i] > 0 && t >= treat_times[i]) ?
                  te * (1.0 + 0.1 * (t - treat_times[i])) : 0.0
            data[row, 1] = ai + 0.1 * t + eff + 0.5 * randn(rng)
            data[row, 2] = Float64(treat_times[i])
            data[row, 3] = randn(rng)
            gid[row] = i; tid[row] = t; row += 1
        end
    end
    pd = PanelData{Float64}(data, ["outcome", "treat_time", "covariate"], Quarterly,
                            [1, 1, 1], gid, tid, nothing, ["unit_$i" for i in 1:n_units],
                            n_units, 3, N_obs, true, ["DiD fixture"],
                            Dict{String,String}(), Symbol[])
    estimate_did(pd, "outcome", "treat_time"; method = :twfe, leads = 3, horizon = 5)
end

function _gmm_fixture()
    rng = MersenneTwister(71)
    n = 300
    X = randn(rng, n, 2)
    y = X * [1.0, -0.5] .+ randn(rng, n)
    d = hcat(y, X)
    mom(θ, dat) = dat[:, 2:3] .* (dat[:, 1] .- dat[:, 2:3] * θ)
    estimate_gmm(mom, [0.0, 0.0], d; weighting = :two_step)
end

function _dsge_est_fixture()
    rng = MersenneTwister(81)
    T_obs = 300
    y = zeros(T_obs)
    for t in 2:T_obs
        y[t] = 0.8 * y[t-1] + randn(rng)
    end
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    MacroEconometricModels._suppress_warnings() do
        estimate_dsge(spec, reshape(y, :, 1), [:ρ]; method = :irf_matching, irf_horizon = 10)
    end
end

"""
    build_display_fixtures() -> Vector{NamedTuple}

Ordered, deterministic fixtures spanning the display surface: the three bespoke
`report(io, ...)` bodies (VAR/VECM/DSGEEstimation), every `_coef_table` consumer
(reg/logit/probit/ordered/multinomial/panel/gmm/arima/volatility), a reference-row
DiD event study, and the table-bearing show types (ADF/Johansen/factor/LP/BVAR
forecast/normality).
"""
function build_display_fixtures()
    logit, probit = _binary_fixtures()
    fx = NamedTuple[]
    push!(fx, (name = "var",        obj = estimate_var(make_var1_data(T = 80, n = 2), 2),                     stars = false, ref = false))
    push!(fx, (name = "vecm",       obj = estimate_vecm(make_cointegrated_data(T_obs = 90, n = 2, rank = 1), 2), stars = false, ref = false))
    push!(fx, (name = "reg_ols",    obj = _reg_fixture(),                                                     stars = true,  ref = false))
    push!(fx, (name = "logit",      obj = logit,                                                              stars = true,  ref = false))
    push!(fx, (name = "probit",     obj = probit,                                                             stars = true,  ref = false))
    push!(fx, (name = "ologit",     obj = _ologit_fixture(),                                                  stars = true,  ref = false))
    push!(fx, (name = "mlogit",     obj = _mlogit_fixture(),                                                  stars = false, ref = false))
    push!(fx, (name = "arma",       obj = estimate_arma(make_ar1_data(n = 200, seed = 91), 1, 1),             stars = false, ref = false))
    push!(fx, (name = "garch",      obj = estimate_garch(simulate_garch11(n = 500, seed = 92)),               stars = false, ref = false))
    push!(fx, (name = "gmm",        obj = _gmm_fixture(),                                                     stars = true,  ref = false))
    push!(fx, (name = "panel_fe",   obj = _panel_fixture(),                                                   stars = true,  ref = false))
    push!(fx, (name = "did_es",     obj = _did_fixture(),                                                     stars = false, ref = true))
    push!(fx, (name = "adf",        obj = adf_test(make_random_walk(n = 150, seed = 93)),                     stars = false, ref = false))
    push!(fx, (name = "johansen",   obj = johansen_test(make_cointegrated_data(T_obs = 120, n = 3, rank = 1), 2), stars = false, ref = false))
    push!(fx, (name = "factor",     obj = estimate_factors(make_factor_data(T = 100, N = 12, r = 3).X, 3),    stars = false, ref = false))
    push!(fx, (name = "lp",         obj = estimate_lp(make_var1_data(T = 120, n = 3), 1, 8),                  stars = false, ref = false))
    push!(fx, (name = "bvar_fcst",  obj = forecast(estimate_bvar(make_var1_data(T = 90, n = 2), 2; n_draws = 200), 4), stars = false, ref = false))
    # Decisively non-normal data (randexp: skewness ≈ 2, excess kurtosis ≈ 6) so EVERY
    # test rejects with a statistic in the thousands (p ≪ 0.001). Gaussian data sat on the
    # decision boundary, and `randn`'s stream is not stable across Julia versions, so the
    # H₀-decision word flipped 1.10↔1.12 (a numerically-derived categorical the golden
    # canonicalizer cannot mask). A decisive fixture keeps every decision stable everywhere.
    push!(fx, (name = "normality",  obj = normality_test_suite(randexp(MersenneTwister(94), 200, 3)),        stars = false, ref = false))
    push!(fx, (name = "dsge_est",   obj = _dsge_est_fixture(),                                                stars = false, ref = false))
    return fx
end

# Both test files run in the same process (one parallel-runner group); build the
# fixtures once (some — DSGE, BVAR — carry real estimation cost) and share.
const _FIXTURE_CACHE = Ref{Any}(nothing)
display_fixtures() = (_FIXTURE_CACHE[] === nothing && (_FIXTURE_CACHE[] = build_display_fixtures()); _FIXTURE_CACHE[])
