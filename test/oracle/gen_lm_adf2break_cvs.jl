# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Critical-value generator for the break-search unit-root tests (issue #577)
# =============================================================================
#
# `lm_unitroot_test` (Schmidt-Phillips / Lee-Strazicich min-LM) and
# `adf_2break_test` (two-break ADF) were shipped with published critical values
# that do not correspond to the statistics this package actually computes: the
# min-LM detrending was rewritten to the differenced-regression form, the
# two-break ADF is an additive-outlier (Lumsdaine-Popp/Perron) regression rather
# than Narayan-Popp's unobserved-components statistic, and neither table varied
# with T. Sizes at a 5% nominal level ran to 55-85%.
#
# This script simulates the NULL distribution of the package's OWN statistics --
# same search grid, same trimming, same regression -- and prints the Julia table
# literals that are pasted into `src/teststat/critical_values.jl`.
#
# Generation parameters (also recorded above each table in critical_values.jl):
#   * DGP        driftless random walk, y_t = y_{t-1} + N(0,1), y_0 = 0
#   * T grid     100, 150, 250, 500
#   * reps       10000 per cell
#   * RNG        MersenneTwister(BASE_SEED + rep), BASE_SEED = 5770000
#                (per-rep seeding => reproducible independent of thread count)
#   * lags       0 (no augmentation). The tables are conditional on the
#                deterministic search design, which is standard for these tests.
#   * trim       0.15 for the min-LM search, 0.10 for the two-break ADF --
#                the package defaults.
#   * quantiles  1%, 2.5%, 5%, 10% of the minimised statistic
#
# Run (`julia` is not on PATH in this environment):
#   JULIA="$HOME/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia"
#   J="$JULIA --project=. --threads=auto test/oracle/gen_lm_adf2break_cvs.jl"
#   $J validate                        # kernels must match the public API
#   $J cells out.csv                   # every cell, then print the tables
#   $J cell lm2both 500 10000 out.csv  # one cell (they are minutes long at T=500)
#   $J assemble out.csv                # print the tables from collected cells
#
# `cell` appends one line per (cell, T) to the results file, so a run that is
# interrupted resumes by re-running only the missing cells.
#
# The `validate` entry point is the important one: the fast kernels below must
# reproduce `lm_unitroot_test` / `adf_2break_test` to floating-point accuracy,
# otherwise the tables would be calibrated for a different statistic.

using LinearAlgebra, Statistics, Random, Printf

const BASE_SEED = 5_770_000
const T_GRID = (100, 150, 250, 500)
const REPS = 10_000
const QUANTILES = (0.01, 0.025, 0.05, 0.10)

# =============================================================================
# Fast min-LM kernel (lags = 0)
# =============================================================================
#
# With no augmentation the Lee-Strazicich statistic collapses to closed form.
# Stage 1 regresses Δy on the non-zero columns of ΔZ. Those columns are
#   Model A (`:level`): [1, imp(tb1), (imp(tb2))]
#   Model C (`:both`):  [1, imp(tb1), step(tb1), (imp(tb2), step(tb2))]
# whose span is the set of indicators of the blocks between the breaks, with the
# break rows themselves as singletons (a break row is fitted exactly). So the
# stage-1 fit is a block mean and the residual e is block-demeaned Δy with zeros
# at the break rows.
#
# Stage 2 regresses ΔS̃ = e on [S̃_{t-1}, ΔZ]. Since e ⊥ ΔZ by construction, FWL
# gives  φ = q'e / q'q  with q = block-demeaned S̃_{t-1}, and
# SSR = e'e − φ² q'q, so t_φ = φ·sqrt(q'q)·sqrt((N−k)/SSR).

"""
    lm_min_stat(y, breaks, regression; trim=0.15) -> Float64

Minimised LM t-statistic, identical to
`lm_unitroot_test(y; breaks, regression, lags=0, trim).statistic`.
"""
function lm_min_stat(y::Vector{Float64}, breaks::Int, regression::Symbol; trim::Float64=0.15)
    n = length(y)
    N = n - 1
    dy = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        dy[i] = y[i+1] - y[i]
    end
    breaks == 0 && return _lm_stat_nobreak(y, dy, regression)

    S = Vector{Float64}(undef, n)
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))
    best = Inf
    if breaks == 1
        for tb in start_idx:end_idx
            s = _lm_stat_break(y, dy, S, tb, 0, regression)
            s < best && (best = s)
        end
    else
        min_gap = max(2, ceil(Int, trim * n))
        for tb1 in start_idx:end_idx, tb2 in (tb1+min_gap):end_idx
            s = _lm_stat_break(y, dy, S, tb1, tb2, regression)
            s < best && (best = s)
        end
    end
    return best
end

# breaks = 0.  `:level` keeps the Schmidt-Phillips intercept-only design (Z = 1,
# ΔZ empty, S̃ = y − y₁); `:both` is the constant-plus-trend design (ΔZ = 1).
function _lm_stat_nobreak(y::Vector{Float64}, dy::Vector{Float64}, regression::Symbol)
    n = length(y)
    N = n - 1
    if regression == :level
        qe = 0.0; qq = 0.0; ee = 0.0
        @inbounds for i in 1:N
            q = y[i] - y[1]
            qe += q * dy[i]; qq += q * q; ee += dy[i]^2
        end
        return _lm_tstat(qe, qq, ee, N, 1)
    else
        m = sum(dy) / N
        sbar = 0.0
        @inbounds for i in 1:N
            sbar += (y[i] - y[1]) - m * (i - 1)
        end
        sbar /= N
        qe = 0.0; qq = 0.0; ee = 0.0
        @inbounds for i in 1:N
            q = (y[i] - y[1]) - m * (i - 1) - sbar
            e = dy[i] - m
            qe += q * e; qq += q * q; ee += e * e
        end
        return _lm_tstat(qe, qq, ee, N, 2)
    end
end

# One or two breaks (`tb2 == 0` means a single break).
function _lm_stat_break(y::Vector{Float64}, dy::Vector{Float64}, S::Vector{Float64},
                        tb1::Int, tb2::Int, regression::Symbol)
    n = length(y)
    N = n - 1
    two = tb2 > 0
    if regression == :level
        # Stage 1: single mean over all rows except the break rows.
        s = 0.0
        cnt = N - (two ? 2 : 1)
        @inbounds for i in 1:N
            (i == tb1 || (two && i == tb2)) && continue
            s += dy[i]
        end
        m = s / cnt
        a1 = dy[tb1] - m
        a2 = two ? dy[tb2] - m : 0.0
        @inbounds for t in 1:n
            S[t] = (y[t] - y[1]) - m * (t - 1) - (t > tb1 ? a1 : 0.0) -
                   (two && t > tb2 ? a2 : 0.0)
        end
        sbar = 0.0
        @inbounds for i in 1:N
            (i == tb1 || (two && i == tb2)) && continue
            sbar += S[i]
        end
        sbar /= cnt
        qe = 0.0; qq = 0.0; ee = 0.0
        @inbounds for i in 1:N
            (i == tb1 || (two && i == tb2)) && continue
            q = S[i] - sbar; e = dy[i] - m
            qe += q * e; qq += q * q; ee += e * e
        end
        k = 1 + 1 + (two ? 2 : 1)          # S̃_{t-1} + constant + impulses
        return _lm_tstat(qe, qq, ee, N, k)
    else
        # Blocks: B0 = 1:tb1-1, {tb1}, B1 = tb1+1:(tb2-1 | N), {tb2}, B2 = tb2+1:N
        hi1 = two ? tb2 - 1 : N
        (tb1 - 1 < 1 || hi1 < tb1 + 1) && return Inf
        two && (tb2 + 1 > N) && return Inf
        mu0 = _blockmean(dy, 1, tb1 - 1)
        mu1 = _blockmean(dy, tb1 + 1, hi1)
        mu2 = two ? _blockmean(dy, tb2 + 1, N) : 0.0
        b1 = mu1 - mu0
        a1 = dy[tb1] - mu1
        b2 = two ? mu2 - mu1 : 0.0
        a2 = two ? dy[tb2] - mu2 : 0.0
        @inbounds for t in 1:n
            v = (y[t] - y[1]) - mu0 * (t - 1)
            t > tb1 && (v -= a1)
            t > tb1 && (v -= b1 * (t - tb1))
            if two
                t > tb2 && (v -= a2)
                t > tb2 && (v -= b2 * (t - tb2))
            end
            S[t] = v
        end
        s0 = _blockmean(S, 1, tb1 - 1)
        s1 = _blockmean(S, tb1 + 1, hi1)
        s2 = two ? _blockmean(S, tb2 + 1, N) : 0.0
        qe = 0.0; qq = 0.0; ee = 0.0
        @inbounds for i in 1:N
            (i == tb1 || (two && i == tb2)) && continue
            if i < tb1
                q = S[i] - s0; e = dy[i] - mu0
            elseif !two || i < tb2
                q = S[i] - s1; e = dy[i] - mu1
            else
                q = S[i] - s2; e = dy[i] - mu2
            end
            qe += q * e; qq += q * q; ee += e * e
        end
        k = 1 + 1 + (two ? 4 : 2)          # S̃_{t-1} + constant + (imp, step) per break
        return _lm_tstat(qe, qq, ee, N, k)
    end
end

@inline function _blockmean(v::Vector{Float64}, lo::Int, hi::Int)
    hi < lo && return NaN
    s = 0.0
    @inbounds for i in lo:hi
        s += v[i]
    end
    return s / (hi - lo + 1)
end

@inline function _lm_tstat(qe::Float64, qq::Float64, ee::Float64, N::Int, k::Int)
    (qq <= 0.0 || !isfinite(qq)) && return Inf
    phi = qe / qq
    ssr = ee - phi * phi * qq
    dof = N - k
    (dof < 1 || ssr <= 0.0) && return Inf
    return phi / sqrt((ssr / dof) / qq)
end

# =============================================================================
# Fast two-break ADF kernel (lags = 0)
# =============================================================================
#
# Δy_i = c + β i + θ₁·1{i≥tb1} + θ₂·1{i≥tb2} [+ ψ₁·(i−tb1+1)⁺ + ψ₂·(i−tb2+1)⁺]
#        + γ y_i + u_i,   i = 1 … N = n−1,   t-statistic on γ.
#
# Every cross-product of {1, i, step(a), ramp(a), y, Δy} is a suffix sum, so the
# (k+1)×(k+1) Gram matrix of a candidate pair is assembled in O(k²) and the
# t-statistic follows from one small Cholesky.

struct ADF2Sums
    N::Int
    cnt::Vector{Float64}   # Σ_{i≥a} 1
    s1::Vector{Float64}    # Σ_{i≥a} i
    s2::Vector{Float64}    # Σ_{i≥a} i²
    y0::Vector{Float64}    # Σ_{i≥a} y_i
    y1::Vector{Float64}    # Σ_{i≥a} i y_i
    y2::Vector{Float64}    # Σ_{i≥a} y_i²
    d0::Vector{Float64}    # Σ_{i≥a} Δy_i
    d1::Vector{Float64}    # Σ_{i≥a} i Δy_i
    d2::Vector{Float64}    # Σ_{i≥a} Δy_i²
    yd::Vector{Float64}    # Σ_{i≥a} y_i Δy_i
end

function ADF2Sums(y::Vector{Float64})
    n = length(y)
    N = n - 1
    z() = zeros(Float64, N + 2)
    cnt, s1, s2 = z(), z(), z()
    y0, y1, y2 = z(), z(), z()
    d0, d1, d2, yd = z(), z(), z(), z()
    @inbounds for a in N:-1:1
        i = Float64(a)
        yi = y[a]
        di = y[a+1] - y[a]
        cnt[a] = cnt[a+1] + 1.0
        s1[a] = s1[a+1] + i
        s2[a] = s2[a+1] + i * i
        y0[a] = y0[a+1] + yi
        y1[a] = y1[a+1] + i * yi
        y2[a] = y2[a+1] + yi * yi
        d0[a] = d0[a+1] + di
        d1[a] = d1[a+1] + i * di
        d2[a] = d2[a+1] + di * di
        yd[a] = yd[a+1] + yi * di
    end
    ADF2Sums(N, cnt, s1, s2, y0, y1, y2, d0, d1, d2, yd)
end

"""
    adf2break_min_stat(y, model; trim=0.10) -> Float64

Minimised two-break ADF t-statistic, identical to
`adf_2break_test(y; model, lags=0, trim).statistic`.
"""
function adf2break_min_stat(y::Vector{Float64}, model::Symbol; trim::Float64=0.10)
    n = length(y)
    S = ADF2Sums(y)
    k = model == :level ? 5 : 7
    G = Matrix{Float64}(undef, k, k)
    g = Vector{Float64}(undef, k)
    b = Vector{Float64}(undef, k)
    ek = Vector{Float64}(undef, k)
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))
    min_gap = model == :level ? 2 : 3
    best = Inf
    for tb1 in start_idx:end_idx, tb2 in (tb1+min_gap):end_idx
        s = _adf2_stat(S, tb1, tb2, model, G, g, b, ek, k)
        s < best && (best = s)
    end
    return best
end

function _adf2_stat(S::ADF2Sums, tb1::Int, tb2::Int, model::Symbol,
                    G::Matrix{Float64}, g::Vector{Float64}, b::Vector{Float64},
                    ek::Vector{Float64}, k::Int)
    N = S.N
    stp_1(a) = S.cnt[a]                       # step(a)·1
    stp_i(a) = S.s1[a]                        # step(a)·i
    stp_y(a) = S.y0[a]                        # step(a)·y
    stp_d(a) = S.d0[a]                        # step(a)·Δy
    rmp_1(a) = S.s1[a] - (a - 1) * S.cnt[a]
    rmp_i(a) = S.s2[a] - (a - 1) * S.s1[a]
    rmp_y(a) = S.y1[a] - (a - 1) * S.y0[a]
    rmp_d(a) = S.d1[a] - (a - 1) * S.d0[a]
    stp_stp(a, b_) = S.cnt[max(a, b_)]
    rmp_stp(a, b_) = (m = max(a, b_); S.s1[m] - (a - 1) * S.cnt[m])
    rmp_rmp(a, b_) = (m = max(a, b_); S.s2[m] - (a + b_ - 2) * S.s1[m] + (a - 1) * (b_ - 1) * S.cnt[m])

    # Column order matches `_adf_2break_at`: [1, i, DU1, DU2, (DT1, DT2), y]
    if model == :level
        G[1,1] = Float64(N);   G[1,2] = S.s1[1];  G[1,3] = stp_1(tb1); G[1,4] = stp_1(tb2); G[1,5] = S.y0[1]
        G[2,2] = S.s2[1];      G[2,3] = stp_i(tb1); G[2,4] = stp_i(tb2); G[2,5] = S.y1[1]
        G[3,3] = stp_stp(tb1, tb1); G[3,4] = stp_stp(tb1, tb2); G[3,5] = stp_y(tb1)
        G[4,4] = stp_stp(tb2, tb2); G[4,5] = stp_y(tb2)
        G[5,5] = S.y2[1]
        g[1] = S.d0[1]; g[2] = S.d1[1]; g[3] = stp_d(tb1); g[4] = stp_d(tb2); g[5] = S.yd[1]
    else
        G[1,1] = Float64(N);   G[1,2] = S.s1[1];  G[1,3] = stp_1(tb1); G[1,4] = stp_1(tb2)
        G[1,5] = rmp_1(tb1);   G[1,6] = rmp_1(tb2); G[1,7] = S.y0[1]
        G[2,2] = S.s2[1];      G[2,3] = stp_i(tb1); G[2,4] = stp_i(tb2)
        G[2,5] = rmp_i(tb1);   G[2,6] = rmp_i(tb2); G[2,7] = S.y1[1]
        G[3,3] = stp_stp(tb1, tb1); G[3,4] = stp_stp(tb1, tb2)
        G[3,5] = rmp_stp(tb1, tb1); G[3,6] = rmp_stp(tb2, tb1); G[3,7] = stp_y(tb1)
        G[4,4] = stp_stp(tb2, tb2)
        G[4,5] = rmp_stp(tb1, tb2); G[4,6] = rmp_stp(tb2, tb2); G[4,7] = stp_y(tb2)
        G[5,5] = rmp_rmp(tb1, tb1); G[5,6] = rmp_rmp(tb1, tb2); G[5,7] = rmp_y(tb1)
        G[6,6] = rmp_rmp(tb2, tb2); G[6,7] = rmp_y(tb2)
        G[7,7] = S.y2[1]
        g[1] = S.d0[1]; g[2] = S.d1[1]; g[3] = stp_d(tb1); g[4] = stp_d(tb2)
        g[5] = rmp_d(tb1); g[6] = rmp_d(tb2); g[7] = S.yd[1]
    end
    @inbounds for i in 2:k, j in 1:i-1
        G[i,j] = G[j,i]
    end

    F = cholesky!(Symmetric(G), check=false)
    issuccess(F) || return Inf
    copyto!(b, g)
    ldiv!(F, b)
    ssr = S.d2[1] - dot(g, b)
    dof = N - k
    (dof < 1 || ssr <= 0.0) && return Inf
    fill!(ek, 0.0); ek[k] = 1.0
    ldiv!(F, ek)
    vkk = ek[k]
    vkk <= 0.0 && return Inf
    se = sqrt((ssr / dof) * vkk)
    se <= 0.0 && return Inf
    return b[k] / se
end

# =============================================================================
# Simulation drivers
# =============================================================================

randomwalk(rng, T) = cumsum(randn(rng, T))

"""Null draws of a minimised statistic; `f(y)` computes the statistic."""
function null_draws(f::Function, T::Int, reps::Int, seed::Int)
    out = Vector{Float64}(undef, reps)
    Threads.@threads for r in 1:reps
        rng = MersenneTwister(seed + r)
        out[r] = f(randomwalk(rng, T))
    end
    return out
end

function quantiles_of(draws::Vector{Float64})
    d = filter(isfinite, draws)
    return [quantile(d, q) for q in QUANTILES]
end

const LM_CASES = [(0, :level), (0, :both), (1, :level), (1, :both), (2, :level), (2, :both)]

# Every (test, model, breaks) cell has a name so it can be run in its own
# process and appended to a results file: the two-break cells at T = 500 take
# minutes, and a single long-running job that dies loses everything.
cell_name(breaks::Int, reg::Symbol) = "lm$(breaks)$(reg)"
cell_name(model::Symbol) = "adf$(model)"

const CELL_NAMES = vcat([cell_name(b, r) for (b, r) in LM_CASES],
                        [cell_name(m) for m in (:level, :both)])

cell_seed(name::AbstractString, T::Int) = BASE_SEED + 1000 * T + sum(Int, codeunits(name))

"""Statistic function for a named cell."""
function cell_spec(name::AbstractString)
    for (breaks, reg) in LM_CASES
        cell_name(breaks, reg) == name && return (y -> lm_min_stat(y, breaks, reg))
    end
    for model in (:level, :both)
        cell_name(model) == name && return (y -> adf2break_min_stat(y, model))
    end
    error("unknown cell $name")
end

"""Run one cell and append `name,T,reps,q1,q2.5,q5,q10` to `outfile`."""
function run_cell(name::AbstractString, T::Int, reps::Int, outfile::AbstractString)
    f = cell_spec(name)
    t0 = time()
    q = quantiles_of(null_draws(f, T, reps, cell_seed(name, T)))
    open(outfile, "a") do io
        @printf(io, "%s,%d,%d,%.4f,%.4f,%.4f,%.4f\n", name, T, reps, q[1], q[2], q[3], q[4])
    end
    @printf(stderr, "%-10s T=%3d  %6.1fs   %7.3f %7.3f %7.3f %7.3f\n",
            name, T, time() - t0, q[1], q[2], q[3], q[4])
    flush(stderr)
    return q
end

"""Read a results file written by `run_cell` into the two table dictionaries."""
function collect_cells(infile::AbstractString; T_grid=T_GRID)
    got = Dict{Tuple{String,Int},Vector{Float64}}()
    for line in eachline(infile)
        isempty(strip(line)) && continue
        parts = split(strip(line), ',')
        got[(String(parts[1]), parse(Int, parts[2]))] = parse.(Float64, parts[4:7])
    end
    lm = Dict{Tuple{Int,Symbol},Matrix{Float64}}()
    for (breaks, reg) in LM_CASES
        M = Matrix{Float64}(undef, length(T_grid), 4)
        for (row, T) in enumerate(T_grid)
            key = (cell_name(breaks, reg), T)
            haskey(got, key) || error("missing cell $key in $infile")
            M[row, :] = got[key]
        end
        lm[(breaks, reg)] = M
    end
    adf = Dict{Symbol,Matrix{Float64}}()
    for model in (:level, :both)
        M = Matrix{Float64}(undef, length(T_grid), 4)
        for (row, T) in enumerate(T_grid)
            key = (cell_name(model), T)
            haskey(got, key) || error("missing cell $key in $infile")
            M[row, :] = got[key]
        end
        adf[model] = M
    end
    return lm, adf
end

function print_tables(lm, adf; T_grid=T_GRID)
    println("# T grid: ", T_grid, "   columns: 1%, 2.5%, 5%, 10%")
    println("const BREAK_TEST_SIM_T = ", T_grid)
    println("const LM_UNITROOT_SIM_CV = Dict{Tuple{Int,Symbol},Matrix{Float64}}(")
    for (breaks, reg) in LM_CASES
        M = lm[(breaks, reg)]
        println("    ($breaks, :$reg) => [")
        for row in axes(M, 1)
            @printf("        %8.3f %8.3f %8.3f %8.3f;   # T = %d\n",
                    M[row,1], M[row,2], M[row,3], M[row,4], T_grid[row])
        end
        println("    ],")
    end
    println(")")
    println("const ADF_2BREAK_SIM_CV = Dict{Symbol,Matrix{Float64}}(")
    for model in (:level, :both)
        M = adf[model]
        println("    :$model => [")
        for row in axes(M, 1)
            @printf("        %8.3f %8.3f %8.3f %8.3f;   # T = %d\n",
                    M[row,1], M[row,2], M[row,3], M[row,4], T_grid[row])
        end
        println("    ],")
    end
    println(")")
end

# =============================================================================
# Kernel validation against the public API
# =============================================================================

function validate(; nseries::Int=6, Ts=(60, 100, 150))
    @eval Main using MacroEconometricModels
    # Resolve the public functions AFTER the `using` above lands in a newer world
    # age — a bare `Main.lm_unitroot_test` in this function body is looked up at
    # the function's own (older) world and throws UndefVarError.
    lm_f = Base.invokelatest(getglobal, Main, :lm_unitroot_test)
    adf_f = Base.invokelatest(getglobal, Main, :adf_2break_test)
    worst = 0.0
    for T in Ts, s in 1:nseries
        rng = MersenneTwister(4242 + 31 * T + s)
        y = randomwalk(rng, T)
        for (breaks, reg) in LM_CASES
            ref = Base.invokelatest(lm_f, y; breaks=breaks,
                                    regression=reg, lags=0).statistic
            got = lm_min_stat(y, breaks, reg)
            d = abs(ref - got) / max(1.0, abs(ref))
            d > worst && (worst = d)
            d > 1e-8 && @printf("MISMATCH lm T=%d s=%d breaks=%d %s: %.10f vs %.10f\n",
                                T, s, breaks, reg, ref, got)
        end
        for model in (:level, :both)
            ref = Base.invokelatest(adf_f, y; model=model, lags=0).statistic
            got = adf2break_min_stat(y, model)
            d = abs(ref - got) / max(1.0, abs(ref))
            d > worst && (worst = d)
            d > 1e-8 && @printf("MISMATCH adf2 T=%d s=%d %s: %.10f vs %.10f\n",
                                T, s, model, ref, got)
        end
    end
    @printf("kernel validation: worst relative deviation %.3e over %d series\n",
            worst, nseries * length(Ts))
    return worst
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "cells" : ARGS[1]
    if mode == "validate"
        w = validate()
        w < 1e-8 || error("kernel validation failed (worst = $w)")
    elseif mode == "cell"
        # cell <name> <T> <reps> <outfile>
        run_cell(ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5])
    elseif mode == "cells"
        # cells <outfile> [reps] -- every cell in one process
        outfile = length(ARGS) > 1 ? ARGS[2] : "lm_adf2break_cells.csv"
        reps = length(ARGS) > 2 ? parse(Int, ARGS[3]) : REPS
        for name in CELL_NAMES, T in T_GRID
            run_cell(name, T, reps, outfile)
        end
        print_tables(collect_cells(outfile)...)
    elseif mode == "assemble"
        # assemble <infile>
        print_tables(collect_cells(ARGS[2])...)
    else
        error("unknown mode $mode")
    end
end
