# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-04 (#412): variable selection — stepwise / best-subset / GETS.

using Test
using MacroEconometricModels
using MacroEconometricModels: _sel_ols, _sel_ic
using LinearAlgebra, Statistics, Random, DelimitedFiles
import StatsAPI

# =============================================================================
# Deterministic DGPs (fixed-seed MersenneTwister — reproducible across groups)
# =============================================================================

# Strong-signal, near-orthogonal design for stepwise ≡ best-subset agreement.
# QR-orthogonalising the regressors makes greedy forward/backward search reach
# the global best-subset optimum (a classical guarantee for orthogonal designs).
function _sel_dgp_orth(; n=120, k=8, seed=11, active=[2, 4, 6], b=3.0)
    rng = MersenneTwister(seed)
    Q = Matrix(qr(randn(rng, n, k)).Q)[:, 1:k] .* sqrt(n)
    X = hcat(ones(n), Q)
    beta = zeros(k + 1); beta[1] = 1.0
    for a in active; beta[a] = b; end
    y = X * beta + 0.5 * randn(rng, n)
    (y, X, active)
end

# GETS oracle DGP: 5 true regressors among 20 candidates + intercept, iid N(0,1)
# errors (so the GETS misspecification gate — Breusch–Godfrey + Jarque–Bera —
# passes throughout and the reduction is driven purely by significance).
function _sel_dgp_sparse(; n=200, k=20, seed=42, active=[2, 5, 8, 11, 14], b=1.5)
    # The default (seed 42) configuration is the GETS oracle; its exact data is
    # pinned to a committed fixture because MersenneTwister is not stable across
    # Julia versions (see test/gen_ev_fixtures.jl). Other seeds (the Monte-Carlo
    # retention-rate loop) stay RNG-driven — those assertions are distributional.
    if seed == 42 && n == 200 && k == 20 && active == [2, 5, 8, 11, 14] && b == 1.5
        d = readdlm(joinpath(@__DIR__, "data", "reg_sel_sparse42.csv"), ',', Float64)
        X = hcat(ones(size(d, 1)), d[:, 2:end])
        return (d[:, 1], X, active)
    end
    rng = MersenneTwister(seed)
    X = hcat(ones(n), randn(rng, n, k))
    beta = zeros(k + 1); beta[1] = 0.5
    for a in active; beta[a] = b; end
    y = X * beta + randn(rng, n)
    (y, X, active)
end

# Independent brute-force best subset (does NOT use `select_variables`'s internal
# `_best_subset`): exhaustive OLS over all subsets of the non-intercept columns,
# always keeping column 1, minimising the information criterion via `estimate_reg`.
function _bruteforce_best(y, X, keep::Vector{Int}, pool::Vector{Int}, which::Symbol)
    best_cols = sort(copy(keep)); best_ic = Inf
    m = length(pool)
    for mask in 0:(2^m - 1)
        subset = Int[pool[b] for b in 1:m if (mask >> (b - 1)) & 1 == 1]
        cols = sort(vcat(keep, subset))
        length(cols) < length(y) || continue
        mdl = estimate_reg(y, X[:, cols]; cov_type=:ols)
        ic = which === :bic ? mdl.bic : mdl.aic
        if ic < best_ic - 1e-9
            best_ic = ic; best_cols = cols
        end
    end
    best_cols
end

@testset "Variable Selection (EV-04, #412)" begin

    # =========================================================================
    # Internal kernels: IC matches estimate_reg exactly (best-subset ties agree)
    # =========================================================================
    @testset "IC kernel ≡ estimate_reg" begin
        y, X, _ = _sel_dgp_orth()
        cols = [1, 2, 4, 6]
        m = estimate_reg(y, X[:, cols]; cov_type=:ols)
        f = _sel_ols(y, X, cols)
        @test isapprox(f.ssr, m.ssr; rtol=1e-10)
        @test isapprox(_sel_ic(f.ssr, f.n, f.k, :aic), m.aic; rtol=1e-10)
        @test isapprox(_sel_ic(f.ssr, f.n, f.k, :bic), m.bic; rtol=1e-10)
    end

    # =========================================================================
    # ORACLE 1 — stepwise :forward/:backward agree with exhaustive best-subset
    # (analytic property on a near-orthogonal, strong-signal design). Verified
    # against an INDEPENDENT brute-force search, not the internal :best_subset.
    # =========================================================================
    @testset "stepwise ≡ best-subset (k=8)" begin
        y, X, _ = _sel_dgp_orth()
        keep = [1]; pool = collect(2:9)
        for crit in (:aic, :bic)
            brute = _bruteforce_best(y, X, keep, pool, crit)
            bs = select_variables(y, X; method=:best_subset, criterion=crit)
            fw = select_variables(y, X; method=:forward,     criterion=crit)
            bw = select_variables(y, X; method=:backward,    criterion=crit)
            @test bs.selected == brute
            @test fw.selected == brute
            @test bw.selected == brute
            # Final refit IC equals the brute-force optimum's IC.
            opt = estimate_reg(y, X[:, brute]; cov_type=:ols)
            @test isapprox((crit === :bic ? fw.final.bic : fw.final.aic),
                           (crit === :bic ? opt.bic : opt.aic); rtol=1e-10)
        end
    end

    # =========================================================================
    # ORACLE 2 — GETS retention on the sparse DGP.
    #
    # R `gets` is NOT installed in this environment (checked:
    #   Rscript -e 'requireNamespace("gets")'  →  FALSE), so the reference is
    # hard-coded per the oracle convention. The equivalent R call is:
    #
    #   library(gets)
    #   X  <- <the columns 2:21 of the design below>      # 20 candidates
    #   fit <- arx(y, mxreg = X, mc = TRUE)               # GUM w/ intercept
    #   red <- getsm(fit, t.pval = 0.05,                  # deletion p-value
    #                ar.LjungB = NULL, arch.LjungB = NULL, # (we gate on BG + JB)
    #                normality.JarqueB = 0.05,
    #                info.method = "sic")                 # BIC tie-break
    #   # getsm retains the full true active set {const, x2, x5, x8, x11, x14};
    #   # spurious retentions occur at ≈ t.pval per irrelevant regressor.
    #
    # Documented GETS property (Hoover & Perez 1999; Hendry & Krolzig 2005):
    # the reduction retains ALL truly-relevant regressors with probability →1 as
    # signal grows, and drops the vast majority of irrelevant ones. We assert
    # the retention (superset) property and the low false-retention count.
    # =========================================================================
    @testset "GETS retains the true active set" begin
        y, X, active = _sel_dgp_sparse()
        truth = sort(vcat(1, active))          # {const} ∪ true regressors
        r = select_variables(y, X; method=:gets)

        @test r.method === :gets
        @test issubset(Set(truth), Set(r.selected))          # retains ALL true regressors
        @test 1 in r.selected                                # intercept always kept
        # Irrelevant regressors are the complement; GETS keeps only a handful.
        irrelevant = setdiff(2:21, active)
        false_kept = intersect(r.selected, irrelevant)
        @test length(false_kept) <= 2                        # ≈ t.pval × 15 ≈ 0.75 expected
        @test !isempty(r.terminal_models)                    # multi-path terminals recorded
        # Final model refit and usable downstream.
        @test r.final isa MacroEconometricModels.RegModel
        @test length(StatsAPI.coef(r.final)) == length(r.selected)
        @test length(StatsAPI.predict(r.final)) == length(y)
    end

    # =========================================================================
    # PROPERTY — seeded Monte Carlo retention rate of the true active set.
    # =========================================================================
    @testset "GETS retention rate (seeded MC)" begin
        reps = 25
        hits = 0
        for s in 1:reps
            y, X, active = _sel_dgp_sparse(seed=100 + s)
            r = select_variables(y, X; method=:gets)
            issubset(Set(vcat(1, active)), Set(r.selected)) && (hits += 1)
        end
        rate = hits / reps
        @test rate >= 0.9        # retains all true regressors with high probability
    end

    # =========================================================================
    # Parsimonious-encompassing F-test: selected ⊆ GUM ⇒ nested F stored, and on
    # a correctly-reduced model it does NOT reject at 5%.
    # =========================================================================
    @testset "encompassing F-test" begin
        y, X, active = _sel_dgp_sparse()
        r = select_variables(y, X; method=:gets)
        @test r.encompassing_f !== nothing
        @test r.encompassing_df[1] == r.n_gum - length(r.selected)
        @test r.encompassing_df[2] == length(y) - r.n_gum
        @test 0.0 <= r.encompassing_pval <= 1.0
        @test r.encompassing_pval > 0.05        # selection encompasses the GUM
        # Reconstruct the F-statistic from the two nested SSRs (machine tol).
        sf = _sel_ols(y, X, r.selected)
        gf = _sel_ols(y, X, sort(vcat(r.keep, setdiff(1:21, r.keep))))
        q = r.encompassing_df[1]; df2 = r.encompassing_df[2]
        fman = ((sf.ssr - gf.ssr) / q) / (gf.ssr / df2)
        @test isapprox(r.encompassing_f, fman; rtol=1e-8)
    end

    # =========================================================================
    # Stepwise :pvalue mechanics — entry/removal and the search path.
    # =========================================================================
    @testset "stepwise :pvalue path" begin
        y, X, active = _sel_dgp_sparse()
        truth = sort(vcat(1, active))
        fw = select_variables(y, X; method=:forward, criterion=:pvalue, p_enter=0.05)
        @test issubset(Set(truth), Set(fw.selected))
        @test all(s -> s[1] === :enter, fw.path)          # forward only enters
        @test !isempty(fw.path)
        # p-values of entered variables are below the entry threshold.
        @test all(s -> s[3] < 0.05 + 1e-12, fw.path)

        bw = select_variables(y, X; method=:backward, criterion=:pvalue, p_remove=0.10)
        @test issubset(Set(truth), Set(bw.selected))
        @test all(s -> s[1] === :remove, bw.path)         # backward only removes

        bi = select_variables(y, X; method=:bidirectional, criterion=:pvalue)
        @test issubset(Set(truth), Set(bi.selected))
        @test 1 in bi.selected                            # intercept retained
    end

    # =========================================================================
    # Forced keep + intercept handling; varnames propagate to the final model.
    # =========================================================================
    @testset "keep / intercept / varnames" begin
        y, X, _ = _sel_dgp_sparse()
        names = vcat("const", ["z$i" for i in 1:20])
        # Force an irrelevant regressor (col 3) to stay in.
        r = select_variables(y, X; method=:gets, keep=[3], varnames=names)
        @test 1 in r.selected && 3 in r.selected          # intercept + forced keep
        @test 3 in r.keep && 1 in r.keep
        @test r.final.varnames == names[r.selected]

        # bidirectional :pvalue requires p_remove ≥ p_enter.
        @test_throws ArgumentError select_variables(y, X; method=:bidirectional,
                                                     criterion=:pvalue, p_enter=0.10, p_remove=0.05)
        @test_throws ArgumentError select_variables(y, X; method=:nonsense)
        @test_throws ArgumentError select_variables(y, X; method=:forward, criterion=:bogus)
    end

    # =========================================================================
    # report() / refs() render without error.
    # =========================================================================
    @testset "report / refs render" begin
        y, X, _ = _sel_dgp_orth()
        r = select_variables(y, X; method=:forward, criterion=:bic)
        io = IOBuffer()
        report(io, r)
        s = String(take!(io))
        @test occursin("Variable Selection", s)
        @test occursin("Search Path", s)
        @test occursin("Selected model", s)

        g = select_variables(y, X; method=:gets)
        io2 = IOBuffer()
        show(io2, g)
        @test occursin("GETS", String(take!(io2)))

        io3 = IOBuffer()
        refs(io3, r)
        rs = String(take!(io3))
        @test occursin("Hoover", rs) || occursin("Hendry", rs) || occursin("Pretis", rs)
    end

    # =========================================================================
    # Float-fallback: Int/Any inputs convert internally.
    # =========================================================================
    @testset "abstract input fallback" begin
        y, X, _ = _sel_dgp_orth()
        r = select_variables(collect(y), Matrix(X); method=:backward, criterion=:aic)
        @test r isa MacroEconometricModels.SelectionResult
    end
end
