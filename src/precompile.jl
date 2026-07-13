# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# PrecompileTools workload (#253): pay the JIT/inference cost for the most common entry
# points at PRECOMPILE time so users' first-call latency (TTFX) drops. Deliberately limited
# to small, stable, robust entry points on tiny SEEDED synthetic inputs — a workload that
# errors at build time would break precompilation/load for everyone. Per the #255 evaluation
# this targets the monolith's top entry points, not speculative package boundaries. Fuller
# coverage (did/lp_did/gmm/DSGE/HA solve) is a follow-up: those need heavier setup and their
# first-call latency is a smaller share of typical use.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Seeded so the precompile inputs are deterministic (and rng-lint clean, #243).
    rng = Random.MersenneTwister(0)
    Y = randn(rng, 60, 3)
    X = Y[:, 2:3]
    yv = collect(@view Y[:, 1])
    @compile_workload begin
        m = estimate_var(Y, 2)          # VAR(p) via OLS — the most common entry point
        irf(m, 8)                        # innovation accounting: IRF
        fevd(m, 8)                       # innovation accounting: FEVD
        estimate_reg(yv, X)              # cross-sectional OLS
        estimate_bvar(Y, 2; n_draws=10, rng=rng)   # Bayesian VAR (tiny draw count)
        estimate_lp(Y, 1, 4)             # local projection
    end
end
