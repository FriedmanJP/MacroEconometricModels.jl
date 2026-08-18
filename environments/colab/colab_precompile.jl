# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Colab-only execution file (#610). Tiny seeded workloads for the high-TTFX
# paths Colab demos actually hit. Keep this aligned with src/precompile.jl;
# a failure here should not be able to break a normal Pkg.add install.

using Random
using DataFrames
using MacroEconometricModels

rng = MersenneTwister(0)
Y = randn(rng, 60, 3)
yv = collect(Y[:, 1])
X = Y[:, 2:3]

m = estimate_var(Y, 2)
irf(m, 8)
fevd(m, 8)
forecast(m, 4; ci_method=:none, rng=rng)
report(devnull, m)
estimate_reg(yv, X)
estimate_bvar(Y, 2; n_draws=10, rng=rng)
estimate_lp(Y, 1, 4)
