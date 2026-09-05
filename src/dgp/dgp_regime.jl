# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-11 (#800): regime-switching simulators (MS, SETAR,
# LSTAR, ESTR) returning the true state/transition path for classification
# and correlation asserts.


"""
    dgp_regime_switching(rng; kind, T, burn, kwargs...) -> NamedTuple

- `:ms`: mean-switching MS-AR(1) `(y−μ_s) = φ(y_{−1}−μ_{s_{−1}}) + σε`
  (promoted from `test_markov_switching.jl::_sim_ms_ar1`); returns `(y, s)`.
- `:setar`: `y_t = φ_lo y_{t-1} + e` if `y_{t-d} ≤ c` else `φ_hi y_{t-1} + e`;
  returns `(y, regime)`.
- `:lstar`: logistic `G(y_{t-d}; γ, c)` blending `(φ_lo, c_lo)` / `(φ_hi, c_hi)`;
  returns `(y, G)`.
- `:estr`: symmetric `G = 1 − exp(−γ(y_{t-d} − c)²)`; returns `(y, G)`.
"""
function dgp_regime_switching(rng::AbstractRNG; kind::Symbol=:ms, T::Int=600,
                              burn::Int=100, mu=(-1.0, 3.0), phi::Float64=0.4,
                              sigma::Float64=0.6, P=[0.9 0.1; 0.15 0.85],
                              phi_lo::Float64=0.8, phi_hi::Float64=0.3,
                              c::Float64=0.0, d::Int=1, gamma::Float64=3.0,
                              c_lo::Float64=0.0, c_hi::Float64=0.0)
    N = T + burn
    if kind === :ms
        Pm = Matrix{Float64}(P)
        s = Vector{Int}(undef, N)
        s[1] = 1
        for t in 2:N
            s[t] = rand(rng) < Pm[s[t - 1], 1] ? 1 : 2
        end
        y = zeros(N)
        z = 0.0
        for t in 2:N
            z = phi * z + sigma * randn(rng)
            y[t] = mu[s[t]] + z
        end
        keep = (burn + 1):N
        return (y=y[keep], s=s[keep], mu=mu, phi=phi, P=Pm)
    elseif kind === :setar
        y = zeros(N)
        reg = zeros(Int, N)
        for t in 2:N
            lo = y[max(t - d, 1)] <= c
            reg[t] = lo ? 1 : 2
            y[t] = (lo ? phi_lo : phi_hi) * y[t - 1] + sigma * randn(rng)
        end
        keep = (burn + 1):N
        return (y=y[keep], regime=reg[keep], phi_lo=phi_lo, phi_hi=phi_hi, c=c)
    elseif kind === :lstar || kind === :estr
        y = zeros(N)
        G = zeros(N)
        for t in 2:N
            yd = y[max(t - d, 1)]
            G[t] = kind === :lstar ? 1 / (1 + exp(-gamma * (yd - c))) :
                                     1 - exp(-gamma * (yd - c)^2)
            y[t] = c_lo + phi_lo * y[t - 1] +
                   G[t] * ((c_hi - c_lo) + (phi_hi - phi_lo) * y[t - 1]) +
                   sigma * randn(rng)
        end
        keep = (burn + 1):N
        return (y=y[keep], G=G[keep], phi_lo=phi_lo, phi_hi=phi_hi,
                gamma=gamma, c=c)
    else
        throw(ArgumentError("unknown kind :$kind (ms|setar|lstar|estr)"))
    end
end
