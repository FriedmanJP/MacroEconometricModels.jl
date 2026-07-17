# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
MCMC convergence diagnostics for Bayesian DSGE chains.

Provides:
- `mcmc_diagnostics` — per-parameter rank-normalized split-R̂, bulk/tail ESS
  (Vehtari et al. 2021), and Geweke (1992) z-statistics
- `trace` — per-parameter draw sequence accessor
- `acf(result, param)` — autocorrelation accessor reusing the spectral `acf`

References:
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
  Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing
  Convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
- Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the
  Calculation of Posterior Moments. In *Bayesian Statistics 4*.
- Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. *Statistical Science*,
  7(4), 473-483.
"""

# =============================================================================
# Rank normalization + chain splitting (Vehtari et al. 2021)
# =============================================================================

"""
    _tied_ranks(v::AbstractVector{T}) → Vector{T}

Average ranks (ties share the mean rank), 1-based.
"""
function _tied_ranks(v::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(v)
    idx = sortperm(v)
    ranks = zeros(T, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[idx[j+1]] == v[idx[i]]
            j += 1
        end
        avg = T(i + j) / 2
        for k in i:j
            ranks[idx[k]] = avg
        end
        i = j + 1
    end
    return ranks
end

"""
    _rank_normalize(x::AbstractMatrix{T}) → Matrix{T}

Rank-normalize pooled draws (N × M, draws × chains): rank across ALL draws,
then map to z-scores via `Φ⁻¹((r − 3/8)/(S + 1/4))` (Blom offsets).
"""
function _rank_normalize(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    S = length(x)
    r = _tied_ranks(vec(x))
    z = [T(quantile(Normal(), (r[i] - T(3) / 8) / (S + T(1) / 4))) for i in eachindex(r)]
    return reshape(z, size(x))
end

"""
    _split_chain(x::AbstractVector{T}) → Matrix{T}

Split a single chain into two half-chains (N÷2 × 2), dropping the middle draw
when the length is odd.
"""
function _split_chain(x::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x) ÷ 2
    return hcat(x[1:N], x[end-N+1:end])
end

# =============================================================================
# Split-R̂ and ESS on prepared (split, possibly rank-normalized) chains
# =============================================================================

"""
    _rhat_chains(z::AbstractMatrix{T}) → T

Potential scale reduction factor `√(var⁺/W)` for an N × M draws-by-chains
matrix. Returns `NaN` for degenerate input (N < 4 or zero within-variance).
"""
function _rhat_chains(z::AbstractMatrix{T}) where {T<:AbstractFloat}
    N, M = size(z)
    (N < 4 || M < 2) && return T(NaN)
    chain_means = vec(mean(z; dims=1))
    W = mean(vec(var(z; dims=1)))
    W > 0 || return T(NaN)
    B_over_N = var(chain_means)
    var_plus = (N - 1) / T(N) * W + B_over_N
    return sqrt(var_plus / W)
end

"""
    _ess_chains(z::AbstractMatrix{T}) → T

Effective sample size for an N × M draws-by-chains matrix via the multi-chain
autocorrelation estimate with Geyer's initial-monotone-sequence truncation
(Vehtari et al. 2021, §3.2; matches Stan's `compute_effective_sample_size`).
"""
function _ess_chains(z::AbstractMatrix{T}) where {T<:AbstractFloat}
    N, M = size(z)
    N < 4 && return T(NaN)

    # Per-chain biased autocovariances (denominator N), computed on demand
    mus = vec(mean(z; dims=1))
    function acov_lag(t::Int)
        s = zero(T)
        for m in 1:M
            zm = @view z[:, m]
            mu = mus[m]
            a = zero(T)
            @inbounds for i in 1:(N-t)
                a += (zm[i] - mu) * (zm[i+t] - mu)
            end
            s += a / N
        end
        return s / M
    end

    acov0 = acov_lag(0)
    acov0 > 0 || return T(NaN)
    W = acov0 * N / (N - 1)                    # mean within-chain variance
    chain_means = vec(mean(z; dims=1))
    B_over_N = M > 1 ? var(chain_means) : zero(T)
    var_plus = (N - 1) / T(N) * W + B_over_N
    var_plus > 0 || return T(NaN)

    rho = zeros(T, N)
    rho[1] = one(T)                            # ρ̂_0
    rho[2] = 1 - (W - acov_lag(1)) / var_plus  # ρ̂_1
    rho_even, rho_odd = rho[1], rho[2]
    t = 0
    while t < N - 4 && rho_even + rho_odd > 0
        t += 2
        rho_even = 1 - (W - acov_lag(t)) / var_plus
        rho_odd = 1 - (W - acov_lag(t + 1)) / var_plus
        if rho_even + rho_odd >= 0
            rho[t+1] = rho_even
            rho[t+2] = rho_odd
        end
    end
    max_t = t

    # Geyer initial monotone sequence: successive pair sums non-increasing
    for k in 2:2:(max_t-2)
        if rho[k+1] + rho[k+2] > rho[k-1] + rho[k]
            rho[k+1] = (rho[k-1] + rho[k]) / 2
            rho[k+2] = rho[k+1]
        end
    end

    S = T(N * M)
    tau_hat = -1 + 2 * sum(@view rho[1:max_t]) + rho[max_t+1]
    tau_hat = max(tau_hat, one(T) / log10(S))  # antithetic-chain guard
    return S / tau_hat
end

# =============================================================================
# Public per-vector diagnostics (single chain → split + rank-normalize)
# =============================================================================

"""
    _rhat_rank(x::AbstractVector{T}) → T

Rank-normalized split-R̂ of a single chain: the maximum of the split-R̂ on the
rank-normalized draws and on the rank-normalized folded draws `|x − median|`
(catches scale, not just location, non-convergence).
"""
function _rhat_rank(x::AbstractVector{T}) where {T<:AbstractFloat}
    length(x) < 8 && return T(NaN)
    z_bulk = _rank_normalize(_split_chain(x))
    folded = abs.(x .- median(x))
    z_fold = _rank_normalize(_split_chain(folded))
    return max(_rhat_chains(z_bulk), _rhat_chains(z_fold))
end

"""
    _ess_bulk(x::AbstractVector{T}) → T

Bulk-ESS: ESS of the rank-normalized split chain (Vehtari et al. 2021).
"""
function _ess_bulk(x::AbstractVector{T}) where {T<:AbstractFloat}
    length(x) < 8 && return T(NaN)
    return _ess_chains(_rank_normalize(_split_chain(x)))
end

"""
    _ess_tail(x::AbstractVector{T}) → T

Tail-ESS: minimum ESS of the 5% and 95% quantile-exceedance indicators
(Vehtari et al. 2021).
"""
function _ess_tail(x::AbstractVector{T}) where {T<:AbstractFloat}
    length(x) < 8 && return T(NaN)
    ess_q = map((T(0.05), T(0.95))) do q
        cutoff = quantile(x, q)
        _ess_chains(_split_chain(T.(x .<= cutoff)))
    end
    return min(ess_q...)
end

"""
    _geweke_nse(x::AbstractVector{T}) → T

Numerical standard error of the mean via a Bartlett-window (Newey-West)
spectral density estimate at frequency zero.
"""
function _geweke_nse(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    mu = mean(x)
    L = min(n - 1, max(1, floor(Int, 4 * (n / 100)^(T(2) / 9))))
    g0 = sum(abs2, x .- mu) / n
    s = g0
    for k in 1:L
        gk = zero(T)
        @inbounds for i in 1:(n-k)
            gk += (x[i+k] - mu) * (x[i] - mu)
        end
        gk /= n
        s += 2 * (1 - T(k) / (L + 1)) * gk
    end
    return sqrt(max(s, zero(T)) / n)
end

"""
    _geweke_z(x::AbstractVector{T}; first=0.1, last=0.5) → T

Geweke (1992) convergence z-statistic comparing the mean of the first `first`
fraction of the chain against the last `last` fraction, with spectral (NSE)
variance estimates. Under convergence, z ~ N(0,1).
"""
function _geweke_z(x::AbstractVector{T}; first::Real=0.1, last::Real=0.5) where {T<:AbstractFloat}
    n = length(x)
    n1 = max(2, floor(Int, first * n))
    n2 = max(2, floor(Int, last * n))
    n1 + n2 <= n || return T(NaN)
    xa = @view x[1:n1]
    xb = @view x[(n-n2+1):n]
    denom = sqrt(_geweke_nse(xa)^2 + _geweke_nse(xb)^2)
    denom > 0 || return T(NaN)
    return (mean(xa) - mean(xb)) / denom
end

# =============================================================================
# MCMCDiagnostics result + public API
# =============================================================================

"""
    MCMCDiagnostics{T}

Per-parameter MCMC convergence diagnostics for a `BayesianDSGE` chain.

Fields:
- `param_names::Vector{Symbol}` — parameter names
- `rhat::Vector{T}` — rank-normalized split-R̂ (max of bulk and folded)
- `ess_bulk::Vector{T}` — rank-normalized bulk effective sample size
- `ess_tail::Vector{T}` — tail effective sample size (min of 5%/95% indicators)
- `geweke_z::Vector{T}` — Geweke (1992) z-statistic (first 10% vs last 50%)
- `geweke_p::Vector{T}` — two-sided p-value of the Geweke z
- `mean::Vector{T}` — posterior mean per parameter
- `sd::Vector{T}` — posterior standard deviation per parameter
- `n_draws::Int` — number of retained (post-burn-in) draws
- `method::Symbol` — sampler that produced the draws
"""
struct MCMCDiagnostics{T<:AbstractFloat}
    param_names::Vector{Symbol}
    rhat::Vector{T}
    ess_bulk::Vector{T}
    ess_tail::Vector{T}
    geweke_z::Vector{T}
    geweke_p::Vector{T}
    mean::Vector{T}
    sd::Vector{T}
    n_draws::Int
    method::Symbol
end

"""
    mcmc_diagnostics(result::BayesianDSGE) → MCMCDiagnostics

Compute per-parameter MCMC convergence diagnostics on the retained
(post-burn-in) posterior draws:

- **Rank-normalized split-R̂** (Vehtari et al. 2021): the chain is split in
  half, pooled draws are rank-normalized (rank → z-score via the inverse normal
  CDF with Blom offsets), and R̂ is the maximum of the bulk and folded
  (`|θ − median|`) statistics. Values ≲ 1.01 indicate convergence.
- **Bulk/tail ESS**: effective sample size from the integrated autocorrelation
  time `ESS = S / (1 + 2Σ ρ̂ₖ)` with Geyer's initial-monotone-sequence
  truncation; bulk on rank-normalized draws, tail as the minimum ESS of the
  5% / 95% quantile indicators. Vehtari et al. recommend ESS ≥ 400.
- **Geweke z**: spectral test comparing the mean of the first 10% vs the last
  50% of the chain; |z| > 1.96 flags non-convergence at the 5% level.

Diagnostics assume Markov chain draws — for `:smc`/`:smc2` results (weighted
particle systems, not chains) a warning is emitted and autocorrelation-based
quantities should be interpreted with caution.

# Example
```julia
fit = estimate_dsge_bayes(spec, data, θ0; priors=priors, method=:mh)
diag = mcmc_diagnostics(fit)
diag.rhat, diag.ess_bulk
```

# References
- Vehtari, A. et al. (2021). Rank-Normalization, Folding, and Localization:
  An Improved R̂ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2).
- Geweke, J. (1992). In *Bayesian Statistics 4*.
"""
function mcmc_diagnostics(result::BayesianDSGE{T}) where {T<:AbstractFloat}
    if result.method != :rwmh
        @warn "mcmc_diagnostics: draws come from method=$(result.method) (weighted " *
              "particles, not a Markov chain); autocorrelation-based diagnostics " *
              "are approximate"
    end
    n_params = length(result.param_names)
    n_draws = size(result.theta_draws, 1)

    rhat_v = zeros(T, n_params)
    essb = zeros(T, n_params)
    esst = zeros(T, n_params)
    gz = zeros(T, n_params)
    gp = zeros(T, n_params)
    mu = zeros(T, n_params)
    sd = zeros(T, n_params)

    for i in 1:n_params
        x = result.theta_draws[:, i]
        rhat_v[i] = _rhat_rank(x)
        essb[i] = _ess_bulk(x)
        esst[i] = _ess_tail(x)
        gz[i] = _geweke_z(x)
        gp[i] = isfinite(gz[i]) ? 2 * (1 - cdf(Normal(), abs(gz[i]))) : T(NaN)
        mu[i] = mean(x)
        sd[i] = std(x)
    end

    return MCMCDiagnostics{T}(copy(result.param_names), rhat_v, essb, esst,
                              gz, gp, mu, sd, n_draws, result.method)
end

"""
    trace(result::BayesianDSGE, param::Symbol) → Vector

Return the retained (post-burn-in) draw sequence for `param`, for trace plots
and custom diagnostics.
"""
function trace(result::BayesianDSGE{T}, param::Symbol) where {T<:AbstractFloat}
    i = findfirst(==(param), result.param_names)
    i === nothing &&
        throw(ArgumentError("unknown parameter :$param; available: $(result.param_names)"))
    return result.theta_draws[:, i]
end

"""
    acf(result::BayesianDSGE, param::Symbol; lags=0, conf_level=0.95) → ACFResult

Autocorrelation function of the retained draw sequence for `param` (dispatches
to the spectral [`acf`](@ref)); useful for assessing chain mixing.
"""
acf(result::BayesianDSGE, param::Symbol; kwargs...) = acf(trace(result, param); kwargs...)

function Base.show(io::IO, d::MCMCDiagnostics{T}) where {T}
    header = Any[
        "Draws"        d.n_draws;
        "Parameters"   length(d.param_names);
        "Method"       string(d.method);
    ]
    _pretty_table(io, header;
        title="MCMC Convergence Diagnostics",
        column_labels=["", ""],
        alignment=[:l, :r])

    data = hcat([string(p) for p in d.param_names],
                round.(d.mean; digits=4), round.(d.sd; digits=4),
                round.(d.rhat; digits=4),
                round.(d.ess_bulk; digits=1), round.(d.ess_tail; digits=1),
                round.(d.geweke_z; digits=3), round.(d.geweke_p; digits=4))
    _pretty_table(io, data;
        title="Per-Parameter Diagnostics",
        column_labels=["Parameter", "Mean", "Std", "R-hat", "ESS (bulk)",
                       "ESS (tail)", "Geweke z", "Geweke p"],
        alignment=[:l, :r, :r, :r, :r, :r, :r, :r])

    bad_rhat = [string(d.param_names[i]) for i in eachindex(d.rhat)
                if isfinite(d.rhat[i]) && d.rhat[i] > 1.01]
    if !isempty(bad_rhat)
        println(io, "Note: R-hat > 1.01 for: ", join(bad_rhat, ", "),
                " — chains may not have converged (Vehtari et al. 2021).")
    end
end
