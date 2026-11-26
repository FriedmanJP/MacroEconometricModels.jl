# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Econometric estimation and Coibion–Gorodnichenko (2015) information-rigidity
diagnostics for steady-state LQG Rational Inattention models:
- `estimate_ri`: Kalman prediction-error-decomposition MLE (`Optim.LBFGS`) and
  adaptive RWMH Bayesian estimation of Shannon capacity `κ` or marginal attention
  cost `ω` alongside structural parameters `θ`, supporting `NaN` missing
  observations and a likelihood-ratio test against full information (`κ = ∞`).
- `coibion_gorodnichenko_test`: Coibion & Gorodnichenko (2015) forecast-error
  on forecast-revision regression with Newey–West HAC standard errors.
- `implied_cg_slope`: exact population Coibion–Gorodnichenko slope `β_CG`
  implied by a multivariate `SimsKalmanSolution`.
"""

# =============================================================================
# Result Structs
# =============================================================================

"""
    RIEstimationResult{T<:AbstractFloat} <: StatsAPI.StatisticalModel

Result of maximum-likelihood (`method = :mle`) or Bayesian MCMC (`method = :bayes`)
econometric estimation of a steady-state LQG Rational Inattention model.

# Fields
- `params::Vector{T}`: estimated parameter vector `[κ_or_ω; extra_params...]`.
- `param_names::Vector{String}`: names for each element of `params`.
- `vcov_mat::Matrix{T}`: `k × k` parameter covariance matrix (Hessian/sandwich for `:mle`, posterior covariance for `:bayes`).
- `stderr::Vector{T}`: length-`k` standard errors (posterior standard deviations for `:bayes`).
- `capacity::T`: estimated/implied Shannon channel capacity `κ` (in `solution.problem.units`).
- `omega::T`: estimated/implied marginal shadow attention cost `ω` per nat.
- `loglik::T`: Kalman prediction-error-decomposition log-likelihood at `params`.
- `loglik_fullinfo::T`: full-information (`κ = ∞` / `ω = 0`) Kalman log-likelihood.
- `lr_stat::T`: likelihood-ratio statistic `2 * (loglik - loglik_fullinfo)` testing `H₀: κ = ∞`.
- `lr_pvalue::T`: asymptotic `χ²(1)` p-value for `H₀: κ = ∞`.
- `aic::T`: Akaike information criterion `-2 * loglik + 2 * k`.
- `bic::T`: Bayesian information criterion `-2 * loglik + log(n_obs) * k`.
- `n_obs::Int`: number of time-series periods in `data`.
- `method::Symbol`: estimation method (`:mle` or `:bayes`).
- `estimate_param::Symbol`: primary attention parameter estimated (`:capacity` or `:omega`).
- `solution::SimsKalmanSolution{T}`: equilibrium `SimsKalmanSolution` at `params`.
- `draws::Matrix{T}`: `n_draws × k` posterior draws for `:bayes` (`0 × 0` for `:mle`).
- `acceptance_rate::T`: RWMH acceptance rate for `:bayes` (`NaN` for `:mle`).
- `converged::Bool`: whether optimization / estimation converged cleanly.
- `manifest::Union{ReproManifest,Nothing}`: reproducibility provenance manifest (`nothing` for `:mle`).
"""
struct RIEstimationResult{T<:AbstractFloat} <: StatsAPI.StatisticalModel
    params::Vector{T}
    param_names::Vector{String}
    vcov_mat::Matrix{T}
    stderr::Vector{T}
    capacity::T
    omega::T
    loglik::T
    loglik_fullinfo::T
    lr_stat::T
    lr_pvalue::T
    aic::T
    bic::T
    n_obs::Int
    method::Symbol
    estimate_param::Symbol
    solution::SimsKalmanSolution{T}
    draws::Matrix{T}
    acceptance_rate::T
    converged::Bool
    manifest::Union{ReproManifest,Nothing}

    function RIEstimationResult{T}(
        params::Vector{T},
        param_names::Vector{String},
        vcov_mat::Matrix{T},
        stderr::Vector{T},
        capacity::T,
        omega::T,
        loglik::T,
        loglik_fullinfo::T,
        lr_stat::T,
        lr_pvalue::T,
        aic::T,
        bic::T,
        n_obs::Int,
        method::Symbol,
        estimate_param::Symbol,
        solution::SimsKalmanSolution{T},
        draws::Matrix{T},
        acceptance_rate::T,
        converged::Bool,
        manifest::Union{ReproManifest,Nothing},
    ) where {T<:AbstractFloat}
        k = length(params)
        k >= 1 || throw(ArgumentError("params must be non-empty"))
        length(param_names) == k || throw(ArgumentError(
            "param_names must have length $k, got $(length(param_names))"))
        size(vcov_mat) == (k, k) || throw(ArgumentError(
            "vcov_mat must be $k×$k, got $(size(vcov_mat))"))
        length(stderr) == k || throw(ArgumentError(
            "stderr must have length $k, got $(length(stderr))"))
        n_obs >= 1 || throw(ArgumentError("n_obs must be >= 1, got $n_obs"))
        method in (:mle, :bayes) || throw(ArgumentError(
            "method must be :mle or :bayes, got :$method"))
        estimate_param in (:capacity, :omega) || throw(ArgumentError(
            "estimate_param must be :capacity or :omega, got :$estimate_param"))
        return new{T}(
            params, param_names, vcov_mat, stderr,
            capacity, omega, loglik, loglik_fullinfo, lr_stat, lr_pvalue,
            aic, bic, n_obs, method, estimate_param, solution,
            draws, acceptance_rate, converged, manifest,
        )
    end
end

"""
    CGTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest

Result of the Coibion & Gorodnichenko (2015) information-rigidity regression:

```math
y_{t+h} - F_t y_{t+h} = \\alpha + \\beta_{\\text{CG}} \\left(F_t y_{t+h} - F_{t-1} y_{t+h}\\right) + e_{t+h}
```

where:
- ``y_{t+h}`` is the realized outcome at horizon ``t+h``
- ``F_t y_{t+h}`` is the time-``t`` forecast of ``y_{t+h}``
- ``F_{t-1} y_{t+h}`` is the lagged (time-``t-1``) forecast of ``y_{t+h}``
- ``\\alpha`` is the regression intercept (zero under unbiasedness)
- ``\\beta_{\\text{CG}}`` is the Coibion–Gorodnichenko rigidity slope (``\\beta_{\\text{CG}} = (1 - G)/G > 0`` under noisy information)
- ``e_{t+h}`` is the forecast-error regression residual evaluated with Newey–West HAC standard errors

# Fields
- `alpha::T`: OLS intercept estimate.
- `se_alpha::T`: Newey–West HAC standard error of `alpha`.
- `beta_cg::T`: OLS slope estimate ``\\hat{\\beta}_{\\text{CG}}``.
- `se_cg::T`: Newey–West HAC standard error of `beta_cg`.
- `t_stat::T`: t-statistic `beta_cg / se_cg` testing ``H_0: \\beta_{\\text{CG}} = 0``.
- `pvalue::T`: two-sided `t(n_obs - 2)` p-value for ``H_0: \\beta_{\\text{CG}} = 0``.
- `n_obs::Int`: number of complete observations used in the regression.
- `hac_lags::Int`: Newey–West Bartlett truncation lag used.
- `implied_gain::T`: implied Kalman gain ``\\hat{G} = 1 / (1 + \\hat{\\beta}_{\\text{CG}})``.
- `implied_capacity::T`: implied scalar AR(1) capacity ``\\hat{\\kappa}_{\\text{CG}} = -\\tfrac{1}{2} \\log_2(1 - \\hat{G})`` in bits (`NaN` if ``\\hat{G} \\notin (0, 1)``).
"""
struct CGTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    alpha::T
    se_alpha::T
    beta_cg::T
    se_cg::T
    t_stat::T
    pvalue::T
    n_obs::Int
    hac_lags::Int
    implied_gain::T
    implied_capacity::T

    function CGTestResult{T}(
        alpha::T,
        se_alpha::T,
        beta_cg::T,
        se_cg::T,
        t_stat::T,
        pvalue::T,
        n_obs::Int,
        hac_lags::Int,
        implied_gain::T,
        implied_capacity::T,
    ) where {T<:AbstractFloat}
        n_obs >= 4 || throw(ArgumentError("n_obs must be >= 4, got $n_obs"))
        hac_lags >= 0 || throw(ArgumentError("hac_lags must be >= 0, got $hac_lags"))
        return new{T}(
            alpha, se_alpha, beta_cg, se_cg, t_stat, pvalue,
            n_obs, hac_lags, implied_gain, implied_capacity,
        )
    end
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(r::RIEstimationResult) = copy(r.params)
StatsAPI.coefnames(r::RIEstimationResult) = copy(r.param_names)
StatsAPI.vcov(r::RIEstimationResult) = copy(r.vcov_mat)
StatsAPI.stderror(r::RIEstimationResult) = copy(r.stderr)
StatsAPI.loglikelihood(r::RIEstimationResult) = r.loglik
StatsAPI.aic(r::RIEstimationResult) = r.aic
StatsAPI.bic(r::RIEstimationResult) = r.bic
StatsAPI.nobs(r::RIEstimationResult) = r.n_obs
StatsAPI.dof(r::RIEstimationResult) = length(r.params)
StatsAPI.dof_residual(r::RIEstimationResult) = r.n_obs - length(r.params)
StatsAPI.islinear(::RIEstimationResult) = false

function StatsAPI.confint(r::RIEstimationResult{T}; level::Real=DEFAULT_CI_MACRO) where {T<:AbstractFloat}
    lv = _validate_conf_level(level)
    k = length(r.params)
    alpha_lv = one(T) - T(lv)
    if r.method === :bayes && size(r.draws, 1) >= 2
        lo = Vector{T}(undef, k)
        hi = Vector{T}(undef, k)
        for j in 1:k
            col = @view r.draws[:, j]
            lo[j] = T(quantile(col, alpha_lv / 2))
            hi[j] = T(quantile(col, one(T) - alpha_lv / 2))
        end
        return hcat(lo, hi)
    end
    crit = T(_conf_z(lv))
    return hcat(r.params .- crit .* r.stderr, r.params .+ crit .* r.stderr)
end

StatsAPI.pvalue(r::CGTestResult) = r.pvalue
StatsAPI.nobs(r::CGTestResult) = r.n_obs
StatsAPI.dof(::CGTestResult) = 1
StatsAPI.dof_residual(r::CGTestResult) = r.n_obs - 2

# =============================================================================
# Internal Helpers (_riest_*)
# =============================================================================

@inline _riest_symmetrize(M::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Matrix{T}((M + M') / 2)

function _riest_promote_prob(::Type{TW}, prob::RationalInattentionProblem{T}) where {TW<:AbstractFloat, T<:AbstractFloat}
    TW === T && return prob
    return RationalInattentionProblem{TW}(
        Matrix{TW}(prob.A),
        Matrix{TW}(prob.B),
        Matrix{TW}(prob.H),
        Matrix{TW}(prob.M),
        prob.B_u === nothing ? nothing : Matrix{TW}(prob.B_u),
        TW(prob.beta),
        prob.capacity === nothing ? nothing : TW(prob.capacity),
        prob.omega === nothing ? nothing : TW(prob.omega),
        prob.units,
        copy(prob.state_names),
        copy(prob.action_names),
        copy(prob.shock_names),
    )
end

function _riest_rebuild_prob(prob::RationalInattentionProblem{TW}, val::TW, estimate_param::Symbol) where {TW<:AbstractFloat}
    if estimate_param === :capacity
        return RationalInattentionProblem{TW}(
            prob.A, prob.B, prob.H, prob.M, prob.B_u, prob.beta,
            val, nothing, prob.units,
            prob.state_names, prob.action_names, prob.shock_names,
        )
    else
        return RationalInattentionProblem{TW}(
            prob.A, prob.B, prob.H, prob.M, prob.B_u, prob.beta,
            nothing, val, prob.units,
            prob.state_names, prob.action_names, prob.shock_names,
        )
    end
end

function _riest_call_build(build_or_prob, val::TW, extra::AbstractVector{TW}, estimate_param::Symbol) where {TW<:AbstractFloat}
    if build_or_prob isa RationalInattentionProblem
        isempty(extra) || throw(ArgumentError(
            "extra_params_init must be empty when build_or_prob is a fixed RationalInattentionProblem"))
        prob_w = _riest_promote_prob(TW, build_or_prob)
        return _riest_rebuild_prob(prob_w, val, estimate_param)
    end
    build_or_prob isa Function || throw(ArgumentError(
        "build_or_prob must be a RationalInattentionProblem or a builder Function, got $(typeof(build_or_prob))"))
    p = if isempty(extra)
        if applicable(build_or_prob, val)
            build_or_prob(val)
        elseif applicable(build_or_prob, val, extra)
            build_or_prob(val, extra)
        elseif applicable(build_or_prob, TW[val])
            build_or_prob(TW[val])
        else
            throw(ArgumentError(
                "Builder function must accept `(val)` or `(val, extra_params)` when `extra_params_init` is empty."))
        end
    else
        if applicable(build_or_prob, val, extra)
            build_or_prob(val, extra)
        elseif applicable(build_or_prob, vcat(val, extra))
            build_or_prob(vcat(val, extra))
        else
            throw(ArgumentError(
                "Builder function must accept `(val, extra_params)` or `vcat(val, extra_params)`."))
        end
    end
    p isa RationalInattentionProblem || throw(ArgumentError(
        "Builder function must return a RationalInattentionProblem, got $(typeof(p))."))
    return _riest_promote_prob(TW, p)
end

function _riest_resolve_obs_indices(prob::RationalInattentionProblem, observables, k_data::Int)
    n = size(prob.A, 1)
    d = size(prob.H, 2)
    all_names = vcat(prob.state_names, [nm * "_hat" for nm in prob.state_names], prob.action_names)
    total_all = 2 * n + d

    if observables === nothing
        if k_data == d
            return (:actions, collect((2 * n + 1):total_all))
        elseif k_data == n + d
            return (:actions_and_states, vcat(collect(1:n), collect((2 * n + 1):total_all)))
        elseif k_data == total_all
            return (:all, collect(1:total_all))
        else
            throw(ArgumentError(
                "Cannot infer observables from data with $k_data columns (expected d=$d for :actions, " *
                "n+d=$(n+d) for :actions_and_states, or 2n+d=$(total_all) for :all). Pass `observables` explicitly."))
        end
    elseif observables isa Symbol
        observables in (:actions, :actions_and_states, :all) || throw(ArgumentError(
            "observables Symbol must be :actions, :actions_and_states, or :all; got :$observables"))
        idx = observables === :actions ? collect((2 * n + 1):total_all) :
              observables === :actions_and_states ? vcat(collect(1:n), collect((2 * n + 1):total_all)) :
              collect(1:total_all)
        length(idx) == k_data || throw(ArgumentError(
            "data has $k_data columns but observables=:$observables produces $(length(idx)) series"))
        return (observables, idx)
    elseif observables isa AbstractVector{<:Integer}
        idx = Int[Int(i) for i in observables]
        !isempty(idx) || throw(ArgumentError("observables index vector must be non-empty"))
        all(i -> 1 <= i <= total_all, idx) || throw(ArgumentError(
            "observables indices must lie in 1:$total_all"))
        length(idx) == k_data || throw(ArgumentError(
            "data has $k_data columns but observables selects $(length(idx)) series"))
        return (:custom, idx)
    elseif observables isa AbstractVector{<:Union{Symbol,AbstractString}}
        req = String[string(s) for s in observables]
        !isempty(req) || throw(ArgumentError("observables name vector must be non-empty"))
        idx = Int[]
        for nm in req
            pos = findfirst(==(nm), all_names)
            pos === nothing && throw(ArgumentError(
                "Observable '$nm' not found in model variable names $all_names"))
            push!(idx, pos)
        end
        length(idx) == k_data || throw(ArgumentError(
            "data has $k_data columns but observables selects $(length(idx)) series"))
        return (:custom, idx)
    else
        throw(ArgumentError(
            "observables must be nothing, Symbol, or Vector of Int/Symbol/String; got $(typeof(observables))"))
    end
end

function _riest_build_Hobs(::Type{TW}, measurement_error, k_obs::Int) where {TW<:AbstractFloat}
    if measurement_error === nothing
        return zeros(TW, k_obs, k_obs)
    elseif measurement_error isa Real
        me = TW(measurement_error)
        (isfinite(me) && me >= zero(TW)) || throw(ArgumentError(
            "measurement_error must be finite and non-negative, got $measurement_error"))
        return Matrix{TW}(Diagonal(fill(me^2, k_obs)))
    elseif measurement_error isa AbstractVector{<:Real}
        length(measurement_error) == k_obs || throw(ArgumentError(
            "measurement_error vector must have length $k_obs, got $(length(measurement_error))"))
        mev = TW[TW(x) for x in measurement_error]
        all(x -> isfinite(x) && x >= zero(TW), mev) || throw(ArgumentError(
            "All measurement_error entries must be finite and non-negative"))
        return Matrix{TW}(Diagonal(mev .^ 2))
    else
        throw(ArgumentError(
            "measurement_error must be nothing, a non-negative Real, or a vector of length $k_obs"))
    end
end

function _riest_kalman_loglik(
    sol::SimsKalmanSolution{TW},
    ykern::Matrix{TW},
    obs_idx::Vector{Int},
    Hobs::Matrix{TW};
    return_contribs::Bool=false,
) where {TW<:AbstractFloat}
    ssm = to_statespace(sol; observables=:all)
    Z_sub = Matrix{TW}(ssm.Z[obs_idx, :])
    RQR = _riest_symmetrize(Matrix{TW}(ssm.R * ssm.Q * ssm.R'))
    if !return_contribs
        ll = _suppress_warnings() do
            _kalman_filter!(nothing, ykern, Z_sub, ssm.Tt, RQR, Hobs;
                            a0=ssm.a1, P0=ssm.P1)
        end
        return ll
    end

    T_obs = size(ykern, 2)
    store = KalmanFilterStore{TW}(ssm.n_state, T_obs; innovations=true)
    ll = _suppress_warnings() do
        _kalman_filter!(store, ykern, Z_sub, ssm.Tt, RQR, Hobs;
                        a0=ssm.a1, P0=ssm.P1)
    end
    contribs = zeros(TW, T_obs)
    log2pi = TW(log(2 * pi))
    @inbounds for t in 1:T_obs
        isassigned(store.v, t) || continue
        v = store.v[t]
        F = store.F[t]
        m = length(v)
        m == 0 && continue
        if m == 1
            f = max(F[1, 1], eps(TW))
            contribs[t] = -TW(0.5) * (log2pi + log(f) + (v[1]^2) / f)
        else
            L = safe_cholesky(Hermitian(Matrix{TW}(F)); silent=true)
            logdetS = TW(2) * sum(i -> log(L[i, i]), 1:m)
            w = L \ v
            contribs[t] = -TW(0.5) * (TW(m) * log2pi + logdetS + dot(w, w))
        end
    end
    return (ll = ll, contribs = contribs)
end

function _riest_fullinfo_loglik(
    prob::RationalInattentionProblem{TW},
    ykern::Matrix{TW},
    obs_idx::Vector{Int},
    Hobs::Matrix{TW},
) where {TW<:AbstractFloat}
    n = size(prob.A, 1)
    d = size(prob.H, 2)
    A = Matrix{TW}(prob.A)
    B = Matrix{TW}(prob.B)
    H = Matrix{TW}(prob.H)
    F = Matrix{TW}(H')
    Bu = prob.B_u === nothing ? zeros(TW, n, d) : Matrix{TW}(prob.B_u)
    T_fi = A + Bu * F
    RQR_fi = _riest_symmetrize(B * B')

    # Full-information Z maps x_t identically onto both x_t and x̂_{t|t}
    In = Matrix{TW}(I, n, n)
    Z_all_fi = vcat(In, In, F)
    Z_fi = Matrix{TW}(Z_all_fi[obs_idx, :])

    rho_fi = maximum(abs, eigvals(Matrix{Float64}(T_fi)); init=0.0)
    mode_fi = rho_fi < 1.0 - 1e-6 ? :stationary : :kappa
    a1_fi, P1_fi = _kalman_init(mode_fi, T_fi, RQR_fi, n; kappa=TW(1e6))

    return _suppress_warnings() do
        _kalman_filter!(nothing, ykern, Z_fi, T_fi, RQR_fi, Hobs; a0=a1_fi, P0=P1_fi)
    end
end

function _riest_eval_u(
    u::AbstractVector{TW},
    build_or_prob,
    ykern::Matrix{TW},
    obs_idx::Vector{Int},
    Hobs::Matrix{TW},
    estimate_param::Symbol;
    allow_zero_signal::Bool=false,
    return_contribs::Bool=false,
) where {TW<:AbstractFloat}
    u1 = u[1]
    u_lo = TW(-14)
    u_hi = estimate_param === :capacity ? TW(3.2) : TW(12)
    if !isfinite(u1) || any(!isfinite, u)
        return return_contribs ? (TW(-Inf), zeros(TW, size(ykern, 2)), nothing) : TW(-Inf)
    end
    if u1 < u_lo || u1 > u_hi
        return return_contribs ? (TW(-Inf), zeros(TW, size(ykern, 2)), nothing) : TW(-Inf)
    end

    val = exp(u1)
    extra = length(u) > 1 ? Vector{TW}(@view u[2:end]) : TW[]

    try
        prob = _riest_call_build(build_or_prob, val, extra, estimate_param)
        sol = solve_ri_kalman(prob; tol=TW(1e-10), max_iter=1500)
        if !sol.converged || (!allow_zero_signal && sol.signal_rank == 0)
            return return_contribs ? (TW(-Inf), zeros(TW, size(ykern, 2)), nothing) : TW(-Inf)
        end
        if return_contribs
            out = _riest_kalman_loglik(sol, ykern, obs_idx, Hobs; return_contribs=true)
            return (out.ll, out.contribs, sol)
        else
            ll = _riest_kalman_loglik(sol, ykern, obs_idx, Hobs; return_contribs=false)
            return ll
        end
    catch e
        if e isa ArgumentError || e isa DomainError ||
           e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException
            return return_contribs ? (TW(-Inf), zeros(TW, size(ykern, 2)), nothing) : TW(-Inf)
        end
        rethrow()
    end
end

function _riest_fd_hessian(f, x::Vector{TW}) where {TW<:AbstractFloat}
    k = length(x)
    H = Matrix{TW}(undef, k, k)
    f0 = f(x)
    h = TW[TW(1e-4) * max(abs(x[i]), one(TW)) for i in 1:k]
    for i in 1:k
        ei = zeros(TW, k)
        ei[i] = h[i]
        fp = f(x + ei)
        fm = f(x - ei)
        H[i, i] = (fp - TW(2) * f0 + fm) / (h[i]^2)
        for j in (i + 1):k
            ej = zeros(TW, k)
            ej[j] = h[j]
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            hij = (fpp - fpm - fmp + fmm) / (TW(4) * h[i] * h[j])
            H[i, j] = hij
            H[j, i] = hij
        end
    end
    return _riest_symmetrize(H)
end

function _riest_fd_scores(
    u_hat::Vector{TW},
    build_or_prob,
    ykern::Matrix{TW},
    obs_idx::Vector{Int},
    Hobs::Matrix{TW},
    estimate_param::Symbol,
) where {TW<:AbstractFloat}
    T_obs = size(ykern, 2)
    k = length(u_hat)
    S = zeros(TW, T_obs, k)
    for j in 1:k
        hj = TW(1e-5) * max(abs(u_hat[j]), one(TW))
        up = copy(u_hat); up[j] += hj
        um = copy(u_hat); um[j] -= hj
        llp, cp, _ = _riest_eval_u(up, build_or_prob, ykern, obs_idx, Hobs, estimate_param; return_contribs=true)
        llm, cm, _ = _riest_eval_u(um, build_or_prob, ykern, obs_idx, Hobs, estimate_param; return_contribs=true)
        if isfinite(llp) && isfinite(llm)
            S[:, j] .= (cp .- cm) ./ (TW(2) * hj)
        end
    end
    return S
end

function _riest_resolve_priors(priors, k::Int, pnames::Vector{String}, estimate_param::Symbol)
    if priors === nothing
        dists = Distribution[LogNormal(0.0, 1.5)]
        for _ in 2:k
            push!(dists, Normal(0.0, 10.0))
        end
        return dists
    elseif priors isa AbstractVector{<:Distribution}
        length(priors) == k || throw(ArgumentError(
            "priors vector must have length $k (1 + extra_params), got $(length(priors))"))
        return Distribution[d for d in priors]
    elseif priors isa Tuple && all(d -> d isa Distribution, priors)
        length(priors) == k || throw(ArgumentError(
            "priors tuple must have length $k, got $(length(priors))"))
        return Distribution[d for d in priors]
    elseif priors isa AbstractDict
        dists = Vector{Distribution}(undef, k)
        for (j, nm) in enumerate(pnames)
            if haskey(priors, nm)
                dists[j] = priors[nm]
            elseif haskey(priors, Symbol(nm))
                dists[j] = priors[Symbol(nm)]
            elseif j == 1 && haskey(priors, estimate_param)
                dists[j] = priors[estimate_param]
            else
                throw(ArgumentError("priors dict missing entry for parameter '$nm'"))
            end
        end
        return dists
    elseif priors isa NamedTuple
        dists = Vector{Distribution}(undef, k)
        for (j, nm) in enumerate(pnames)
            sym = Symbol(nm)
            if haskey(priors, sym)
                dists[j] = priors[sym]
            elseif j == 1 && haskey(priors, estimate_param)
                dists[j] = priors[estimate_param]
            else
                throw(ArgumentError("priors NamedTuple missing entry for parameter :$sym"))
            end
        end
        return dists
    else
        throw(ArgumentError(
            "priors must be nothing, a Vector/Tuple of Distributions, a Dict, or a NamedTuple"))
    end
end

function _riest_initial_param(
    build_or_prob,
    extra_init::Vector{TW},
    ykern::Matrix{TW},
    obs_idx::Vector{Int},
    Hobs::Matrix{TW},
    estimate_param::Symbol,
    init_param,
) where {TW<:AbstractFloat}
    if init_param !== nothing
        v0 = TW(init_param)
        (isfinite(v0) && v0 > zero(TW)) || throw(ArgumentError(
            "Initial parameter must be finite and strictly positive, got $init_param"))
        return v0
    end
    if build_or_prob isa RationalInattentionProblem
        prob_w = _riest_promote_prob(TW, build_or_prob)
        if estimate_param === :capacity && prob_w.capacity !== nothing
            return prob_w.capacity
        elseif estimate_param === :omega && prob_w.omega !== nothing
            return prob_w.omega
        else
            sol0 = solve_ri_kalman(prob_w)
            if estimate_param === :capacity
                return prob_w.units === :bits ? max(sol0.capacity_bits, TW(0.25)) : max(sol0.capacity_nats, TW(0.25))
            else
                return max(sol0.omega, TW(0.1))
            end
        end
    end

    # Grid probe for builder function to find a valid high-likelihood starting point
    candidates = estimate_param === :capacity ?
        TW[0.25, 0.5, 0.75, 1.25, 2.5, 4.0] :
        TW[0.05, 0.2, 0.5, 1.0, 2.0, 5.0]
    best_v = candidates[3]
    best_ll = TW(-Inf)
    for v in candidates
        u_try = vcat(log(v), extra_init)
        ll = _riest_eval_u(u_try, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
        if isfinite(ll) && ll > best_ll
            best_ll = ll
            best_v = v
        end
    end
    if !isfinite(best_ll)
        # Try explicit verification at 1.0 to raise informative builder error if invalid
        prob_test = _riest_call_build(build_or_prob, one(TW), extra_init, estimate_param)
        solve_ri_kalman(prob_test)
    end
    return best_v
end

# =============================================================================
# Public API: estimate_ri
# =============================================================================

"""
    estimate_ri(build_or_prob, data::AbstractMatrix;
                method::Symbol=:mle,
                estimate_param::Symbol=:capacity,
                extra_params_init=Float64[],
                extra_param_names=String[],
                priors=nothing,
                observables=nothing,
                cov_type::Symbol=:hessian,
                init_param=nothing,
                measurement_error=nothing,
                n_draws::Int=1000,
                burnin::Int=250,
                step_scale::Real=0.1,
                rng::AbstractRNG=Random.default_rng(),
                seed::Union{Integer,Nothing}=nothing) -> RIEstimationResult

Estimate the unobserved Shannon information capacity ``\\kappa`` (or marginal
attention shadow cost ``\\omega``) and optional structural parameters ``\\theta``
of a steady-state LQG Rational Inattention model from observed macro and/or
action/survey series ``y_t`` via the Kalman prediction-error-decomposition
log-likelihood (`_kalman_filter!`, supporting `NaN` missing entries):

```math
\\ell(\\kappa, \\theta) = \\sum_{t=1}^{T} \\left( -\\frac{k_t}{2}\\log(2\\pi) - \\frac{1}{2}\\log\\lvert S_t \\rvert - \\frac{1}{2} v_t' S_t^{-1} v_t \\right)
```

where:
- ``\\ell(\\kappa, \\theta)`` is the Kalman prediction-error-decomposition log-likelihood evaluated on `to_statespace(solve_ri_kalman(prob))`
- ``k_t`` is the number of non-missing (`!isnan`) observables in period ``t``
- ``v_t = y_t^{\\text{obs}} - Z_t \\hat{\\xi}_{t|t-1}`` is the one-step-ahead innovation vector
- ``S_t = Z_t P_{t|t-1} Z_t' + H_{\\text{obs}, t}`` is the innovation covariance matrix

# Arguments
- `build_or_prob`: either a fixed [`RationalInattentionProblem`](@ref) (only `κ` or `ω` is estimated) or a builder function `prob(val)` / `prob(val, extra_params)` / `prob(full_params)` returning a `RationalInattentionProblem`.
- `data::AbstractMatrix`: `T × k_obs` matrix of observed data (time along rows; `NaN` indicates missing entries).

# Keyword Arguments
- `method::Symbol=:mle`: `:mle` (`Optim.LBFGS` on unconstrained `(log κ_or_ω, θ...)` with delta-method SEs and full-information LR test) or `:bayes` (adaptive Random-Walk Metropolis–Hastings).
- `estimate_param::Symbol=:capacity`: whether the primary RI parameter is `:capacity` (``\\kappa``) or `:omega` (``\\omega``).
- `extra_params_init`: initial values for structural parameters `θ` (empty by default).
- `extra_param_names`: display names for `θ` (defaults to `["θ1", ...]`).
- `priors`: prior distributions for `:bayes` (`nothing` uses `LogNormal(0, 1.5)` for `κ`/`ω` and `Normal(0, 10)` for extra `θ`; also accepts a vector/tuple of `Distribution`s, `Dict`, or `NamedTuple`).
- `observables=nothing`: observable selection (`:actions`, `:actions_and_states`, `:all`, integer indices, variable names, or `nothing` to infer from `size(data, 2)`).
- `cov_type::Symbol=:hessian`: MLE covariance estimator (`:hessian` or `:sandwich`).
- `init_param=nothing`: optional positive initial guess for `κ` or `ω`.
- `measurement_error=nothing`: optional scalar or vector of measurement-error standard deviations added to observation noise covariance `H_obs`.
- `n_draws::Int=1000`: number of retained post-burn-in draws for `:bayes`.
- `burnin::Int=250`: number of adaptive burn-in draws discarded for `:bayes`.
- `step_scale::Real=0.1`: initial RWMH step scale in unconstrained log-parameter space.
- `rng::AbstractRNG=Random.default_rng()`: random number generator for `:bayes`.
- `seed::Union{Integer,Nothing}=nothing`: optional integer seed recorded in `manifest` (`seed` overrides `rng` via `Xoshiro(seed)`).
"""
function estimate_ri(
    build_or_prob,
    data::AbstractMatrix{<:Real};
    method::Symbol=:mle,
    estimate_param::Symbol=:capacity,
    extra_params_init::AbstractVector{<:Real}=Float64[],
    extra_param_names::AbstractVector{<:AbstractString}=String[],
    priors=nothing,
    observables=nothing,
    cov_type::Symbol=:hessian,
    init_param::Union{Nothing,Real}=nothing,
    measurement_error::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    n_draws::Int=1000,
    burnin::Int=250,
    step_scale::Real=0.1,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
)
    method in (:mle, :bayes) || throw(ArgumentError(
        "method must be :mle or :bayes, got :$method"))
    estimate_param in (:capacity, :omega) || throw(ArgumentError(
        "estimate_param must be :capacity or :omega, got :$estimate_param"))
    cov_type in (:hessian, :sandwich) || throw(ArgumentError(
        "cov_type must be :hessian or :sandwich, got :$cov_type"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be >= 1, got $n_draws"))
    burnin >= 0 || throw(ArgumentError("burnin must be >= 0, got $burnin"))
    (isfinite(step_scale) && step_scale > 0) || throw(ArgumentError(
        "step_scale must be finite and > 0, got $step_scale"))

    T_obs, k_data = size(data)
    T_obs >= 2 || throw(ArgumentError("data must have at least 2 time periods (rows), got $T_obs"))
    k_data >= 1 || throw(ArgumentError("data must have at least 1 column, got $k_data"))
    any(isinf, data) && throw(ArgumentError("data contains Inf/-Inf values"))
    any(isfinite, data) || throw(ArgumentError("data contains no finite observations"))

    T = float(eltype(data))
    if build_or_prob isa RationalInattentionProblem
        T = promote_type(T, eltype(build_or_prob.A))
    elseif !isempty(extra_params_init)
        T = promote_type(T, float(eltype(extra_params_init)))
    end
    T <: AbstractFloat || (T = Float64)
    TW = T <: Union{Float16, Float32} ? Float64 : T

    extra_init = TW[TW(x) for x in extra_params_init]
    p = length(extra_init)
    k = 1 + p

    if !isempty(extra_param_names)
        length(extra_param_names) == p || throw(ArgumentError(
            "extra_param_names must have length $p matching extra_params_init, got $(length(extra_param_names))"))
        extra_names = String[string(s) for s in extra_param_names]
    else
        extra_names = String["θ$(j)" for j in 1:p]
    end
    primary_name = estimate_param === :capacity ? "capacity" : "omega"
    pnames = vcat([primary_name], extra_names)

    ykern = Matrix{TW}(permutedims(Matrix{TW}(data))) # k_data × T_obs
    Hobs = _riest_build_Hobs(TW, measurement_error, k_data)

    # Probe structural dimensions via a reference problem
    probe_val = init_param !== nothing ? TW(init_param) :
        (build_or_prob isa RationalInattentionProblem ?
            TW(something(build_or_prob.capacity, build_or_prob.omega, one(TW))) : one(TW))
    prob_ref = _riest_call_build(build_or_prob, probe_val, extra_init, estimate_param)
    _, obs_idx = _riest_resolve_obs_indices(prob_ref, observables, k_data)

    val0 = _riest_initial_param(build_or_prob, extra_init, ykern, obs_idx, Hobs, estimate_param, init_param)
    u0 = vcat(log(val0), extra_init)

    if method === :mle
        nll = u -> begin
            ll = _riest_eval_u(u, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
            isfinite(ll) ? -ll : TW(1e15) + sum(abs2, u)
        end

        function g!(G, u)
            @inbounds for j in 1:k
                hj = TW(1e-5) * max(abs(u[j]), one(TW))
                up = copy(u); up[j] += hj
                um = copy(u); um[j] -= hj
                fp = nll(up)
                fm = nll(um)
                G[j] = (fp - fm) / (TW(2) * hj)
            end
            return G
        end

        opts = Optim.Options(iterations=500, g_tol=TW(1e-7), f_reltol=TW(1e-10))
        res = Optim.optimize(nll, g!, u0, Optim.LBFGS(), opts)
        u_hat = Vector{TW}(Optim.minimizer(res))
        converged = Optim.converged(res)

        if !converged
            res_nm = Optim.optimize(nll, u_hat, Optim.NelderMead(),
                                    Optim.Options(iterations=400, f_abstol=TW(1e-9)))
            if Optim.minimum(res_nm) <= Optim.minimum(res)
                u_nm = Vector{TW}(Optim.minimizer(res_nm))
                res2 = Optim.optimize(nll, g!, u_nm, Optim.LBFGS(), opts)
                if Optim.minimum(res2) <= Optim.minimum(res_nm)
                    u_hat = Vector{TW}(Optim.minimizer(res2))
                    converged = Optim.converged(res2) || Optim.converged(res_nm)
                else
                    u_hat = u_nm
                    converged = Optim.converged(res_nm)
                end
            end
        end

        val_hat = exp(u_hat[1])
        extra_hat = p > 0 ? u_hat[2:end] : TW[]
        prob_hat = _riest_call_build(build_or_prob, val_hat, extra_hat, estimate_param)
        sol_hat = solve_ri_kalman(prob_hat; tol=TW(1e-11), max_iter=2000)
        ll_hat = _riest_kalman_loglik(sol_hat, ykern, obs_idx, Hobs)

        # Hessian & optional sandwich covariance in unconstrained u-space, then delta method onto (val, θ)
        H_nll = _riest_fd_hessian(nll, u_hat)
        ev_H = eigvals(Symmetric(H_nll))
        V_u = if minimum(ev_H) > TW(1e-10) * max(maximum(ev_H), one(TW))
            H_inv = Matrix{TW}(robust_inv(Symmetric(H_nll); silent=true))
            if cov_type === :sandwich
                S_sc = _riest_fd_scores(u_hat, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
                J_opg = _riest_symmetrize(S_sc' * S_sc)
                _riest_symmetrize(H_inv * J_opg * H_inv)
            else
                _riest_symmetrize(H_inv)
            end
        else
            S_sc = _riest_fd_scores(u_hat, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
            J_opg = _riest_symmetrize(S_sc' * S_sc)
            _riest_symmetrize(Matrix{TW}(robust_inv(Symmetric(J_opg); silent=true)))
        end

        D_jac = Diagonal(vcat(val_hat, ones(TW, p)))
        vcov_nat = _riest_symmetrize(D_jac * V_u * D_jac)
        se_nat = sqrt.(max.(diag(vcov_nat), zero(TW)))
        params_hat = vcat(val_hat, extra_hat)

        # Full-information null likelihood (κ = ∞ / ω = 0)
        ll_fi = if p == 0
            _riest_fullinfo_loglik(prob_hat, ykern, obs_idx, Hobs)
        else
            nll_fi = th -> begin
                try
                    pr_th = _riest_call_build(build_or_prob, val_hat, th, estimate_param)
                    llv = _riest_fullinfo_loglik(pr_th, ykern, obs_idx, Hobs)
                    isfinite(llv) ? -llv : TW(1e15)
                catch e
                    if e isa ArgumentError || e isa DomainError ||
                       e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException
                        TW(1e15)
                    else
                        rethrow()
                    end
                end
            end
            res_fi = Optim.optimize(nll_fi, copy(extra_hat), Optim.LBFGS(),
                                    Optim.Options(iterations=300, g_tol=TW(1e-6)))
            max(-TW(Optim.minimum(res_fi)), _riest_fullinfo_loglik(prob_hat, ykern, obs_idx, Hobs))
        end

        lr_stat = max(zero(TW), TW(2) * (ll_hat - ll_fi))
        lr_pval = TW(ccdf(Chisq(1), lr_stat))
        aic_val = -TW(2) * ll_hat + TW(2) * TW(k)
        bic_val = -TW(2) * ll_hat + log(TW(T_obs)) * TW(k)

        cap_out = prob_hat.units === :bits ? sol_hat.capacity_bits : sol_hat.capacity_nats
        sol_out = solve_ri_kalman(_riest_promote_prob(T, prob_hat))

        return RIEstimationResult{T}(
            Vector{T}(params_hat),
            pnames,
            Matrix{T}(vcov_nat),
            Vector{T}(se_nat),
            T(cap_out),
            T(sol_hat.omega),
            T(ll_hat),
            T(ll_fi),
            T(lr_stat),
            T(lr_pval),
            T(aic_val),
            T(bic_val),
            T_obs,
            :mle,
            estimate_param,
            sol_out,
            zeros(T, 0, 0),
            T(NaN),
            converged,
            nothing,
        )
    else
        # ---- Bayesian Adaptive RWMH ----
        rng_use = _resolve_repro_rng(rng, seed)
        prior_dists = _riest_resolve_priors(priors, k, pnames, estimate_param)

        function logprior_u(u)
            val = exp(u[1])
            lp = TW(logpdf(prior_dists[1], val)) + u[1]  # Jacobian |d(exp(u1))/du1| = exp(u1)
            isfinite(lp) || return TW(-Inf)
            for j in 2:k
                lp_j = TW(logpdf(prior_dists[j], u[j]))
                isfinite(lp_j) || return TW(-Inf)
                lp += lp_j
            end
            return lp
        end

        u_cur = copy(u0)
        lp_cur = logprior_u(u_cur)
        ll_cur = _riest_eval_u(u_cur, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
        if !isfinite(lp_cur) || !isfinite(ll_cur)
            # Perturb gently to locate finite posterior support
            found = false
            for _ in 1:60
                u_try = u0 .+ TW(0.25) .* randn(rng_use, TW, k)
                lp_try = logprior_u(u_try)
                isfinite(lp_try) || continue
                ll_try = _riest_eval_u(u_try, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
                if isfinite(ll_try)
                    u_cur = u_try
                    lp_cur = lp_try
                    ll_cur = ll_try
                    found = true
                    break
                end
            end
            found || throw(ArgumentError(
                "Could not find a finite-posterior starting point for Bayesian RWMH; check priors and initial values."))
        end
        post_cur = ll_cur + lp_cur

        scale = TW(step_scale)
        opt_c = k == 1 ? TW(1.0) : TW(2.38) / sqrt(TW(k))
        prop_L = Matrix{TW}(opt_c * Matrix{TW}(I, k, k))

        total_iters = burnin + n_draws
        burn_hist = Matrix{TW}(undef, max(burnin, 1), k)
        draws_nat = Matrix{TW}(undef, n_draws, k)

        adapt_interval = max(25, min(50, div(max(burnin, 1), 4)))
        win_acc = 0
        kept_acc = 0

        for iter in 1:total_iters
            z = randn(rng_use, TW, k)
            u_prop = u_cur + scale * (prop_L * z)
            lp_prop = logprior_u(u_prop)
            accepted = false
            if isfinite(lp_prop)
                ll_prop = _riest_eval_u(u_prop, build_or_prob, ykern, obs_idx, Hobs, estimate_param)
                if isfinite(ll_prop)
                    post_prop = ll_prop + lp_prop
                    if log(rand(rng_use, TW)) < post_prop - post_cur
                        u_cur = u_prop
                        lp_cur = lp_prop
                        ll_cur = ll_prop
                        post_cur = post_prop
                        accepted = true
                    end
                end
            end

            if iter <= burnin
                win_acc += Int(accepted)
                burn_hist[iter, :] .= u_cur
                if iter % adapt_interval == 0
                    rate = TW(win_acc) / TW(adapt_interval)
                    win_acc = 0
                    target = k == 1 ? TW(0.35) : TW(0.234)
                    if rate > target + TW(0.08)
                        scale *= TW(1.15)
                    elseif rate < target - TW(0.08)
                        scale *= TW(0.85)
                    end
                    scale = clamp(scale, TW(1e-4), TW(5.0))
                    if iter >= max(20, 2 * k + 5)
                        S_emp = _riest_symmetrize(cov(@view burn_hist[1:iter, :]) + TW(1e-6) * I)
                        prop_L = Matrix{TW}(safe_cholesky(Hermitian(opt_c^2 * S_emp); silent=true))
                    end
                end
            else
                kept_acc += Int(accepted)
                idx = iter - burnin
                draws_nat[idx, 1] = exp(u_cur[1])
                for j in 2:k
                    draws_nat[idx, j] = u_cur[j]
                end
            end
        end

        acc_rate = TW(kept_acc) / TW(n_draws)
        params_mean = vec(mean(draws_nat; dims=1))
        vcov_post = n_draws > 1 ? _riest_symmetrize(cov(draws_nat)) : zeros(TW, k, k)
        se_post = sqrt.(max.(diag(vcov_post), zero(TW)))

        val_hat = params_mean[1]
        extra_hat = p > 0 ? params_mean[2:end] : TW[]
        prob_hat = _riest_call_build(build_or_prob, val_hat, extra_hat, estimate_param)
        sol_hat = solve_ri_kalman(prob_hat; tol=TW(1e-11), max_iter=2000)
        ll_hat = _riest_kalman_loglik(sol_hat, ykern, obs_idx, Hobs)
        ll_fi = _riest_fullinfo_loglik(prob_hat, ykern, obs_idx, Hobs)
        lr_stat = max(zero(TW), TW(2) * (ll_hat - ll_fi))
        lr_pval = TW(ccdf(Chisq(1), lr_stat))
        aic_val = -TW(2) * ll_hat + TW(2) * TW(k)
        bic_val = -TW(2) * ll_hat + log(TW(T_obs)) * TW(k)

        cap_out = prob_hat.units === :bits ? sol_hat.capacity_bits : sol_hat.capacity_nats
        sol_out = solve_ri_kalman(_riest_promote_prob(T, prob_hat))

        seed_rec = seed === nothing ? nothing : Int(seed)
        man = capture_manifest(; seed=seed_rec,
            settings=Dict{String,Any}(
                "method" => "bayes",
                "estimate_param" => String(estimate_param),
                "n_draws" => n_draws,
                "burnin" => burnin,
                "step_scale" => Float64(step_scale),
            ))

        return RIEstimationResult{T}(
            Vector{T}(params_mean),
            pnames,
            Matrix{T}(vcov_post),
            Vector{T}(se_post),
            T(cap_out),
            T(sol_hat.omega),
            T(ll_hat),
            T(ll_fi),
            T(lr_stat),
            T(lr_pval),
            T(aic_val),
            T(bic_val),
            T_obs,
            :bayes,
            estimate_param,
            sol_out,
            Matrix{T}(draws_nat),
            T(acc_rate),
            isfinite(ll_hat) && sol_out.converged,
            man,
        )
    end
end

# =============================================================================
# Public API: coibion_gorodnichenko_test & implied_cg_slope
# =============================================================================

"""
    coibion_gorodnichenko_test(realized::AbstractVector,
                               forecast_now::AbstractVector,
                               forecast_lag::AbstractVector;
                               hac_lags=:auto) -> CGTestResult

Estimate the Coibion & Gorodnichenko (2015) information-rigidity regression:

```math
y_{t+h} - F_t y_{t+h} = \\alpha + \\beta_{\\text{CG}} \\left(F_t y_{t+h} - F_{t-1} y_{t+h}\\right) + e_{t+h}
```

where:
- ``y_{t+h}`` is `realized`
- ``F_t y_{t+h}`` is `forecast_now`
- ``F_{t-1} y_{t+h}`` is `forecast_lag`
- ``\\alpha`` is the regression intercept
- ``\\beta_{\\text{CG}}`` is the Coibion–Gorodnichenko slope coefficient
- ``e_{t+h}`` is the regression disturbance with Newey–West HAC standard errors

Under Full Information Rational Expectations (FIRE), ``\\beta_{\\text{CG}} = 0``.
Under noisy information with scalar AR(1) Kalman gain ``G``,
``\\beta_{\\text{CG}} = (1 - G) / G``, yielding implied gain
``\\hat{G} = 1 / (1 + \\hat{\\beta}_{\\text{CG}})`` and implied Shannon capacity
``\\hat{\\kappa}_{\\text{CG}} = -\\tfrac{1}{2} \\log_2(1 - \\hat{G})`` bits.

# Arguments
- `realized::AbstractVector`: realized outcome series ``y_{t+h}``.
- `forecast_now::AbstractVector`: current-period forecast series ``F_t y_{t+h}``.
- `forecast_lag::AbstractVector`: lagged forecast series ``F_{t-1} y_{t+h}``.

# Keyword Arguments
- `hac_lags=:auto`: Newey–West Bartlett truncation lag (`:auto` uses Andrews 1991 automatic bandwidth via `optimal_bandwidth_nw`, or pass a non-negative integer).
"""
function coibion_gorodnichenko_test(
    realized::AbstractVector{<:Real},
    forecast_now::AbstractVector{<:Real},
    forecast_lag::AbstractVector{<:Real};
    hac_lags::Union{Symbol,Integer}=:auto,
)
    n_raw = length(realized)
    (length(forecast_now) == n_raw && length(forecast_lag) == n_raw) ||
        throw(DimensionMismatch(
            "realized, forecast_now, and forecast_lag must have equal length (got $n_raw, $(length(forecast_now)), $(length(forecast_lag)))"))

    if hac_lags isa Symbol
        hac_lags === :auto || throw(ArgumentError(
            "hac_lags Symbol must be :auto, got :$hac_lags"))
    elseif hac_lags isa Integer
        hac_lags >= 0 || throw(ArgumentError("hac_lags must be >= 0, got $hac_lags"))
    end

    T = float(promote_type(eltype(realized), eltype(forecast_now), eltype(forecast_lag)))
    T <: AbstractFloat || (T = Float64)
    TW = T <: Union{Float16, Float32} ? Float64 : T

    (any(isinf, realized) || any(isinf, forecast_now) || any(isinf, forecast_lag)) &&
        throw(ArgumentError("Input series contain Inf/-Inf values"))

    keep = [isfinite(realized[i]) && isfinite(forecast_now[i]) && isfinite(forecast_lag[i])
            for i in 1:n_raw]
    n_obs = count(keep)
    n_obs >= 4 || throw(ArgumentError(
        "Need at least 4 complete (non-NaN) observations for Coibion–Gorodnichenko regression, got $n_obs"))

    y_vec = TW[TW(realized[i]) for i in 1:n_raw if keep[i]]
    f_now = TW[TW(forecast_now[i]) for i in 1:n_raw if keep[i]]
    f_lag = TW[TW(forecast_lag[i]) for i in 1:n_raw if keep[i]]

    fe = y_vec .- f_now
    fr = f_now .- f_lag

    var(fr) > eps(TW) * max(one(TW), var(fe)) || throw(ArgumentError(
        "Forecast revisions (forecast_now - forecast_lag) have near-zero variance; β_CG is unidentified"))

    X = hcat(ones(TW, n_obs), fr)
    beta_ols = (X' * X) \ (X' * fe)
    resid = fe .- X * beta_ols

    bw = hac_lags === :auto ? optimal_bandwidth_nw(resid; kernel=:bartlett) : Int(hac_lags)
    V = bw == 0 ?
        white_vcov(X, resid; variant=:hc0) :
        newey_west(X, resid; bandwidth=bw, kernel=:bartlett)

    alpha_hat = beta_ols[1]
    beta_cg = beta_ols[2]
    se_alpha = sqrt(max(V[1, 1], zero(TW)))
    se_cg = sqrt(max(V[2, 2], zero(TW)))
    t_stat = se_cg > zero(TW) ? beta_cg / se_cg : TW(NaN)
    pval = isfinite(t_stat) ? TW(2) * TW(ccdf(TDist(n_obs - 2), abs(t_stat))) : TW(NaN)

    denom_g = one(TW) + beta_cg
    implied_gain = abs(denom_g) > eps(TW) ? one(TW) / denom_g : TW(NaN)
    implied_cap = (isfinite(implied_gain) && zero(TW) < implied_gain < one(TW)) ?
        -TW(0.5) * log2(one(TW) - implied_gain) : TW(NaN)

    return CGTestResult{T}(
        T(alpha_hat),
        T(se_alpha),
        T(beta_cg),
        T(se_cg),
        T(t_stat),
        T(pval),
        n_obs,
        bw,
        T(implied_gain),
        T(implied_cap),
    )
end

"""
    implied_cg_slope(sol::SimsKalmanSolution{T};
                     horizon::Int=1,
                     weights::Union{Nothing,AbstractVector{<:Real}}=nothing) where {T<:AbstractFloat} -> T

Compute the exact population Coibion–Gorodnichenko (2015) regression slope
``\\beta_{\\text{CG}}(h)`` implied by a steady-state Rational Inattention
[`SimsKalmanSolution`](@ref) under consensus noisy-information filtering across
ex-post forecast horizon `horizon`:

```math
\\beta_{\\text{CG}}(h) = \\frac{\\operatorname{Cov}\\!\\left(v' (x_{t+h} - F_t x_{t+h}),\\; v' (F_t x_{t+h} - F_{t-1} x_{t+h})\\right)}{\\operatorname{Var}\\!\\left(v' (F_t x_{t+h} - F_{t-1} x_{t+h})\\right)}
```

where:
- ``x_{t+h}`` is the ``n \\times 1`` true state vector at ``t + h``
- ``F_t x_{t+h} = A_{\\text{cl}}^h \\bar{x}_{t|t}`` is the time-``t`` consensus forecast (with ``A_{\\text{cl}} = A + B_u F``)
- ``F_{t-1} x_{t+h} = A_{\\text{cl}}^h \\bar{x}_{t|t-1}`` is the lagged consensus forecast
- ``v`` is the ``n \\times 1`` target state linear-combination vector (`weights`, defaulting to the first column of ``H``)

For a scalar AR(1) system with steady-state Kalman gain ``G = K_{\\text{gain}} Y' = 1 - 2^{-2\\kappa}``,
this returns ``(1 - G) / G = 1 / (2^{2\\kappa} - 1)`` for all horizons ``h \\ge 1``.
Returns `NaN` when the endogenous signal rank is zero (`sol.signal_rank == 0`).

# Arguments
- `sol::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution.

# Keyword Arguments
- `horizon::Int=1`: forecast horizon ``h \\ge 1``.
- `weights=nothing`: optional length-`n` state projection vector `v` (defaults to the first non-zero column of `sol.problem.H`).
"""
function implied_cg_slope(
    sol::SimsKalmanSolution{T};
    horizon::Int=1,
    weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
    prob = sol.problem
    n = size(prob.A, 1)
    TW = T <: Union{Float16, Float32} ? Float64 : T

    v = if weights !== nothing
        length(weights) == n || throw(ArgumentError(
            "weights must have length n = $n, got $(length(weights))"))
        vw = TW[TW(w) for w in weights]
        all(isfinite, vw) || throw(ArgumentError("weights contains non-finite values"))
        norm(vw) > zero(TW) || throw(ArgumentError("weights must be non-zero"))
        vw
    else
        vh = Vector{TW}(@view prob.H[:, 1])
        if norm(vh) <= sqrt(eps(TW))
            for c in 2:size(prob.H, 2)
                vc = Vector{TW}(@view prob.H[:, c])
                if norm(vc) > sqrt(eps(TW))
                    vh = vc
                    break
                end
            end
        end
        if norm(vh) <= sqrt(eps(TW))
            vh = zeros(TW, n); vh[1] = one(TW)
        end
        vh
    end

    sol.signal_rank == 0 && return T(NaN)

    A = Matrix{TW}(prob.A)
    B = Matrix{TW}(prob.B)
    Q = _riest_symmetrize(B * B')
    F = Matrix{TW}(sol.F_policy)
    Bu = prob.B_u === nothing ? zeros(TW, n, size(prob.H, 2)) : Matrix{TW}(prob.B_u)
    Acl = A + Bu * F

    KY = Matrix{TW}(sol.K_gain) * Matrix{TW}(sol.Y_signal')
    IKY = Matrix{TW}(I, n, n) - KY
    M_cl = IKY * A

    jm = _riss_joint_matrices(sol)
    T_joint_h = Matrix{TW}(jm.T_joint)^horizon
    Tjh_11 = @view T_joint_h[1:n, 1:n]
    Acl_h = Acl^horizon

    rho_cl = maximum(abs, eigvals(M_cl); init=zero(TW))
    Sigma_cons_prior = if rho_cl < one(TW) - TW(1e-7)
        Q_cons = _riest_symmetrize(IKY * Q * IKY')
        Sigma_cons_post = _riest_symmetrize(_dlyap(M_cl, Q_cons; tol=TW(1e-13), warn_label="implied_cg_slope"))
        _riest_symmetrize(A * Sigma_cons_post * A' + Q)
    else
        Matrix{TW}(sol.Sigma_prior)
    end

    # Covariance of consensus forecast revision K Y' e_{t|t-1} and cross-covariance with e_t = (I - K Y') e_{t|t-1}
    Cov_err_rev = IKY * Sigma_cons_prior * KY'
    Cov_rev = KY * Sigma_cons_prior * KY'

    num = dot(v, Tjh_11 * Cov_err_rev * Acl_h' * v)
    denom = dot(v, Acl_h * Cov_rev * Acl_h' * v)
    denom <= eps(TW) * max(abs(num), one(TW)) && return T(NaN)
    return T(num / denom)
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, r::RIEstimationResult{T}) where {T}
    meth_str = r.method === :mle ? "Kalman MLE (L-BFGS)" : "Bayesian Adaptive RWMH"
    spec = Any[
        "Estimation method"    meth_str;
        "Primary parameter"    string(r.estimate_param);
        "Capacity κ"           "$(_fmt(r.capacity)) $(r.solution.problem.units)";
        "Shadow cost ω (/nat)" _fmt(r.omega);
        "Log-likelihood"       _fmt(r.loglik; digits=2);
        "Full-info LL (κ=∞)"   _fmt(r.loglik_fullinfo; digits=2);
        "LR stat vs FIRE"      _fmt(r.lr_stat; digits=2);
        "LR p-value"           _format_pvalue(r.lr_pvalue);
        "AIC"                  _fmt(r.aic; digits=2);
        "BIC"                  _fmt(r.bic; digits=2);
        "Observations"         r.n_obs;
        "Converged"            _yesno(r.converged);
    ]
    if r.method === :bayes
        spec = vcat(spec, Any[
            "Posterior draws"  size(r.draws, 1);
            "Acceptance rate"  _fmt(r.acceptance_rate);
        ])
    end
    _pretty_table(io, spec;
        title = "Rational Inattention Econometric Estimation",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

function report(io::IO, r::RIEstimationResult{T}; conf_level::Real=DEFAULT_CI_MACRO) where {T}
    lv = _validate_conf_level(conf_level)
    show(io, r)
    println(io)
    if r.method === :mle
        _coef_table(io, "Parameter Estimates (Delta-Method SEs)",
                    r.param_names, r.params, r.stderr; dist=:z, level=lv)
        _sig_legend(io)
    else
        q_lo, _, q_hi = _conf_level_to_quantiles(lv)
        lo_lbl = "$(_quantile_pct_label(q_lo)) CI"
        hi_lbl = "$(_quantile_pct_label(q_hi)) CI"
        k = length(r.params)
        post_tbl = Matrix{Any}(undef, k, 6)
        for j in 1:k
            col = @view r.draws[:, j]
            post_tbl[j, 1] = r.param_names[j]
            post_tbl[j, 2] = _fmt(r.params[j])
            post_tbl[j, 3] = _fmt(T(median(col)))
            post_tbl[j, 4] = _fmt(r.stderr[j])
            post_tbl[j, 5] = _fmt(T(quantile(col, q_lo)))
            post_tbl[j, 6] = _fmt(T(quantile(col, q_hi)))
        end
        _pretty_table(io, post_tbl;
            title = "Posterior Parameter Summary",
            column_labels = ["Parameter", "Mean", "Median", "Std.Dev.", lo_lbl, hi_lbl],
            alignment = [:l, :r, :r, :r, :r, :r],
        )
        _manifest_footer(io, r.manifest)
    end
    return nothing
end
report(r::RIEstimationResult; repro::Bool=false, conf_level::Real=DEFAULT_CI_MACRO) =
    report(IOContext(stdout, :repro => repro), r; conf_level=conf_level)

function Base.show(io::IO, r::CGTestResult{T}) where {T}
    spec = Any[
        "Slope β_CG"             _fmt(r.beta_cg);
        "HAC Std.Err."           _fmt(r.se_cg);
        "t-statistic (H₀: β=0)"  _fmt(r.t_stat);
        "p-value"                _format_pvalue(r.pvalue);
        "Implied Kalman gain Ĝ"  _fmt(r.implied_gain);
        "Implied capacity κ̂"     "$(_fmt(r.implied_capacity)) bits";
        "Newey–West lags"        r.hac_lags;
        "Observations"           r.n_obs;
    ]
    _pretty_table(io, spec;
        title = "Coibion–Gorodnichenko (2015) Information Rigidity Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

function report(io::IO, r::CGTestResult{T}; conf_level::Real=DEFAULT_CI_MACRO) where {T}
    lv = _validate_conf_level(conf_level)
    show(io, r)
    println(io)
    _coef_table(io, "CG Regression: y_{t+h} - F_t y_{t+h} = α + β_CG (F_t y_{t+h} - F_{t-1} y_{t+h}) + e_{t+h}",
                ["(Intercept)", "Revision (β_CG)"],
                T[r.alpha, r.beta_cg],
                T[r.se_alpha, r.se_cg];
                dist=:t, dof_r=r.n_obs - 2, level=lv)
    _sig_legend(io)
    return nothing
end
report(r::CGTestResult; conf_level::Real=DEFAULT_CI_MACRO) = report(stdout, r; conf_level=conf_level)
