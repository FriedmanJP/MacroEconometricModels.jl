# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Del Negro & Schorfheide (2004, 2007) Bayesian DSGE-VAR(λ) prior construction,
closed-form Normal-Inverse-Wishart marginal likelihood, misspecification weight selection,
conjugate/hierarchical posterior sampling, and structural DSGE-QR innovation accounting.

References:
- Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models for VARs.
  *International Economic Review*, 45(2), 643–673.
- Del Negro, M., Schorfheide, F., Smets, F., & Wouters, R. (2007). On the Fit of New
  Keynesian Models. *Journal of Business & Economic Statistics*, 25(2), 123–143.
"""

# =============================================================================
# Structs
# =============================================================================

"""
    DSGEVARPrior{T<:AbstractFloat}

Del Negro & Schorfheide (2004, §2.1–2.2) population-moment Normal-Inverse-Wishart prior
and QR structural rotation induced by a DSGE model at hyperparameter ``\\lambda``:

```math
\\Phi \\mid \\Sigma_u \\sim \\mathcal{MN}\\!\\left(\\Phi^*,\\; \\Sigma_u \\otimes (\\lambda T \\Gamma_{xx})^{-1}\\right), \\qquad
\\Sigma_u \\sim \\mathcal{IW}\\!\\left(\\lambda T \\Sigma_u^*,\\; \\lambda T - k\\right)
```

where:
- ``\\Phi^* = \\Gamma_{xx}^{-1} \\Gamma_{xy}`` is the ``k \\times n`` population `VAR(p)` coefficient projection
- ``\\Sigma_u^* = \\Gamma_{yy} - \\Gamma_{xy}' \\Phi^*`` is the ``n \\times n`` population innovation covariance
- ``\\Gamma_{yy}, \\Gamma_{xy}, \\Gamma_{xx}`` are the exact theoretical second-moment matrices of ``(y_t, x_t)``
- ``\\lambda`` is the DSGE prior weight (dummy observations ``T_d = \\lambda T``)
- ``T`` is the effective VAR sample size (`T_obs`)
- ``k`` is the number of regressors per equation (`1 + n p` with intercept, `n p` without)
- ``\\Sigma_u`` is the ``n \\times n`` reduced-form VAR innovation covariance

# Fields
- `Gamma_yy::Matrix{T}` — `n × n` theoretical second moment ``E[y_t y_t']``
- `Gamma_xy::Matrix{T}` — `k × n` theoretical cross moment ``E[x_t y_t']``
- `Gamma_xx::Matrix{T}` — `k × k` theoretical regressor second moment ``E[x_t x_t']``
- `Phi_star::Matrix{T}` — `k × n` population VAR coefficient projection
- `Sigma_u_star::Matrix{T}` — `n × n` population VAR innovation covariance
- `Omega_star::Matrix{T}` — `n × n` or `n × m` orthogonal QR structural rotation from ``D = L^* \\Omega^*``
- `lambda::T` — DSGE prior weight hyperparameter (finite ``\\ge \\lambda_{\\min}`` or `Inf`)
- `lambda_min::T` — minimum proper-prior threshold ``(n + k + 1) / T_{\\text{obs}}``
- `T_obs::Int` — effective VAR sample size ``T_{\\text{eff}}``
- `p::Int` — VAR lag order
- `obs_names::Vector{String}` — length-`n` observable variable names
- `shock_names::Vector{String}` — length-`m` (or `n`) structural shock names

# References
- Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models for VARs.
  *International Economic Review*, 45(2), 643–673.
"""
struct DSGEVARPrior{T<:AbstractFloat}
    Gamma_yy::Matrix{T}
    Gamma_xy::Matrix{T}
    Gamma_xx::Matrix{T}
    Phi_star::Matrix{T}
    Sigma_u_star::Matrix{T}
    Omega_star::Matrix{T}
    lambda::T
    lambda_min::T
    T_obs::Int
    p::Int
    obs_names::Vector{String}
    shock_names::Vector{String}

    function DSGEVARPrior{T}(
        Gamma_yy::AbstractMatrix,
        Gamma_xy::AbstractMatrix,
        Gamma_xx::AbstractMatrix,
        Phi_star::AbstractMatrix,
        Sigma_u_star::AbstractMatrix,
        Omega_star::AbstractMatrix,
        lambda::Real,
        lambda_min::Real,
        T_obs::Int,
        p::Int,
        obs_names::AbstractVector,
        shock_names::AbstractVector,
    ) where {T<:AbstractFloat}
        Gyy = Matrix{T}(Gamma_yy)
        Gxy = Matrix{T}(Gamma_xy)
        Gxx = Matrix{T}(Gamma_xx)
        Ps = Matrix{T}(Phi_star)
        Su = Matrix{T}(Sigma_u_star)
        Om = Matrix{T}(Omega_star)
        on = String[string(s) for s in obs_names]
        sn = String[string(s) for s in shock_names]

        n = size(Gyy, 1)
        k = size(Gxx, 1)
        m = size(Om, 2)
        p >= 1 || throw(ArgumentError("p must be >= 1; got $p"))
        T_obs >= 1 || throw(ArgumentError("T_obs must be >= 1; got $T_obs"))
        size(Gyy) == (n, n) || throw(ArgumentError("Gamma_yy must be square, got $(size(Gyy))"))
        size(Gxx) == (k, k) || throw(ArgumentError("Gamma_xx must be square, got $(size(Gxx))"))
        size(Gxy) == (k, n) || throw(ArgumentError("Gamma_xy must be k×n = ($k, $n), got $(size(Gxy))"))
        size(Ps) == (k, n) || throw(ArgumentError("Phi_star must be k×n = ($k, $n), got $(size(Ps))"))
        size(Su) == (n, n) || throw(ArgumentError("Sigma_u_star must be n×n = ($n, $n), got $(size(Su))"))
        size(Om, 1) == n || throw(ArgumentError("Omega_star must have n=$n rows, got $(size(Om, 1))"))
        length(on) == n || throw(ArgumentError("obs_names must have length n=$n, got $(length(on))"))
        length(sn) == m || throw(ArgumentError("shock_names must have length m=$m, got $(length(sn))"))

        new{T}(Gyy, Gxy, Gxx, Ps, Su, Om, T(lambda), T(lambda_min), T_obs, p, on, sn)
    end
end

"""
    DSGEVARLambdaProfile{T<:AbstractFloat}

Marginal likelihood profile across candidate DSGE prior weights ``\\lambda`` produced by
[`select_dsge_var_lambda`](@ref):

```math
\\hat{\\lambda} = \\arg\\max_{\\lambda \\in [\\lambda_{\\min}, \\infty]} \\log p_\\lambda(Y \\mid \\theta)
```

where:
- ``\\log p_\\lambda(Y \\mid \\theta)`` is the closed-form Normal-Inverse-Wishart log marginal likelihood
- ``\\lambda_{\\min} = (n + k + 1) / T_{\\text{obs}}`` is the minimum proper-prior weight
- ``\\hat{\\lambda}`` measures inverse DSGE misspecification (larger ``\\hat{\\lambda}`` favors DSGE restrictions)

# Fields
- `lambdas::Vector{T}` — evaluated finite ``\\lambda`` grid values
- `log_marginal_likelihoods::Vector{T}` — marginal likelihood ``\\log p_\\lambda(Y \\mid \\theta)`` on `lambdas`
- `lambda_hat::T` — optimal ``\\hat{\\lambda}`` (refined via Brent optimization or `Inf` when dogmatic limit wins)
- `logml_hat::T` — log marginal likelihood at `lambda_hat`
- `logml_inf::T` — Gaussian log-likelihood at the dogmatic DSGE limit ``\\lambda = \\infty``
- `lambda_min::T` — minimum proper-prior threshold ``(n + k + 1) / T_{\\text{obs}}``
- `p::Int` — VAR lag order
- `T_obs::Int` — effective VAR sample size ``T_{\\text{eff}}``
- `obs_names::Vector{String}` — length-`n` observable variable names
"""
struct DSGEVARLambdaProfile{T<:AbstractFloat}
    lambdas::Vector{T}
    log_marginal_likelihoods::Vector{T}
    lambda_hat::T
    logml_hat::T
    logml_inf::T
    lambda_min::T
    p::Int
    T_obs::Int
    obs_names::Vector{String}

    function DSGEVARLambdaProfile{T}(
        lambdas::AbstractVector,
        log_marginal_likelihoods::AbstractVector,
        lambda_hat::Real,
        logml_hat::Real,
        logml_inf::Real,
        lambda_min::Real,
        p::Int,
        T_obs::Int,
        obs_names::AbstractVector,
    ) where {T<:AbstractFloat}
        lams = Vector{T}(lambdas)
        lmls = Vector{T}(log_marginal_likelihoods)
        on = String[string(s) for s in obs_names]

        p >= 1 || throw(ArgumentError("p must be >= 1; got $p"))
        T_obs >= 1 || throw(ArgumentError("T_obs must be >= 1; got $T_obs"))
        length(lams) == length(lmls) || throw(ArgumentError(
            "lambdas ($(length(lams))) and log_marginal_likelihoods ($(length(lmls))) must have equal length"))
        !isempty(on) || throw(ArgumentError("obs_names cannot be empty"))

        new{T}(lams, lmls, T(lambda_hat), T(logml_hat), T(logml_inf), T(lambda_min), p, T_obs, on)
    end
end

"""
    DSGEVARPosterior{T<:AbstractFloat} <: AbstractBayesianResult

Posterior draws from Del Negro & Schorfheide (2004, 2007) Bayesian DSGE-VAR(``\\lambda``)
estimation in either fixed-``\\theta`` conjugate mode or joint hierarchical ``(\\theta, \\Phi, \\Sigma_u)``
mode:

```math
p(\\Phi, \\Sigma_u \\mid Y, \\theta, \\lambda) = p(\\Sigma_u \\mid Y, \\theta, \\lambda)\\, p(\\Phi \\mid \\Sigma_u, Y, \\theta, \\lambda), \\qquad
B_0^{(s)} = L_u^{(s)} \\Omega^*(\\theta^{(s)})
```

where:
- ``\\Phi^{(s)}`` is the `k × n` VAR coefficient draw
- ``\\Sigma_u^{(s)} = L_u^{(s)} (L_u^{(s)})'`` is the `n × n` reduced-form innovation covariance draw
- ``B_0^{(s)}`` is the `n × m` structural impact matrix rotated via the DSGE QR factor ``\\Omega^*(\\theta^{(s)})``
- ``\\theta^{(s)}`` is the deep DSGE parameter draw (present in joint mode, `nothing` in fixed-``\\theta`` mode)

# Fields
- `B_draws::Array{T,3}` — `n_draws × k × n` posterior VAR coefficient draws
- `Sigma_draws::Array{T,3}` — `n_draws × n × n` posterior innovation covariance draws
- `B0_draws::Array{T,3}` — `n_draws × n × n_shocks` DSGE-QR structural impact draws ``L_u^{(s)} \\Omega^*``
- `lambda::T` — DSGE prior weight used for estimation
- `log_marginal_likelihood::T` — closed-form (fixed-``\\theta``) or SMC/MHM (joint) log marginal likelihood
- `n_draws::Int` — number of stored posterior draws
- `p::Int` — VAR lag order
- `n::Int` — number of observable variables
- `data::Matrix{T}` — observed `T_raw × n` data matrix
- `method::Symbol` — estimation mode/sampler (`:direct`, `:dogmatic`, `:rwmh`, or `:smc`)
- `varnames::Vector{String}` — length-`n` observable variable names
- `shock_names::Vector{String}` — length-`n_shocks` structural shock names
- `theta_draws::Union{Nothing,Matrix{T}}` — `n_draws × d` DSGE parameter draws (or `nothing` in fixed-``\\theta`` mode)
- `param_names::Vector{Symbol}` — length-`d` sorted estimated DSGE parameter names (empty in fixed-``\\theta`` mode)
- `log_posterior::Union{Nothing,Vector{T}}` — length-`n_draws` log posterior kernel values at ``\\theta^{(s)}`` (or `nothing`)
- `acceptance_rate::T` — MCMC/SMC acceptance rate (`1` for direct/dogmatic samplers)
- `dsge_prior::DSGEVARPrior{T}` — reference [`DSGEVARPrior`](@ref) (evaluated at fixed ``\\theta`` or posterior mean/HPD ``\\theta``)
- `manifest::Union{ReproManifest,Nothing}` — reproducibility manifest recording seed and settings

# References
- Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models for VARs.
  *International Economic Review*, 45(2), 643–673.
- Del Negro, M., Schorfheide, F., Smets, F., & Wouters, R. (2007). On the Fit of New
  Keynesian Models. *Journal of Business & Economic Statistics*, 25(2), 123–143.
"""
struct DSGEVARPosterior{T<:AbstractFloat} <: AbstractBayesianResult
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    B0_draws::Array{T,3}
    lambda::T
    log_marginal_likelihood::T
    n_draws::Int
    p::Int
    n::Int
    data::Matrix{T}
    method::Symbol
    varnames::Vector{String}
    shock_names::Vector{String}
    theta_draws::Union{Nothing,Matrix{T}}
    param_names::Vector{Symbol}
    log_posterior::Union{Nothing,Vector{T}}
    acceptance_rate::T
    dsge_prior::DSGEVARPrior{T}
    manifest::Union{ReproManifest,Nothing}

    function DSGEVARPosterior{T}(
        B_draws::AbstractArray{<:Real,3},
        Sigma_draws::AbstractArray{<:Real,3},
        B0_draws::AbstractArray{<:Real,3},
        lambda::Real,
        log_marginal_likelihood::Real,
        n_draws::Int,
        p::Int,
        n::Int,
        data::AbstractMatrix,
        method::Union{Symbol,AbstractString},
        varnames::AbstractVector,
        shock_names::AbstractVector,
        theta_draws::Union{Nothing,AbstractMatrix},
        param_names::AbstractVector,
        log_posterior::Union{Nothing,AbstractVector},
        acceptance_rate::Real,
        dsge_prior::DSGEVARPrior{T},
        manifest::Union{ReproManifest,Nothing}=nothing,
    ) where {T<:AbstractFloat}
        Bd = Array{T,3}(B_draws)
        Sd = Array{T,3}(Sigma_draws)
        B0d = Array{T,3}(B0_draws)
        dat = Matrix{T}(data)
        msym = Symbol(method)
        vn = String[string(s) for s in varnames]
        sn = String[string(s) for s in shock_names]
        pnames = isempty(param_names) ? Symbol[] : Symbol[Symbol(s) for s in param_names]
        td = theta_draws === nothing ? nothing : Matrix{T}(theta_draws)
        lp = log_posterior === nothing ? nothing : Vector{T}(log_posterior)

        n_draws >= 1 || throw(ArgumentError("n_draws must be >= 1; got $n_draws"))
        p >= 1 || throw(ArgumentError("p must be >= 1; got $p"))
        n >= 1 || throw(ArgumentError("n must be >= 1; got $n"))
        size(Bd, 1) == n_draws || throw(ArgumentError("B_draws first dim must equal n_draws=$n_draws"))
        size(Bd, 3) == n || throw(ArgumentError("B_draws third dim must equal n=$n"))
        size(Sd) == (n_draws, n, n) || throw(ArgumentError("Sigma_draws size $(size(Sd)) != ($n_draws, $n, $n)"))
        size(B0d, 1) == n_draws && size(B0d, 2) == n ||
            throw(ArgumentError("B0_draws size $(size(B0d)) incompatible with ($n_draws, $n, m)"))
        length(vn) == n || throw(ArgumentError("varnames length must equal n=$n"))
        length(sn) == size(B0d, 3) || throw(ArgumentError("shock_names length must equal size(B0_draws, 3)"))
        if td !== nothing
            size(td, 1) == n_draws || throw(ArgumentError("theta_draws rows must equal n_draws=$n_draws"))
            size(td, 2) == length(pnames) || throw(ArgumentError("theta_draws cols must equal length(param_names)"))
        end
        if lp !== nothing
            length(lp) == n_draws || throw(ArgumentError("log_posterior length must equal n_draws=$n_draws"))
        end

        new{T}(Bd, Sd, B0d, T(lambda), T(log_marginal_likelihood),
               n_draws, p, n, dat, msym, vn, sn, td, pnames, lp, T(acceptance_rate),
               dsge_prior, manifest)
    end
end

# =============================================================================
# Internal Helpers (_dsgevar_*)
# =============================================================================

_dsgevar_eltype(abcd::ABCDSystem{T}) where {T<:AbstractFloat} = T

function _dsgevar_to_abcd(sys::ABCDSystem; observables=nothing, measurement_error=nothing,
                          reduction::Symbol=:predetermined, abcd_tol::Real=1e-10)
    base = _abcd_var_subset(sys, observables)
    if measurement_error !== nothing
        B_aug, D_aug, sn_aug = _augment_abcd_measurement_error(
            base.B, base.D, base.shock_names, base.obs_names, measurement_error)
        return ABCDSystem{eltype(base.A)}(
            base.A, B_aug, base.C, D_aug, base.c_state, base.d_obs,
            base.state_names, base.obs_names, sn_aug)
    end
    return base
end

function _dsgevar_to_abcd(sol::DSGESolution; observables=nothing, measurement_error=nothing,
                          reduction::Symbol=:predetermined, abcd_tol::Real=1e-10)
    is_determined(sol) || throw(ArgumentError(
        "DSGESolution is indeterminate or has no solution; cannot construct DSGE-VAR prior"))
    obs_arg = observables === nothing ? Symbol[] :
        (observables isa AbstractVector ? observables : [observables])
    return abcd_representation(sol, obs_arg;
                               measurement_error=measurement_error,
                               reduction=reduction,
                               tol=abcd_tol)
end

function _dsgevar_to_abcd(sol::Union{PerturbationSolution,HADSGESolution};
                          observables=nothing, measurement_error=nothing,
                          reduction::Symbol=:predetermined, abcd_tol::Real=1e-10)
    obs_arg = observables === nothing ? Symbol[] :
        (observables isa AbstractVector ? observables : [observables])
    return abcd_representation(sol, obs_arg;
                               measurement_error=measurement_error,
                               reduction=reduction,
                               tol=abcd_tol)
end

function _dsgevar_to_abcd(ss::StateSpaceModel; observables=nothing,
                          reduction::Symbol=:none, abcd_tol::Real=1e-10, kwargs...)
    obs_arg = observables === nothing ? Symbol[] :
        (observables isa AbstractVector ? observables : [observables])
    return abcd_representation(ss, obs_arg; reduction=reduction, tol=abcd_tol)
end

function _dsgevar_to_abcd(spec::ModelSpec; observables=nothing, measurement_error=nothing,
                          reduction::Symbol=:predetermined, abcd_tol::Real=1e-10,
                          solver::Symbol=:gensys, solver_kwargs::NamedTuple=NamedTuple())
    spec_ss = isempty(spec.steady_state) ? compute_steady_state(spec) : spec
    sol = solve(spec_ss; method=solver, solver_kwargs...)
    return _dsgevar_to_abcd(sol; observables=observables,
                            measurement_error=measurement_error,
                            reduction=reduction, abcd_tol=abcd_tol)
end

"""Compute exact population moments `(Gamma_yy, Gamma_xy, Gamma_xx, Phi_star, Sigma_u_star, Omega_star, shock_names)` in `WT`."""
function _dsgevar_moments(abcd::ABCDSystem{T}, p::Int, include_intercept::Bool,
                          measurement_jitter::Real;
                          unit_root_damping::Real=1e-5) where {T<:AbstractFloat}
    n = nobservables(abcd)
    n >= 1 || throw(ArgumentError("ABCDSystem must have at least 1 observable; got $n"))
    m = nshocks(abcd)
    WT = promote_type(T, Float64)
    k = include_intercept ? 1 + n * p : n * p

    Gamma0, Gammas, mu_y = _abcd_var_autocovariances(abcd, p; unit_root_damping=unit_root_damping)
    mu_outer = mu_y * transpose(mu_y)

    Gamma_yy_raw = Gamma0 .+ mu_outer
    Gamma_yy = Matrix{WT}((Gamma_yy_raw + transpose(Gamma_yy_raw)) / 2)

    Gamma_xy = zeros(WT, k, n)
    Gamma_xx = zeros(WT, k, k)
    offset = include_intercept ? 1 : 0

    if include_intercept
        Gamma_xx[1, 1] = one(WT)
        @views Gamma_xy[1, :] .= mu_y
    end

    @inbounds for j in 1:p
        cols = (offset + (j - 1) * n + 1):(offset + j * n)
        Gamma_xy[cols, :] .= transpose(Gammas[j]) .+ mu_outer
        if include_intercept
            @views Gamma_xx[1, cols] .= mu_y
            @views Gamma_xx[cols, 1] .= mu_y
        end
        for i in 1:p
            rows = (offset + (i - 1) * n + 1):(offset + i * n)
            if i == j
                Gamma_xx[rows, cols] .= Gamma0 .+ mu_outer
            elseif j > i
                Gamma_xx[rows, cols] .= Gammas[j - i] .+ mu_outer
            else
                Gamma_xx[rows, cols] .= transpose(Gammas[i - j]) .+ mu_outer
            end
        end
    end
    Gamma_xx = Matrix{WT}((Gamma_xx + transpose(Gamma_xx)) / 2)

    Gamma_xx_inv = Matrix{WT}(robust_inv(Symmetric(Gamma_xx); silent=true, rcond_tol=WT(1e-12)))
    Phi_star = Matrix{WT}(Gamma_xx_inv * Gamma_xy)

    Sigma_diff = Gamma_yy - transpose(Gamma_xy) * Phi_star
    Sigma_raw = Matrix{WT}((Sigma_diff + transpose(Sigma_diff)) / 2)

    F_s = eigen(Hermitian(Sigma_raw))
    max_ev = maximum(abs, F_s.values; init=zero(WT))
    min_ev = minimum(F_s.values)
    tol_pd = WT(1e-12) * max(max_ev, one(WT))

    if n > m || min_ev <= tol_pd
        vals = max.(F_s.values, zero(WT))
        Sigma_clean = Matrix{WT}(F_s.vectors * Diagonal(vals) * transpose(F_s.vectors))
        Sigma_sym = Matrix{WT}((Sigma_clean + transpose(Sigma_clean)) / 2)
        jit = WT(measurement_jitter)
        Sigma_u_star = jit > zero(WT) ? Matrix{WT}(Sigma_sym + jit * Matrix{WT}(I, n, n)) : Sigma_sym
    else
        Sigma_u_star = Sigma_raw
    end

    # Del Negro & Schorfheide (2004 §2.2) QR structural rotation: D_sub = L_star * Omega_star
    D_mat = Matrix{WT}(abcd.D)
    if m >= n
        F_qr = qr(Matrix{WT}(transpose(D_mat)))
        Q_R = Matrix{WT}(F_qr.Q * Matrix{WT}(I, m, n))
        R_R = Matrix{WT}(F_qr.R)
        s_diag = WT[R_R[i, i] < zero(WT) ? -one(WT) : one(WT) for i in 1:n]
        Omega_star = Matrix{WT}(Diagonal(s_diag) * transpose(Q_R))
        shock_names_out = copy(abcd.shock_names)
    else
        D_sub = hcat(D_mat, zeros(WT, n, n - m))
        F_qr = qr(Matrix{WT}(transpose(D_sub)))
        Q_R = Matrix{WT}(F_qr.Q * Matrix{WT}(I, n, n))
        R_R = Matrix{WT}(F_qr.R)
        s_diag = WT[R_R[i, i] < zero(WT) ? -one(WT) : one(WT) for i in 1:n]
        Omega_star = Matrix{WT}(Diagonal(s_diag) * transpose(Q_R))
        shock_names_out = vcat(copy(abcd.shock_names), ["me_aux_$j" for j in 1:(n - m)])
    end

    return Gamma_yy, Gamma_xy, Gamma_xx, Phi_star, Sigma_u_star, Omega_star, shock_names_out
end

"""Build `(Y_eff, X, YtY, XtY, XtX)` from data matrix `Y` matching `include_intercept`."""
function _dsgevar_data_stats(::Type{WT}, Y::AbstractMatrix, p::Int, n_expected::Int,
                              include_intercept::Bool) where {WT<:AbstractFloat}
    _validate_data(Y, "Y")
    T_raw, n_y = size(Y)
    n_y == n_expected || throw(ArgumentError(
        "Data matrix Y has $n_y columns, expected $n_expected observables"))
    T_raw > p || throw(ArgumentError("Not enough observations (T=$T_raw) for p=$p lags"))

    Yw = Matrix{WT}(Y)
    Y_eff, X_full = construct_var_matrices(Yw, p)
    X = include_intercept ? X_full : Matrix{WT}(@view X_full[:, 2:end])
    T_eff = size(Y_eff, 1)
    YtY = Matrix{WT}(transpose(Y_eff) * Y_eff)
    XtY = Matrix{WT}(transpose(X) * Y_eff)
    XtX = Matrix{WT}((transpose(X) * X + transpose(transpose(X) * X)) / 2)
    return Y_eff, X, YtY, XtY, XtX, T_eff
end

"""Infer whether a `DSGEVARPrior` includes an intercept column from `k` vs `n * p`."""
function _dsgevar_has_intercept(prior::DSGEVARPrior)
    n = length(prior.obs_names)
    k = size(prior.Gamma_xx, 1)
    if k == 1 + n * prior.p
        return true
    elseif k == n * prior.p
        return false
    else
        throw(ArgumentError(
            "Incompatible Gamma_xx dimension k=$k for n=$n observables and p=$(prior.p) lags"))
    end
end

"""Compute conjugate NIW posterior parameters in working type `WT`."""
function _dsgevar_posterior_params(
    Gamma_yy::Matrix{WT},
    Gamma_xy::Matrix{WT},
    Gamma_xx::Matrix{WT},
    Phi_star::Matrix{WT},
    Sigma_u_star::Matrix{WT},
    T_prior::Int,
    lambda_w::WT,
    Y_eff::Matrix{WT},
    X::Matrix{WT},
    XtY::Matrix{WT},
    XtX::Matrix{WT},
    T_eff::Int,
) where {WT<:AbstractFloat}
    n = size(Gamma_yy, 1)
    k = size(Gamma_xx, 1)
    T_d = lambda_w * WT(T_prior)
    nu_prior = T_d - WT(k)
    nu_post = WT(T_eff) + nu_prior

    K_prior = Matrix{WT}((T_d * Gamma_xx + transpose(T_d * Gamma_xx)) / 2)
    K_post = Matrix{WT}((K_prior + XtX + transpose(K_prior + XtX)) / 2)

    V_post_raw = Matrix{WT}(robust_inv(Symmetric(K_post); silent=true))
    V_post = Matrix{WT}((V_post_raw + transpose(V_post_raw)) / 2)
    B_post = Matrix{WT}(V_post * (T_d * Gamma_xy + XtY))

    S_prior = Matrix{WT}((T_d * Sigma_u_star + transpose(T_d * Sigma_u_star)) / 2)
    resid_data = Y_eff - X * B_post
    dB = B_post - Phi_star
    # Cancellation-free positive-definite posterior scale update equivalent to
    # (T_d * Gamma_yy + Y'Y) - B_post' K_post B_post when Sigma_u_star = Gamma_yy - Gamma_xy' Phi_star
    S_post_raw = S_prior + transpose(resid_data) * resid_data + transpose(dB) * K_prior * dB
    S_post = Matrix{WT}((S_post_raw + transpose(S_post_raw)) / 2)

    return B_post, V_post, K_prior, K_post, S_prior, S_post, nu_prior, nu_post, n, k
end

"""Evaluate closed-form NIW or Gaussian (`lambda == Inf`) log marginal likelihood."""
function _dsgevar_logml_eval(
    Gamma_yy::Matrix{WT},
    Gamma_xy::Matrix{WT},
    Gamma_xx::Matrix{WT},
    Phi_star::Matrix{WT},
    Sigma_u_star::Matrix{WT},
    T_prior::Int,
    lambda_w::WT,
    Y_eff::Matrix{WT},
    X::Matrix{WT},
    XtY::Matrix{WT},
    XtX::Matrix{WT},
    T_eff::Int,
) where {WT<:AbstractFloat}
    n = size(Gamma_yy, 1)
    k = size(Gamma_xx, 1)

    if isinf(lambda_w)
        U_star = Y_eff - X * Phi_star
        L_u = safe_cholesky(Hermitian(Sigma_u_star); silent=true)
        Z_u = L_u \ transpose(U_star)
        quad = sum(abs2, Z_u)
        logdet_S = _suppress_warnings() do
            logdet_safe(Sigma_u_star)
        end
        return -WT(0.5) * WT(T_eff * n) * log(WT(2π)) -
               WT(0.5) * WT(T_eff) * WT(logdet_S) -
               WT(0.5) * WT(quad)
    end

    lambda_w <= zero(WT) && return WT(-Inf)
    _, _, K_prior, K_post, S_prior, S_post, nu_prior, nu_post, _, _ =
        _dsgevar_posterior_params(Gamma_yy, Gamma_xy, Gamma_xx, Phi_star, Sigma_u_star,
                                  T_prior, lambda_w, Y_eff, X, XtY, XtX, T_eff)

    nu_prior <= WT(n - 1) && return WT(-Inf)

    logmvgamma(a) = WT(n * (n - 1)) / WT(4) * log(WT(π)) +
        sum(SpecialFunctions.loggamma(a + WT(1 - j) / WT(2)) for j in 1:n)

    ld_Kp = _suppress_warnings() do; logdet_safe(K_prior); end
    ld_Kpost = _suppress_warnings() do; logdet_safe(K_post); end
    ld_Sp = _suppress_warnings() do; logdet_safe(S_prior); end
    ld_Spost = _suppress_warnings() do; logdet_safe(S_post); end

    return WT(0.5) * WT(n) * WT(ld_Kp - ld_Kpost) -
           WT(0.5) * nu_post * WT(ld_Spost) +
           WT(0.5) * nu_prior * WT(ld_Sp) -
           WT(0.5) * WT(T_eff * n) * log(WT(π)) +
           WT(logmvgamma(nu_post / WT(2))) - WT(logmvgamma(nu_prior / WT(2)))
end

"""Draw `Sigma ~ IW(nu, S)` via scale-invariant triangular Bartlett decomposition (supports real `nu`)."""
function _dsgevar_draw_iw(nu::WT, L_S::AbstractMatrix{WT}, rng::AbstractRNG) where {WT<:AbstractFloat}
    n = size(L_S, 1)
    nu > WT(n - 1) || throw(ArgumentError("Inverse-Wishart dof nu=$nu must exceed n-1=$(n-1)"))
    A = zeros(WT, n, n)
    @inbounds for i in 1:n
        df_i = nu - WT(i - 1)
        A[i, i] = sqrt(WT(rand(rng, Distributions.Chisq(Float64(df_i)))))
        for j in 1:(i - 1)
            A[i, j] = randn(rng, WT)
        end
    end
    # Z = L_S * (A')^{-1} => Z Z' = L_S (A A')^{-1} L_S' ~ IW(nu, L_S L_S')
    Z = L_S / UpperTriangular(transpose(A))
    Sigma_raw = Z * transpose(Z)
    return Matrix{WT}((Sigma_raw + transpose(Sigma_raw)) / 2)
end

"""Pad `B_draws` with a zero intercept row if `include_intercept=false` when delegating to `BVARPosterior`."""
function _dsgevar_pad_intercept(B_draws::Array{T,3}, p::Int, n::Int) where {T<:AbstractFloat}
    k = size(B_draws, 2)
    k == 1 + n * p && return B_draws
    k == n * p || throw(ArgumentError("Unexpected B_draws second dimension $k for n=$n, p=$p"))
    nd = size(B_draws, 1)
    out = zeros(T, nd, 1 + n * p, n)
    out[:, 2:end, :] .= B_draws
    return out
end

function _dsgevar_pad_B(B::AbstractMatrix{T}, p::Int, n::Int) where {T<:AbstractFloat}
    k = size(B, 1)
    k == 1 + n * p && return Matrix{T}(B)
    k == n * p || throw(ArgumentError("Unexpected B first dimension $k for n=$n, p=$p"))
    return vcat(zeros(T, 1, n), Matrix{T}(B))
end

# =============================================================================
# Public API 1: dsge_var_prior
# =============================================================================

"""
    dsge_var_prior(sys_or_sol, p::Int, lambda::Real, T_obs::Int;
                   observables=nothing, include_intercept::Bool=true,
                   measurement_jitter::Real=1e-8) -> DSGEVARPrior

Construct a Del Negro & Schorfheide (2004) Normal-Inverse-Wishart prior [`DSGEVARPrior`](@ref)
from the theoretical population autocovariances of a DSGE or canonical [`ABCDSystem`](@ref):

```math
\\Gamma_{yy} = \\Gamma_y(0) + \\mu_y \\mu_y', \\qquad
\\Phi^* = \\Gamma_{xx}^{-1} \\Gamma_{xy}, \\qquad
\\Sigma_u^* = \\Gamma_{yy} - \\Gamma_{xy}' \\Phi^*, \\qquad
D = L^* \\Omega^*
```

where:
- ``\\Gamma_y(0), \\mu_y`` are the theoretical zero-lag autocovariance and unconditional mean of the observables
- ``\\Gamma_{xx}`` is the ``k \\times k`` second-moment matrix matching `construct_var_matrices(Y, p)`
- ``\\Gamma_{xy}`` is the ``k \\times n`` cross-moment matrix matching `X' Y`
- ``\\Phi^*`` is the ``k \\times n`` population `VAR(p)` slope projection
- ``\\Sigma_u^*`` is the ``n \\times n`` population innovation covariance (regularized by `measurement_jitter * I` only if `n > m` or singular)
- ``D = L^* \\Omega^*`` is the QR structural impact factorization with lower-triangular positive-diagonal ``L^*`` and orthogonal ``\\Omega^*``

Throws `ArgumentError` if ``\\lambda < \\lambda_{\\min} = (n + k + 1) / T_{\\text{obs}}`` (unless `isinf(lambda)`).

# Arguments
- `sys_or_sol`: `ABCDSystem`, `DSGESolution`, `PerturbationSolution`, `HADSGESolution`, `StateSpaceModel`, or `ModelSpec`
- `p::Int`: VAR lag order (`p >= 1`)
- `lambda::Real`: DSGE prior weight (finite ``\\ge \\lambda_{\\min}`` or `Inf`)
- `T_obs::Int`: effective VAR sample size ``T_{\\text{eff}}``

# Keywords
- `observables=nothing`: subset of observables (`Symbol`, `String`, or integer indices; `nothing` keeps defaults)
- `include_intercept::Bool=true`: prepend a constant column matching `construct_var_matrices`
- `measurement_jitter::Real=1e-8`: minimal diagonal regularization added to `Sigma_u_star` only when `n > m` or singular

# References
- Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models for VARs.
  *International Economic Review*, 45(2), 643–673.
"""
function dsge_var_prior(
    sys_or_sol::Union{ABCDSystem,DSGESolution,PerturbationSolution,HADSGESolution,StateSpaceModel,ModelSpec},
    p::Int,
    lambda::Real,
    T_obs::Int;
    observables=nothing,
    include_intercept::Bool=true,
    measurement_jitter::Real=1e-8,
)
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    T_obs >= 1 || throw(ArgumentError("Effective sample size T_obs must be >= 1; got $T_obs"))
    isnan(lambda) && throw(ArgumentError("lambda cannot be NaN"))
    (lambda > 0 || isinf(lambda)) || throw(ArgumentError("lambda must be positive (or Inf); got $lambda"))
    measurement_jitter >= 0 || throw(ArgumentError(
        "measurement_jitter must be non-negative; got $measurement_jitter"))

    abcd = _dsgevar_to_abcd(sys_or_sol; observables=observables)
    T = _dsgevar_eltype(abcd)
    WT = promote_type(T, Float64)

    n = nobservables(abcd)
    n >= 1 || throw(ArgumentError("Model must have at least 1 observable; got $n"))
    k = include_intercept ? 1 + n * p : n * p
    lambda_min_w = WT(n + k + 1) / WT(T_obs)

    if !isinf(lambda)
        lambda_w = WT(lambda)
        lambda_w < lambda_min_w * (one(WT) - WT(2) * WT(eps(T))) && throw(ArgumentError(
            "lambda=$lambda is below the proper-prior minimum lambda_min = (n + k + 1) / T_obs = $(T(lambda_min_w))"))
    end

    Gamma_yy, Gamma_xy, Gamma_xx, Phi_star, Sigma_u_star, Omega_star, shock_names_out =
        _dsgevar_moments(abcd, p, include_intercept, measurement_jitter)

    lam_T = isinf(lambda) ? T(Inf) : T(lambda)
    return DSGEVARPrior{T}(
        Matrix{T}(Gamma_yy),
        Matrix{T}(Gamma_xy),
        Matrix{T}(Gamma_xx),
        Matrix{T}(Phi_star),
        Matrix{T}(Sigma_u_star),
        Matrix{T}(Omega_star),
        lam_T,
        T(lambda_min_w),
        T_obs,
        p,
        copy(abcd.obs_names),
        shock_names_out,
    )
end

# =============================================================================
# Public API 2: dsge_var_logml & select_dsge_var_lambda
# =============================================================================

"""
    dsge_var_logml(prior::DSGEVARPrior, Y::AbstractMatrix)
    dsge_var_logml(sys_or_sol, Y::AbstractMatrix, p::Int, lambda::Real;
                   observables=nothing, include_intercept::Bool=true,
                   measurement_jitter::Real=1e-8)

Compute the closed-form Normal-Inverse-Wishart log marginal likelihood ``\\log p_\\lambda(Y \\mid \\theta)``
for a DSGE-VAR(``\\lambda``) model (Del Negro & Schorfheide 2004, eqs. 11–12):

```math
\\log p_\\lambda(Y \\mid \\theta) = \\frac{n}{2}\\left(\\log|K_0| - \\log|K_1|\\right)
- \\frac{\\nu_1}{2}\\log|S_1| + \\frac{\\nu_0}{2}\\log|S_0|
- \\frac{T_{\\text{eff}} n}{2}\\log\\pi
+ \\log\\Gamma_n\\!\\left(\\frac{\\nu_1}{2}\\right)
- \\log\\Gamma_n\\!\\left(\\frac{\\nu_0}{2}\\right)
```

where:
- ``K_0 = T_d \\Gamma_{xx}`` and ``K_1 = K_0 + X'X`` are the prior and posterior regressor precision matrices
- ``S_0 = T_d \\Sigma_u^*`` and ``S_1 = (T_d \\Gamma_{yy} + Y_{\\text{eff}}' Y_{\\text{eff}}) - B_1' K_1 B_1`` are the prior and posterior IW scale matrices
- ``\\nu_0 = T_d - k`` and ``\\nu_1 = T_{\\text{eff}} + \\nu_0`` with ``T_d = \\lambda T_{\\text{obs}}``
- ``\\Gamma_n(\\cdot)`` is the ``n``-variate multivariate Gamma function

When `isinf(lambda)`, returns the exact Gaussian VAR(``p``) log-likelihood evaluated at ``(\\Phi^*, \\Sigma_u^*)``.

# References
- Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models for VARs.
  *International Economic Review*, 45(2), 643–673.
"""
function dsge_var_logml(prior::DSGEVARPrior{Tp}, Y::AbstractMatrix) where {Tp<:AbstractFloat}
    T = promote_type(Tp, float(eltype(Y)))
    WT = promote_type(T, Float64)
    n = length(prior.obs_names)
    include_intercept = _dsgevar_has_intercept(prior)

    Y_eff, X, _, XtY, XtX, T_eff = _dsgevar_data_stats(WT, Y, prior.p, n, include_intercept)
    lml = _dsgevar_logml_eval(
        Matrix{WT}(prior.Gamma_yy),
        Matrix{WT}(prior.Gamma_xy),
        Matrix{WT}(prior.Gamma_xx),
        Matrix{WT}(prior.Phi_star),
        Matrix{WT}(prior.Sigma_u_star),
        prior.T_obs,
        WT(prior.lambda),
        Y_eff, X, XtY, XtX, T_eff,
    )
    return T(lml)
end

function dsge_var_logml(
    sys_or_sol::Union{ABCDSystem,DSGESolution,PerturbationSolution,HADSGESolution,StateSpaceModel,ModelSpec},
    Y::AbstractMatrix,
    p::Int,
    lambda::Real;
    observables=nothing,
    include_intercept::Bool=true,
    measurement_jitter::Real=1e-8,
)
    _validate_data(Y, "Y")
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    T_raw = size(Y, 1)
    T_raw > p || throw(ArgumentError("Not enough observations (T=$T_raw) for p=$p lags"))
    T_eff = T_raw - p

    prior = dsge_var_prior(sys_or_sol, p, lambda, T_eff;
                           observables=observables,
                           include_intercept=include_intercept,
                           measurement_jitter=measurement_jitter)
    return dsge_var_logml(prior, Y)
end

"""
    select_dsge_var_lambda(sol_or_sys, Y::AbstractMatrix, p::Int;
                           lambda_grid=nothing, observables=nothing,
                           refine::Bool=true, include_intercept::Bool=true,
                           measurement_jitter::Real=1e-8) -> DSGEVARLambdaProfile

Evaluate the closed-form DSGE-VAR(``\\lambda``) log marginal likelihood over a grid of
prior weights ``\\lambda`` and refine the maximizer via `Optim.Brent()`, comparing against the
dogmatic DSGE limit ``\\lambda = \\infty``:

```math
\\hat{\\lambda} = \\arg\\max_{\\lambda \\in [\\lambda_{\\min}, \\infty]} \\log p_\\lambda(Y \\mid \\theta)
```

where:
- ``\\lambda`` balances empirical VAR flexibility against DSGE cross-equation restrictions
- ``\\log p_\\lambda(Y \\mid \\theta)`` is evaluated by [`dsge_var_logml`](@ref)

# Arguments
- `sol_or_sys`: `ABCDSystem`, `DSGESolution`, `PerturbationSolution`, `HADSGESolution`, `StateSpaceModel`, or `ModelSpec`
- `Y::AbstractMatrix`: `T_raw × n` data matrix
- `p::Int`: VAR lag order

# Keywords
- `lambda_grid=nothing`: optional vector of candidate ``\\lambda`` values (defaults to a multi-scale grid from ``\\lambda_{\\min}`` to `50.0`)
- `observables=nothing`: observable subset selector
- `refine::Bool=true`: refine around the best finite grid point using `Optim.optimize(..., Optim.Brent())`
- `include_intercept::Bool=true`: include a constant term in the VAR design matrix
- `measurement_jitter::Real=1e-8`: minimal jitter if `n > m` or singular innovation covariance

# Returns
[`DSGEVARLambdaProfile`](@ref) containing the evaluated grid, marginal likelihoods, optimal
`lambda_hat`, `logml_hat`, and dogmatic `logml_inf`.
"""
function select_dsge_var_lambda(
    sol_or_sys::Union{ABCDSystem,DSGESolution,PerturbationSolution,HADSGESolution,StateSpaceModel,ModelSpec},
    Y::AbstractMatrix,
    p::Int;
    lambda_grid=nothing,
    observables=nothing,
    refine::Bool=true,
    include_intercept::Bool=true,
    measurement_jitter::Real=1e-8,
)
    _validate_data(Y, "Y")
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    T_raw = size(Y, 1)
    T_raw > p || throw(ArgumentError("Not enough observations (T=$T_raw) for p=$p lags"))
    T_eff = T_raw - p

    prior_inf = dsge_var_prior(sol_or_sys, p, Inf, T_eff;
                               observables=observables,
                               include_intercept=include_intercept,
                               measurement_jitter=measurement_jitter)
    T = promote_type(eltype(prior_inf.Gamma_yy), float(eltype(Y)))
    WT = promote_type(T, Float64)
    n = length(prior_inf.obs_names)
    lmin_w = WT(prior_inf.lambda_min)

    Gyy = Matrix{WT}(prior_inf.Gamma_yy)
    Gxy = Matrix{WT}(prior_inf.Gamma_xy)
    Gxx = Matrix{WT}(prior_inf.Gamma_xx)
    Pstar = Matrix{WT}(prior_inf.Phi_star)
    Sstar = Matrix{WT}(prior_inf.Sigma_u_star)

    Y_eff, X, _, XtY, XtX, _ = _dsgevar_data_stats(WT, Y, p, n, include_intercept)

    eval_ml(lam::Real) = _dsgevar_logml_eval(Gyy, Gxy, Gxx, Pstar, Sstar, T_eff, WT(lam),
                                             Y_eff, X, XtY, XtX, T_eff)

    logml_inf_w = eval_ml(WT(Inf))

    if lambda_grid === nothing
        base_grid = WT[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75,
                       1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5,
                       10.0, 15.0, 25.0, 50.0]
        lmin_safe = lmin_w * (one(WT) + WT(1e-6))
        lam_vec = sort!(unique!(vcat([lmin_safe], filter(>(lmin_safe), base_grid))))
        if maximum(lam_vec) < lmin_safe * WT(10)
            hi = max(WT(50.0), lmin_safe * WT(20))
            lam_vec = sort!(unique!(vcat(lam_vec, collect(range(lmin_safe, hi; length=15)))))
        end
    else
        raw = collect(lambda_grid)
        isempty(raw) && throw(ArgumentError("lambda_grid cannot be empty"))
        for v in raw
            (isnan(v) || v <= 0) && throw(ArgumentError("All lambda_grid entries must be > 0; got $v"))
        end
        finite_raw = WT[WT(v) for v in raw if isfinite(v)]
        lam_vec = sort!(unique!(filter(>=(lmin_w * (one(WT) - WT(1e-12))), finite_raw)))
        if isempty(lam_vec) && !any(isinf, raw)
            throw(ArgumentError(
                "No finite lambda_grid entry satisfies lambda >= lambda_min = $(T(lmin_w))"))
        end
    end

    lml_vec = WT[eval_ml(l) for l in lam_vec]

    if isempty(lam_vec)
        return DSGEVARLambdaProfile{T}(
            T[], T[], T(Inf), T(logml_inf_w), T(logml_inf_w), T(lmin_w),
            p, T_eff, copy(prior_inf.obs_names))
    end

    idx_best = argmax(lml_vec)
    best_lam_w = lam_vec[idx_best]
    best_ml_w = lml_vec[idx_best]

    if refine && length(lam_vec) >= 2
        lo_b = idx_best == 1 ? lam_vec[1] : lam_vec[idx_best - 1]
        hi_b = idx_best == length(lam_vec) ? lam_vec[end] : lam_vec[idx_best + 1]
        if hi_b > lo_b
            res = Optim.optimize(lam -> -eval_ml(WT(lam)), Float64(lo_b), Float64(hi_b), Optim.Brent())
            cand_lam = WT(Optim.minimizer(res))
            cand_ml = eval_ml(cand_lam)
            if isfinite(cand_ml) && cand_ml > best_ml_w
                best_lam_w = cand_lam
                best_ml_w = cand_ml
            end
        end
    end

    if isfinite(logml_inf_w) && logml_inf_w > best_ml_w
        lam_hat_w = WT(Inf)
        ml_hat_w = logml_inf_w
    else
        lam_hat_w = best_lam_w
        ml_hat_w = best_ml_w
    end

    return DSGEVARLambdaProfile{T}(
        Vector{T}(lam_vec),
        Vector{T}(lml_vec),
        T(lam_hat_w),
        T(ml_hat_w),
        T(logml_inf_w),
        T(lmin_w),
        p,
        T_eff,
        copy(prior_inf.obs_names),
    )
end

# =============================================================================
# Public API 3: estimate_dsge_var (Fixed-θ Mode and Joint (θ, Φ, Σ_u) Mode)
# =============================================================================

"""
    estimate_dsge_var(sol_or_sys, Y::AbstractMatrix, p::Int;
                      lambda::Union{Symbol,Real}=:auto, observables=nothing,
                      include_intercept::Bool=true, measurement_jitter::Real=1e-8,
                      n_draws::Int=1000, rng=Random.default_rng(), seed=nothing) -> DSGEVARPosterior

    estimate_dsge_var(spec::ModelSpec, Y::AbstractMatrix, p::Int;
                      priors::Union{DSGEPrior,Dict}, lambda::Union{Symbol,Real}=1.0,
                      method::Symbol=:rwmh, observables=nothing,
                      include_intercept::Bool=true, measurement_jitter::Real=1e-8,
                      n_draws::Int=400, burnin::Int=150, n_particles::Int=150,
                      step_scale::Real=0.05, solver::Symbol=:gensys,
                      solver_kwargs::NamedTuple=NamedTuple(),
                      rng=Random.default_rng(), seed=nothing) -> DSGEVARPosterior

Estimate a Bayesian DSGE-VAR(``\\lambda``) following Del Negro & Schorfheide (2004, 2007):

1. **Fixed-``\\theta`` mode** (given a solved model or [`ABCDSystem`](@ref)): draws i.i.d.
   samples from the conjugate Normal-Inverse-Wishart posterior (or returns deterministic
   dogmatic draws at ``(\\Phi^*, \\Sigma_u^*, L_u^* \\Omega^*)`` when ``\\lambda = \\infty``):

```math
\\Sigma_u^{(s)} \\mid Y, \\theta \\sim \\mathcal{IW}(S_1, \\nu_1), \\qquad
\\Phi^{(s)} \\mid \\Sigma_u^{(s)}, Y, \\theta \\sim \\mathcal{MN}\\!\\left(B_1, \\Sigma_u^{(s)} \\otimes K_1^{-1}\\right), \\qquad
B_0^{(s)} = L_u^{(s)} \\Omega^*
```

where:
- ``\\Sigma_u^{(s)} = L_u^{(s)} (L_u^{(s)})'`` is the `n × n` covariance draw with lower-triangular Cholesky factor ``L_u^{(s)}``
- ``\\Phi^{(s)}`` is the `k × n` VAR coefficient draw
- ``B_0^{(s)}`` is the `n × m` structural impact matrix rotated via the DSGE QR matrix ``\\Omega^*``
- ``B_1, K_1, S_1, \\nu_1`` are the conjugate Normal-Inverse-Wishart posterior parameters

2. **Joint ``(\\theta, \\Phi, \\Sigma_u)`` hierarchical Bayes mode** (given a `ModelSpec` and
   `priors`): samples structural parameters from the marginal posterior
   ``p(\\theta \\mid Y, \\lambda) \\propto p_\\lambda(Y \\mid \\theta)\\, p(\\theta)`` via
   `:rwmh` or `:smc`, then draws conditional ``(\\Phi^{(s)}, \\Sigma_u^{(s)}, B_0^{(s)})`` for
   each posterior parameter draw ``\\theta^{(s)}``.

# Keywords (Fixed-``\\theta``)
- `lambda::Union{Symbol,Real}=:auto`: DSGE prior weight (`:auto` runs [`select_dsge_var_lambda`](@ref), or supply a finite ``\\lambda \\ge \\lambda_{\\min}`` or `Inf`)
- `observables=nothing`: observable subset selector
- `include_intercept::Bool=true`: include constant intercept column matching `construct_var_matrices`
- `measurement_jitter::Real=1e-8`: minimal regularization added to `Sigma_u_star` only if `n > m` or singular
- `n_draws::Int=1000`: number of posterior draws
- `rng::AbstractRNG=Random.default_rng()`: random number generator
- `seed::Union{Integer,Nothing}=nothing`: optional RNG seed recorded in `manifest`

# Additional Keywords (Joint ``(\\theta, \\Phi, \\Sigma_u)`` on `ModelSpec`)
- `priors::Union{DSGEPrior,Dict}`: prior specification over estimated DSGE parameters
- `method::Symbol=:rwmh`: posterior sampler over ``\\theta`` (`:rwmh` or `:smc`)
- `burnin::Int=150`: burn-in draws discarded by `:rwmh`
- `n_particles::Int=150`: SMC particle count for `method=:smc`
- `step_scale::Real=0.05`: proposal step scale in unconstrained parameter space for `:rwmh`
- `solver::Symbol=:gensys`: DSGE linear rational-expectations solver (`:gensys`, `:blanchard_kahn`, `:klein`, etc.)
- `solver_kwargs::NamedTuple=NamedTuple()`: keyword arguments forwarded to `solve`

# References
- Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models for VARs.
  *International Economic Review*, 45(2), 643–673.
- Del Negro, M., Schorfheide, F., Smets, F., & Wouters, R. (2007). On the Fit of New
  Keynesian Models. *Journal of Business & Economic Statistics*, 25(2), 123–143.
"""
function estimate_dsge_var(
    sol_or_sys::Union{ABCDSystem,DSGESolution,PerturbationSolution,HADSGESolution,StateSpaceModel},
    Y::AbstractMatrix,
    p::Int;
    lambda::Union{Symbol,Real}=:auto,
    observables=nothing,
    include_intercept::Bool=true,
    measurement_jitter::Real=1e-8,
    n_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
)
    _validate_data(Y, "Y")
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be >= 1; got $n_draws"))
    T_raw = size(Y, 1)
    T_raw > p + 1 || throw(ArgumentError("Not enough observations (T=$T_raw) for p=$p lags"))
    T_eff = T_raw - p

    rng = _resolve_repro_rng(rng, seed)

    if lambda isa Symbol
        lambda === :auto || throw(ArgumentError("lambda Symbol must be :auto; got :$lambda"))
        prof = select_dsge_var_lambda(sol_or_sys, Y, p;
                                      observables=observables,
                                      include_intercept=include_intercept,
                                      measurement_jitter=measurement_jitter)
        lam_val = prof.lambda_hat
        logml_val = prof.logml_hat
    else
        lam_val = lambda
        logml_val = nothing
    end

    prior = dsge_var_prior(sol_or_sys, p, lam_val, T_eff;
                           observables=observables,
                           include_intercept=include_intercept,
                           measurement_jitter=measurement_jitter)
    T = promote_type(eltype(prior.Gamma_yy), float(eltype(Y)))
    WT = promote_type(T, Float64)
    n = length(prior.obs_names)
    k = size(prior.Gamma_xx, 1)
    m = size(prior.Omega_star, 2)

    Y_eff, X, _, XtY, XtX, _ = _dsgevar_data_stats(WT, Y, p, n, include_intercept)
    prior_T = DSGEVARPrior{T}(
        Matrix{T}(prior.Gamma_yy), Matrix{T}(prior.Gamma_xy), Matrix{T}(prior.Gamma_xx),
        Matrix{T}(prior.Phi_star), Matrix{T}(prior.Sigma_u_star), Matrix{T}(prior.Omega_star),
        T(prior.lambda), T(prior.lambda_min), prior.T_obs, prior.p,
        copy(prior.obs_names), copy(prior.shock_names),
    )

    if logml_val === nothing
        logml_val = dsge_var_logml(prior_T, Y)
    end

    B_draws = Array{T,3}(undef, n_draws, k, n)
    Sigma_draws = Array{T,3}(undef, n_draws, n, n)
    B0_draws = Array{T,3}(undef, n_draws, n, m)

    Omega_w = Matrix{WT}(prior.Omega_star)

    if isinf(lam_val)
        Phi_star_T = Matrix{T}(prior.Phi_star)
        Sigma_star_w = Matrix{WT}(prior.Sigma_u_star)
        Sigma_star_T = Matrix{T}(Sigma_star_w)
        L_star = Matrix{WT}(safe_cholesky(Hermitian(Sigma_star_w); silent=true))
        B0_star_T = Matrix{T}(L_star * Omega_w)
        @inbounds for s in 1:n_draws
            B_draws[s, :, :] = Phi_star_T
            Sigma_draws[s, :, :] = Sigma_star_T
            B0_draws[s, :, :] = B0_star_T
        end
        method_sym = :dogmatic
    else
        B_post, V_post, _, _, _, S_post, _, nu_post, _, _ =
            _dsgevar_posterior_params(
                Matrix{WT}(prior.Gamma_yy),
                Matrix{WT}(prior.Gamma_xy),
                Matrix{WT}(prior.Gamma_xx),
                Matrix{WT}(prior.Phi_star),
                Matrix{WT}(prior.Sigma_u_star),
                prior.T_obs,
                WT(lam_val),
                Y_eff, X, XtY, XtX, T_eff,
            )
        L_V = Matrix{WT}(safe_cholesky(Hermitian(V_post); silent=true))
        L_S = Matrix{WT}(safe_cholesky(Hermitian(S_post); silent=true))
        Z_buf = Matrix{WT}(undef, k, n)

        @inbounds for s in 1:n_draws
            Sigma_s = _dsgevar_draw_iw(nu_post, L_S, rng)
            L_Sigma = Matrix{WT}(safe_cholesky(Hermitian(Sigma_s); silent=true))
            randn!(rng, Z_buf)
            Phi_s = B_post + L_V * Z_buf * transpose(L_Sigma)
            B0_s = L_Sigma * Omega_w

            B_draws[s, :, :] = Matrix{T}(Phi_s)
            Sigma_draws[s, :, :] = Matrix{T}(Sigma_s)
            B0_draws[s, :, :] = Matrix{T}(B0_s)
        end
        method_sym = :direct
    end

    manifest = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "mode" => "fixed",
        "method" => String(method_sym),
        "lambda" => Float64(lam_val),
        "n_draws" => n_draws,
        "p" => p,
        "include_intercept" => include_intercept,
    ))

    return DSGEVARPosterior{T}(
        B_draws, Sigma_draws, B0_draws,
        T(lam_val), T(logml_val),
        n_draws, p, n, Matrix{T}(Y),
        method_sym, copy(prior_T.obs_names), copy(prior_T.shock_names),
        nothing, Symbol[], nothing, one(T),
        prior_T, manifest,
    )
end

# -----------------------------------------------------------------------------
# Joint (θ, Φ, Σ_u) Hierarchical Bayes Mode on ModelSpec
# -----------------------------------------------------------------------------

"""Solve `spec` at `theta` catching only benign numerical/solve errors."""
function _dsgevar_solve_at_theta(
    spec::ModelSpec{Ts},
    param_names::Vector{Symbol},
    theta::AbstractVector,
    solver::Symbol,
    solver_kwargs::NamedTuple,
) where {Ts<:AbstractFloat}
    new_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        new_pv[pn] = Ts(theta[i])
    end
    new_spec = _respec(spec, new_pv)
    sol = try
        ss_spec = compute_steady_state(new_spec)
        solve(ss_spec; method=solver, solver_kwargs...)
    catch e
        if e isa DSGESolveError || e isa LinearAlgebra.SingularException ||
           e isa LinearAlgebra.PosDefException || e isa LinearAlgebra.LAPACKException
            return nothing
        end
        rethrow(e)
    end
    if sol isa DSGESolution && !is_determined(sol)
        return nothing
    end
    return sol
end

"""Evaluate `(prior_moments, logml)` at `theta`; return `nothing` if solve/determinacy/stability fails."""
function _dsgevar_eval_theta(
    spec::ModelSpec{Ts},
    param_names::Vector{Symbol},
    theta::AbstractVector,
    obs_syms::Vector{Symbol},
    p::Int,
    lam_w::WT,
    include_intercept::Bool,
    measurement_jitter::Real,
    Y_eff::Matrix{WT},
    X::Matrix{WT},
    XtY::Matrix{WT},
    XtX::Matrix{WT},
    T_eff::Int,
    solver::Symbol,
    solver_kwargs::NamedTuple,
) where {Ts<:AbstractFloat,WT<:AbstractFloat}
    sol = _dsgevar_solve_at_theta(spec, param_names, theta, solver, solver_kwargs)
    sol === nothing && return nothing
    abcd = _dsgevar_to_abcd(sol; observables=obs_syms)
    if nstates(abcd) > 0
        max_eig = maximum(abs, eigvals(Matrix{WT}(abcd.A)); init=zero(WT))
        max_eig > one(WT) + WT(1e-4) && return nothing
    end
    Gyy, Gxy, Gxx, Pstar, Sstar, Ostar, sn =
        _dsgevar_moments(abcd, p, include_intercept, measurement_jitter)
    lml = _dsgevar_logml_eval(Gyy, Gxy, Gxx, Pstar, Sstar, T_eff, lam_w,
                              Y_eff, X, XtY, XtX, T_eff)
    isfinite(lml) || return nothing
    return (Gyy=Gyy, Gxy=Gxy, Gxx=Gxx, Phi_star=Pstar, Sigma_u_star=Sstar,
            Omega_star=Ostar, obs_names=abcd.obs_names, shock_names=sn, logml=lml)
end

function _dsgevar_normalize_prior(priors::Dict, ::Type{T}) where {T<:AbstractFloat}
    isempty(priors) && throw(ArgumentError("priors cannot be empty"))
    dp = Dict{Symbol,Distribution}(Symbol(k) => v for (k, v) in priors)
    pr = _build_bayes_prior(dp)
    return DSGEPrior{T}(pr.param_names, pr.distributions, Vector{T}(pr.lower), Vector{T}(pr.upper))
end

function _dsgevar_normalize_prior(priors::DSGEPrior, ::Type{T}) where {T<:AbstractFloat}
    return DSGEPrior{T}(copy(priors.param_names), copy(priors.distributions),
                        Vector{T}(priors.lower), Vector{T}(priors.upper))
end

function _dsgevar_check_spec_estimable(spec::ModelSpec)
    if has_kind(spec, FirmSystem) || has_kind(spec, IntermediarySystem) ||
       has_kind(spec, DCEGMSystem) || has_kind(spec, LifeCycleSystem) ||
       has_kind(spec, ContinuousHouseholdSystem) ||
       (spec.n_endog == 0 && !has_kind(spec, HouseholdSystem))
        throw(ArgumentError(
            "estimate_dsge_var does not support this ModelSpec agent family"))
    end
    return nothing
end

function estimate_dsge_var(
    spec::ModelSpec{Ts},
    Y::AbstractMatrix,
    p::Int;
    priors::Union{DSGEPrior,Dict},
    lambda::Union{Symbol,Real}=1.0,
    method::Symbol=:rwmh,
    observables=nothing,
    include_intercept::Bool=true,
    measurement_jitter::Real=1e-8,
    n_draws::Int=400,
    burnin::Int=150,
    n_particles::Int=150,
    step_scale::Real=0.05,
    solver::Symbol=:gensys,
    solver_kwargs::NamedTuple=NamedTuple(),
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
) where {Ts<:AbstractFloat}
    _dsgevar_check_spec_estimable(spec)
    _validate_data(Y, "Y")
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    method in (:rwmh, :smc) || throw(ArgumentError(
        "method must be :rwmh or :smc; got :$method"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be >= 1; got $n_draws"))
    burnin >= 0 || throw(ArgumentError("burnin must be >= 0; got $burnin"))
    n_particles >= 1 || throw(ArgumentError("n_particles must be >= 1; got $n_particles"))
    step_scale > 0 || throw(ArgumentError("step_scale must be > 0; got $step_scale"))
    measurement_jitter >= 0 || throw(ArgumentError("measurement_jitter must be >= 0; got $measurement_jitter"))

    T_raw, n_y = size(Y)
    T_raw > p + 1 || throw(ArgumentError("Not enough observations (T=$T_raw) for p=$p lags"))
    T_eff = T_raw - p

    T = promote_type(Ts, float(eltype(Y)))
    WT = promote_type(T, Float64)
    rng = _resolve_repro_rng(rng, seed)

    prior_T = _dsgevar_normalize_prior(priors, T)
    prior_W = _dsgevar_normalize_prior(priors, WT)
    param_names = prior_T.param_names
    d = length(param_names)
    d >= 1 || throw(ArgumentError("priors must contain at least one parameter"))
    for pn in param_names
        haskey(spec.param_values, pn) || throw(ArgumentError(
            "Prior parameter :$pn not found in spec.param_values"))
    end

    # Resolve & validate observables eagerly before running any sampler
    default_obs = spec.augmented ? spec.original_endog : spec.endog
    obs_sel = observables === nothing ? default_obs :
        (observables isa AbstractVector ? observables : [observables])
    _, obs_str_names = _resolve_abcd_observables(spec.endog, default_obs, obs_sel)
    obs_syms = Symbol.(obs_str_names)
    n = length(obs_syms)
    n_y == n || throw(ArgumentError(
        "Data matrix Y has $n_y columns, expected n=$n observables"))

    k = include_intercept ? 1 + n * p : n * p
    lmin_w = WT(n + k + 1) / WT(T_eff)

    # Resolve lambda
    if lambda isa Symbol
        lambda === :auto || throw(ArgumentError("lambda Symbol must be :auto; got :$lambda"))
        sol0 = _dsgevar_solve_at_theta(spec, param_names, [spec.param_values[pn] for pn in param_names], solver, solver_kwargs)
        sol0 === nothing && throw(ArgumentError(
            "Cannot resolve lambda=:auto: baseline ModelSpec failed to solve or is indeterminate"))
        lam_w = WT(select_dsge_var_lambda(sol0, Y, p;
                                          observables=obs_syms,
                                          include_intercept=include_intercept,
                                          measurement_jitter=measurement_jitter).lambda_hat)
    else
        isnan(lambda) && throw(ArgumentError("lambda cannot be NaN"))
        (lambda > 0 || isinf(lambda)) || throw(ArgumentError("lambda must be positive (or Inf); got $lambda"))
        lam_w = WT(lambda)
        if !isinf(lam_w) && lam_w < lmin_w * (one(WT) - WT(2) * WT(eps(T)))
            throw(ArgumentError(
                "lambda=$lambda is below the proper-prior minimum lambda_min = (n + k + 1) / T_obs = $(T(lmin_w))"))
        end
    end

    Y_eff, X, _, XtY, XtX, _ = _dsgevar_data_stats(WT, Y, p, n, include_intercept)

    function ll_fn(theta::AbstractVector)
        res = _dsgevar_eval_theta(spec, param_names, theta, obs_syms, p, lam_w,
                                  include_intercept, measurement_jitter,
                                  Y_eff, X, XtY, XtX, T_eff, solver, solver_kwargs)
        return res === nothing ? WT(-Inf) : res.logml
    end

    # Initial theta inside prior support and determinacy region
    theta_init = WT[WT(spec.param_values[pn]) for pn in param_names]
    pt = ParameterTransform(prior_W.lower, prior_W.upper)
    for i in 1:d
        lo, hi = prior_W.lower[i], prior_W.upper[i]
        span = isfinite(lo) && isfinite(hi) ? hi - lo : one(WT)
        margin = WT(1e-8) * span
        isfinite(lo) && theta_init[i] <= lo && (theta_init[i] = lo + margin)
        isfinite(hi) && theta_init[i] >= hi && (theta_init[i] = hi - margin)
    end
    lp_init = _log_prior(theta_init, prior_W)
    ll_init = isfinite(lp_init) ? ll_fn(theta_init) : WT(-Inf)

    if !isfinite(ll_init) || !isfinite(lp_init)
        found = false
        for _ in 1:100
            cand = WT[WT(rand(rng, prior_W.distributions[i])) for i in 1:d]
            lp_c = _log_prior(cand, prior_W)
            isfinite(lp_c) || continue
            ll_c = ll_fn(cand)
            if isfinite(ll_c)
                theta_init = cand
                lp_init = lp_c
                ll_init = ll_c
                found = true
                break
            end
        end
        found || throw(ArgumentError(
            "Initial parameters and 100 prior draws all failed to yield a finite DSGE-VAR log posterior"))
    end

    # Sample θ | Y, λ
    theta_draws_w = Matrix{WT}(undef, n_draws, d)
    log_post_w = Vector{WT}(undef, n_draws)
    acc_rate_w = zero(WT)
    logml_est_w = zero(WT)

    if method === :rwmh
        # Unconstrained-space proposal scaled by step_scale (and prior SD on unbounded coords)
        prop_sd = Vector{WT}(undef, d)
        for i in 1:d
            lo, hi = prior_W.lower[i], prior_W.upper[i]
            if isfinite(lo) || isfinite(hi)
                prop_sd[i] = WT(step_scale)
            else
                sd_i = try
                    WT(std(prior_W.distributions[i]))
                catch
                    one(WT)
                end
                prop_sd[i] = WT(step_scale) * (isfinite(sd_i) && sd_i > 0 ? sd_i : one(WT))
            end
        end

        y_curr = to_unconstrained(pt, theta_init)
        theta_curr = copy(theta_init)
        ll_curr = ll_init
        lp_curr = lp_init
        lj_curr = log_jacobian(pt, y_curr)

        total_iters = burnin + n_draws
        accepted_post = 0
        accepted_all = 0

        for it in 1:total_iters
            y_star = y_curr .+ prop_sd .* randn(rng, WT, d)
            theta_star = to_constrained(pt, y_star)
            lp_star = _log_prior(theta_star, prior_W)
            if isfinite(lp_star)
                ll_star = ll_fn(theta_star)
                if isfinite(ll_star)
                    lj_star = log_jacobian(pt, y_star)
                    log_alpha = (ll_star + lp_star + lj_star) - (ll_curr + lp_curr + lj_curr)
                    if log(rand(rng, WT)) < log_alpha
                        y_curr = y_star
                        theta_curr = theta_star
                        ll_curr = ll_star
                        lp_curr = lp_star
                        lj_curr = lj_star
                        accepted_all += 1
                        it > burnin && (accepted_post += 1)
                    end
                end
            end

            if it > burnin
                idx = it - burnin
                theta_draws_w[idx, :] = theta_curr
                log_post_w[idx] = ll_curr + lp_curr
            end
        end

        acc_rate_w = WT(accepted_post) / WT(n_draws)
        logml_est_w = _suppress_warnings() do
            _geweke_mhm(theta_draws_w, log_post_w)
        end
        if !isfinite(logml_est_w)
            logml_est_w = _logsumexp(log_post_w) - log(WT(n_draws))
        end
    else # :smc
        smc_state = _smc_from_likelihood(ll_fn, param_names, prior_W, theta_init;
                                         n_smc=n_particles, n_mh_steps=1,
                                         ess_target=0.5, max_stages=500,
                                         min_dphi=1e-10, rng=rng)
        acc_rate_w = isempty(smc_state.acceptance_rates) ? zero(WT) : last(smc_state.acceptance_rates)
        logml_est_w = smc_state.log_marginal_likelihood
        lp_smc = smc_state.log_likelihoods .+ smc_state.log_priors

        if n_draws == n_particles
            theta_draws_w .= transpose(smc_state.theta_particles)
            log_post_w .= lp_smc
        elseif n_draws < n_particles
            sel = randperm(rng, n_particles)[1:n_draws]
            for (i, idx) in enumerate(sel)
                theta_draws_w[i, :] = @view smc_state.theta_particles[:, idx]
                log_post_w[i] = lp_smc[idx]
            end
        else
            sel = rand(rng, 1:n_particles, n_draws)
            for (i, idx) in enumerate(sel)
                theta_draws_w[i, :] = @view smc_state.theta_particles[:, idx]
                log_post_w[i] = lp_smc[idx]
            end
        end
    end

    # Conditional draws (Φ^(s), Σ_u^(s), B0^(s)) | θ^(s), Y, λ (caching consecutive equal θ^(s))
    first_eval = _dsgevar_eval_theta(spec, param_names, theta_draws_w[1, :], obs_syms, p, lam_w,
                                     include_intercept, measurement_jitter,
                                     Y_eff, X, XtY, XtX, T_eff, solver, solver_kwargs)
    first_eval === nothing && throw(ArgumentError(
        "Unexpected solve failure at accepted posterior draw"))
    m = size(first_eval.Omega_star, 2)

    B_draws = Array{T,3}(undef, n_draws, k, n)
    Sigma_draws = Array{T,3}(undef, n_draws, n, n)
    B0_draws = Array{T,3}(undef, n_draws, n, m)
    Z_buf = Matrix{WT}(undef, k, n)

    cached_theta = Vector{WT}(undef, d)
    fill!(cached_theta, WT(NaN))
    cached_eval = first_eval
    cached_B_post = Matrix{WT}(undef, 0, 0)
    cached_L_V = Matrix{WT}(undef, 0, 0)
    cached_L_S = Matrix{WT}(undef, 0, 0)
    cached_nu_post = zero(WT)
    cached_Omega = Matrix{WT}(undef, 0, 0)

    for s in 1:n_draws
        th_s = @view theta_draws_w[s, :]
        if th_s != cached_theta
            ev = _dsgevar_eval_theta(spec, param_names, th_s, obs_syms, p, lam_w,
                                     include_intercept, measurement_jitter,
                                     Y_eff, X, XtY, XtX, T_eff, solver, solver_kwargs)
            if ev !== nothing
                cached_eval = ev
                copyto!(cached_theta, th_s)
                cached_Omega = ev.Omega_star
                if !isinf(lam_w)
                    Bp, Vp, _, _, _, Sp, _, nup, _, _ =
                        _dsgevar_posterior_params(
                            ev.Gyy, ev.Gxy, ev.Gxx, ev.Phi_star, ev.Sigma_u_star,
                            T_eff, lam_w, Y_eff, X, XtY, XtX, T_eff,
                        )
                    cached_B_post = Bp
                    cached_L_V = Matrix{WT}(safe_cholesky(Hermitian(Vp); silent=true))
                    cached_L_S = Matrix{WT}(safe_cholesky(Hermitian(Sp); silent=true))
                    cached_nu_post = nup
                end
            end
        end

        if isinf(lam_w)
            L_u = Matrix{WT}(safe_cholesky(Hermitian(cached_eval.Sigma_u_star); silent=true))
            B_draws[s, :, :] = Matrix{T}(cached_eval.Phi_star)
            Sigma_draws[s, :, :] = Matrix{T}(cached_eval.Sigma_u_star)
            B0_draws[s, :, :] = Matrix{T}(L_u * cached_Omega)
        else
            Sigma_s = _dsgevar_draw_iw(cached_nu_post, cached_L_S, rng)
            L_Sigma = Matrix{WT}(safe_cholesky(Hermitian(Sigma_s); silent=true))
            randn!(rng, Z_buf)
            Phi_s = cached_B_post + cached_L_V * Z_buf * transpose(L_Sigma)
            B0_s = L_Sigma * cached_Omega

            B_draws[s, :, :] = Matrix{T}(Phi_s)
            Sigma_draws[s, :, :] = Matrix{T}(Sigma_s)
            B0_draws[s, :, :] = Matrix{T}(B0_s)
        end
    end

    # Reference DSGEVARPrior at posterior mean (fallback to highest-posterior draw)
    theta_mean = vec(mean(theta_draws_w; dims=1))
    ref_eval = _dsgevar_eval_theta(spec, param_names, theta_mean, obs_syms, p, lam_w,
                                   include_intercept, measurement_jitter,
                                   Y_eff, X, XtY, XtX, T_eff, solver, solver_kwargs)
    if ref_eval === nothing
        hpd_idx = argmax(log_post_w)
        ref_eval = _dsgevar_eval_theta(spec, param_names, theta_draws_w[hpd_idx, :], obs_syms, p, lam_w,
                                       include_intercept, measurement_jitter,
                                       Y_eff, X, XtY, XtX, T_eff, solver, solver_kwargs)
        ref_eval === nothing && (ref_eval = first_eval)
    end

    ref_prior = DSGEVARPrior{T}(
        Matrix{T}(ref_eval.Gyy), Matrix{T}(ref_eval.Gxy), Matrix{T}(ref_eval.Gxx),
        Matrix{T}(ref_eval.Phi_star), Matrix{T}(ref_eval.Sigma_u_star),
        Matrix{T}(ref_eval.Omega_star),
        T(lam_w), T(lmin_w), T_eff, p,
        copy(ref_eval.obs_names), copy(ref_eval.shock_names),
    )

    manifest = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "mode" => "joint",
        "method" => String(method),
        "lambda" => Float64(lam_w),
        "n_draws" => n_draws,
        "burnin" => burnin,
        "n_particles" => n_particles,
        "step_scale" => Float64(step_scale),
        "p" => p,
        "include_intercept" => include_intercept,
        "solver" => String(solver),
    ))

    return DSGEVARPosterior{T}(
        B_draws, Sigma_draws, B0_draws,
        T(lam_w), T(logml_est_w),
        n_draws, p, n, Matrix{T}(Y),
        method, copy(ref_prior.obs_names), copy(ref_prior.shock_names),
        Matrix{T}(theta_draws_w), copy(param_names), Vector{T}(log_post_w),
        T(acc_rate_w), ref_prior, manifest,
    )
end

# =============================================================================
# Ecosystem Delegation & Innovation Accounting on DSGEVARPosterior
# =============================================================================

function _to_bvar_posterior(post::DSGEVARPosterior{T}) where {T<:AbstractFloat}
    B_pad = _dsgevar_pad_intercept(post.B_draws, post.p, post.n)
    return BVARPosterior{T}(
        B_pad,
        post.Sigma_draws,
        post.n_draws,
        post.p,
        post.n,
        post.data,
        :dsge_var,
        post.method,
        post.varnames;
        manifest=post.manifest,
    )
end

Base.size(post::DSGEVARPosterior, dim::Int) =
    dim == 1 ? post.n_draws : error("DSGEVARPosterior has 1 dimension (n_draws)")
Base.length(post::DSGEVARPosterior) = post.n_draws
varnames(post::DSGEVARPosterior) = post.varnames
marginal_likelihood(post::DSGEVARPosterior) = post.log_marginal_likelihood

function point_estimate(post::DSGEVARPosterior{T}) where {T<:AbstractFloat}
    return Matrix{T}(dropdims(mean(post.B_draws; dims=1); dims=1))
end

has_uncertainty(post::DSGEVARPosterior) = !isinf(post.lambda) && post.n_draws > 1

function uncertainty_bounds(post::DSGEVARPosterior{T}) where {T<:AbstractFloat}
    has_uncertainty(post) || return nothing
    _, k, n = size(post.B_draws)
    lo = Matrix{T}(undef, k, n)
    hi = Matrix{T}(undef, k, n)
    @inbounds for j in 1:n, i in 1:k
        d = @view post.B_draws[:, i, j]
        lo[i, j] = T(quantile(d, 0.025))
        hi[i, j] = T(quantile(d, 0.975))
    end
    return (lo, hi)
end

function posterior_mean_model(post::DSGEVARPosterior{T};
                              data::AbstractMatrix=Matrix{T}(undef, 0, 0)) where {T<:AbstractFloat}
    return posterior_mean_model(_to_bvar_posterior(post); data=data)
end

function posterior_median_model(post::DSGEVARPosterior{T};
                                data::AbstractMatrix=Matrix{T}(undef, 0, 0)) where {T<:AbstractFloat}
    return posterior_median_model(_to_bvar_posterior(post); data=data)
end

"""Compute structural MA IRFs `(horizon × n × n_shocks)` from `(B_pad, B0)`."""
function _dsgevar_structural_irf(B_pad::AbstractMatrix{T}, B0::AbstractMatrix{T},
                                 n::Int, p::Int, horizon::Int) where {T<:AbstractFloat}
    Phi = ma_coefficients(B_pad, n, p, horizon)
    n_shocks = size(B0, 2)
    out = zeros(T, horizon, n, n_shocks)
    @inbounds for h in 1:horizon
        out[h, :, :] = Phi[h] * B0
    end
    return out
end

"""
    irf(post::DSGEVARPosterior, horizon::Int;
        method::Symbol=:dsge_qr, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
        point_estimate::Symbol=:mean, shock_names=nothing, threaded::Bool=false,
        normalize=nothing, shock_size=nothing, kwargs...) -> BayesianImpulseResponse

Compute Bayesian impulse response functions from a [`DSGEVARPosterior`](@ref).

When `method === :dsge_qr` (default), uses the stored DSGE-VAR QR-rotated structural impact
draws `post.B0_draws` directly. For any other identification scheme (`:cholesky`, `:sign`,
`:long_run`, `:arias`, etc.), delegates to [`BVARPosterior`](@ref).
"""
function irf(post::DSGEVARPosterior{T}, horizon::Int;
             method::Symbol=:dsge_qr,
             quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
             point_estimate::Symbol=:mean,
             shock_names::Union{Nothing,Vector{String}}=nothing,
             threaded::Bool=false,
             normalize::Union{Nothing,Symbol}=nothing,
             shock_size::Union{Nothing,Pair}=nothing,
             kwargs...) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1; got $horizon"))
    point_estimate in (:mean, :median) || throw(ArgumentError(
        "point_estimate must be :mean or :median; got :$point_estimate"))

    if method !== :dsge_qr
        return irf(_to_bvar_posterior(post), horizon;
                   method=method, quantiles=quantiles, point_estimate=point_estimate,
                   shock_names=shock_names, threaded=threaded,
                   normalize=normalize, shock_size=shock_size, kwargs...)
    end

    n = post.n
    p = post.p
    n_req = post.n_draws
    m = size(post.B0_draws, 3)
    snames = if shock_names !== nothing
        length(shock_names) == m || throw(ArgumentError(
            "shock_names length ($(length(shock_names))) must match n_shocks ($m)"))
        Vector{String}(shock_names)
    else
        copy(post.shock_names)
    end

    B_padded = _dsgevar_pad_intercept(post.B_draws, p, n)
    all_irfs = Array{T,4}(undef, n_req, horizon, n, m)
    valid = 0
    @inbounds for s in 1:n_req
        ir_s = _dsgevar_structural_irf(@view(B_padded[s, :, :]), @view(post.B0_draws[s, :, :]), n, p, horizon)
        if all(isfinite, ir_s)
            valid += 1
            all_irfs[valid, :, :, :] = ir_s
        end
    end
    valid == 0 && throw(IdentificationError("All DSGE-VAR posterior draws yielded non-finite IRFs"))
    irfs_valid = valid == n_req ? all_irfs : all_irfs[1:valid, :, :, :]

    if _unit_effect_requested(normalize) && n == m
        irf_ref = dropdims(mean(irfs_valid; dims=1); dims=1)
        specs = _unit_effect_specs_from_irf(irf_ref, post.varnames, shock_size)
        _scale_irf_draws!(irfs_valid, specs)
    end

    q_vec = T.(quantiles)
    use_threaded = threaded || (valid * horizon * n * m > 100000)
    irf_q, irf_m = compute_posterior_quantiles(irfs_valid, q_vec;
                                               threaded=use_threaded, central=point_estimate)

    return BayesianImpulseResponse{T}(
        irf_q, irf_m, horizon, copy(post.varnames), snames, q_vec, irfs_valid,
        n_req, valid, n_req - valid,
    )
end

"""
    fevd(post::DSGEVARPosterior, horizon::Int;
         method::Symbol=:dsge_qr, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
         point_estimate::Symbol=:mean, shock_names=nothing,
         threaded::Bool=false, kwargs...) -> BayesianFEVD

Compute Bayesian forecast error variance decomposition from a [`DSGEVARPosterior`](@ref).

When `method === :dsge_qr` (default), accumulates forecast error variances from `post.B0_draws`.
For any other identification method, delegates to [`BVARPosterior`](@ref).
"""
function fevd(post::DSGEVARPosterior{T}, horizon::Int;
              method::Symbol=:dsge_qr,
              quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
              point_estimate::Symbol=:mean,
              shock_names::Union{Nothing,Vector{String}}=nothing,
              threaded::Bool=false,
              kwargs...) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1; got $horizon"))
    point_estimate in (:mean, :median) || throw(ArgumentError(
        "point_estimate must be :mean or :median; got :$point_estimate"))

    if method !== :dsge_qr
        return fevd(_to_bvar_posterior(post), horizon;
                    method=method, quantiles=quantiles, point_estimate=point_estimate,
                    shock_names=shock_names, threaded=threaded, kwargs...)
    end

    n = post.n
    p = post.p
    n_req = post.n_draws
    m = size(post.B0_draws, 3)
    snames = if shock_names !== nothing
        length(shock_names) == m || throw(ArgumentError(
            "shock_names length ($(length(shock_names))) must match n_shocks ($m)"))
        Vector{String}(shock_names)
    else
        copy(post.shock_names)
    end

    B_padded = _dsgevar_pad_intercept(post.B_draws, p, n)
    all_fevds = Array{T,4}(undef, n_req, n, m, horizon)
    valid = 0
    @inbounds for s in 1:n_req
        ir_s = _dsgevar_structural_irf(@view(B_padded[s, :, :]), @view(post.B0_draws[s, :, :]), n, p, horizon)
        all(isfinite, ir_s) || continue
        _, props = n == m ? _compute_fevd(ir_s, n, horizon) : _compute_fevd_rect(ir_s, n, m, horizon)
        valid += 1
        all_fevds[valid, :, :, :] = props
    end
    valid == 0 && throw(IdentificationError("All DSGE-VAR posterior draws yielded non-finite FEVDs"))
    fevds_valid = valid == n_req ? all_fevds : all_fevds[1:valid, :, :, :]

    q_vec = T.(quantiles)
    use_threaded = threaded || (valid * horizon * n * m > 100000)
    fevd_q, fevd_m = compute_posterior_quantiles(fevds_valid, q_vec;
                                                 threaded=use_threaded, central=point_estimate)

    return BayesianFEVD{T}(
        fevd_q, fevd_m, horizon, copy(post.varnames), snames, q_vec,
        n_req, valid, n_req - valid,
    )
end

"""
    historical_decomposition(post::DSGEVARPosterior, horizon::Int=0;
                             method::Symbol=..., quantiles=[0.16, 0.5, 0.84],
                             point_estimate::Symbol=:mean, shock_names=nothing,
                             seed=nothing, kwargs...) -> BayesianHistoricalDecomposition

Compute Bayesian historical decomposition from a [`DSGEVARPosterior`](@ref).

Uses `post.B0_draws` when `method === :dsge_qr` (default when `n_shocks == n`) and delegates
to [`BVARPosterior`](@ref) for any other identification method.
"""
function historical_decomposition(
    post::DSGEVARPosterior{T},
    horizon::Int=0;
    method::Symbol=(size(post.B0_draws, 3) == post.n ? :dsge_qr : :cholesky),
    data::AbstractMatrix=Matrix{T}(undef, 0, 0),
    quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
    point_estimate::Symbol=:mean,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    seed::Union{Integer,Nothing}=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    if method !== :dsge_qr
        return historical_decomposition(_to_bvar_posterior(post), horizon;
                                        data=data, method=method, quantiles=quantiles,
                                        point_estimate=point_estimate,
                                        shock_names=shock_names, seed=seed, kwargs...)
    end

    m = size(post.B0_draws, 3)
    n = post.n
    m == n || throw(ArgumentError(
        "historical_decomposition with method=:dsge_qr requires a square system (n_shocks == n); " *
        "got n=$n observables and m=$m shocks. Use method=:cholesky instead."))

    use_data = isempty(data) ? post.data : Matrix{T}(data)
    isempty(use_data) && throw(ArgumentError("Data required for historical decomposition"))
    p = post.p
    T_eff = size(use_data, 1) - p
    h_eff = horizon <= 0 ? T_eff : min(horizon, T_eff)
    actual = Matrix{T}(use_data[(p + 1):end, :])

    n_req = post.n_draws
    B_padded = _dsgevar_pad_intercept(post.B_draws, p, n)

    all_contributions = zeros(T, n_req, T_eff, n, n)
    all_initial = zeros(T, n_req, T_eff, n)
    all_shocks = zeros(T, n_req, T_eff, n)
    valid = 0

    for s in 1:n_req
        B_s = Matrix{T}(@view B_padded[s, :, :])
        S_s = Matrix{T}(@view post.Sigma_draws[s, :, :])
        B0_s = Matrix{T}(@view post.B0_draws[s, :, :])
        _, X_s = construct_var_matrices(use_data, p)
        Y_eff_s = @view use_data[(p + 1):end, :]
        U_s = Y_eff_s - X_s * B_s

        m_s = VARModel(use_data, p, B_s, Matrix{T}(U_s), S_s, zero(T), zero(T), zero(T), post.varnames)
        L_u = Matrix{T}(safe_cholesky(Hermitian(S_s); silent=true))
        Q_s = Matrix{T}(robust_inv(L_u; silent=true) * B0_s)
        shocks = compute_structural_shocks(m_s, Q_s)
        Theta = _compute_structural_ma_coefficients(m_s, Q_s, h_eff)
        contrib = _compute_hd_contributions(shocks, Theta)
        init_c = _compute_initial_conditions(actual, contrib)

        if all(isfinite, contrib) && all(isfinite, shocks)
            valid += 1
            all_contributions[valid, :, :, :] = contrib
            all_initial[valid, :, :] = init_c
            all_shocks[valid, :, :] = shocks
        end
    end
    valid == 0 && throw(IdentificationError("All DSGE-VAR posterior draws failed historical decomposition"))

    q_vec = T.(quantiles)
    nq = length(q_vec)
    contrib_q = zeros(T, T_eff, n, n, nq)
    contrib_m = zeros(T, T_eff, n, n)
    initial_q = zeros(T, T_eff, n, nq)
    initial_m = zeros(T, T_eff, n)
    shocks_m = zeros(T, T_eff, n)
    _central = point_estimate == :median ? median : mean

    @inbounds for t in 1:T_eff, i in 1:n
        d_init = @view all_initial[1:valid, t, i]
        initial_q[t, i, :] = quantile(d_init, q_vec)
        initial_m[t, i] = _central(d_init)

        d_shock = @view all_shocks[1:valid, t, i]
        shocks_m[t, i] = _central(d_shock)

        for j in 1:n
            d_c = @view all_contributions[1:valid, t, i, j]
            contrib_q[t, i, j, :] = quantile(d_c, q_vec)
            contrib_m[t, i, j] = _central(d_c)
        end
    end

    snames = if shock_names !== nothing
        length(shock_names) == n || throw(ArgumentError(
            "shock_names length ($(length(shock_names))) must match n ($n)"))
        Vector{String}(shock_names)
    else
        copy(post.shock_names)
    end

    result = BayesianHistoricalDecomposition{T}(
        contrib_q, contrib_m, initial_q, initial_m, shocks_m, actual, T_eff,
        copy(post.varnames), snames, q_vec, :dsge_qr,
        n_req, valid, n_req - valid,
    )
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("method" => "dsge_qr")))
end

"""
    forecast(post::DSGEVARPosterior, h::Int;
             reps=nothing, conf_level::Real=0.95, point_estimate::Symbol=:mean,
             store_draws::Bool=false, rng::AbstractRNG=Random.default_rng(),
             seed::Union{Integer,Nothing}=nothing) -> BVARForecast

Compute out-of-sample forecasts with posterior credible intervals from a
[`DSGEVARPosterior`](@ref) by delegating to [`BVARPosterior`](@ref).
"""
function forecast(post::DSGEVARPosterior{T}, h::Int;
                  reps::Union{Nothing,Int}=nothing,
                  conf_level::Real=0.95,
                  point_estimate::Symbol=:mean,
                  store_draws::Bool=false,
                  rng::AbstractRNG=Random.default_rng(),
                  seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    return forecast(_to_bvar_posterior(post), h;
                    reps=reps, conf_level=conf_level,
                    point_estimate=point_estimate,
                    store_draws=store_draws, rng=rng)
end

function reproduce(post::DSGEVARPosterior)
    m = post.manifest
    m === nothing && return _no_manifest_report("DSGEVARPosterior")
    m.seed === nothing && return _no_seed_report(m, "estimate_dsge_var(...; seed=N)")
    if post.theta_draws !== nothing
        return _needs_source_report("joint DSGEVARPosterior",
            "estimate_dsge_var(spec, Y, p; priors=..., seed=N)")
    end

    s = m.settings
    T = eltype(post.B_draws)
    WT = promote_type(T, Float64)
    rng = _resolve_repro_rng(Random.default_rng(), m.seed)
    n_draws = Int(get(s, "n_draws", post.n_draws))
    prior = post.dsge_prior
    n = post.n
    p = post.p
    k = size(prior.Gamma_xx, 1)
    sh_m = size(prior.Omega_star, 2)
    include_intercept = _dsgevar_has_intercept(prior)
    Y_eff, X, _, XtY, XtX, T_eff = _dsgevar_data_stats(WT, post.data, p, n, include_intercept)

    B_fresh = Array{T,3}(undef, n_draws, k, n)
    S_fresh = Array{T,3}(undef, n_draws, n, n)
    B0_fresh = Array{T,3}(undef, n_draws, n, sh_m)
    Omega_w = Matrix{WT}(prior.Omega_star)

    if isinf(post.lambda)
        L_star = Matrix{WT}(safe_cholesky(Hermitian(Matrix{WT}(prior.Sigma_u_star)); silent=true))
        B0_star_T = Matrix{T}(L_star * Omega_w)
        @inbounds for i in 1:n_draws
            B_fresh[i, :, :] = prior.Phi_star
            S_fresh[i, :, :] = prior.Sigma_u_star
            B0_fresh[i, :, :] = B0_star_T
        end
    else
        B_post, V_post, _, _, _, S_post, _, nu_post, _, _ =
            _dsgevar_posterior_params(
                Matrix{WT}(prior.Gamma_yy),
                Matrix{WT}(prior.Gamma_xy),
                Matrix{WT}(prior.Gamma_xx),
                Matrix{WT}(prior.Phi_star),
                Matrix{WT}(prior.Sigma_u_star),
                prior.T_obs,
                WT(post.lambda),
                Y_eff, X, XtY, XtX, T_eff,
            )
        L_V = Matrix{WT}(safe_cholesky(Hermitian(V_post); silent=true))
        L_S = Matrix{WT}(safe_cholesky(Hermitian(S_post); silent=true))
        Z_buf = Matrix{WT}(undef, k, n)
        @inbounds for i in 1:n_draws
            Sigma_i = _dsgevar_draw_iw(nu_post, L_S, rng)
            L_Sigma = Matrix{WT}(safe_cholesky(Hermitian(Sigma_i); silent=true))
            randn!(rng, Z_buf)
            Phi_i = B_post + L_V * Z_buf * transpose(L_Sigma)
            B0_i = L_Sigma * Omega_w
            B_fresh[i, :, :] = Matrix{T}(Phi_i)
            S_fresh[i, :, :] = Matrix{T}(Sigma_i)
            B0_fresh[i, :, :] = Matrix{T}(B0_i)
        end
    end

    diffs = [
        _repro_field_diff("B_draws", post.B_draws, B_fresh),
        _repro_field_diff("Sigma_draws", post.Sigma_draws, S_fresh),
        _repro_field_diff("B0_draws", post.B0_draws, B0_fresh),
    ]
    return _finalize_repro(diffs, m)
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, prior::DSGEVARPrior{T}) where {T}
    n = length(prior.obs_names)
    k = size(prior.Gamma_xx, 1)
    m = length(prior.shock_names)
    has_int = (k == 1 + n * prior.p)
    lam_str = isinf(prior.lambda) ? "Inf" : _fmt(prior.lambda; digits=4)

    spec_data = Any[
        "Observables (n)"      n;
        "Lags (p)"             prior.p;
        "Parameters / eq (k)"  k;
        "Shocks (m)"           m;
        "Effective sample (T)" prior.T_obs;
        "Prior weight (λ)"     lam_str;
        "Min weight (λ_min)"   _fmt(prior.lambda_min; digits=4);
        "Intercept"            _yesno(has_int);
        "Observables"          isempty(prior.obs_names) ? "—" : join(prior.obs_names, ", ");
        "Shocks"               isempty(prior.shock_names) ? "—" : join(prior.shock_names, ", ");
    ]
    _pretty_table(io, spec_data;
        title = "Del Negro–Schorfheide DSGE-VAR(λ) Prior",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    if 0 < n <= 12
        println(io)
        _matrix_table(io, prior.Sigma_u_star, "Population Innovation Covariance Σ_u*";
            row_labels=prior.obs_names, col_labels=prior.obs_names)
    end
    if 0 < n <= 12 && 0 < m <= 12
        println(io)
        _matrix_table(io, prior.Omega_star, "QR Structural Rotation Ω*";
            row_labels=prior.obs_names, col_labels=prior.shock_names)
    end
    return nothing
end

"""
    report([io::IO], prior::DSGEVARPrior)

Print a detailed summary of a [`DSGEVARPrior`](@ref) including population moments,
implied VAR projection ``\\Phi^*``, innovation covariance ``\\Sigma_u^*``, and QR rotation ``\\Omega^*``.
"""
function report(io::IO, prior::DSGEVARPrior{T}) where {T}
    show(io, prior)
    n = length(prior.obs_names)
    k = size(prior.Gamma_xx, 1)
    has_int = (k == 1 + n * prior.p)
    if 0 < n <= 12 && 0 < k <= 24
        row_labels = String[]
        has_int && push!(row_labels, _INTERCEPT_LABEL)
        for l in 1:prior.p, v in 1:n
            push!(row_labels, "$(prior.obs_names[v]).L$l")
        end
        println(io)
        _matrix_table(io, prior.Phi_star, "Population VAR Projection Φ*";
            row_labels=row_labels, col_labels=prior.obs_names)
    end
    if 0 < n <= 12
        println(io)
        _matrix_table(io, prior.Gamma_yy, "Theoretical Second Moment Γ_yy";
            row_labels=prior.obs_names, col_labels=prior.obs_names)
    end
    return nothing
end

report(prior::DSGEVARPrior) = report(stdout, prior)

function Base.show(io::IO, prof::DSGEVARLambdaProfile{T}) where {T}
    n = length(prof.obs_names)
    lam_hat_str = isinf(prof.lambda_hat) ? "Inf" : _fmt(prof.lambda_hat; digits=4)
    delta_ml = prof.logml_hat - prof.logml_inf

    spec_data = Any[
        "Observables (n)"      n;
        "Lags (p)"             prof.p;
        "Effective sample (T)" prof.T_obs;
        "Grid points"          length(prof.lambdas);
        "Min weight (λ_min)"   _fmt(prof.lambda_min; digits=4);
        "Optimal λ̂"            lam_hat_str;
        "Log ML (λ̂)"           _fmt(prof.logml_hat; digits=4);
        "Log ML (λ = ∞)"       _fmt(prof.logml_inf; digits=4);
        "Δ Log ML (λ̂ vs ∞)"   _fmt(delta_ml; digits=4);
        "Observables"          join(prof.obs_names, ", ");
    ]
    _pretty_table(io, spec_data;
        title = "DSGE-VAR(λ) Marginal Likelihood Profile",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

"""
    report([io::IO], prof::DSGEVARLambdaProfile)

Print a summary and grid table of a [`DSGEVARLambdaProfile`](@ref).
"""
function report(io::IO, prof::DSGEVARLambdaProfile{T}) where {T}
    show(io, prof)
    ng = length(prof.lambdas)
    if ng > 0
        println(io)
        idx_show = ng <= 30 ? collect(1:ng) : unique!(round.(Int, range(1, ng; length=30)))
        tbl = Matrix{Any}(undef, length(idx_show) + 1, 3)
        for (ri, gi) in enumerate(idx_show)
            tbl[ri, 1] = _fmt(prof.lambdas[gi]; digits=4)
            tbl[ri, 2] = _fmt(prof.log_marginal_likelihoods[gi]; digits=4)
            tbl[ri, 3] = _fmt(prof.log_marginal_likelihoods[gi] - prof.logml_inf; digits=4)
        end
        tbl[end, 1] = "Inf"
        tbl[end, 2] = _fmt(prof.logml_inf; digits=4)
        tbl[end, 3] = _fmt(zero(T); digits=4)
        _pretty_table(io, tbl;
            title = "Marginal Likelihood across DSGE Prior Weights",
            column_labels = ["λ", "Log ML", "Δ vs λ=∞"],
            alignment = [:r, :r, :r],
        )
    end
    return nothing
end

report(prof::DSGEVARLambdaProfile) = report(stdout, prof)

function Base.show(io::IO, post::DSGEVARPosterior{T}) where {T}
    k = size(post.B_draws, 2)
    m = length(post.shock_names)
    lam_str = isinf(post.lambda) ? "Inf" : _fmt(post.lambda; digits=4)

    spec_data = if post.theta_draws === nothing
        Any[
            "Method"               _label(post.method);
            "Variables (n)"        post.n;
            "Lags (p)"             post.p;
            "Shocks (m)"           m;
            "Draws"                post.n_draws;
            "Prior weight (λ)"     lam_str;
            "Log marginal lik."    _fmt(post.log_marginal_likelihood; digits=4);
            "Parameters / eq (k)"  k;
        ]
    else
        Any[
            "Method"               _label(post.method);
            "Variables (n)"        post.n;
            "Lags (p)"             post.p;
            "Shocks (m)"           m;
            "Draws"                post.n_draws;
            "DSGE params (d)"      length(post.param_names);
            "Prior weight (λ)"     lam_str;
            "Log marginal lik."    _fmt(post.log_marginal_likelihood; digits=4);
            "Acceptance rate"      _fmt(post.acceptance_rate; digits=4);
            "Parameters / eq (k)"  k;
        ]
    end
    _pretty_table(io, spec_data;
        title = "Bayesian DSGE-VAR($(post.p))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    if post.theta_draws !== nothing && !isempty(post.param_names)
        d = length(post.param_names)
        th_tbl = Matrix{Any}(undef, d, 6)
        for i in 1:d
            col = @view post.theta_draws[:, i]
            th_tbl[i, 1] = string(post.param_names[i])
            th_tbl[i, 2] = _fmt(mean(col))
            th_tbl[i, 3] = _fmt(std(col))
            th_tbl[i, 4] = _fmt(T(quantile(col, 0.025)))
            th_tbl[i, 5] = _fmt(T(quantile(col, 0.50)))
            th_tbl[i, 6] = _fmt(T(quantile(col, 0.975)))
        end
        println(io)
        _pretty_table(io, th_tbl;
            title = "DSGE Structural Parameter Posterior",
            column_labels = ["Parameter", "Mean", "Std", "2.5%", "50%", "97.5%"],
            alignment = [:l, :r, :r, :r, :r, :r],
        )
    end

    has_int = (k == 1 + post.n * post.p)
    coef_names = String[]
    has_int && push!(coef_names, _INTERCEPT_LABEL)
    for l in 1:post.p, v in 1:post.n
        push!(coef_names, "$(post.varnames[v]).L$l")
    end

    for eq in 1:post.n
        draws_eq = @view post.B_draws[:, :, eq]
        n_show = min(k, length(coef_names))
        tbl = Matrix{Any}(undef, n_show, 6)
        for i in 1:n_show
            col = @view draws_eq[:, i]
            tbl[i, 1] = coef_names[i]
            tbl[i, 2] = _fmt(mean(col))
            tbl[i, 3] = _fmt(std(col))
            tbl[i, 4] = _fmt(T(quantile(col, 0.025)))
            tbl[i, 5] = _fmt(T(quantile(col, 0.50)))
            tbl[i, 6] = _fmt(T(quantile(col, 0.975)))
        end
        println(io)
        _pretty_table(io, tbl;
            title = "Equation: $(post.varnames[eq])",
            column_labels = ["", "Mean", "Std", "2.5%", "50%", "97.5%"],
            alignment = [:l, :r, :r, :r, :r, :r],
        )
    end

    Sigma_mean = dropdims(mean(post.Sigma_draws; dims=1); dims=1)
    println(io)
    _matrix_table(io, Sigma_mean, "Posterior Mean Σ_u";
        row_labels=post.varnames, col_labels=post.varnames)

    _manifest_footer(io, post.manifest)
    return nothing
end

"""
    report([io::IO], post::DSGEVARPosterior; repro::Bool=false)

Print a comprehensive summary of a [`DSGEVARPosterior`](@ref), including DSGE structural
parameter summaries (in joint mode), per-equation VAR coefficient posteriors, posterior mean
innovation covariance ``\\Sigma_u``, and posterior mean DSGE-QR structural impact matrix ``B_0``.
"""
function report(io::IO, post::DSGEVARPosterior{T}) where {T}
    show(io, post)
    m = length(post.shock_names)
    if 0 < post.n <= 12 && 0 < m <= 12
        B0_mean = dropdims(mean(post.B0_draws; dims=1); dims=1)
        println(io)
        _matrix_table(io, B0_mean, "Posterior Mean Structural Impact B₀";
            row_labels=post.varnames, col_labels=post.shock_names)
    end
    return nothing
end

report(post::DSGEVARPosterior; repro::Bool=false) =
    report(IOContext(stdout, :repro => repro), post)
