# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
GMM estimation of DSGE model parameters.

Supports two methods:
- IRF matching (Christiano, Eichenbaum & Evans 2005)
- Euler equation GMM (Hansen & Singleton 1982)
"""

"""
    estimate_dsge(spec, data, param_names; method=:irf_matching, ...) -> DSGEEstimation

Estimate DSGE deep parameters via GMM.

# Arguments
- `spec::DSGESpec{T}` -- model specification (initial param values as starting point)
- `data::AbstractMatrix` -- T_obs x n_vars data matrix
- `param_names::Vector{Symbol}` -- which parameters to estimate

# Keywords
- `method::Symbol` -- `:irf_matching` or `:euler_gmm`
- `target_irfs::ImpulseResponse` -- pre-computed target IRFs (optional, for IRF matching)
- `var_lags::Int=4` -- lag order for VAR when computing target IRFs
- `irf_horizon::Int=20` -- horizon for IRF matching
- `weighting::Symbol=:two_step` -- GMM weighting
- `n_lags_instruments::Int=4` -- instrument lags for Euler equation GMM

# References
- Christiano, L., Eichenbaum, M., & Evans, C. (2005). "Nominal Rigidities and the
  Dynamic Effects of a Shock to Monetary Policy." *Journal of Political Economy*, 113(1), 1-45.
- Hansen, L. P. & Singleton, K. J. (1982). "Generalized Instrumental Variables Estimation
  of Nonlinear Rational Expectations Models." *Econometrica*, 50(5), 1269-1286.
"""
function estimate_dsge(spec::DSGESpec{T}, data::AbstractMatrix,
                        param_names::Vector{Symbol};
                        method::Symbol=:irf_matching,
                        target_irfs::Union{Nothing,ImpulseResponse}=nothing,
                        var_lags::Int=4, irf_horizon::Int=20,
                        weighting::Symbol=:two_step,
                        n_lags_instruments::Int=4,
                        sim_ratio::Int=5, burn::Int=100,
                        moments_fn::Function=d -> autocovariance_moments(d; lags=1),
                        bounds::Union{Nothing,ParameterTransform}=nothing,
                        lags::Int=1,
                        rng=Random.default_rng()) where {T<:AbstractFloat}
    data_T = Matrix{T}(data)

    if method == :irf_matching
        return _estimate_irf_matching(spec, data_T, param_names;
                                       target_irfs=target_irfs,
                                       var_lags=var_lags,
                                       irf_horizon=irf_horizon,
                                       weighting=weighting)
    elseif method == :euler_gmm
        return _estimate_euler_gmm(spec, data_T, param_names;
                                    n_lags=n_lags_instruments,
                                    weighting=weighting)
    elseif method == :smm
        return _estimate_dsge_smm(spec, data_T, param_names;
                                    sim_ratio=sim_ratio, burn=burn,
                                    weighting=weighting, moments_fn=moments_fn,
                                    bounds=bounds, rng=rng)
    elseif method == :analytical_gmm
        return _estimate_dsge_analytical_gmm(spec, data_T, param_names;
                                              lags=lags,
                                              bounds=bounds)
    else
        throw(ArgumentError("method must be :irf_matching, :euler_gmm, :smm, or :analytical_gmm"))
    end
end

# Convenience method for Float64 data
function estimate_dsge(spec::DSGESpec{T}, data::AbstractMatrix{T},
                        param_names::Vector{Symbol};
                        kwargs...) where {T<:AbstractFloat}
    invoke(estimate_dsge, Tuple{DSGESpec{T}, AbstractMatrix, Vector{Symbol}},
           spec, data, param_names; kwargs...)
end

# =============================================================================
# IRF Matching (Christiano, Eichenbaum & Evans 2005)
# =============================================================================

"""
    _estimate_irf_matching(spec, data, param_names; ...) -> DSGEEstimation

Internal: IRF matching GMM estimation.

Matches model-implied IRFs to empirical VAR IRFs by minimizing the distance
under a GMM weighting matrix.
"""
function _estimate_irf_matching(spec::DSGESpec{T}, data::Matrix{T},
                                 param_names::Vector{Symbol};
                                 target_irfs=nothing, var_lags=4,
                                 irf_horizon=20, weighting=:two_step) where {T}
    n_est = length(param_names)

    # Step 1: Compute target IRFs from VAR if not provided
    if target_irfs === nothing
        var_model = estimate_var(data, var_lags)
        target_irfs = irf(var_model, irf_horizon; method=:cholesky)
    end
    target_vec = vec(target_irfs.values)
    n_moments = length(target_vec)

    # Extract initial parameter values as starting point
    theta0 = T[spec.param_values[p] for p in param_names]

    # Compute model IRF distance for a given parameter vector
    function irf_distance(theta)
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[], spec.ss_fn
        )
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if !is_determined(sol)
                return fill(T(1e6), n_moments)
            end
            model_irfs = irf(sol, irf_horizon)
            model_vec = vec(model_irfs.values)
            if length(model_vec) != n_moments
                return fill(T(1e6), n_moments)
            end
            model_vec .- target_vec
        catch
            fill(T(1e6), n_moments)
        end
    end

    # IRF matching is a minimum-distance estimator: min_θ g(θ)' W g(θ)
    # Use identity weighting matrix (standard for IRF matching)
    W = Matrix{T}(I, n_moments, n_moments)
    result = minimize_gmm(
        (theta, _data) -> reshape(irf_distance(theta), 1, :),
        theta0, data, W; max_iter=100, tol=T(1e-8)
    )
    theta_hat = result.theta
    converged = result.converged

    # Compute vcov via numerical Hessian of the objective
    # Q(θ) = g(θ)'g(θ), so ∂²Q/∂θ² gives the curvature
    g_hat = irf_distance(theta_hat)
    dg = numerical_gradient(irf_distance, theta_hat)  # n_moments x n_est

    # For minimum-distance with identity W:
    # V_θ = (G'G)^{-1} where G = ∂g/∂θ evaluated at θ_hat
    bread = dg' * W * dg
    bread_inv = robust_inv(bread)
    T_obs = size(data, 1)
    vcov_hat = bread_inv / T(T_obs)

    # J-statistic (overidentification test)
    J_stat, J_pvalue = if n_moments > n_est
        J = T(T_obs) * (g_hat' * W * g_hat)
        df = n_moments - n_est
        (J, one(T) - cdf(Chisq(df), J))
    else
        (zero(T), one(T))
    end

    # Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = theta_hat[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[], spec.ss_fn
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        theta_hat, vcov_hat, param_names,
        :irf_matching, J_stat, J_pvalue,
        final_sol, converged, final_spec
    )
end

# =============================================================================
# Euler Equation GMM (Hansen & Singleton 1982)
# =============================================================================

"""
    _estimate_euler_gmm(spec, data, param_names; ...) -> DSGEEstimation

Internal: Euler equation GMM estimation.

Uses the structural Euler equations as moment conditions with lagged
variables as instruments.
"""
function _estimate_euler_gmm(spec::DSGESpec{T}, data::Matrix{T},
                              param_names::Vector{Symbol};
                              n_lags=4, weighting=:two_step) where {T}
    T_obs, n_vars = size(data)
    theta0 = T[spec.param_values[p] for p in param_names]

    # Need n_lags lags + 1 lead
    t_start = n_lags + 2
    T_eff = T_obs - t_start

    n_eq = spec.n_endog
    n_inst = n_vars * n_lags

    function moment_fn(theta, _data)
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end

        moments = zeros(T, T_eff, n_eq * n_inst)

        for (idx, t) in enumerate(t_start:(T_obs-1))
            y_t = _data[t, :]
            y_lag = _data[t-1, :]
            y_lead = _data[t+1, :]
            eps_zero = zeros(T, spec.n_exog)

            resids = zeros(T, n_eq)
            for i in 1:n_eq
                resids[i] = spec.residual_fns[i](y_t, y_lag, y_lead, eps_zero, new_pv)
            end

            # Instrument vector: lagged values
            Z = zeros(T, n_inst)
            col = 1
            for lag in 1:n_lags
                for v in 1:n_vars
                    Z[col] = _data[t-lag, v]
                    col += 1
                end
            end

            # Moment conditions: kron(residuals, instruments)
            moments[idx, :] = kron(resids, Z)
        end
        moments
    end

    gmm_result = estimate_gmm(moment_fn, theta0, data; weighting=weighting)

    # Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = gmm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[], spec.ss_fn
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        gmm_result.theta, gmm_result.vcov, param_names,
        :euler_gmm, gmm_result.J_stat, gmm_result.J_pvalue,
        final_sol, gmm_result.converged, final_spec
    )
end

# =============================================================================
# SMM Estimation (Ruge-Murcia 2012)
# =============================================================================

"""
    _estimate_dsge_smm(spec, data, param_names; ...) -> DSGEEstimation

Internal: SMM estimation of DSGE parameters.

Builds a simulator from the DSGE spec: for each candidate θ, updates spec
parameters, solves the model, and simulates data.
"""
function _estimate_dsge_smm(spec::DSGESpec{T}, data::Matrix{T},
                              param_names::Vector{Symbol};
                              sim_ratio=5, burn=100, weighting=:two_step,
                              moments_fn=d -> autocovariance_moments(d; lags=1),
                              bounds=nothing, rng=Random.default_rng()) where {T}
    theta0 = T[spec.param_values[p] for p in param_names]

    # Build DSGE simulator: for each candidate theta, update spec, solve, simulate.
    # The burn-in is handled here (using the captured `burn` kwarg from the outer
    # scope), so we pass burn=0 to estimate_smm below.
    #
    # When the model does not solve (e.g. explosive parameters), return a large-
    # variance random matrix instead of NaN.  NaN moments cause NaN objective
    # values which break Nelder-Mead simplex comparisons.  A large-variance draw
    # produces moments far from the data, yielding a high (but finite) objective
    # that the optimizer can move away from.
    dsge_burn = burn
    function dsge_simulator(theta, T_periods, _burn_in; rng=Random.default_rng())
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[], spec.ss_fn
        )
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if !is_determined(sol)
                # Return large-variance noise so objective is finite but penalized
                return T(1e4) .* randn(copy(rng), T, T_periods, spec.n_endog)
            end
            sim_full = simulate(sol, T_periods + dsge_burn; rng=rng)
            sim_full[(dsge_burn+1):end, :]
        catch
            T(1e4) .* randn(copy(rng), T, T_periods, spec.n_endog)
        end
    end

    smm_result = estimate_smm(dsge_simulator, moments_fn, theta0, data;
                               sim_ratio=sim_ratio, burn=0,  # burn handled inside simulator
                               weighting=weighting, bounds=bounds, rng=rng)

    # Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = smm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[], spec.ss_fn
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        smm_result.theta, smm_result.vcov, param_names,
        :smm, smm_result.J_stat, smm_result.J_pvalue,
        final_sol, smm_result.converged, final_spec
    )
end

# =============================================================================
# Data moment computation for GMM format
# =============================================================================

"""
    _compute_data_moments(data::Matrix{T}; lags::Vector{Int}=[1],
                           observable_indices::Union{Nothing,Vector{Int}}=nothing) → Vector{T}

Compute data moment vector matching the GMM format from model moments:
1. Means: E[y_i]
2. Product moments: E[y_i * y_j] for i ≤ j (upper triangle)
3. Diagonal autocovariances: E[y_i,t * y_i,t-k] for each lag k

This matches the moment ordering in `analytical_moments(...; format=:gmm)`.
"""
function _compute_data_moments(data::Matrix{T};
                                lags::Vector{Int}=[1],
                                observable_indices::Union{Nothing,Vector{Int}}=nothing) where {T}
    Y = observable_indices === nothing ? data : data[:, observable_indices]
    T_obs, ny = size(Y)

    moments = T[]

    # 1. Means
    Ey = vec(sum(Y; dims=1)) / T_obs
    append!(moments, Ey)

    # 2. Product moments (upper triangle of Y'Y/T)
    Eyy = Y' * Y / T_obs
    for i in 1:ny
        for j in i:ny
            push!(moments, Eyy[i, j])
        end
    end

    # 3. Diagonal autocovariances at each lag
    num_lags = length(lags)
    for k in 1:num_lags
        lag = lags[k]
        autoEyy = Y[1+lag:T_obs, :]' * Y[1:T_obs-lag, :] / (T_obs - lag)
        for i in 1:ny
            push!(moments, autoEyy[i, i])
        end
    end

    return moments
end

# =============================================================================
# Analytical GMM Estimation (Lyapunov equation moments)
# =============================================================================

"""
    _estimate_dsge_analytical_gmm(spec, data, param_names; ...) -> DSGEEstimation

Internal: Analytical GMM estimation using Lyapunov equation moments.

Matches model-implied analytical moments (from `analytical_moments`) to data moments
(from `autocovariance_moments`). No simulation required — uses the discrete Lyapunov
equation to compute unconditional covariances exactly.
"""
function _estimate_dsge_analytical_gmm(spec::DSGESpec{T}, data::Matrix{T},
                                         param_names::Vector{Symbol};
                                         lags=1, weighting=:identity,
                                         bounds=nothing) where {T}
    theta0 = T[spec.param_values[p] for p in param_names]

    # Data moments
    m_data = autocovariance_moments(data; lags=lags)
    n_moments = length(m_data)

    # GMM moment function: returns 1 × n_moments matrix
    # Analytical moments are deterministic given θ — single "observation"
    function analytical_moment_fn(theta, _data)
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[], spec.ss_fn
        )
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if !is_determined(sol) || !is_stable(sol)
                return fill(T(1e6), 1, n_moments)
            end
            m_model = analytical_moments(sol; lags=lags)
            g = reshape(m_data .- m_model, 1, n_moments)
            return g
        catch
            return fill(T(1e6), 1, n_moments)
        end
    end

    gmm_result = estimate_gmm(analytical_moment_fn, theta0, data;
                                weighting=weighting, bounds=bounds)

    # Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = gmm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[], spec.ss_fn
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        gmm_result.theta, gmm_result.vcov, param_names,
        :analytical_gmm, gmm_result.J_stat, gmm_result.J_pvalue,
        final_sol, gmm_result.converged, final_spec
    )
end
