# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Large BVAR nowcasting with GLP-style priors (Cimadomo et al. 2022).

Estimates a mixed-frequency BVAR with Normal-Inverse-Wishart prior.
Hyperparameters (lambda, theta, miu, alpha) optimized via marginal
log-likelihood maximization.
"""

using Distributions: loggamma

# =============================================================================
# Public API
# =============================================================================

"""
    nowcast_bvar(Y, nM, nQ; lags=5, thresh=1e-6, max_iter=200,
                lambda0=0.2, theta0=1.0, miu0=1.0, alpha0=2.0) -> NowcastBVAR{T}

Estimate a large BVAR for mixed-frequency nowcasting.

The first `nM` columns are monthly variables; the next `nQ` columns are
quarterly (observed every 3rd month). The BVAR is estimated on the complete
(non-NaN) portion, then the Kalman smoother fills the ragged edge.

# Arguments
- `Y::AbstractMatrix` — T_obs × N data matrix (NaN for missing)
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables

# Keyword Arguments
- `lags::Int=5` — number of lags
- `thresh::Real=1e-6` — optimization convergence threshold
- `max_iter::Union{Int,Nothing}=nothing` — max Nelder-Mead iterations. Defaults to `200`
  under `:conjugate` (4 hyperparameters) and `2000` under `:litterman`, whose search is
  5-dimensional and typically needs ~1000 iterations to reach a stationary point. An
  explicit value is always honoured.
- `lambda0::Real=0.2` — initial overall shrinkage
- `theta0::Real=1.0` — initial lag-decay exponent (Litterman `d` / GLP `α`; larger ⇒
  higher lags shrunk harder). See `_bvar_dummy_obs` for why own-vs-cross relative
  tightness is not a separate hyperparameter under the conjugate NIW prior (#572).
- `miu0::Real=1.0` — initial sum-of-coefficients weight
- `alpha0::Real=2.0` — initial co-persistence weight
- `prior::Symbol=:conjugate` — `:conjugate` for the GLP Normal-Inverse-Wishart
  dummy-observation prior, or `:litterman` for Litterman's (1986) non-conjugate prior with
  `Σ` fixed at `diag(σ̂²)`. Only `:litterman` admits `theta_cross`; see
  `_litterman_prior_rows` (#602).
- `theta_cross0::Real=1.0` — initial cross-variable relative tightness. `:litterman` only;
  passing it under `:conjugate` throws, because no dummy-observation prior can express it.

# Returns
`NowcastBVAR{T}` with smoothed data and posterior parameters.

!!! warning "`loglik` is not comparable across priors"
    The conjugate value integrates out both the coefficients and `Σ`; the Litterman value
    integrates out the coefficients with `Σ` held fixed. They are different objectives on
    different parameter spaces. Compare lag orders or hyperparameters *within* a prior, and
    compare priors by out-of-sample performance instead.

# References
- Cimadomo, J., Giannone, D., Lenza, M., Monti, F. & Sokol, A. (2022).
  Nowcasting with Large Bayesian Vector Autoregressions.
- Litterman, R. B. (1986). Forecasting with Bayesian Vector Autoregressions —
  Five Years of Experience. *Journal of Business & Economic Statistics* 4(1), 25-38.
- Kadiyala, K. R. & Karlsson, S. (1997). Numerical Methods for Estimation and Inference
  in Bayesian VAR-Models. *Journal of Applied Econometrics* 12(2), 99-132.
"""
function nowcast_bvar(Y::AbstractMatrix, nM::Int, nQ::Int;
                      lags::Int=5, thresh::Real=1e-6,
                      max_iter::Union{Int,Nothing}=nothing,
                      lambda0::Real=0.2, theta0::Real=1.0,
                      miu0::Real=1.0, alpha0::Real=2.0,
                      prior::Symbol=:conjugate,
                      theta_cross0::Union{Real,Nothing}=nothing)
    T_obs, N = size(Y)
    N == nM + nQ || throw(ArgumentError("nM ($nM) + nQ ($nQ) must equal number of columns ($N)"))
    lags >= 1 || throw(ArgumentError("lags must be >= 1, got $lags"))
    prior in (:conjugate, :litterman) ||
        throw(ArgumentError("prior must be :conjugate or :litterman, got :$prior"))
    if prior == :conjugate && theta_cross0 !== nothing
        throw(ArgumentError(
            "theta_cross is not a parameter of the conjugate prior: dummy observations " *
            "imply Var(vec(B)) = Σ ⊗ (X_d'X_d)⁻¹, so for a given regressor the cross/own " *
            "prior SD ratio is √(Σ_mm/Σ_jj) whatever the dummy rows are. Use " *
            "prior=:litterman to make it a free hyperparameter."))
    end

    Tf = eltype(Y) <: AbstractFloat ? eltype(Y) : Float64
    Ymat = Matrix{Tf}(Y)

    # Find the last complete row (no NaN)
    t_complete = T_obs
    while t_complete > 0 && any(isnan, Ymat[t_complete, :])
        t_complete -= 1
    end

    # Need at least lags+1 complete rows for VAR construction
    if t_complete < lags + 2
        t_complete = T_obs
    end

    # Fill NaN in balanced panel (quarterly columns have NaN at non-quarter months)
    Ybal = copy(Ymat[1:t_complete, :])
    for j in 1:N
        col = Ybal[:, j]
        valid_mask = .!isnan.(col)
        if any(valid_mask) && !all(valid_mask)
            m = mean(col[valid_mask])
            for i in 1:t_complete
                if isnan(Ybal[i, j])
                    Ybal[i, j] = m
                end
            end
        elseif !any(valid_mask)
            Ybal[:, j] .= zero(Tf)
        end
    end

    # Compute AR(1) residual standard deviations for prior scaling
    sigma_ar = zeros(Tf, N)
    for j in 1:N
        col = Ybal[:, j]
        valid = filter(!isnan, col)
        if length(valid) > 2
            y_dep = valid[2:end]
            y_lag = valid[1:end-1]
            b = dot(y_lag, y_dep) / max(dot(y_lag, y_lag), Tf(1e-10))
            resid_j = y_dep - b * y_lag
            sigma_ar[j] = max(std(resid_j), Tf(1e-6))
        else
            sigma_ar[j] = one(Tf)
        end
    end

    # Optimize hyperparameters via marginal log-likelihood. The Litterman path carries a
    # fifth coordinate, log(theta_cross) — the knob the conjugate prior cannot express.
    par0 = prior == :litterman ?
        [log(Tf(lambda0)), log(Tf(theta0)), log(Tf(theta_cross0 === nothing ? 1.0 : theta_cross0)),
         log(Tf(miu0)), log(Tf(alpha0))] :
        [log(Tf(lambda0)), log(Tf(theta0)), log(Tf(miu0)), log(Tf(alpha0))]

    log_ml = prior == :litterman ? _litterman_log_ml : _bvar_log_ml
    obj = par -> begin
        val = -log_ml(par, Ybal, lags, sigma_ar)
        isfinite(val) ? val : Tf(1e10)
    end

    # The Litterman search carries a fifth coordinate and needs roughly 1000 Nelder-Mead
    # iterations on a typical macro panel; 200 stops it well short of a stationary point
    # (and the box-edge `converged` flag would not catch that). The conjugate default is
    # left at 200 so existing fits are bit-identical.
    iters = max_iter === nothing ? (prior == :litterman ? 2000 : 200) : max_iter
    result = Optim.optimize(obj, par0, Optim.NelderMead(),
                            Optim.Options(iterations=iters, f_reltol=Tf(thresh)))

    par_opt = Optim.minimizer(result)
    lambda_opt = exp(par_opt[1])
    theta_opt = exp(par_opt[2])
    theta_cross_opt = prior == :litterman ? exp(par_opt[3]) : Tf(NaN)
    miu_opt = exp(par_opt[prior == :litterman ? 4 : 3])
    alpha_opt = exp(par_opt[prior == :litterman ? 5 : 4])

    # The log-hyperparameter box is |par| ≤ 5; a hit at the corner (λ = exp(5) ≈ 148.4)
    # means the marginal-likelihood optimizer diverged to the boundary rather than an
    # interior optimum — flag it (the box is the documented cause and is NOT altered, which
    # would perturb the whole nowcast pipeline). (B4/T173)
    converged = !any(x -> abs(x) >= Tf(5) - Tf(1e-3), par_opt)

    # Estimate BVAR with optimal hyperparameters
    beta, sigma, ml = prior == :litterman ?
        _litterman_estimate(Ybal, lags, sigma_ar, lambda_opt, theta_opt,
                            theta_cross_opt, miu_opt, alpha_opt) :
        _bvar_estimate(Ybal, lags, sigma_ar, lambda_opt, theta_opt, miu_opt, alpha_opt)

    # Use Kalman smoother to fill missing data
    X_sm = _bvar_smooth_missing(Ymat, beta, sigma, lags, t_complete)

    NowcastBVAR{Tf}(X_sm, beta, sigma, lambda_opt, theta_opt, miu_opt,
                     alpha_opt, lags, ml, nM, nQ, Ymat, converged,
                     Tf(theta_cross_opt), prior)
end

# =============================================================================
# BVAR Marginal Log-Likelihood
# =============================================================================

"""Compute log marginal likelihood for Normal-IW BVAR."""
function _bvar_log_ml(par::AbstractVector{T}, Y::Matrix{T}, lags::Int,
                      sigma_ar::Vector{T}) where {T<:AbstractFloat}
    any(x -> abs(x) > T(5), par) && return -T(1e10)
    lambda = exp(par[1])
    theta = exp(par[2])
    miu = exp(par[3])
    alpha = exp(par[4])

    _, _, ml = _bvar_estimate(Y, lags, sigma_ar, lambda, theta, miu, alpha)
    return isfinite(ml) ? ml : -T(1e10)
end

# =============================================================================
# BVAR Estimation with Minnesota-style Prior
# =============================================================================

"""
    _bvar_estimate(Y, lags, sigma_ar, lambda, theta, miu, alpha) -> (beta, sigma, logml)

Estimate BVAR with Normal-Inverse-Wishart prior.

Uses Minnesota-style dummy observations for prior implementation.
"""
function _bvar_estimate(Y::Matrix{T}, lags::Int, sigma_ar::Vector{T},
                        lambda::T, theta::T, miu::T, alpha::T) where {T<:AbstractFloat}
    T_obs, N = size(Y)

    # Construct VAR matrices
    Y_dep = Y[(lags + 1):end, :]
    T_eff = size(Y_dep, 1)
    X_reg = ones(T, T_eff, 1)  # intercept
    for lag in 1:lags
        X_reg = hcat(X_reg, Y[(lags + 1 - lag):(end - lag), :])
    end

    k = size(X_reg, 2)  # 1 + N*lags

    # Minnesota prior: dummy observations
    # Prior mean: random walk for each variable
    Y_d, X_d = _bvar_dummy_obs(Y[1:lags, :], lags, sigma_ar, lambda, theta, miu, alpha)

    # Stack data + dummy observations
    Y_star = vcat(Y_dep, Y_d)
    X_star = vcat(X_reg, X_d)

    if !all(isfinite, Y_star) || !all(isfinite, X_star)
        return zeros(T, k, N), Matrix{T}(I(N)), -T(1e10)
    end

    # OLS on augmented system (= posterior mode of Normal-IW)
    XtX = X_star' * X_star
    XtX_reg = XtX + T(1e-6) * I(k)
    beta = XtX_reg \ (X_star' * Y_star)

    if !all(isfinite, beta)
        return zeros(T, k, N), Matrix{T}(I(N)), -T(1e10)
    end

    # Residuals and posterior sigma (on the augmented system)
    resid = Y_star - X_star * beta
    sigma = (resid' * resid) / T(size(Y_star, 1) - k)
    sigma = (sigma + sigma') / T(2)

    # Full Normal-Inverse-Wishart closed-form log marginal likelihood (#571).
    # The earlier plug-in Gaussian likelihood had no |X'X| ratio / Gamma terms, so
    # it rose monotonically in lag order and pushed λ to the box wall.
    logml = try
        T_d = size(Y_d, 1)
        K_post = X_star' * X_star
        K_prior = X_d' * X_d
        # Prior mean by column-equilibrated least squares. The lag^theta scaling spreads
        # the dummy columns over many orders of magnitude; inverting the raw Gram matrix
        # sends robust_inv into its truncated pseudo-inverse, whose residual inflates
        # S_prior by a large step and hands the optimizer a spurious log-ML jump toward
        # the box wall (#571).
        col_scale = [max(norm(@view X_d[:, j]), eps(T)) for j in 1:k]
        B_prior = ((X_d ./ col_scale') \ Y_d) ./ col_scale
        S_post = (Y_star - X_star * beta)' * (Y_star - X_star * beta)
        S_prior = (Y_d - X_d * B_prior)' * (Y_d - X_d * B_prior)
        nu_prior = T_d - k
        if nu_prior <= N - 1
            -T(1e10)
        else
            nu_post = T_eff + nu_prior
            logmvgamma(a) = T(N * (N - 1)) / 4 * log(T(π)) +
                            sum(loggamma(a + T(1 - j) / 2) for j in 1:N)
            T(0.5) * N * (logdet_safe(K_prior) - logdet_safe(K_post)) -
            T(0.5) * nu_post * logdet_safe(S_post) +
            T(0.5) * nu_prior * logdet_safe(S_prior) -
            T(0.5) * T_eff * N * log(T(π)) +
            logmvgamma(T(nu_post) / 2) - logmvgamma(T(nu_prior) / 2)
        end
    catch
        -T(1e10)
    end

    if !isfinite(logml)
        logml = -T(1e10)
    end

    return beta, sigma, logml
end

"""
    _bvar_dummy_obs(Y0, lags, sigma_ar, lambda, theta, miu, alpha) -> (Y_d, X_d)

Construct Minnesota prior dummy observations (GLP / stacked-dummy form).

Every Minnesota row carries exactly ONE nonzero regressor entry, as in `gen_dummy_obs`
(`src/bvar/priors.jl`): the row for lag `l` and variable `i` scales the coefficient on
`y_{t-l}[i]` by `σ_i · l^θ / λ` and targets 1 (own equation, lag 1) or 0. Putting the
own-lag and all cross-lag entries of one equation into the SAME row instead restricts
their *sum*: every lag block collapses to rank 1, `X_d'X_d` is singular, and the NIW
marginal likelihood degenerates to the `-1e10` sentinel on the default start (#571/#572).

- `lambda` — overall tightness (smaller ⇒ tighter; inverse of the prior SD scale)
- `theta` — lag-decay exponent (Litterman `d`, GLP `α`): larger ⇒ higher lags shrunk
  harder toward zero. Own-vs-cross *relative* tightness is not a free hyperparameter of a
  conjugate NIW prior: dummy observations imply prior variance `Σ_mm · (X_d'X_d)⁻¹_cc` for
  coefficient `(c, m)`, so for regressor `j` the cross/own prior SD ratio is `√(Σ_mm/Σ_jj)`
  — the standard Minnesota asymmetry — under *any* row scale (#572).
- `miu` — sum-of-coefficients (unit root prior)
- `alpha` — co-persistence (common stochastic trend prior); also the intercept prior scale

The closing block is the inverse-Wishart scale prior (`Y = diag(σ)`, `X = 0`). Without it
the random-walk `B` satisfies every dummy row exactly, the prior SSR is singular, and the
marginal likelihood collapses again — this is the block that fixes the prior location
`E[Σ] ∝ diag(σ̂²)` and supplies the prior degrees of freedom.
"""
function _bvar_dummy_obs(Y0::AbstractMatrix{T}, lags::Int, sigma_ar::Vector{T},
                         lambda::T, theta::T, miu::T, alpha::T) where {T<:AbstractFloat}
    N = size(Y0, 2)
    k = 1 + N * lags

    # Mean of initial observations (NaN-safe for mixed-frequency data)
    y_bar = zeros(T, N)
    for j in 1:N
        col = Y0[:, j]
        valid = filter(!isnan, col)
        y_bar[j] = isempty(valid) ? zero(T) : mean(valid)
    end

    dummy_Y = Matrix{T}(undef, 0, N)
    dummy_X = Matrix{T}(undef, 0, k)

    # 1. Minnesota tightness dummies (Litterman 1986 / BGR 2010): one row per
    # (lag, variable), single nonzero. Dummy ∝ lag^theta / lambda so higher lags
    # receive MORE prior weight (#572).
    Y_mn = zeros(T, N * lags, N)
    X_mn = zeros(T, N * lags, k)
    row = 0
    for lag in 1:lags, i in 1:N
        row += 1
        scale = sigma_ar[i] * T(lag)^theta / lambda
        lag == 1 && (Y_mn[row, i] = scale)   # random-walk mean on lag 1 only
        X_mn[row, 1 + (lag - 1) * N + i] = scale
    end
    dummy_Y = vcat(dummy_Y, Y_mn)
    dummy_X = vcat(dummy_X, X_mn)

    # 2. Sum-of-coefficients prior (unit root)
    if miu > 0
        Y_d = zeros(T, N, N)
        X_d = zeros(T, N, k)
        for i in 1:N
            Y_d[i, i] = y_bar[i] / miu
            for lag in 1:lags
                X_d[i, 1 + (lag - 1) * N + i] = y_bar[i] / miu
            end
        end
        dummy_Y = vcat(dummy_Y, Y_d)
        dummy_X = vcat(dummy_X, X_d)
    end

    # 3. Co-persistence prior (common stochastic trend)
    if alpha > 0
        Y_d = y_bar' / alpha
        X_d = zeros(T, 1, k)
        X_d[1, 1] = one(T) / alpha  # intercept
        for lag in 1:lags
            X_d[1, (1 + (lag - 1) * N + 1):(1 + lag * N)] = y_bar' / alpha
        end
        dummy_Y = vcat(dummy_Y, Y_d)
        dummy_X = vcat(dummy_X, X_d)
    end

    # 4. Inverse-Wishart scale prior (Sims varprior / GLP): X = 0, so it constrains Σ only.
    # Blocks 1-3 are all satisfied exactly by the random-walk B, so this block is what keeps
    # the prior SSR (and hence the NIW marginal likelihood) nonsingular (#571).
    dummy_Y = vcat(dummy_Y, diagm(sigma_ar))
    dummy_X = vcat(dummy_X, zeros(T, N, k))

    return dummy_Y, dummy_X
end

# =============================================================================
# Litterman (non-conjugate, fixed diagonal Sigma) prior — #602
# =============================================================================

"""
    _litterman_prior_rows(m, N, lags, sigma_ar, y_bar, lambda, theta, theta_cross,
                          miu, alpha) -> (A, c)

Prior rows `A β_m ≈ c` for equation `m` of the Litterman (1986) prior.

The prior is `β_m ~ N(b_m, (A'A)^{-1})` with `A'A b_m = A'c`, i.e. exactly the
dummy-observation representation — but written **per equation**, which is the whole point:
`A` depends on `m` through the `σ_m/σ_j` cross terms, so the implied `Var(vec(B))` is no
longer `Σ ⊗ V` and the own-versus-cross ratio stops being pinned at `√(Σ_mm/Σ_jj)` (#602).

Prior standard deviations for the coefficient on lag `l` of variable `j` in equation `m`:

| | prior SD | prior mean |
|---|---|---|
| own (`j == m`) | `λ / l^θ` | 1 at `l == 1`, else 0 |
| cross (`j != m`) | `λ · θ_cross · σ_m / (l^θ · σ_j)` | 0 |
| intercept | `λ · _LITTERMAN_INTERCEPT_SCALE` (diffuse) | 0 |

`θ_cross = 1` reproduces the conjugate prior's Minnesota asymmetry; `θ_cross < 1` shrinks
other variables' lags harder than own lags. The `σ_m/σ_j` factor puts the cross coefficient
on the scale of the ratio of residual standard deviations, so the restriction is invariant
to the units each series is measured in.

Rows for the sum-of-coefficients (`miu`) and co-persistence (`alpha`) priors follow the
conjugate construction, restricted to the column of the dummy block that belongs to
equation `m`; they are what pin the long-run behaviour on a persistent macro panel, so the
two priors impose comparable structure and their fits are worth comparing.
"""
const _LITTERMAN_INTERCEPT_SCALE = 100.0    # diffuse but proper intercept prior

function _litterman_prior_rows(m::Int, N::Int, lags::Int, sigma_ar::Vector{T},
                               y_bar::Vector{T}, lambda::T, theta::T, theta_cross::T,
                               miu::T, alpha::T) where {T<:AbstractFloat}
    k = 1 + N * lags
    n_mn = k                                   # one Minnesota row per coefficient
    n_soc = miu > 0 ? N : 0
    n_cop = alpha > 0 ? 1 : 0
    A = zeros(T, n_mn + n_soc + n_cop, k)
    c = zeros(T, n_mn + n_soc + n_cop)

    # --- Minnesota block: one row per coefficient, weight = 1/prior SD ---
    A[1, 1] = one(T) / (lambda * T(_LITTERMAN_INTERCEPT_SCALE))     # intercept, mean 0
    row = 1
    for lag in 1:lags, j in 1:N
        row += 1
        col = 1 + (lag - 1) * N + j
        sd = j == m ? lambda / T(lag)^theta :
                      lambda * theta_cross * sigma_ar[m] / (T(lag)^theta * sigma_ar[j])
        w = one(T) / max(sd, eps(T))
        A[row, col] = w
        (j == m && lag == 1) && (c[row] = w)                        # random-walk prior mean
    end

    # --- Sum-of-coefficients: own lags of variable i sum to 1 (target ȳ_m only for i == m) ---
    if miu > 0
        for i in 1:N
            r = n_mn + i
            for lag in 1:lags
                A[r, 1 + (lag - 1) * N + i] = y_bar[i] / miu
            end
            i == m && (c[r] = y_bar[i] / miu)
        end
    end

    # --- Co-persistence: a single common stochastic trend ---
    if alpha > 0
        r = n_mn + n_soc + 1
        A[r, 1] = one(T) / alpha
        for lag in 1:lags, j in 1:N
            A[r, 1 + (lag - 1) * N + j] = y_bar[j] / alpha
        end
        c[r] = y_bar[m] / alpha
    end

    return A, c
end

"""
    _litterman_estimate(Y, lags, sigma_ar, lambda, theta, theta_cross, miu, alpha)
        -> (beta, sigma, logml)

Posterior mean and log marginal likelihood of the Litterman prior with `Σ = diag(σ̂²)` held
fixed. The equations separate, so this is one `k × k` solve each — no MCMC.

With known `σ_m²` and prior `β_m ~ N(b_m, P_m^{-1})`, completing the square gives

    p(y_m) = (2πσ_m²)^{-T/2} |P_m|^{1/2} |M_m|^{-1/2}
             exp(-½[ y'y/σ_m² + b'P b − g'M^{-1}g ])

with `M_m = P_m + X'X/σ_m²` and `g_m = P_m b_m + X'y_m/σ_m²`. `β̂_m = M_m^{-1} g_m` is the
posterior mean. The reported `logml` is `Σ_m log p(y_m)`: a genuine marginal likelihood over
the coefficients, so it penalizes complexity and is comparable across `(λ, θ, θ_cross, μ,
α)`. It is **not** comparable to the conjugate NIW value — that one also integrates out `Σ`,
which this model holds fixed.
"""
function _litterman_estimate(Y::Matrix{T}, lags::Int, sigma_ar::Vector{T},
                             lambda::T, theta::T, theta_cross::T,
                             miu::T, alpha::T) where {T<:AbstractFloat}
    T_obs, N = size(Y)
    Y_dep = Y[(lags + 1):end, :]
    T_eff = size(Y_dep, 1)
    X_reg = ones(T, T_eff, 1)
    for lag in 1:lags
        X_reg = hcat(X_reg, Y[(lags + 1 - lag):(end - lag), :])
    end
    k = size(X_reg, 2)

    y_bar = zeros(T, N)
    for j in 1:N
        valid = filter(!isnan, @view Y[1:lags, j])
        y_bar[j] = isempty(valid) ? zero(T) : mean(valid)
    end

    beta = zeros(T, k, N)
    sigma = Matrix{T}(Diagonal(sigma_ar .^ 2))
    if !all(isfinite, X_reg) || !all(isfinite, Y_dep)
        return beta, sigma, -T(1e10)
    end

    XtX = X_reg' * X_reg
    logml = zero(T)
    for m in 1:N
        s2 = max(sigma_ar[m]^2, eps(T))
        A, c = _litterman_prior_rows(m, N, lags, sigma_ar, y_bar, lambda, theta,
                                     theta_cross, miu, alpha)
        P = A' * A
        P = (P + P') / T(2)
        b = try
            A \ c                                  # least-squares prior mean; P b = A'c
        catch
            return beta, sigma, -T(1e10)
        end

        M = P + XtX / s2
        M = (M + M') / T(2)
        y_m = @view Y_dep[:, m]
        g = P * b + (X_reg' * y_m) / s2

        chol_M = try
            cholesky(Hermitian(M))
        catch
            return beta, sigma, -T(1e10)
        end
        bm = chol_M \ g
        beta[:, m] = bm

        chol_P = try
            cholesky(Hermitian(P))
        catch
            return beta, sigma, -T(1e10)
        end
        quad = dot(y_m, y_m) / s2 + dot(b, P * b) - dot(g, bm)
        logml += -T(0.5) * T_eff * log(2 * T(π) * s2) +
                 T(0.5) * logdet(chol_P) - T(0.5) * logdet(chol_M) -
                 T(0.5) * quad
    end

    if !all(isfinite, beta) || !isfinite(logml)
        return zeros(T, k, N), Matrix{T}(Diagonal(sigma_ar .^ 2)), -T(1e10)
    end
    return beta, sigma, logml
end

"""Log marginal likelihood of the Litterman prior at log-hyperparameters `par`
(`[log λ, log θ, log θ_cross, log μ, log α]`), boxed to `|par| ≤ 5` like the conjugate path."""
function _litterman_log_ml(par::AbstractVector{T}, Y::Matrix{T}, lags::Int,
                           sigma_ar::Vector{T}) where {T<:AbstractFloat}
    any(x -> abs(x) > T(5), par) && return -T(1e10)
    _, _, ml = _litterman_estimate(Y, lags, sigma_ar, exp(par[1]), exp(par[2]),
                                   exp(par[3]), exp(par[4]), exp(par[5]))
    return isfinite(ml) ? ml : -T(1e10)
end

# =============================================================================
# Kalman Smoother for Ragged Edge
# =============================================================================

"""
    _bvar_smooth_missing(Y, beta, sigma, lags, t_complete) -> Matrix

Fill the missing entries of the panel `Y` (interior NaNs and the ragged edge) with a genuine
Kalman smoother. The estimated BVAR(`lags`) is cast in companion state-space form
(state `[y_t; …; y_{t-lags+1}]`, transition from the lag blocks of `beta`, state-noise `sigma`,
observation `C = [I 0]`, tiny measurement ridge) and the missing-data filter/RTS smoother from
`kalman_missing.jl` is run on the mean-centred panel. Because that smoother drops only the
missing rows each period, contemporaneously OBSERVED variables update the unobserved states
through the state covariance — so a released series informs the fill of an unreleased one,
which the previous interpolation + deterministic-projection routine ignored. `t_complete` is
retained for signature compatibility; the smoother fills every missing entry uniformly.
"""
function _bvar_smooth_missing(Y::Matrix{T}, beta::Matrix{T}, sigma::Matrix{T},
                               lags::Int, t_complete::Int) where {T<:AbstractFloat}
    T_obs, N = size(Y)
    sd = N * lags

    # Companion state-space form of the BVAR. beta is (1 + N*lags) × N (row 1 = intercept,
    # then lag blocks); B[i][j,m] = coefficient of y_{t-i}[m] in equation j.
    c = Vector{T}(beta[1, :])
    B = [permutedims(Matrix{T}(beta[(2 + (i - 1) * N):(1 + i * N), :])) for i in 1:lags]
    A = zeros(T, sd, sd)
    for i in 1:lags
        A[1:N, ((i - 1) * N + 1):(i * N)] = B[i]
    end
    lags > 1 && (A[(N + 1):sd, 1:(sd - N)] = Matrix{T}(I, sd - N, sd - N))
    Q = zeros(T, sd, sd); Q[1:N, 1:N] = T(0.5) * (sigma + sigma')
    C = zeros(T, N, sd); C[1:N, 1:N] = Matrix{T}(I, N, N)
    R = Matrix{T}(I, N, N) * T(1e-8)              # observed series measured (nearly) exactly

    # Centre by the steady-state mean so the centred VAR is intercept-free (mean-zero);
    # fall back to per-column observed means if I - ΣBᵢ is near-singular (near unit root).
    mu_emp = [begin v = filter(!isnan, @view Y[:, j]); isempty(v) ? zero(T) : T(mean(v)) end for j in 1:N]
    Imb = Matrix{T}(I, N, N) - sum(B)
    mu = let ss = try Imb \ c catch; fill(T(Inf), N) end
        (all(isfinite, ss) && maximum(abs, ss) < T(1e6)) ? ss : mu_emp
    end

    Yc = Y .- mu'                                  # NaN stays NaN → dropped per period by _miss_data
    x0 = zeros(T, sd)
    P0 = _compute_unconditional_covariance(A, Q, sd)
    all(isfinite, P0) || (P0 = Matrix{T}(I, sd, sd) * T(1e6))
    x_smooth, _, _, _ = _kalman_smoother_missing(Matrix{T}(Yc'), A, C, Q, R, x0, P0)

    # Fill only the missing entries with the (un-centred) smoothed current-period state.
    X_sm = copy(Y)
    @inbounds for t in 1:T_obs, j in 1:N
        isnan(Y[t, j]) && (X_sm[t, j] = mu[j] + x_smooth[j, t])
    end
    X_sm
end
