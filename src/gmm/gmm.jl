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
Generalized Method of Moments (GMM) estimation framework.

Provides flexible GMM estimation with support for:
- One-step, two-step, and iterated GMM
- Optimal weighting matrix estimation with HAC correction
- Hansen's J-test for overidentification
- Numerical gradient computation

References:
- Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments
  Estimators." Econometrica, 50(4), 1029-1054.
- Newey, W. K., & McFadden, D. (1994). "Large Sample Estimation and Hypothesis
  Testing." Handbook of Econometrics, Vol. 4.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# GMM Types
# =============================================================================

"""Abstract supertype for GMM models."""
abstract type AbstractGMMModel <: StatsAPI.StatisticalModel end

"""
    GMMWeighting{T} <: Any

GMM weighting matrix specification.

Fields:
- method: Weighting method (:identity, :optimal, :two_step, :iterated)
- max_iter: Maximum iterations for iterated GMM
- tol: Convergence tolerance
"""
struct GMMWeighting{T<:AbstractFloat}
    method::Symbol
    max_iter::Int
    tol::T

    function GMMWeighting{T}(method::Symbol, max_iter::Int, tol::T) where {T<:AbstractFloat}
        method ∉ (:identity, :optimal, :two_step, :iterated) &&
            throw(ArgumentError("method must be :identity, :optimal, :two_step, or :iterated"))
        @assert max_iter > 0
        @assert tol > 0
        new{T}(method, max_iter, tol)
    end
end

GMMWeighting(; method::Symbol=:two_step, max_iter::Int=100, tol::Real=1e-8) =
    GMMWeighting{Float64}(method, max_iter, Float64(tol))

"""
    GMMModel{T} <: AbstractGMMModel

Generalized Method of Moments estimator.

Minimizes: g(θ)'W g(θ) where g(θ) = (1/n) Σᵢ gᵢ(θ)

Fields:
- theta: Parameter estimates
- vcov: Asymptotic covariance matrix
- n_moments: Number of moment conditions
- n_params: Number of parameters
- n_obs: Number of observations
- weighting: Weighting specification
- W: Final weighting matrix
- g_bar: Sample moment vector at solution
- J_stat: Hansen's J-test statistic
- J_pvalue: p-value for J-test
- converged: Convergence flag
- iterations: Number of iterations
"""
struct GMMModel{T<:AbstractFloat} <: AbstractGMMModel
    theta::Vector{T}
    vcov::Matrix{T}
    n_moments::Int
    n_params::Int
    n_obs::Int
    weighting::GMMWeighting{T}
    W::Matrix{T}
    g_bar::Vector{T}
    J_stat::T
    J_pvalue::T
    converged::Bool
    iterations::Int

    function GMMModel{T}(theta::Vector{T}, vcov::Matrix{T}, n_moments::Int, n_params::Int,
                         n_obs::Int, weighting::GMMWeighting{T}, W::Matrix{T}, g_bar::Vector{T},
                         J_stat::T, J_pvalue::T, converged::Bool, iterations::Int) where {T<:AbstractFloat}
        @assert length(theta) == n_params
        @assert size(vcov) == (n_params, n_params)
        @assert size(W) == (n_moments, n_moments)
        @assert length(g_bar) == n_moments
        @assert n_moments >= n_params "GMM requires at least as many moments as parameters"
        @assert J_stat >= 0
        @assert 0 <= J_pvalue <= 1
        new{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W, g_bar,
               J_stat, J_pvalue, converged, iterations)
    end
end

is_overidentified(model::GMMModel) = model.n_moments > model.n_params
overid_df(model::GMMModel) = model.n_moments - model.n_params

function Base.show(io::IO, m::GMMModel{T}) where {T}
    spec = Any[
        "Parameters"  m.n_params;
        "Moments"     m.n_moments;
        "Observations" m.n_obs;
        "Weighting"   string(m.weighting.method);
        "Converged"   m.converged ? "Yes" : "No";
        "Iterations"  m.iterations
    ]
    _pretty_table(io, spec;
        title = "GMM Estimation Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # Coefficient table
    se = sqrt.(max.(diag(m.vcov), zero(T)))
    param_names = ["θ[$i]" for i in 1:m.n_params]
    _coef_table(io, "Coefficients", param_names, m.theta, se; dist=:z)
    # J-test
    if is_overidentified(m)
        j_data = Any[
            "J-statistic" _fmt(m.J_stat);
            "P-value"     _format_pvalue(m.J_pvalue);
            "DF"          overid_df(m)
        ]
        _pretty_table(io, j_data;
            title = "Hansen J-test",
            column_labels = ["", ""],
            alignment = [:l, :r],
        )
    end
end

# StatsAPI interface for GMMModel
StatsAPI.coef(model::GMMModel) = model.theta
StatsAPI.vcov(model::GMMModel) = model.vcov
StatsAPI.nobs(model::GMMModel) = model.n_obs
StatsAPI.dof(model::GMMModel) = model.n_params
StatsAPI.islinear(::GMMModel) = false
StatsAPI.stderror(model::GMMModel) = sqrt.(diag(model.vcov))

function StatsAPI.confint(model::GMMModel{T}; level::Real=0.95) where {T}
    se = stderror(model)
    z = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(model.theta .- z .* se, model.theta .+ z .* se)
end

# =============================================================================
# Numerical Gradient
# =============================================================================

"""
    numerical_gradient(f::Function, x::AbstractVector{T}; eps::T=T(1e-7)) -> Matrix{T}

Compute numerical gradient (Jacobian) of function f at point x using central differences.

Arguments:
- f: Function that takes vector x and returns vector (moment conditions)
- x: Point at which to evaluate gradient
- eps: Step size for finite differences

Returns:
- Jacobian matrix (n_moments × n_params)
"""
function numerical_gradient(f::Function, x::AbstractVector{T}; eps::T=T(1e-7)) where {T<:AbstractFloat}
    n = length(x)
    f0 = f(x)
    m = length(f0)

    J = Matrix{T}(undef, m, n)

    @inbounds for j in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[j] += eps
        x_minus[j] -= eps

        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)
    end

    J
end

# =============================================================================
# GMM Objective Function
# =============================================================================

"""
    gmm_objective(theta::AbstractVector{T}, moment_fn::Function, data,
                  W::AbstractMatrix{T}) -> T

Compute GMM objective: Q(θ) = g(θ)'W g(θ)

where g(θ) = (1/n) Σᵢ gᵢ(θ,data)

Arguments:
- theta: Parameter vector
- moment_fn: Function (theta, data) -> Matrix of moment conditions (n × q)
- data: Data passed to moment function
- W: Weighting matrix (q × q)

Returns:
- GMM objective value
"""
function gmm_objective(theta::AbstractVector{T}, moment_fn::Function, data,
                       W::AbstractMatrix{T}) where {T<:AbstractFloat}
    G = moment_fn(theta, data)  # n × q matrix
    g_bar = vec(mean(G, dims=1))  # Sample mean of moments
    g_bar' * W * g_bar
end

# =============================================================================
# Weighting Matrix Estimation
# =============================================================================

"""
    identity_weighting(n_moments::Int, ::Type{T}=Float64) -> Matrix{T}

Identity weighting matrix (one-step GMM).
"""
function identity_weighting(n_moments::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    Matrix{T}(I, n_moments, n_moments)
end

"""
    optimal_weighting_matrix(moment_fn::Function, theta::AbstractVector{T}, data;
                             hac::Bool=true, bandwidth::Int=0) -> Matrix{T}

Compute optimal GMM weighting matrix: W = inv(Var(g)).

For i.i.d. data: W = inv((1/n) Σᵢ gᵢ gᵢ')
For time series: Uses HAC estimation with Newey-West kernel.

Arguments:
- moment_fn: Moment function
- theta: Current parameter estimate
- data: Data
- hac: Use HAC correction for serial correlation
- bandwidth: HAC bandwidth (0 = automatic)

Returns:
- Optimal weighting matrix (q × q)
"""
function optimal_weighting_matrix(moment_fn::Function, theta::AbstractVector{T}, data;
                                  hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    G = moment_fn(theta, data)  # n × q
    n, q = size(G)

    # Demean
    G_demean = G .- mean(G, dims=1)

    if hac
        # HAC covariance with Newey-West kernel
        Omega = long_run_covariance(G_demean; bandwidth=bandwidth, kernel=:bartlett)
    else
        # Simple covariance (i.i.d. assumption)
        Omega = (G_demean' * G_demean) / n
    end

    # Ensure positive definite and invert
    Omega_sym = Hermitian((Omega + Omega') / 2)

    # Regularize if needed
    eigvals_O = eigvals(Omega_sym)
    if minimum(eigvals_O) < eps(T)
        Omega_reg = Omega_sym + T(1e-8) * I
        return inv(Omega_reg)
    end

    robust_inv(Matrix(Omega_sym))
end

# =============================================================================
# GMM Optimization
# =============================================================================

"""
    minimize_gmm(moment_fn::Function, theta0::AbstractVector{T}, data,
                 W::AbstractMatrix{T}; max_iter::Int=100, tol::T=T(1e-8)) -> NamedTuple

Minimize GMM objective using gradient descent with BFGS-like updates.

Returns:
- theta: Minimizer
- objective: Final objective value
- converged: Convergence flag
- iterations: Number of iterations
"""
function minimize_gmm(moment_fn::Function, theta0::AbstractVector{T}, data,
                      W::AbstractMatrix{T}; max_iter::Int=100, tol::T=T(1e-8)) where {T<:AbstractFloat}
    n_params = length(theta0)
    theta = copy(theta0)

    # Objective function wrapper
    function obj(t)
        gmm_objective(t, moment_fn, data, W)
    end

    # Gradient of objective
    function grad(t)
        G = moment_fn(t, data)
        g_bar = vec(mean(G, dims=1))

        # ∂Q/∂θ = 2 * (∂g/∂θ)' * W * g
        dg_dtheta = numerical_gradient(t_ -> vec(mean(moment_fn(t_, data), dims=1)), t)
        2 * dg_dtheta' * W * g_bar
    end

    # Simple gradient descent with line search
    H_inv = Matrix{T}(I, n_params, n_params)  # Approximate inverse Hessian
    obj_prev = obj(theta)

    for iter in 1:max_iter
        g = grad(theta)

        # Check convergence
        if norm(g) < tol
            return (theta=theta, objective=obj_prev, converged=true, iterations=iter)
        end

        # Search direction
        d = -H_inv * g

        # Line search (backtracking)
        alpha = one(T)
        c1 = T(1e-4)
        rho = T(0.5)

        obj_new = obj(theta + alpha * d)
        while obj_new > obj_prev + c1 * alpha * dot(g, d) && alpha > T(1e-10)
            alpha *= rho
            obj_new = obj(theta + alpha * d)
        end

        if alpha < T(1e-10)
            return (theta=theta, objective=obj_prev, converged=false, iterations=iter)
        end

        # Update
        s = alpha * d
        theta_new = theta + s
        g_new = grad(theta_new)
        y = g_new - g

        # BFGS update of inverse Hessian
        if dot(s, y) > T(1e-10)
            rho_bfgs = one(T) / dot(s, y)
            I_mat = Matrix{T}(I, n_params, n_params)
            H_inv = (I_mat - rho_bfgs * s * y') * H_inv * (I_mat - rho_bfgs * y * s') +
                    rho_bfgs * s * s'
        end

        theta = theta_new
        obj_prev = obj_new
    end

    (theta=theta, objective=obj_prev, converged=false, iterations=max_iter)
end

# =============================================================================
# Main GMM Estimation
# =============================================================================

"""
    estimate_gmm(moment_fn::Function, theta0::AbstractVector{T}, data;
                 weighting::Symbol=:two_step, max_iter::Int=100,
                 tol::T=T(1e-8), hac::Bool=true, bandwidth::Int=0) -> GMMModel{T}

Estimate parameters via Generalized Method of Moments.

Minimizes: Q(θ) = g(θ)'W g(θ) where g(θ) = (1/n) Σᵢ gᵢ(θ)

Arguments:
- moment_fn: Function (theta, data) -> Matrix of moment conditions (n × q)
- theta0: Initial parameter guess
- data: Data passed to moment function
- weighting: Weighting method (:identity, :optimal, :two_step, :iterated)
- max_iter: Maximum iterations for optimization and/or iterated GMM
- tol: Convergence tolerance
- hac: Use HAC correction for optimal weighting
- bandwidth: HAC bandwidth (0 = automatic)

Returns:
- GMMModel with estimates, covariance, and J-test results
"""
function estimate_gmm(moment_fn::Function, theta0::AbstractVector{T}, data;
                      weighting::Symbol=:two_step, max_iter::Int=100,
                      tol::T=T(1e-8), hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    n_params = length(theta0)

    # Get dimensions from initial moment evaluation
    G0 = moment_fn(theta0, data)
    n_obs, n_moments = size(G0)

    @assert n_moments >= n_params "GMM requires at least as many moments as parameters"

    # Weighting specification
    weighting_spec = GMMWeighting{T}(weighting, max_iter, tol)

    # Estimation based on weighting method
    if weighting == :identity
        # One-step GMM with identity weighting
        W = identity_weighting(n_moments, T)
        result = minimize_gmm(moment_fn, theta0, data, W; max_iter=max_iter, tol=tol)
        theta_hat = result.theta
        W_final = W
        converged = result.converged
        iterations = result.iterations

    elseif weighting == :optimal
        # Optimal GMM assuming consistent initial estimate available
        W = optimal_weighting_matrix(moment_fn, theta0, data; hac=hac, bandwidth=bandwidth)
        result = minimize_gmm(moment_fn, theta0, data, W; max_iter=max_iter, tol=tol)
        theta_hat = result.theta
        W_final = W
        converged = result.converged
        iterations = result.iterations

    elseif weighting == :two_step
        # Two-step efficient GMM
        # Step 1: Identity weighting
        W1 = identity_weighting(n_moments, T)
        result1 = minimize_gmm(moment_fn, theta0, data, W1; max_iter=max_iter, tol=tol)

        # Step 2: Optimal weighting based on step 1
        W2 = optimal_weighting_matrix(moment_fn, result1.theta, data; hac=hac, bandwidth=bandwidth)
        result2 = minimize_gmm(moment_fn, result1.theta, data, W2; max_iter=max_iter, tol=tol)

        theta_hat = result2.theta
        W_final = W2
        converged = result2.converged
        iterations = result1.iterations + result2.iterations

    elseif weighting == :iterated
        # Continuously updated GMM (iterate until convergence)
        theta_curr = copy(theta0)
        converged = false
        total_iter = 0

        for iter in 1:max_iter
            W_curr = optimal_weighting_matrix(moment_fn, theta_curr, data; hac=hac, bandwidth=bandwidth)
            result = minimize_gmm(moment_fn, theta_curr, data, W_curr; max_iter=max_iter, tol=tol)

            total_iter += result.iterations

            if norm(result.theta - theta_curr) < tol
                theta_curr = result.theta
                W_final = W_curr
                converged = true
                break
            end

            theta_curr = result.theta
            W_final = W_curr
        end

        theta_hat = theta_curr
        iterations = total_iter

    else
        throw(ArgumentError("weighting must be :identity, :optimal, :two_step, or :iterated"))
    end

    # Compute final moment conditions
    G_final = moment_fn(theta_hat, data)
    g_bar = vec(mean(G_final, dims=1))

    # Asymptotic covariance
    # V = (G'WG)^{-1} * G'W * Ω * W * G * (G'WG)^{-1}
    # where G is the Jacobian of g w.r.t. theta and Ω is the moment covariance

    # Jacobian
    dg_dtheta = numerical_gradient(t -> vec(mean(moment_fn(t, data), dims=1)), theta_hat)

    # Sandwich formula
    bread = dg_dtheta' * W_final * dg_dtheta
    bread_inv = robust_inv(bread)

    # For efficient GMM (W = Ω^{-1}), the variance simplifies to (G'WG)^{-1}
    if weighting in (:optimal, :two_step, :iterated)
        vcov = bread_inv / n_obs
    else
        # General sandwich
        Omega = long_run_covariance(G_final .- mean(G_final, dims=1);
                                     bandwidth=bandwidth, kernel=:bartlett)
        meat = dg_dtheta' * W_final * Omega * W_final * dg_dtheta
        vcov = (bread_inv * meat * bread_inv) / n_obs
    end

    # Hansen's J-test for overidentification
    J_stat, J_pvalue = if n_moments > n_params
        J = n_obs * (g_bar' * W_final * g_bar)
        df = n_moments - n_params
        (J, 1 - cdf(Chisq(df), J))
    else
        (zero(T), one(T))  # Just identified
    end

    GMMModel{T}(theta_hat, vcov, n_moments, n_params, n_obs, weighting_spec,
                W_final, g_bar, J_stat, J_pvalue, converged, iterations)
end

# =============================================================================
# J-Test and Model Diagnostics
# =============================================================================

"""
    j_test(model::GMMModel{T}) -> NamedTuple

Hansen's J-test for overidentifying restrictions.

H0: All moment conditions are valid (E[g(θ₀)] = 0)
H1: Some moment conditions are violated

Returns:
- J_stat: Test statistic
- p_value: p-value from chi-squared distribution
- df: Degrees of freedom (n_moments - n_params)
- reject_05: Whether to reject at 5% level
"""
function j_test(model::GMMModel{T}) where {T<:AbstractFloat}
    df = overid_df(model)

    if df <= 0
        return (J_stat=zero(T), p_value=one(T), df=0, reject_05=false,
                message="Model is just-identified, J-test not applicable")
    end

    (J_stat=model.J_stat, p_value=model.J_pvalue, df=df,
     reject_05=model.J_pvalue < T(0.05))
end

"""
    gmm_summary(model::GMMModel{T}) -> NamedTuple

Summary statistics for GMM estimation.
"""
function gmm_summary(model::GMMModel{T}) where {T<:AbstractFloat}
    se = sqrt.(diag(model.vcov))
    t_stats = model.theta ./ se
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_stats)))

    j_result = j_test(model)

    (theta=model.theta, se=se, t_stats=t_stats, p_values=p_values,
     n_moments=model.n_moments, n_params=model.n_params, n_obs=model.n_obs,
     weighting=model.weighting.method, converged=model.converged,
     iterations=model.iterations, j_test=j_result)
end

# =============================================================================
# LP via GMM
# =============================================================================

"""
    lp_gmm_moments(Y::AbstractMatrix{T}, shock_var::Int, h::Int, theta,
                   lags::Int) -> Matrix{T}

Construct moment conditions for LP estimated via GMM.

Moments: E[Z_t * ε_{t+h}] = 0 where ε_{t+h} = y_{t+h} - θ' * X_t
and Z includes all exogenous variables.

This is useful when you need to impose cross-equation restrictions.
"""
function lp_gmm_moments(Y::AbstractMatrix{T}, shock_var::Int, h::Int, theta,
                        lags::Int) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    t_start = lags + 1
    t_end = T_obs - h

    T_eff = t_end - t_start + 1
    k = 2 + n * lags  # intercept + shock + lagged controls

    # theta is k-vector for single response variable
    @assert length(theta) == k

    # Construct moments: Z_t * ε_{t+h}
    # Z = X (exogenous instruments = regressors in LP)
    moments = Matrix{T}(undef, T_eff, k)

    for (i, t) in enumerate(t_start:t_end)
        # Response
        y_th = Y[t + h, shock_var]  # Response variable (could be generalized)

        # Regressors
        x_t = Vector{T}(undef, k)
        x_t[1] = one(T)  # Intercept
        x_t[2] = Y[t, shock_var]  # Shock

        col = 3
        for lag in 1:lags
            for var in 1:n
                x_t[col] = Y[t - lag, var]
                col += 1
            end
        end

        # Residual
        eps_th = y_th - dot(x_t, theta)

        # Moment: Z_t * ε_{t+h}
        moments[i, :] = x_t .* eps_th
    end

    moments
end

"""
    estimate_lp_gmm(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                    lags::Int=4, weighting::Symbol=:two_step) -> Vector{GMMModel{T}}

Estimate Local Projection via GMM.

Returns a GMMModel for each horizon.
"""
function estimate_lp_gmm(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                         lags::Int=4, weighting::Symbol=:two_step) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    k = 2 + n * lags

    models = Vector{GMMModel{T}}(undef, horizon + 1)

    for h in 0:horizon
        # Moment function for horizon h
        function moment_fn(theta, data)
            lp_gmm_moments(data, shock_var, h, theta, lags)
        end

        # Initial estimate from OLS
        theta0 = zeros(T, k)

        # Estimate via GMM
        models[h + 1] = estimate_gmm(moment_fn, theta0, Y; weighting=weighting)
    end

    models
end

# =============================================================================
# Linear GMM Utilities (for Panel VAR and other linear IV models)
# =============================================================================

"""
    linear_gmm_solve(S_ZX::Matrix{T}, S_Zy::AbstractVecOrMat{T},
                      W::Matrix{T}) -> Vector{T}

Solve linear GMM analytically: E[Z'(y - Xβ)] = 0.

Given pre-aggregated cross-products S_ZX = Σ_i Z_i'X_i and S_Zy = Σ_i Z_i'y_i:

    β = (S_ZX' W S_ZX)⁻¹ S_ZX' W S_Zy

# Arguments
- `S_ZX` — q × k aggregated instrument-regressor cross-product
- `S_Zy` — q × 1 (or q-vector) aggregated instrument-response cross-product
- `W` — q × q weighting matrix

# Returns
- Parameter vector β (length k)
"""
function linear_gmm_solve(S_ZX::Matrix{T}, S_Zy::AbstractVecOrMat{T},
                           W::Matrix{T}) where {T<:AbstractFloat}
    bread = S_ZX' * W * S_ZX
    rhs = S_ZX' * W * (S_Zy isa AbstractVector ? S_Zy : vec(S_Zy))
    robust_inv(bread) * rhs
end

"""
    gmm_sandwich_vcov(S_ZX::Matrix{T}, W::Matrix{T},
                       D_e::Matrix{T}) -> Matrix{T}

Robust sandwich covariance for one-step GMM:

    V = (S_ZX' W S_ZX)⁻¹ S_ZX' W D_e W S_ZX (S_ZX' W S_ZX)⁻¹

where D_e = Σ_i (Z_i e_i)(Z_i e_i)' is the robust moment covariance.

# Arguments
- `S_ZX` — q × k aggregated instrument-regressor cross-product
- `W` — q × q weighting matrix
- `D_e` — q × q robust moment covariance

# Returns
- k × k covariance matrix
"""
function gmm_sandwich_vcov(S_ZX::Matrix{T}, W::Matrix{T},
                            D_e::Matrix{T}) where {T<:AbstractFloat}
    bread_inv = robust_inv(S_ZX' * W * S_ZX)
    bread_inv * (S_ZX' * W * D_e * W * S_ZX) * bread_inv
end

"""
    andrews_lu_mmsc(j_stat::T, n_instruments::Int, n_params::Int,
                     n_obs::Int; hq_criterion::Real=2.1) -> NamedTuple

Andrews-Lu (2001) Model and Moment Selection Criteria based on Hansen J-statistic.

    MMSC_BIC  = J - (c - b) × log(n)
    MMSC_AIC  = J - (c - b) × 2
    MMSC_HQIC = J - Q(c - b) × log(log(n))

where c = moment conditions, b = parameters, n = observations.
Lower values indicate better model specification.

# Arguments
- `j_stat` — Hansen J-test statistic
- `n_instruments` — number of moment conditions (c)
- `n_params` — number of parameters (b)
- `n_obs` — number of observations (n)
- `hq_criterion` — penalty parameter Q for HQIC (default 2.1)

# Returns
NamedTuple `(bic, aic, hqic)`.

# References
- Andrews, D. & Lu, B. (2001). Journal of Econometrics 101(1), 123-164.
"""
function andrews_lu_mmsc(j_stat::T, n_instruments::Int, n_params::Int,
                          n_obs::Int; hq_criterion::Real=2.1) where {T<:AbstractFloat}
    excess = n_instruments - n_params
    bic  = j_stat - T(excess) * log(T(n_obs))
    aic  = j_stat - T(excess) * T(2)
    hqic = j_stat - T(hq_criterion) * T(excess) * log(log(T(n_obs)))
    (bic=bic, aic=aic, hqic=hqic)
end
