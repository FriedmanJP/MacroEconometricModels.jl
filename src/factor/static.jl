"""
Static Factor Model via Principal Component Analysis.

Implements Bai & Ng (2002) static factor model: X_t = Λ F_t + e_t

References:
- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models.
  Econometrica, 70(1), 191-221.
"""

using LinearAlgebra, Statistics, StatsAPI

# =============================================================================
# Factor Model Type
# =============================================================================

"""
    FactorModel{T} <: AbstractFactorModel

Static factor model via PCA: Xₜ = Λ Fₜ + eₜ.

Fields: X, factors, loadings, eigenvalues, explained_variance, cumulative_variance, r, standardized.
"""
struct FactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    factors::Matrix{T}
    loadings::Matrix{T}
    eigenvalues::Vector{T}
    explained_variance::Vector{T}
    cumulative_variance::Vector{T}
    r::Int
    standardized::Bool
end

# =============================================================================
# Static Factor Model Estimation
# =============================================================================

"""
    estimate_factors(X, r; standardize=true) -> FactorModel

Estimate static factor model X_t = Λ F_t + e_t via Principal Component Analysis.

# Arguments
- `X`: Data matrix (T × N), observations × variables
- `r`: Number of factors to extract

# Keyword Arguments
- `standardize::Bool=true`: Standardize data before estimation

# Returns
`FactorModel` containing factors, loadings, eigenvalues, and explained variance.

# Example
```julia
X = randn(200, 50)  # 200 observations, 50 variables
fm = estimate_factors(X, 3)  # Extract 3 factors
r2(fm)  # R² for each variable
```
"""
function estimate_factors(X::AbstractMatrix{T}, r::Int; standardize::Bool=true) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    X_orig = copy(X)
    X_proc = standardize ? _standardize(X) : X

    # Eigendecomposition of sample covariance
    Σ = (X_proc'X_proc) / T_obs
    eig = eigen(Symmetric(Σ))
    idx = sortperm(eig.values, rev=true)
    λ, V = eig.values[idx], eig.vectors[:, idx]

    # Extract loadings and factors
    loadings = V[:, 1:r] * Diagonal(sqrt.(λ[1:r]))
    factors = X_proc * V[:, 1:r]

    # Variance explained
    total = sum(λ)
    expl = λ / total
    cumul = cumsum(expl)

    FactorModel{T}(X_orig, factors, loadings, λ, expl, cumul, r, standardize)
end

@float_fallback estimate_factors X

function Base.show(io::IO, m::FactorModel{T}) where {T}
    Tobs, N = size(m.X)
    spec = Any[
        "Factors"       m.r;
        "Variables"     N;
        "Observations"  Tobs;
        "Standardized"  m.standardized ? "Yes" : "No"
    ]
    _pretty_table(io, spec;
        title = "Static Factor Model (r=$(m.r))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    # Variance explained
    n_show = min(m.r, 5)
    var_data = Matrix{Any}(undef, n_show, 3)
    for i in 1:n_show
        var_data[i, 1] = "Factor $i"
        var_data[i, 2] = _fmt_pct(m.explained_variance[i])
        var_data[i, 3] = _fmt_pct(m.cumulative_variance[i])
    end
    _pretty_table(io, var_data;
        title = "Variance Explained",
        column_labels = ["", "Variance", "Cumulative"],
        alignment = [:l, :r, :r],
    )
    # Top loadings per factor (top 5 for up to 3 factors)
    n_factors_show = min(m.r, 3)
    n_top = min(5, N)
    for f in 1:n_factors_show
        loadings_f = m.loadings[:, f]
        sorted_idx = sortperm(abs.(loadings_f); rev=true)
        top_idx = sorted_idx[1:n_top]
        load_data = Matrix{Any}(undef, n_top, 3)
        for (row, idx) in enumerate(top_idx)
            load_data[row, 1] = "Var $idx"
            load_data[row, 2] = _fmt(loadings_f[idx])
            load_data[row, 3] = _fmt(abs(loadings_f[idx]))
        end
        _pretty_table(io, load_data;
            title = "Top Loadings — Factor $f",
            column_labels = ["Variable", "Loading", "|Loading|"],
            alignment = [:l, :r, :r],
        )
    end
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Predicted values: F * Λ'."""
StatsAPI.predict(m::FactorModel) = m.factors * m.loadings'

"""Residuals: X - predicted."""
function StatsAPI.residuals(m::FactorModel{T}) where {T}
    fitted = predict(m)
    m.standardized ? _standardize(m.X) - fitted : m.X - fitted
end

"""R² for each variable."""
function StatsAPI.r2(m::FactorModel{T}) where {T}
    resid = residuals(m)
    X_ref = m.standardized ? _standardize(m.X) : m.X
    [max(zero(T), 1 - var(@view(resid[:, i])) / max(var(@view(X_ref[:, i])), T(1e-10)))
     for i in 1:size(m.X, 2)]
end

"""Number of observations."""
StatsAPI.nobs(m::FactorModel) = size(m.X, 1)

"""Degrees of freedom."""
StatsAPI.dof(m::FactorModel) = size(m.X, 2) * m.r + size(m.X, 1) * m.r - m.r^2

# =============================================================================
# Information Criteria (Bai & Ng 2002)
# =============================================================================

"""
    ic_criteria(X, max_factors; standardize=true)

Compute Bai-Ng (2002) information criteria IC1, IC2, IC3 for selecting the number of factors.

# Arguments
- `X`: Data matrix (T × N)
- `max_factors`: Maximum number of factors to consider

# Returns
Named tuple with IC values and optimal factor counts:
- `IC1`, `IC2`, `IC3`: Information criteria vectors
- `r_IC1`, `r_IC2`, `r_IC3`: Optimal factor counts

# Example
```julia
result = ic_criteria(X, 10)
println("Optimal factors: IC1=", result.r_IC1, ", IC2=", result.r_IC2, ", IC3=", result.r_IC3)
```
"""
function ic_criteria(X::AbstractMatrix{T}, max_factors::Int; standardize::Bool=true) where {T<:AbstractFloat}
    T_obs, N = size(X)
    1 <= max_factors <= min(T_obs, N) || throw(ArgumentError("max_factors must be in [1, min(T,N)]"))

    IC1, IC2, IC3 = Vector{T}(undef, max_factors), Vector{T}(undef, max_factors), Vector{T}(undef, max_factors)
    NT, minNT = N * T_obs, min(N, T_obs)

    for r in 1:max_factors
        resid = residuals(estimate_factors(X, r; standardize))
        V_r = sum(resid .^ 2) / NT
        logV = log(V_r)
        pen_base = r * (N + T_obs) / NT

        IC1[r] = logV + pen_base * log(NT / (N + T_obs))
        IC2[r] = logV + pen_base * log(minNT)
        IC3[r] = logV + r * log(minNT) / minNT
    end

    (IC1=IC1, IC2=IC2, IC3=IC3, r_IC1=argmin(IC1), r_IC2=argmin(IC2), r_IC3=argmin(IC3))
end

# =============================================================================
# Visualization Helpers
# =============================================================================

"""
    scree_plot_data(m::FactorModel)

Return data for scree plot: factor indices, explained variance, cumulative variance.

# Example
```julia
data = scree_plot_data(fm)
# Plot: data.factors vs data.explained_variance
```
"""
scree_plot_data(m::FactorModel) = (factors=1:length(m.eigenvalues), explained_variance=m.explained_variance,
                                    cumulative_variance=m.cumulative_variance)

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::FactorModel, h; p=1, ci_method=:none, conf_level=0.95, n_boot=1000)

Forecast factors and observables h steps ahead from a static factor model.

Internally fits a VAR(p) on the extracted factors, then uses the VAR dynamics
to produce multi-step forecasts and (optionally) confidence intervals.

# Arguments
- `model`: Estimated static factor model
- `h`: Forecast horizon

# Keyword Arguments
- `p::Int=1`: VAR lag order for factor dynamics
- `ci_method::Symbol=:none`: CI method — `:none`, `:theoretical`, or `:bootstrap`
- `conf_level::Real=0.95`: Confidence level for intervals
- `n_boot::Int=1000`: Number of bootstrap replications (if `ci_method=:bootstrap`)

# Returns
`FactorForecast` with factor and observable forecasts (and CIs if requested).
"""
function forecast(m::FactorModel{T}, h::Int; p::Int=1, ci_method::Symbol=:none,
    conf_level::Real=0.95, n_boot::Int=1000) where {T}

    h < 1 && throw(ArgumentError("h must be ≥ 1"))
    p < 1 && throw(ArgumentError("p must be ≥ 1"))
    ci_method ∈ (:none, :theoretical, :bootstrap) || throw(ArgumentError("ci_method must be :none, :theoretical, or :bootstrap"))

    r = m.r
    T_obs, N = size(m.X)
    F = m.factors
    Lambda = m.loadings

    # Fit VAR(p) on extracted factors
    var_model = estimate_var(F, p)
    A = [Matrix{T}(var_model.B[(2+(lag-1)*r):(1+lag*r), :]') for lag in 1:p]
    Sigma_eta = var_model.Sigma

    # Idiosyncratic covariance from PCA residuals
    X_proc = m.standardized ? _standardize(m.X) : m.X
    e = X_proc - F * Lambda'
    Sigma_e = diagm(vec(var(e, dims=1)))

    # Last p factor vectors (most recent first)
    F_last = [F[T_obs-lag+1, :] for lag in 1:p]

    # Point forecasts
    F_fc = zeros(T, h, r)
    X_fc = zeros(T, h, N)
    for step in 1:h
        F_h = sum(A[lag] * (step - lag >= 1 ? F_fc[step - lag, :] : F_last[lag - step + 1]) for lag in 1:p)
        F_fc[step, :] = F_h
        X_fc[step, :] = Lambda * F_h
    end

    conf_T = T(conf_level)

    if ci_method == :none
        z = zeros(T, h, r)
        zx = zeros(T, h, N)
        if m.standardized
            _unstandardize_factor_forecast!(X_fc, zx, zx, zx, m.X)
        end
        return _build_factor_forecast(F_fc, X_fc, z, z, zx, copy(zx), z, copy(zx), h, conf_T, :none)
    end

    if ci_method == :theoretical
        factor_mse = _factor_forecast_var_theoretical(A, Sigma_eta, r, p, h)
        z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))

        F_se = Matrix{T}(undef, h, r)
        for step in 1:h
            F_se[step, :] = sqrt.(max.(diag(factor_mse[step]), zero(T)))
        end
        F_lo = F_fc .- z_val .* F_se
        F_hi = F_fc .+ z_val .* F_se

        X_se = _factor_forecast_obs_se(factor_mse, Lambda, Sigma_e, h)
        X_lo = X_fc .- z_val .* X_se
        X_hi = X_fc .+ z_val .* X_se

        if m.standardized
            _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, m.X)
        end
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :theoretical)
    end

    # Bootstrap
    factor_resids = var_model.U
    f_lo, f_hi, o_lo, o_hi, f_se, o_se = _factor_forecast_bootstrap(
        F_last, A, factor_resids, Sigma_e, Lambda, h, r, p, n_boot, conf_T)

    if m.standardized
        _unstandardize_factor_forecast!(X_fc, o_lo, o_hi, o_se, m.X)
    end
    _build_factor_forecast(F_fc, X_fc, f_lo, f_hi, o_lo, o_hi, f_se, o_se, h, conf_T, :bootstrap)
end
