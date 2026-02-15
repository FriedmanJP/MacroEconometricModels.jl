"""
Type definitions and StatsAPI interface for ARCH models.
"""

# =============================================================================
# ARCH Model Type
# =============================================================================

"""
    ARCHModel{T} <: AbstractVolatilityModel

ARCH(q) model (Engle 1982): εₜ = σₜ zₜ, σ²ₜ = ω + α₁ε²ₜ₋₁ + ... + αqε²ₜ₋q

# Fields
- `y::Vector{T}`: Original data
- `q::Int`: ARCH order
- `mu::T`: Mean (intercept)
- `omega::T`: Variance intercept (ω > 0)
- `alpha::Vector{T}`: ARCH coefficients [α₁, ..., αq]
- `conditional_variance::Vector{T}`: Estimated conditional variances σ²ₜ
- `standardized_residuals::Vector{T}`: Standardized residuals zₜ = εₜ/σₜ
- `residuals::Vector{T}`: Raw residuals εₜ = yₜ - μ
- `fitted::Vector{T}`: Fitted values (mean)
- `loglik::T`: Log-likelihood
- `aic::T`: Akaike Information Criterion
- `bic::T`: Bayesian Information Criterion
- `method::Symbol`: Estimation method
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations
"""
struct ARCHModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    q::Int
    mu::T
    omega::T
    alpha::Vector{T}
    conditional_variance::Vector{T}
    standardized_residuals::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end

# =============================================================================
# Volatility Forecast Result (shared across all volatility models)
# =============================================================================

"""
    VolatilityForecast{T}

Forecast result from a volatility model.

# Fields
- `forecast::Vector{T}`: Point forecasts of conditional variance
- `ci_lower::Vector{T}`: Lower confidence interval bound
- `ci_upper::Vector{T}`: Upper confidence interval bound
- `se::Vector{T}`: Standard errors of forecasts
- `horizon::Int`: Forecast horizon
- `conf_level::T`: Confidence level (e.g., 0.95)
- `model_type::Symbol`: Source model type (:arch, :garch, :egarch, :gjr_garch, :sv)
"""
struct VolatilityForecast{T<:AbstractFloat}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
    model_type::Symbol
end

# =============================================================================
# Type Accessors
# =============================================================================

"""Return ARCH order q."""
arch_order(m::ARCHModel) = m.q

"""Return persistence Σαᵢ for ARCH model."""
persistence(m::ARCHModel) = sum(m.alpha)

"""Return half-life of volatility shocks: log(0.5) / log(persistence)."""
function halflife(m::ARCHModel)
    p = persistence(m)
    p <= zero(p) && return Inf
    p >= one(p) && return Inf
    log(typeof(p)(0.5)) / log(p)
end

"""Return unconditional variance ω / (1 - Σαᵢ)."""
function unconditional_variance(m::ARCHModel)
    p = persistence(m)
    p >= one(p) && return typeof(m.omega)(Inf)
    m.omega / (one(p) - p)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Number of observations."""
StatsAPI.nobs(m::ARCHModel) = length(m.y)

"""Coefficient vector `[μ, ω, α₁, …, αq]`."""
StatsAPI.coef(m::ARCHModel) = vcat(m.mu, m.omega, m.alpha)

"""Raw residuals ``\\hat{\\varepsilon}_t``."""
StatsAPI.residuals(m::ARCHModel) = m.residuals

"""Conditional variance series ``\\hat{\\sigma}^2_t``."""
StatsAPI.predict(m::ARCHModel) = m.conditional_variance

"""Maximized log-likelihood."""
StatsAPI.loglikelihood(m::ARCHModel) = m.loglik

"""Akaike Information Criterion."""
StatsAPI.aic(m::ARCHModel) = m.aic

"""Bayesian Information Criterion."""
StatsAPI.bic(m::ARCHModel) = m.bic

"""Number of estimated parameters: `2 + q` (μ + ω + q alphas)."""
StatsAPI.dof(m::ARCHModel) = 2 + m.q  # mu + omega + q alphas

"""`false` — ARCH models are nonlinear."""
StatsAPI.islinear(::ARCHModel) = false

# =============================================================================
# Display
# =============================================================================

function _show_volatility_model(io::IO, header::String, m;
                                 alpha::Vector=Float64[],
                                 beta::Vector=Float64[],
                                 gamma::Vector=Float64[])
    # Build parameter names and values
    T_float = eltype(m.y)
    param_names = String["μ (mean)", "ω (intercept)"]
    param_vals = T_float[m.mu, m.omega]
    for (i, a) in enumerate(alpha)
        push!(param_names, "α[$i]")
        push!(param_vals, a)
    end
    for (i, g) in enumerate(gamma)
        push!(param_names, "γ[$i]")
        push!(param_vals, g)
    end
    for (i, b) in enumerate(beta)
        push!(param_names, "β[$i]")
        push!(param_vals, b)
    end

    # Compute standard errors
    se = try
        stderror(m)
    catch
        fill(T_float(NaN), length(param_vals))
    end
    while length(se) < length(param_vals)
        push!(se, T_float(NaN))
    end

    # Publication-quality coefficient table
    _coef_table(io, header, param_names, param_vals, se; dist=:z)

    pers = persistence(m)
    uv = unconditional_variance(m)
    fit_data = Any[
        "Observations"     length(m.y);
        "Log-likelihood"   _fmt(m.loglik; digits=4);
        "AIC"              _fmt(m.aic; digits=4);
        "BIC"              _fmt(m.bic; digits=4);
        "Persistence"      _fmt(pers);
        "Unconditional σ²" (isfinite(uv) ? _fmt(uv) : "∞");
        "Converged"        string(m.converged)
    ]
    _pretty_table(io, fit_data;
        column_labels = ["Fit", "Value"],
        alignment = [:l, :r],
    )

    _sig_legend(io)
end

Base.show(io::IO, m::ARCHModel) = _show_volatility_model(io, "ARCH($(m.q)) Model", m; alpha=m.alpha)

function Base.show(io::IO, f::VolatilityForecast)
    h = f.horizon
    ci_pct = round(Int, 100 * f.conf_level)
    n_show = min(10, h)
    nrows = h > n_show ? n_show + 1 : n_show
    data = Matrix{Any}(undef, nrows, 5)
    for i in 1:n_show
        data[i, 1] = i
        data[i, 2] = _fmt(f.forecast[i])
        data[i, 3] = _fmt(f.se[i])
        data[i, 4] = _fmt(f.ci_lower[i])
        data[i, 5] = _fmt(f.ci_upper[i])
    end
    if h > n_show
        data[nrows, 1] = "..."
        data[nrows, 2] = "($(h - n_show) more)"
        data[nrows, 3] = ""
        data[nrows, 4] = ""
        data[nrows, 5] = ""
    end
    _pretty_table(io, data;
        title = "Volatility Forecast ($(f.model_type), h=$h, $(ci_pct)% CI)",
        column_labels = ["h", "σ² Forecast", "Std. Err.", "Lower", "Upper"],
        alignment = [:r, :r, :r, :r, :r],
    )
end
