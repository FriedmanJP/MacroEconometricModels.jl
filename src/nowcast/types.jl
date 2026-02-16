"""
Nowcasting model types — DFM, Large BVAR, Bridge Equations.

Based on the ECB Nowcasting Toolbox (Linzenich & Meunier 2024, WP 3004).
"""

# =============================================================================
# DFM Nowcasting Result
# =============================================================================

"""
    NowcastDFM{T<:AbstractFloat} <: AbstractNowcastModel

Dynamic factor model nowcasting result (Bańbura & Modugno 2014).

Estimates factors from mixed-frequency data with arbitrary missing patterns
using EM algorithm + Kalman smoother.

# Fields
- `X_sm::Matrix{T}` — smoothed data (NaN filled)
- `F::Matrix{T}` — smoothed factors (T_obs × state_dim)
- `C::Matrix{T}` — observation loadings
- `A::Matrix{T}` — state transition matrix
- `Q::Matrix{T}` — state innovation covariance
- `R::Matrix{T}` — observation noise covariance (diagonal)
- `Mx::Vector{T}` — data column means (for standardization)
- `Wx::Vector{T}` — data column stds
- `Z_0::Vector{T}` — initial state mean
- `V_0::Matrix{T}` — initial state covariance
- `r::Int` — number of factors
- `p::Int` — VAR lags in factor dynamics
- `blocks::Matrix{Int}` — block structure (N × n_blocks)
- `loglik::T` — log-likelihood at convergence
- `n_iter::Int` — EM iterations used
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables
- `idio::Symbol` — idiosyncratic spec (:ar1 or :iid)
- `data::Matrix{T}` — original data with NaN
"""
struct NowcastDFM{T<:AbstractFloat} <: AbstractNowcastModel
    X_sm::Matrix{T}
    F::Matrix{T}
    C::Matrix{T}
    A::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    Mx::Vector{T}
    Wx::Vector{T}
    Z_0::Vector{T}
    V_0::Matrix{T}
    r::Int
    p::Int
    blocks::Matrix{Int}
    loglik::T
    n_iter::Int
    nM::Int
    nQ::Int
    idio::Symbol
    data::Matrix{T}
end

# =============================================================================
# Large BVAR Nowcasting Result
# =============================================================================

"""
    NowcastBVAR{T<:AbstractFloat} <: AbstractNowcastModel

Large Bayesian VAR nowcasting result (Cimadomo et al. 2022).

Uses GLP-style normal-inverse-Wishart prior with hyperparameter optimization
via marginal likelihood maximization.

# Fields
- `X_sm::Matrix{T}` — smoothed data (NaN filled)
- `beta::Matrix{T}` — posterior mode VAR coefficients
- `sigma::Matrix{T}` — posterior mode error covariance
- `lambda::T` — overall shrinkage
- `theta::T` — cross-variable shrinkage
- `miu::T` — sum-of-coefficients prior weight
- `alpha::T` — co-persistence prior weight
- `lags::Int` — number of lags
- `loglik::T` — marginal log-likelihood
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables
- `data::Matrix{T}` — original data with NaN
"""
struct NowcastBVAR{T<:AbstractFloat} <: AbstractNowcastModel
    X_sm::Matrix{T}
    beta::Matrix{T}
    sigma::Matrix{T}
    lambda::T
    theta::T
    miu::T
    alpha::T
    lags::Int
    loglik::T
    nM::Int
    nQ::Int
    data::Matrix{T}
end

# =============================================================================
# Bridge Equation Nowcasting Result
# =============================================================================

"""
    NowcastBridge{T<:AbstractFloat} <: AbstractNowcastModel

Bridge equation combination nowcasting result (Bańbura et al. 2023).

Combines multiple OLS bridge regressions (each using a pair of monthly
indicators) via median combination.

# Fields
- `X_sm::Matrix{T}` — smoothed data (NaN filled by interpolation)
- `Y_nowcast::Vector{T}` — combined nowcast for target variable (per quarter)
- `Y_individual::Matrix{T}` — individual equation nowcasts (n_quarters × n_equations)
- `n_equations::Int` — number of bridge equations
- `coefficients::Vector{Vector{T}}` — OLS coefficients per equation
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables
- `lagM::Int` — monthly indicator lags
- `lagQ::Int` — quarterly indicator lags
- `lagY::Int` — autoregressive lags
- `data::Matrix{T}` — original data with NaN
"""
struct NowcastBridge{T<:AbstractFloat} <: AbstractNowcastModel
    X_sm::Matrix{T}
    Y_nowcast::Vector{T}
    Y_individual::Matrix{T}
    n_equations::Int
    coefficients::Vector{Vector{T}}
    nM::Int
    nQ::Int
    lagM::Int
    lagQ::Int
    lagY::Int
    data::Matrix{T}
end

# =============================================================================
# Unified Nowcast Result
# =============================================================================

"""
    NowcastResult{T<:AbstractFloat}

Unified nowcast result wrapping any `AbstractNowcastModel`.

# Fields
- `model::AbstractNowcastModel` — underlying model
- `X_sm::Matrix{T}` — smoothed/nowcasted data
- `target_index::Int` — column index of target variable
- `nowcast::T` — current-quarter nowcast value
- `forecast::T` — next-quarter forecast value
- `method::Symbol` — `:dfm`, `:bvar`, or `:bridge`
"""
struct NowcastResult{T<:AbstractFloat}
    model::AbstractNowcastModel
    X_sm::Matrix{T}
    target_index::Int
    nowcast::T
    forecast::T
    method::Symbol
end

# =============================================================================
# News Decomposition Result
# =============================================================================

"""
    NowcastNews{T<:AbstractFloat}

News decomposition result (Bańbura & Modugno 2014).

Decomposes nowcast revision into contributions from new data releases (news),
data revisions, and parameter re-estimation.

# Fields
- `old_nowcast::T` — previous nowcast value
- `new_nowcast::T` — updated nowcast value
- `impact_news::Vector{T}` — per-release news impact
- `impact_revision::T` — data revision impact
- `impact_reestimation::T` — parameter re-estimation impact (residual)
- `group_impacts::Vector{T}` — news aggregated by variable group
- `variable_names::Vector{String}` — names for each news release
"""
struct NowcastNews{T<:AbstractFloat}
    old_nowcast::T
    new_nowcast::T
    impact_news::Vector{T}
    impact_revision::T
    impact_reestimation::T
    group_impacts::Vector{T}
    variable_names::Vector{String}
end

# =============================================================================
# StatsAPI interface for nowcast types
# =============================================================================

StatsAPI.loglikelihood(m::NowcastDFM) = m.loglik
StatsAPI.loglikelihood(m::NowcastBVAR) = m.loglik

StatsAPI.predict(m::NowcastDFM) = m.X_sm
StatsAPI.predict(m::NowcastBVAR) = m.X_sm
StatsAPI.predict(m::NowcastBridge) = m.X_sm

StatsAPI.nobs(m::NowcastDFM) = size(m.data, 1)
StatsAPI.nobs(m::NowcastBVAR) = size(m.data, 1)
StatsAPI.nobs(m::NowcastBridge) = size(m.data, 1)
