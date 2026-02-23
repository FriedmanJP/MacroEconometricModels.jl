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
Simulated Method of Moments (SMM) estimation.

References:
- Ruge-Murcia, F. (2012). "Estimating Nonlinear DSGE Models by the Simulated Method
  of Moments." Journal of Economic Dynamics and Control, 36(6), 914-938.
- Lee, B.-S. & Ingram, B. F. (1991). "Simulation Estimation of Time-Series Models."
  Journal of Econometrics, 47(2-3), 197-205.
"""

using LinearAlgebra, Statistics, Distributions, Random

# =============================================================================
# SMMModel Type
# =============================================================================

"""
    SMMModel{T} <: AbstractGMMModel

Simulated Method of Moments estimator.

Shares the `AbstractGMMModel` interface with `GMMModel` --- `coef`, `vcov`, `nobs`,
`stderror`, `show`, `refs`, `report`, `j_test` all work.

# Fields
- `theta::Vector{T}` --- estimated parameters
- `vcov::Matrix{T}` --- asymptotic covariance matrix
- `n_moments::Int` --- number of moment conditions
- `n_params::Int` --- number of parameters
- `n_obs::Int` --- number of data observations
- `weighting::GMMWeighting{T}` --- weighting specification
- `W::Matrix{T}` --- final weighting matrix
- `g_bar::Vector{T}` --- moment discrepancy at solution
- `J_stat::T` --- Hansen J-test statistic
- `J_pvalue::T` --- J-test p-value
- `converged::Bool` --- convergence flag
- `iterations::Int` --- optimizer iterations
- `sim_ratio::Int` --- tau = simulation periods / data periods
"""
struct SMMModel{T<:AbstractFloat} <: AbstractGMMModel
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
    sim_ratio::Int

    function SMMModel{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W,
                          g_bar, J_stat, J_pvalue, converged, iterations,
                          sim_ratio) where {T<:AbstractFloat}
        @assert length(theta) == n_params
        @assert size(vcov) == (n_params, n_params)
        @assert size(W) == (n_moments, n_moments)
        @assert length(g_bar) == n_moments
        @assert n_moments >= n_params "SMM requires at least as many moments as parameters"
        @assert sim_ratio >= 1
        new{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W,
               g_bar, J_stat, J_pvalue, converged, iterations, sim_ratio)
    end
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(m::SMMModel) = m.theta
StatsAPI.vcov(m::SMMModel) = m.vcov
StatsAPI.nobs(m::SMMModel) = m.n_obs
StatsAPI.dof(m::SMMModel) = m.n_params
StatsAPI.islinear(::SMMModel) = false
StatsAPI.stderror(m::SMMModel) = sqrt.(max.(diag(m.vcov), zero(eltype(m.theta))))

is_overidentified(m::SMMModel) = m.n_moments > m.n_params
overid_df(m::SMMModel) = m.n_moments - m.n_params

function StatsAPI.confint(m::SMMModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    z = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.theta .- z .* se, m.theta .+ z .* se)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::SMMModel{T}) where {T}
    spec = Any[
        "Parameters"    m.n_params;
        "Moments"       m.n_moments;
        "Observations"  m.n_obs;
        "Sim ratio (tau)" m.sim_ratio;
        "Weighting"     string(m.weighting.method);
        "Converged"     m.converged ? "Yes" : "No";
        "Iterations"    m.iterations
    ]
    _pretty_table(io, spec;
        title = "SMM Estimation Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    se = stderror(m)
    param_names = ["theta[$i]" for i in 1:m.n_params]
    _coef_table(io, "Coefficients", param_names, m.theta, se; dist=:z)
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

# =============================================================================
# J-Test
# =============================================================================

"""
    j_test(m::SMMModel{T}) -> NamedTuple

Hansen's J-test for overidentifying restrictions on an SMM model.

H0: All moment conditions are valid (E[g(theta_0)] = 0)
H1: Some moment conditions are violated

Returns:
- J_stat: Test statistic
- p_value: p-value from chi-squared distribution
- df: Degrees of freedom (n_moments - n_params)
- reject_05: Whether to reject at 5% level
"""
function j_test(m::SMMModel{T}) where {T}
    df = overid_df(m)
    if df <= 0
        return (J_stat=zero(T), p_value=one(T), df=0, reject_05=false,
                message="Model is just-identified, J-test not applicable")
    end
    (J_stat=m.J_stat, p_value=m.J_pvalue, df=df, reject_05=m.J_pvalue < T(0.05))
end

# =============================================================================
# Moment Functions
# =============================================================================

"""
    autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) -> Vector{T}

Compute standard DSGE moment vector from data matrix.

Returns: `[upper-triangle variance-covariance elements; diagonal autocovariances at each lag]`

For k variables, lags L: `k*(k+1)/2 + k*L` moments total.

# Arguments
- `data` --- T_obs x k data matrix
- `lags` --- number of autocovariance lags (default: 1)
"""
function autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) where {T<:AbstractFloat}
    n, k = size(data)
    means = vec(mean(data, dims=1))
    data_c = data .- means'

    moments = T[]

    # Upper triangle of variance-covariance matrix (1/n divisor)
    for i in 1:k
        for j in i:k
            push!(moments, dot(data_c[:, i], data_c[:, j]) / n)
        end
    end

    # Diagonal autocovariances at each lag (1/n divisor)
    for lag in 1:lags
        for i in 1:k
            acov = dot(data_c[(lag+1):n, i], data_c[1:(n-lag), i]) / n
            push!(moments, acov)
        end
    end

    moments
end

autocovariance_moments(data::AbstractMatrix{<:Real}; kwargs...) =
    autocovariance_moments(Float64.(data); kwargs...)

# =============================================================================
# Weighting Matrix
# =============================================================================

"""
    smm_weighting_matrix(data::AbstractMatrix{T}, moments_fn::Function;
                          hac::Bool=true, bandwidth::Int=0) -> Matrix{T}

Compute optimal SMM weighting matrix from data moment contributions.
Centers the per-observation moment contributions and applies HAC with Bartlett kernel.

# Arguments
- `data` --- T_obs x k data matrix
- `moments_fn` --- function computing moment vector from data
- `hac` --- use HAC correction (default: true)
- `bandwidth` --- HAC bandwidth, 0 = automatic: `floor(4*(n/100)^(2/9))`
"""
function smm_weighting_matrix(data::AbstractMatrix{T}, moments_fn::Function;
                               hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    n = size(data, 1)
    m_full = moments_fn(data)
    q = length(m_full)

    # Compute per-observation moment contributions
    G = Matrix{T}(undef, n, q)
    for t in 1:n
        G[t, :] = moments_fn(data[t:t, :])
    end
    G_demean = G .- mean(G, dims=1)

    if hac
        Omega = long_run_covariance(G_demean; bandwidth=bandwidth, kernel=:bartlett)
    else
        Omega = (G_demean' * G_demean) / n
    end

    Omega_sym = Hermitian((Omega + Omega') / 2)
    eigvals_O = eigvals(Omega_sym)
    if minimum(eigvals_O) < eps(T)
        Omega_reg = Omega_sym + T(1e-8) * I
        return Matrix{T}(inv(Omega_reg))
    end

    robust_inv(Matrix(Omega_sym))
end
