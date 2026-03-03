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

Fields: X, factors, loadings, eigenvalues, explained_variance, cumulative_variance, r, standardized, block_names.
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
    block_names::Union{Vector{Symbol}, Nothing}
end

# =============================================================================
# Static Factor Model Estimation
# =============================================================================

"""
    estimate_factors(X, r; standardize=true, blocks=nothing) -> FactorModel

Estimate static factor model X_t = Λ F_t + e_t via Principal Component Analysis,
or via block-restricted EM when `blocks` is provided.

# Arguments
- `X`: Data matrix (T × N), observations × variables
- `r`: Number of factors to extract

# Keyword Arguments
- `standardize::Bool=true`: Standardize data before estimation
- `blocks::Union{Nothing, Dict{Symbol, Vector{Int}}}=nothing`: Block restriction map.
  When provided, each key is a block (factor) name and the value is a vector of variable
  indices that load on that factor. Variables not in a block have zero loadings on that
  factor. The number of blocks must equal `r`.

# Returns
`FactorModel` containing factors, loadings, eigenvalues, and explained variance.
When `blocks` is provided, `block_names` is set on the returned model.

# Example
```julia
X = randn(200, 50)  # 200 observations, 50 variables
fm = estimate_factors(X, 3)  # Extract 3 factors via PCA
r2(fm)  # R² for each variable

# Block-restricted estimation
blocks = Dict(:real => [1,2,3,4,5], :nominal => [6,7,8,9,10])
fm_restricted = estimate_factors(X, 2; blocks=blocks)
```
"""
function estimate_factors(X::AbstractMatrix{T}, r::Int;
        standardize::Bool=true,
        blocks::Union{Nothing, Dict{Symbol, Vector{Int}}}=nothing) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    # Route to block-restricted EM if blocks provided
    if blocks !== nothing
        return _estimate_restricted_em(X, r, blocks, standardize)
    end

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

    FactorModel{T}(X_orig, factors, loadings, λ, expl, cumul, r, standardize, nothing)
end

@float_fallback estimate_factors X

# =============================================================================
# Block-Restricted Factor Estimation via EM
# =============================================================================

"""
    _estimate_restricted_em(X, r, blocks, standardize; max_iter=500, tol=1e-6) -> FactorModel

Estimate block-restricted factor model via EM algorithm.

Each factor loads only on its assigned block of variables. The restriction mask R
enforces zero loadings outside each block.

# Algorithm
1. Validate block structure (count, overlap, range, minimum size)
2. Build N × r restriction mask R from block assignments
3. Initialize Λ via block-wise PCA (first eigenvector per block)
4. EM iteration:
   - E-step: F = X Λ (Λ'Λ)⁻¹ (posterior mean of factors given loadings)
   - M-step: Λ_new = (F'F)⁻¹ F'X, masked by R (zero out restricted entries)
5. Convergence when max absolute change in Λ < tol
"""
function _estimate_restricted_em(X::AbstractMatrix{T}, r::Int,
        blocks::Dict{Symbol, Vector{Int}}, standardize::Bool;
        max_iter::Int=500, tol::T=T(1e-6)) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # --- Validate blocks ---
    block_names = collect(keys(blocks))
    block_indices = collect(values(blocks))

    length(blocks) == r || throw(ArgumentError(
        "Number of blocks ($(length(blocks))) must equal number of factors r=$r"))

    # Check for overlapping indices
    all_idx = Int[]
    for (name, idx) in blocks
        for i in idx
            i in all_idx && throw(ArgumentError(
                "Variable index $i appears in multiple blocks"))
            push!(all_idx, i)
        end
    end

    # Check index range and minimum block size
    for (name, idx) in blocks
        length(idx) < 2 && throw(ArgumentError(
            "Block :$name has $(length(idx)) variable(s); minimum is 2"))
        for i in idx
            (1 <= i <= N) || throw(ArgumentError(
                "Variable index $i in block :$name is out of range [1, $N]"))
        end
    end

    X_orig = copy(X)
    X_proc = standardize ? _standardize(X) : X

    # --- Build restriction mask R (N × r): 1 where loading is allowed, 0 otherwise ---
    R = zeros(T, N, r)
    for (f, name) in enumerate(block_names)
        idx = blocks[name]
        for i in idx
            R[i, f] = one(T)
        end
    end

    # --- Initialize Λ via block-wise PCA ---
    Λ = zeros(T, N, r)
    for (f, name) in enumerate(block_names)
        idx = blocks[name]
        X_block = X_proc[:, idx]
        Σ_block = (X_block'X_block) / T_obs
        eig_block = eigen(Symmetric(Σ_block))
        # First eigenvector (largest eigenvalue)
        max_idx = argmax(eig_block.values)
        v1 = eig_block.vectors[:, max_idx]
        scale = sqrt(eig_block.values[max_idx])
        for (row, i) in enumerate(idx)
            Λ[i, f] = v1[row] * scale
        end
    end

    # --- EM iteration ---
    factors = zeros(T, T_obs, r)
    for iter in 1:max_iter
        # E-step: compute factors given loadings
        # F = X_proc * Λ * inv(Λ'Λ)
        ΛtΛ = Λ' * Λ
        ΛtΛ_inv = Matrix{T}(robust_inv(ΛtΛ))
        factors .= X_proc * Λ * ΛtΛ_inv

        # M-step: update loadings, respecting block restrictions
        # Λ_new = (F'F)⁻¹ * F'X_proc  (transposed: each column = factor loadings)
        FtF = factors' * factors
        FtF_inv = Matrix{T}(robust_inv(FtF))
        Λ_new = (FtF_inv * (factors' * X_proc))'  # N × r

        # Apply restriction mask
        Λ_new .*= R

        # Check convergence
        max_change = maximum(abs.(Λ_new .- Λ))
        Λ .= Λ_new
        max_change < tol && break
    end

    # Final factor estimates
    ΛtΛ = Λ' * Λ
    ΛtΛ_inv = Matrix{T}(robust_inv(ΛtΛ))
    factors .= X_proc * Λ * ΛtΛ_inv

    # Compute eigenvalues from factor covariance (for variance explained)
    Σ_full = (X_proc'X_proc) / T_obs
    eig_full = eigen(Symmetric(Σ_full))
    idx_sort = sortperm(eig_full.values, rev=true)
    λ_all = eig_full.values[idx_sort]

    total_var = sum(λ_all)
    expl = λ_all / total_var
    cumul = cumsum(expl)

    FactorModel{T}(X_orig, factors, Λ, λ_all, expl, cumul, r, standardize, block_names)
end

function Base.show(io::IO, m::FactorModel{T}) where {T}
    Tobs, N = size(m.X)
    estimation_type = m.block_names !== nothing ? "Block-Restricted" : "PCA"
    spec = Any[
        "Factors"       m.r;
        "Variables"     N;
        "Observations"  Tobs;
        "Standardized"  m.standardized ? "Yes" : "No";
        "Estimation"    estimation_type
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
        fname = m.block_names !== nothing ? string(m.block_names[i]) : "Factor $i"
        var_data[i, 1] = fname
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
        ftitle = m.block_names !== nothing ? "Top Loadings — $(m.block_names[f])" : "Top Loadings — Factor $f"
        _pretty_table(io, load_data;
            title = ftitle,
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
    forecast(model::FactorModel, h; p=1, ci_method=:theoretical, conf_level=0.95, n_boot=1000)

Forecast factors and observables h steps ahead from a static factor model.

Internally fits a VAR(p) on the extracted factors, then uses the VAR dynamics
to produce multi-step forecasts and confidence intervals.

# Arguments
- `model`: Estimated static factor model
- `h`: Forecast horizon

# Keyword Arguments
- `p::Int=1`: VAR lag order for factor dynamics
- `ci_method::Symbol=:theoretical`: CI method — `:none`, `:theoretical`, or `:bootstrap`
- `conf_level::Real=0.95`: Confidence level for intervals
- `n_boot::Int=1000`: Number of bootstrap replications (if `ci_method=:bootstrap`)

# Returns
`FactorForecast` with factor and observable forecasts (and CIs if requested).
"""
function forecast(m::FactorModel{T}, h::Int; p::Int=1, ci_method::Symbol=:theoretical,
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
