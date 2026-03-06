# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Structural Dynamic Factor Model (Structural DFM).

Combines the Generalized Dynamic Factor Model (GDFM) of Forni et al. (2000, 2005)
with structural identification of common shocks via a VAR on the extracted factors,
following the approach in Forni et al. (2009).

The procedure:
1. Estimate GDFM to extract q common factors from a large panel X.
2. Fit a VAR(p) on the q time-domain factors.
3. Apply structural identification (Cholesky or sign restrictions) to the factor VAR.
4. Map the identified factor IRFs to all N panel variables through time-domain loadings.

References:
- Forni, M., Giannone, D., Lippi, M., & Reichlin, L. (2009). Opening the Black Box:
  Structural Factor Models with Large Cross-Sections. Econometric Theory, 25(5), 1319-1347.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The Generalized Dynamic-Factor
  Model: Identification and Estimation. Review of Economics and Statistics.
"""

using LinearAlgebra, Statistics

# =============================================================================
# StructuralDFM Type
# =============================================================================

"""
    StructuralDFM{T} <: AbstractFactorModel

Structural Dynamic Factor Model combining GDFM with structural identification.

# Fields
- `gdfm::GeneralizedDynamicFactorModel{T}`: Underlying GDFM estimation
- `factor_var::VARModel{T}`: VAR(p) fitted on q common factors
- `B0::Matrix{T}`: Impact matrix (q x q), B0 = chol(Sigma) * Q
- `Q::Matrix{T}`: Rotation/identification matrix (q x q)
- `identification::Symbol`: Identification method (:cholesky or :sign)
- `structural_irf::Array{T,3}`: Panel-wide structural IRFs (H x N x q)
- `loadings_td::Matrix{T}`: Time-domain loadings (N x q), Lambda = (F'F)^{-1} F' X
- `p_var::Int`: VAR lag order on factors
- `shock_names::Vector{String}`: Names for structural shocks
"""
struct StructuralDFM{T<:AbstractFloat} <: AbstractFactorModel
    gdfm::GeneralizedDynamicFactorModel{T}
    factor_var::VARModel{T}
    B0::Matrix{T}
    Q::Matrix{T}
    identification::Symbol
    structural_irf::Array{T,3}
    loadings_td::Matrix{T}
    p_var::Int
    shock_names::Vector{String}
end

# =============================================================================
# Estimation — From Raw Data
# =============================================================================

"""
    estimate_structural_dfm(X, q; identification=:cholesky, p=1, H=40, sign_check=nothing, max_draws=1000) -> StructuralDFM

Estimate a Structural DFM from raw panel data.

This is a one-step convenience wrapper that first estimates a GDFM via
`estimate_gdfm(X, q)`, then applies structural identification to the
factor dynamics.

# Arguments
- `X`: Panel data matrix (T x N)
- `q`: Number of dynamic factors

# Keyword Arguments
- `identification::Symbol=:cholesky`: Identification method (:cholesky or :sign)
- `p::Int=1`: VAR lag order on factors
- `H::Int=40`: IRF horizon
- `sign_check::Union{Nothing,Function}=nothing`: Sign restriction check function (required for :sign)
- `max_draws::Int=1000`: Maximum draws for sign restriction search
- `standardize::Bool=true`: Standardize data for GDFM estimation
- `bandwidth::Int=0`: GDFM kernel bandwidth (0 = automatic)
- `kernel::Symbol=:bartlett`: GDFM spectral kernel

# Returns
`StructuralDFM{T}` with identified factor IRFs mapped to all N panel variables.

# Example
```julia
using FFTW
X = randn(200, 50)
sdfm = estimate_structural_dfm(X, 3; identification=:cholesky, p=2, H=20)
irf_result = irf(sdfm, 20)  # panel-wide structural IRFs
```

# References
- Forni, M., Giannone, D., Lippi, M., & Reichlin, L. (2009). Opening the Black Box:
  Structural Factor Models with Large Cross-Sections. Econometric Theory, 25(5), 1319-1347.
"""
function estimate_structural_dfm(X::AbstractMatrix{T}, q::Int;
    identification::Symbol=:cholesky,
    p::Int=1,
    H::Int=40,
    sign_check::Union{Nothing,Function}=nothing,
    max_draws::Int=1000,
    standardize::Bool=true,
    bandwidth::Int=0,
    kernel::Symbol=:bartlett
) where {T<:AbstractFloat}

    # Estimate GDFM
    gdfm = estimate_gdfm(X, q; standardize=standardize, bandwidth=bandwidth, kernel=kernel)

    # Delegate to GDFM-based method
    estimate_structural_dfm(gdfm;
        identification=identification, p=p, H=H,
        sign_check=sign_check, max_draws=max_draws)
end

@float_fallback estimate_structural_dfm X

# =============================================================================
# Estimation — From Existing GDFM
# =============================================================================

"""
    estimate_structural_dfm(gdfm::GeneralizedDynamicFactorModel; identification=:cholesky, p=1, H=40, sign_check=nothing, max_draws=1000) -> StructuralDFM

Estimate a Structural DFM from an existing GDFM estimation.

# Arguments
- `gdfm`: Pre-estimated GDFM

# Keyword Arguments
- `identification::Symbol=:cholesky`: Identification method (:cholesky or :sign)
- `p::Int=1`: VAR lag order on factors
- `H::Int=40`: IRF horizon
- `sign_check::Union{Nothing,Function}=nothing`: Sign restriction check function (required for :sign)
- `max_draws::Int=1000`: Maximum draws for sign restriction search

# Returns
`StructuralDFM{T}` with identified factor IRFs mapped to all N panel variables.
"""
function estimate_structural_dfm(gdfm::GeneralizedDynamicFactorModel{T};
    identification::Symbol=:cholesky,
    p::Int=1,
    H::Int=40,
    sign_check::Union{Nothing,Function}=nothing,
    max_draws::Int=1000
) where {T<:AbstractFloat}

    # Validate inputs
    identification in (:cholesky, :sign) || throw(ArgumentError(
        "identification must be :cholesky or :sign, got :$identification"))
    p >= 1 || throw(ArgumentError("VAR lag order p must be >= 1"))
    H >= 1 || throw(ArgumentError("IRF horizon H must be >= 1"))

    if identification == :sign && isnothing(sign_check)
        throw(ArgumentError("sign_check function is required for :sign identification"))
    end

    q = gdfm.q
    F = gdfm.factors  # T_obs x q
    T_obs, N = size(gdfm.X)

    T_obs > p + q || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for VAR($p) with $q factors"))

    # Step 1: Fit VAR(p) on the q time-domain factors
    factor_varnames = ["Factor $i" for i in 1:q]
    factor_var = estimate_var(F, p; check_stability=false, varnames=factor_varnames)

    # Step 2: Structural identification
    if identification == :cholesky
        Q = Matrix{T}(I, q, q)  # Cholesky: Q = I (compute_irf handles the Cholesky internally)
    else
        # Sign restrictions
        Q, _ = identify_sign(factor_var, H, sign_check; max_draws=max_draws)
    end

    # Step 3: Compute structural factor IRFs (H x q x q)
    factor_irf = compute_irf(factor_var, Q, H)

    # Step 4: Compute time-domain loadings via regression: Lambda = (F'F)^{-1} F' X
    FtF_inv = Matrix{T}(robust_inv(F' * F))
    Lambda = (FtF_inv * (F' * gdfm.X))'  # N x q

    # Step 5: Map factor IRFs to panel variables
    # structural_irf[h, i, j] = sum_k Lambda[i, k] * factor_irf[h, k, j]
    structural_irf = zeros(T, H, N, q)
    for h in 1:H
        for j in 1:q
            factor_irfs_h = @view factor_irf[h, :, j]  # q-vector
            structural_irf[h, :, j] = Lambda * factor_irfs_h
        end
    end

    # Step 6: Build impact matrix B0 = chol(Sigma) * Q
    L = safe_cholesky(factor_var.Sigma)
    B0 = Matrix{T}(L * Q)

    shock_names = ["Shock $i" for i in 1:q]

    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification,
        structural_irf, Lambda, p, shock_names)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.nobs(m::StructuralDFM) = size(m.gdfm.X, 1)

StatsAPI.dof(m::StructuralDFM) = dof(m.gdfm) + dof(m.factor_var)

function StatsAPI.r2(m::StructuralDFM{T}) where {T}
    r2(m.gdfm)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::StructuralDFM{T}) where {T}
    T_obs, N = size(m.gdfm.X)
    q = m.gdfm.q
    H = size(m.structural_irf, 1)

    spec = Any[
        "Dynamic factors (q)"   q;
        "Panel variables (N)"   N;
        "Observations (T)"      T_obs;
        "VAR lags (p)"          m.p_var;
        "IRF horizon (H)"       H;
        "Identification"        string(m.identification);
        "Kernel"                string(m.gdfm.kernel);
        "Bandwidth"             m.gdfm.bandwidth
    ]
    _pretty_table(io, spec;
        title = "Structural DFM (q=$q, p=$(m.p_var), $(_id_label(m.identification)))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Show variance explained by factors
    n_show = min(q, 5)
    cum_var = cumsum(m.gdfm.variance_explained)
    var_data = Matrix{Any}(undef, n_show, 3)
    for i in 1:n_show
        var_data[i, 1] = "Factor $i"
        var_data[i, 2] = _fmt_pct(m.gdfm.variance_explained[i])
        var_data[i, 3] = _fmt_pct(cum_var[i])
    end
    _pretty_table(io, var_data;
        title = "Variance Explained",
        column_labels = ["", "Variance", "Cumulative"],
        alignment = [:l, :r, :r],
    )

    # Show impact matrix B0
    if q <= 6
        b0_data = Matrix{Any}(undef, q, q + 1)
        for i in 1:q
            b0_data[i, 1] = "Factor $i"
            for j in 1:q
                b0_data[i, j + 1] = _fmt(m.B0[i, j])
            end
        end
        _pretty_table(io, b0_data;
            title = "Impact Matrix B0",
            column_labels = vcat([""], m.shock_names),
            alignment = vcat([:l], fill(:r, q)),
        )
    end
end

"""Label for identification method in display."""
_id_label(s::Symbol) = s == :cholesky ? "Cholesky" : s == :sign ? "Sign" : string(s)
