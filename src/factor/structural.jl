# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Structural Dynamic Factor Model (Structural DFM).

Default `method=:fglr` is Forni–Giannone–Lippi–Reichlin (2009):
1. Extract `r ≥ q` static PCA factors from the panel.
2. Fit a VAR(p) on those factors.
3. Rank-`q` reduction `u_t = K ε_t` of the factor-VAR residuals.
4. Identify `H` on the panel impact `Λ K` (Cholesky on `order`, or sign
   restrictions on the factor-space IRF `Ψ_h K H`).
5. Map to observables: panel IRFs are `Λ Ψ_h K H`.

Pass `method=:gdfm_var` for the legacy two-sided GDFM-factor VAR.

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
- `factor_var::VARModel{T}`: VAR(p) on static (`:fglr`) or GDFM (`:gdfm_var`) factors
- `B0::Matrix{T}`: Impact on the VAR factors — `r×q` (`K H`) under `:fglr`, `q×q` under `:gdfm_var`
- `Q::Matrix{T}`: Identification rotation `H` (`q×q`)
- `identification::Symbol`: Identification method (`:cholesky` or `:sign`)
- `structural_irf::Array{T,3}`: Panel-wide structural IRFs (`H × N × q`)
- `loadings_td::Matrix{T}`: Projection loadings (`N×r` FGLR, `N×q` legacy)
- `p_var::Int`: VAR lag order on factors
- `shock_names::Vector{String}`: Names for structural shocks
- `varnames::Vector{String}`: Panel variable names (length N; #538)
- `K::Matrix{T}`: Rank-`q` loading of factor-VAR residuals (`r×q`)
- `r::Int`: Number of static factors
- `method::Symbol`: `:fglr` (default) or `:gdfm_var` (legacy two-sided)
- `static_factors::Matrix{T}`: `T×r` static (PCA) factors
- `loadings_static::Matrix{T}`: `N×r` static loadings
- `shock_variance_share::T`: Share of `tr Σ̂_u` captured by the `q` shocks
- `units::Symbol`: `:raw` or `:standardized` IRF units
- `identified_set::Union{Nothing,SignIdentifiedSet{T}}`: Accepted panel (or factor) IRFs when `store_all=true`
- `acceptance_rate::T`: Fraction of Haar draws accepted under `:sign` (1 for Cholesky)
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
    varnames::Vector{String}
    K::Matrix{T}
    r::Int
    method::Symbol
    static_factors::Matrix{T}
    loadings_static::Matrix{T}
    shock_variance_share::T
    units::Symbol
    identified_set::Union{Nothing,SignIdentifiedSet{T}}
    acceptance_rate::T
    id_order::Vector{Int}
    lag_criterion::Symbol
    max_eigenvalue_modulus::T
    instrument::Union{Nothing,Vector{T}}
    first_stage_F::T
end

# Backward-compatible constructor (pre-SDFM-18, no proxy fields)
StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                 p_var, shock_names, varnames, K, r, method, static_factors,
                 loadings_static, shock_variance_share, units, identified_set,
                 acceptance_rate, id_order, lag_criterion, max_eigenvalue_modulus) where {T} =
    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                     p_var, shock_names, varnames, K, r, method, static_factors,
                     loadings_static, shock_variance_share, units, identified_set,
                     acceptance_rate, id_order, lag_criterion, max_eigenvalue_modulus,
                     nothing, T(NaN))

# Backward-compatible constructor (pre-SDFM-09, no lag/stability fields)
StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                 p_var, shock_names, varnames, K, r, method, static_factors,
                 loadings_static, shock_variance_share, units, identified_set,
                 acceptance_rate) where {T} =
    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                     p_var, shock_names, varnames, K, r, method, static_factors,
                     loadings_static, shock_variance_share, units, identified_set,
                     acceptance_rate, collect(1:gdfm.q), :fixed,
                     _factor_var_modulus(factor_var))

# Backward-compatible constructor (pre-SDFM-03, has units)
StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                 p_var, shock_names, varnames, K, r, method, static_factors,
                 loadings_static, shock_variance_share, units) where {T} =
    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                     p_var, shock_names, varnames, K, r, method, static_factors,
                     loadings_static, shock_variance_share, units, nothing, one(T))

# Backward-compatible constructor (pre-SDFM-02, has varnames)
StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                 p_var, shock_names, varnames) where {T} =
    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                     p_var, shock_names, varnames,
                     Matrix{T}(I, size(B0, 1), size(B0, 2)), size(B0, 1), :gdfm_var,
                     gdfm.factors, loadings_td, one(T), :raw)

# Backward-compatible constructor (pre-#538)
StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                 p_var, shock_names) where {T} =
    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification, structural_irf, loadings_td,
                     p_var, shock_names, ["Var $i" for i in 1:size(gdfm.X, 2)])

"""Spectral radius of the factor-VAR companion matrix."""
function _factor_var_modulus(fv::VARModel{T}) where {T<:AbstractFloat}
    n, p = nvars(fv), fv.p
    maximum(abs.(eigvals(companion_matrix(fv.B, n, p))))
end

"""Resolve `p::Int` or `p=:aic/:bic/:hq` via `select_lag_order` on the factor path."""
function _resolve_factor_lag(F::AbstractMatrix, p::Integer, ::Int)
    p < 1 && throw(ArgumentError("VAR lag order p must be >= 1"))
    Int(p), :fixed
end
function _resolve_factor_lag(F::AbstractMatrix, p::Symbol, p_max::Int)
    p_max < 1 && throw(ArgumentError("p_max must be >= 1"))
    crit = p === :hq ? :hqic : p
    crit in (:aic, :bic, :hqic) || throw(ArgumentError(
        "p must be a positive integer or :aic/:bic/:hq, got :$p"))
    select_lag_order(F, p_max; criterion=crit), crit
end

const _SDFM_ID_METHODS = (
    :cholesky, :sign, :narrative, :long_run, :proxy,
    :fastica, :jade, :sobi, :dcov, :hsic,
    :student_t, :mixture_normal, :pml, :skew_normal, :nongaussian_ml,
    :markov_switching, :garch, :smooth_transition, :external_volatility,
    :arias, :uhlig,
)
const _SDFM_COMPUTE_Q_METHODS = (
    :narrative, :fastica, :jade, :sobi, :dcov, :hsic,
    :student_t, :mixture_normal, :pml, :skew_normal, :nongaussian_ml,
    :markov_switching, :garch, :smooth_transition, :external_volatility,
)

# =============================================================================
# Estimation — From Raw Data
# =============================================================================

"""
    estimate_structural_dfm(X, q; identification=:cholesky, p=1, H=40, r=0, method=:fglr, ...) -> StructuralDFM

Estimate a Structural DFM from raw panel data.

Default `method=:fglr` is Forni–Giannone–Lippi–Reichlin (2009): `r ≥ q` static
PCA factors, a VAR(p) on those factors, rank-`q` reduction `u_t = K ε_t`, and
identification `H` on the panel impact `Λ K`. Pass `method=:gdfm_var` for the
legacy two-sided GDFM-factor VAR.

# Arguments
- `X`: Panel data matrix (T x N)
- `q`: Number of dynamic / common shocks

# Keyword Arguments
- `identification::Symbol=:cholesky`: Identification method (:cholesky or :sign)
- `p::Union{Int,Symbol}=1`: VAR lag order, or `:aic`/`:bic`/`:hq` over `1:p_max`
- `p_max::Int=8`: Grid upper bound when `p` is a criterion
- `check_stability::Bool=true`: Warn when the factor-VAR companion modulus is ≥ 1
- `H::Int=40`: IRF horizon
- `r::Int=0`: Static factors (0 = `q`; must satisfy `r ≥ q`)
- `method::Symbol=:fglr`: `:fglr` or `:gdfm_var`
- `order::Union{Nothing,Vector{Int}}=nothing`: Observable indices for Cholesky (`:fglr`; default `1:q`)
- `units::Symbol=:raw`: `:raw` (original units) or `:standardized`
- `sign_check::Union{Nothing,Function}=nothing`: Predicate on the IRF array (`H×N×q` panel under `restriction_space=:panel`, `H×r×q` factors under `:factor`)
- `sign_restrictions=nothing`: Declarative restrictions — `SVARRestrictions` or a vector of `(variable, shock, horizons, sign)` tuples (`variable` is a name or index; `sign` is `:positive`/`:negative`). `SVARRestrictions` keep only `SignRestriction`s and treat `horizon` as 0-based (impact = 0 → IRF row 1); tuple `horizons` are 1-based IRF indices.
- `restriction_space::Symbol=:panel`: `:panel` (observables) or `:factor` (static / GDFM factors)
- `store_all::Bool=false`: Keep the full accepted set as `identified_set` (`SignIdentifiedSet`)
- `max_draws::Int=1000`: Maximum draws for sign restriction search
- `standardize::Bool=true`: Standardize data for PCA / GDFM
- `bandwidth::Int=0`: GDFM kernel bandwidth (0 = automatic)
- `kernel::Symbol=:bartlett`: GDFM spectral kernel
- `spectral::Symbol=:lag_window`: GDFM spectral estimator
- `varnames::Union{Nothing,Vector{String}}=nothing`: Panel variable names (length N)
- `shock_names::Union{Nothing,Vector{String}}=nothing`: Structural shock names (length q)
- `rng::AbstractRNG=Random.default_rng()`: RNG for sign-restriction search

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
    p::Union{Int,Symbol}=1,
    p_max::Int=8,
    check_stability::Bool=true,
    H::Int=40,
    r::Int=0,
    method::Symbol=:fglr,
    order::Union{Nothing,Vector{Int}}=nothing,
    units::Symbol=:raw,
    sign_check::Union{Nothing,Function}=nothing,
    sign_restrictions=nothing,
    restriction_space::Symbol=:panel,
    store_all::Bool=false,
    max_draws::Int=1000,
    narrative_check::Union{Nothing,Function}=nothing,
    target_vars::Union{Nothing,Vector{Int},Vector{<:AbstractString}}=nothing,
    restrictions::Union{Nothing,SVARRestrictions}=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    standardize::Bool=true,
    bandwidth::Int=0,
    kernel::Symbol=:bartlett,
    spectral::Symbol=:lag_window,
    varnames::Union{Nothing,Vector{String}}=nothing,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    instrument::Union{Nothing,AbstractVector}=nothing,
    normalize::Union{Nothing,Tuple}=nothing,
) where {T<:AbstractFloat}

    _validate_data(X, "X")
    r_use = r <= 0 ? q : r
    r_gdfm = r_use

    gdfm = estimate_gdfm(X, q; standardize=standardize, bandwidth=bandwidth, kernel=kernel,
                        varnames=varnames, r=r_gdfm, spectral=spectral)

    estimate_structural_dfm(gdfm;
        identification=identification, p=p, p_max=p_max, check_stability=check_stability,
        H=H, r=r_use, method=method, instrument=instrument, normalize=normalize,
        order=order, units=units,
        sign_check=sign_check, sign_restrictions=sign_restrictions,
        restriction_space=restriction_space, store_all=store_all,
        max_draws=max_draws, narrative_check=narrative_check,
        target_vars=target_vars, restrictions=restrictions,
        transition_var=transition_var, regime_indicator=regime_indicator,
        varnames=varnames, shock_names=shock_names, rng=rng, standardize=standardize)
end

@float_fallback estimate_structural_dfm X

"""
    estimate_structural_dfm(X, :auto; q_method=:hallin_liska, q_max=8, ...) -> StructuralDFM

Select `q` with [`hallin_liska`](@ref) (`:hallin_liska`), [`bai_ng_q`](@ref)
(`:bai_ng`), or [`amengual_watson_q`](@ref) (`:amengual_watson`), then estimate
as usual. `r` defaults to `max(q, r)` after selection.
"""
function estimate_structural_dfm(X::AbstractMatrix{T}, q::Symbol;
    q_method::Symbol=:hallin_liska,
    q_max::Int=8,
    r::Int=0,
    p::Union{Int,Symbol}=1,
    standardize::Bool=true,
    bandwidth::Int=0,
    kernel::Symbol=:bartlett,
    spectral::Symbol=:lag_window,
    kwargs...,
) where {T<:AbstractFloat}
    q === :auto || throw(ArgumentError("q must be a positive integer or :auto, got :$q"))
    validate_option(q_method, "q_method", (:hallin_liska, :bai_ng, :amengual_watson))
    T_obs, N = size(X)
    q_cap = min(q_max, max(1, N - 1), max(1, T_obs - 1))
    q_hat = if q_method === :hallin_liska
        hallin_liska(X, q_cap; standardize=standardize, bandwidth=bandwidth,
            kernel=kernel, spectral=spectral).q
    elseif q_method === :bai_ng
        r_use = r > 0 ? r : min(2 * q_cap, N, T_obs)
        p_bn = p isa Integer ? Int(p) : 1
        bn = bai_ng_q(X, r_use; p=p_bn, standardize=standardize)
        max(bn.q_D1, 1)
    else
        r_use = r > 0 ? r : min(2 * q_cap, N, T_obs)
        p_bn = p isa Integer ? Int(p) : 1
        max(amengual_watson_q(X, r_use, p_bn; standardize=standardize).q, 1)
    end
    q_hat = max(q_hat, 1)
    estimate_structural_dfm(X, q_hat; r=r, p=p, standardize=standardize,
        bandwidth=bandwidth, kernel=kernel, spectral=spectral, kwargs...)
end

function bai_ng_q(sdfm::StructuralDFM{T}; p::Int=sdfm.p_var, δ::Real=0.1, m::Real=1) where {T<:AbstractFloat}
    F = sdfm.method === :fglr ? sdfm.static_factors : sdfm.gdfm.factors
    N = size(sdfm.gdfm.X, 2)
    T_obs = size(F, 1)
    bai_ng_q(F, N, T_obs; p=p, δ=δ, m=m)
end

# =============================================================================
# Estimation — From Existing GDFM
# =============================================================================

"""
    estimate_structural_dfm(gdfm::GeneralizedDynamicFactorModel; identification=:cholesky, p=1, H=40, method=:fglr, ...) -> StructuralDFM

Estimate a Structural DFM from an existing GDFM estimation.

# Arguments
- `gdfm`: Pre-estimated GDFM

# Keyword Arguments
- `identification::Symbol=:cholesky`: Identification method (:cholesky or :sign)
- `p::Union{Int,Symbol}=1`: VAR lag order, or `:aic`/`:bic`/`:hq` over `1:p_max`
- `p_max::Int=8`: Grid upper bound when `p` is a criterion
- `check_stability::Bool=true`: Warn when the factor-VAR companion modulus is ≥ 1
- `H::Int=40`: IRF horizon
- `r::Int=0`: Static factors (0 = `gdfm.r`; must satisfy `r ≥ q`)
- `method::Symbol=:fglr`: `:fglr` or `:gdfm_var`
- `order`: Observable indices for Cholesky under `:fglr`
- `sign_check::Union{Nothing,Function}=nothing`: Predicate on the IRF array (`H×N×q` under `:panel`, `H×r×q` under `:factor`)
- `sign_restrictions=nothing`: Declarative restrictions (`SVARRestrictions` or `(variable, shock, horizons, sign)` tuples). `SVARRestrictions` keep only `SignRestriction`s with 0-based `horizon`; tuples use 1-based IRF indices.
- `restriction_space::Symbol=:panel`: `:panel` or `:factor`
- `store_all::Bool=false`: Keep the accepted set as `identified_set`
- `max_draws::Int=1000`: Maximum draws for sign restriction search

# Returns
`StructuralDFM{T}` with identified factor IRFs mapped to all N panel variables.
"""
function estimate_structural_dfm(gdfm::GeneralizedDynamicFactorModel{T};
    identification::Symbol=:cholesky,
    p::Union{Int,Symbol}=1,
    p_max::Int=8,
    check_stability::Bool=true,
    H::Int=40,
    r::Int=0,
    method::Symbol=:fglr,
    order::Union{Nothing,Vector{Int}}=nothing,
    units::Symbol=:raw,
    sign_check::Union{Nothing,Function}=nothing,
    sign_restrictions=nothing,
    restriction_space::Symbol=:panel,
    store_all::Bool=false,
    max_draws::Int=1000,
    narrative_check::Union{Nothing,Function}=nothing,
    target_vars::Union{Nothing,Vector{Int},Vector{<:AbstractString}}=nothing,
    restrictions::Union{Nothing,SVARRestrictions}=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    varnames::Union{Nothing,Vector{String}}=nothing,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    standardize::Bool=true,
    instrument::Union{Nothing,AbstractVector}=nothing,
    normalize::Union{Nothing,Tuple}=nothing,
) where {T<:AbstractFloat}

    identification in _SDFM_ID_METHODS || throw(ArgumentError(
        "identification must be one of $(_SDFM_ID_METHODS), got :$identification"))
    method in (:fglr, :gdfm_var) || throw(ArgumentError(
        "method must be :fglr or :gdfm_var, got :$method"))
    units in (:raw, :standardized) || throw(ArgumentError(
        "units must be :raw or :standardized, got :$units"))
    restriction_space in (:panel, :factor) || throw(ArgumentError(
        "restriction_space must be :panel or :factor, got :$restriction_space"))
    H >= 1 || throw(ArgumentError("IRF horizon H must be >= 1"))

    if identification === :sign && isnothing(sign_check) && isnothing(sign_restrictions)
        throw(ArgumentError("sign_check or sign_restrictions is required for :sign identification"))
    end
    if identification === :narrative && (isnothing(sign_check) || isnothing(narrative_check))
        throw(ArgumentError("Need check_func and narrative_check for narrative"))
    end
    if identification in (:arias, :uhlig) && restrictions === nothing && sign_restrictions === nothing
        throw(ArgumentError(":$identification requires `restrictions::SVARRestrictions`"))
    end
    if identification === :proxy && instrument === nothing
        throw(ArgumentError("identification=:proxy requires `instrument`"))
    end

    q = gdfm.q
    T_obs, N = size(gdfm.X)
    r_use = r <= 0 ? (method === :fglr ? gdfm.r : q) : r
    r_use < q && throw(ArgumentError("r must be >= q, got r=$r_use, q=$q"))
    validate_factor_inputs(T_obs, N, r_use)

    vn = something(varnames, gdfm.varnames)
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but panel has $N columns"))

    sn = something(shock_names, identification === :proxy ?
        vcat(["Proxy"], ["Unidentified $i" for i in 2:q]) :
        ["Shock $i" for i in 1:q])
    length(sn) == q || throw(ArgumentError(
        "shock_names has $(length(sn)) entries but q=$q"))

    if method === :fglr
        return _estimate_sdfm_fglr(gdfm, q, r_use, p, p_max, check_stability, H, identification, order, units,
            sign_check, sign_restrictions, restriction_space, store_all,
            max_draws, narrative_check, target_vars, restrictions,
            transition_var, regime_indicator, vn, sn, rng, standardize,
            instrument, normalize)
    end
    _estimate_sdfm_gdfm_var(gdfm, q, p, p_max, check_stability, H, identification, sign_check, sign_restrictions,
        restriction_space, store_all, max_draws, narrative_check, target_vars,
        restrictions, transition_var, regime_indicator, vn, sn, rng,
        instrument, normalize)
end

# =============================================================================
# FGLR path — static PCA, VAR(r), rank-q reduction, panel identification
# =============================================================================

function _estimate_sdfm_fglr(gdfm::GeneralizedDynamicFactorModel{T}, q::Int, r::Int,
    p, p_max::Int, check_stability::Bool, H::Int, identification::Symbol, order, units::Symbol,
    sign_check, sign_restrictions, restriction_space::Symbol, store_all::Bool,
    max_draws::Int, narrative_check, target_vars, restrictions,
    transition_var, regime_indicator, vn::Vector{String}, sn::Vector{String},
    rng::AbstractRNG, standardize::Bool,
    instrument, normalize) where {T<:AbstractFloat}

    X = gdfm.X
    T_obs, N = size(X)

    ord = something(order, collect(1:q))
    length(ord) == q || throw(ArgumentError(
        "order has $(length(ord)) entries but q=$q"))
    all(1 .<= ord .<= N) || throw(ArgumentError(
        "order indices must be in 1:$N"))

    fm = estimate_factors(X, r; standardize=standardize, varnames=vn)
    F = fm.factors                      # T × r
    Lambda = fm.loadings                # N × r
    p_use, lag_crit = _resolve_factor_lag(F, p, p_max)
    T_obs > p_use + r || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for VAR($p_use) with $r static factors"))

    factor_varnames = ["Static factor $i" for i in 1:r]
    factor_var = estimate_var(F, p_use; check_stability=false, varnames=factor_varnames)

    K, share = _rank_q_reduction(factor_var.Sigma, q)
    H_id, idset, rate, from_var_Q, z_store, Fstat = _fglr_identify_H(Lambda, K, factor_var, identification, ord, H,
                            sign_check, sign_restrictions, restriction_space, store_all,
                            max_draws, narrative_check, target_vars, restrictions,
                            transition_var, regime_indicator,
                            rng, vn, sn, X, standardize, units, instrument, normalize)
    B0 = from_var_Q ? Matrix{T}(safe_cholesky(factor_var.Sigma) * H_id) : Matrix{T}(K * H_id)

    structural_irf = _fglr_panel_irf(factor_var, Lambda, B0, H; X=X,
                                     standardized=standardize, units=units)

    modu = _factor_var_modulus(factor_var)
    if check_stability && modu >= one(T)
        @warn "Estimated factor VAR is non-stationary (max eigenvalue modulus = $(round(modu, digits=4))); IRFs may be explosive. Pass check_stability=false to silence."
    end
    StructuralDFM{T}(gdfm, factor_var, B0, H_id, identification,
        structural_irf, Lambda, p_use, sn, vn,
        K, r, :fglr, F, Lambda, share, units, idset, rate,
        collect(Int, ord), lag_crit, T(modu), z_store, Fstat)
end

"""Leading-q eigendecomposition of Σ̂_u: K = V_q Λ_q^{1/2} (`r×q`)."""
function _rank_q_reduction(Sigma::AbstractMatrix{T}, q::Int) where {T<:AbstractFloat}
    E = eigen(Hermitian(Matrix{T}(Sigma)))
    idx = sortperm(real.(E.values); rev=true)
    evals = max.(real.(E.values[idx]), zero(T))
    evecs = real.(E.vectors[:, idx])
    K = evecs[:, 1:q] * Diagonal(sqrt.(evals[1:q]))
    tot = sum(evals)
    share = tot > 0 ? T(sum(evals[1:q]) / tot) : one(T)
    K, share
end

"""q×q identification H. Cholesky / long-run on selected observables; Haar / compute_Q otherwise.
Fourth return is `true` when `H` is a VAR-space `compute_Q` rotation (`B0 = chol(Σ) H`)."""
function _fglr_identify_H(Lambda::AbstractMatrix{T}, K::AbstractMatrix{T},
    factor_var::VARModel{T}, identification::Symbol, order::Vector{Int},
    H_irf::Int, sign_check, sign_restrictions, restriction_space::Symbol,
    store_all::Bool, max_draws::Int, narrative_check, target_vars, restrictions,
    transition_var, regime_indicator, rng::AbstractRNG,
    vn::Vector{String}, sn::Vector{String}, X, standardize::Bool,
    units::Symbol, instrument=nothing, normalize=nothing) where {T<:AbstractFloat}

    q = size(K, 2)
    r = size(K, 1)
    N = size(Lambda, 1)
    nanF = T(NaN)

    if identification === :cholesky
        C0 = Lambda * K
        P = C0[order, :]
        L = safe_cholesky(P * P')
        return Matrix{T}(P \ L), nothing, one(T), false, nothing, nanF
    end

    if identification === :long_run
        Hlr = _sdfm_long_run_H(Lambda, K, factor_var, target_vars, vn, q)
        return Hlr, nothing, one(T), false, nothing, nanF
    end

    if identification === :proxy
        Hpr, z_store, Fstat = _sdfm_proxy_H(Lambda, K, factor_var, instrument, normalize, vn)
        return Hpr, nothing, one(T), false, z_store, Fstat
    end

    if identification === :sign
        Psi = _ma_array(factor_var, H_irf)
        scale = (standardize && units === :raw && X !== nothing) ?
            max.(vec(std(X; dims=1)), T(1e-10)) : nothing
        n_check = restriction_space === :panel ? N : r
        names_check = restriction_space === :panel ? vn : ["Static factor $i" for i in 1:r]
        rules = _sdfm_parse_sign_rules(sign_restrictions, names_check, n_check, q, H_irf)
        fac = zeros(T, H_irf, r, q)
        panel = zeros(T, H_irf, N, q)
        function irf_from_Q(Htry)
            B0try = K * Htry
            @inbounds for h in 1:H_irf
                fac[h, :, :] = @view(Psi[h, :, :]) * B0try
                panel[h, :, :] = Lambda * @view(fac[h, :, :])
            end
            if scale !== nothing
                panel .*= reshape(scale, 1, N, 1)
            end
            restriction_space === :panel ? panel : fac
        end
        H_id, idset, rate = _sdfm_sign_search(irf_from_Q, q, H_irf, n_check, T, sign_check, rules,
                                              store_all, max_draws, rng, names_check, sn)
        return H_id, idset, rate, false, nothing, nanF
    end

    r == q || throw(ArgumentError(
        "identification :$identification requires r==q, got r=$r, q=$q; use :cholesky, :sign, :long_run, or :proxy when r>q"))

    rest = restrictions === nothing ? sign_restrictions : restrictions
    if identification === :arias
        rest isa SVARRestrictions || throw(ArgumentError(":arias requires restrictions::SVARRestrictions"))
        res = identify_arias(factor_var, rest, H_irf; n_draws=1, n_rotations=max_draws, rng=rng)
        return Matrix{T}(res.Q_draws[1]), nothing, T(res.acceptance_rate), true, nothing, nanF
    end
    if identification === :uhlig
        rest isa SVARRestrictions || throw(ArgumentError(":uhlig requires restrictions::SVARRestrictions"))
        res = identify_uhlig(factor_var, rest, H_irf; rng=rng)
        return Matrix{T}(res.Q), nothing, one(T), true, nothing, nanF
    end

    identification in _SDFM_COMPUTE_Q_METHODS || throw(ArgumentError(
        "identification must be one of $(_SDFM_ID_METHODS), got :$identification"))
    Q = compute_Q(factor_var, identification; horizon=H_irf, check_func=sign_check,
                  narrative_check=narrative_check, max_draws=max_draws,
                  transition_var=transition_var, regime_indicator=regime_indicator, rng=rng)
    return Matrix{T}(Q), nothing, one(T), true, nothing, nanF
end

"""Panel-space proxy rotation: first column of `K H` ∝ Cov(û, z), unit-effect normalised."""
function _sdfm_proxy_H(Lambda::AbstractMatrix{T}, K::AbstractMatrix{T},
    factor_var::VARModel{T}, instrument, normalize, vn::Vector{String}) where {T<:AbstractFloat}
    instrument === nothing && throw(ArgumentError("identification=:proxy requires `instrument`"))
    N = size(Lambda, 1)
    q = size(K, 2)
    i_norm, nval = _sdfm_proxy_normalize(normalize, vn, N)
    U = factor_var.U
    T_eff = size(U, 1)
    z_eff = _align_instrument(instrument, size(factor_var.Y, 1), factor_var.p, T_eff)
    mask = [isfinite(z_eff[t]) && all(isfinite, @view U[t, :]) for t in 1:T_eff]
    count(mask) < 8 && throw(ArgumentError("instrument has too few finite observations"))
    Uu = U[mask, :]
    zz = z_eff[mask]
    zc = zz .- mean(zz)
    Uc = Uu .- mean(Uu; dims=1)
    nobs = length(zz)
    Suz = vec(Uc' * zc) / T(nobs - 1)
    k_raw = K * (K \ Suz)
    scale = dot(Lambda[i_norm, :], k_raw)
    abs(scale) < T(1e-12) && throw(ArgumentError(
        "proxy impact on the normalising observable is numerically zero"))
    k1 = k_raw ./ scale .* T(nval)
    h1 = K \ k1
    H = Matrix{T}(I, q, q)
    H[:, 1] = h1
    Fq = qr(H)
    H = Matrix{T}(Fq.Q)
    s = [Fq.R[i, i] < zero(T) ? -one(T) : one(T) for i in 1:q]
    H = H * Diagonal(s)
    impact = Lambda * (K * H)
    sc = impact[i_norm, 1]
    abs(sc) < T(1e-12) && throw(ArgumentError("proxy impact on the normalising observable is zero"))
    H[:, 1] .*= T(nval) / sc
    y = Uc * (Suz ./ (norm(Suz) + T(1e-12)))
    Fstat = _ols_first_stage_F(y, zc)
    if Fstat < T(10)
        @warn "Weak instrument: first-stage F = $(round(Fstat, digits=2)) < 10 (Montiel Olea, Stock & Watson 2021)"
    end
    H, Vector{T}(z_eff), T(Fstat)
end

function _sdfm_proxy_normalize(normalize, vn::Vector{String}, N::Int)
    if normalize === nothing
        return 1, 1.0
    end
    var, val = normalize
    i = if var isa Integer
        Int(var)
    else
        idx = findfirst(==(String(var)), vn)
        idx === nothing && throw(ArgumentError(
            "normalize variable '$var' not found. Available: $vn"))
        idx
    end
    (1 <= i <= N) || throw(ArgumentError("normalize variable out of range 1:$N"))
    i, Float64(val)
end

"""Panel long-run H: Cholesky of the long-run covariance of `target_vars` (length q)."""
function _sdfm_long_run_H(Lambda::AbstractMatrix{T}, K::AbstractMatrix{T},
    factor_var::VARModel{T}, target_vars, vn::Vector{String}, q::Int) where {T<:AbstractFloat}
    N = size(Lambda, 1)
    idx = _sdfm_resolve_targets(target_vars, vn, N, q)
    n, p = nvars(factor_var), factor_var.p
    A_sum = sum(extract_ar_coefficients(factor_var.B, n, p))
    M = Matrix{T}(I(n) - A_sum)
    Psi_inf = robust_inv(M; silent=true)
    C_lr = Lambda * Psi_inf * K          # N × q
    P = C_lr[idx, :]                     # q × q
    L = safe_cholesky(P * P')
    H = Matrix{T}(P \ L)
    Clr = C_lr * H
    if Clr[idx[1], 1] < 0
        H[:, 1] .*= -one(T)
    end
    H
end

function _sdfm_resolve_targets(target_vars, vn::Vector{String}, N::Int, q::Int)
    if target_vars === nothing
        return collect(1:q)
    end
    idx = Int[]
    for v in target_vars
        if v isa Integer
            push!(idx, Int(v))
        else
            i = findfirst(==(String(v)), vn)
            i === nothing && throw(ArgumentError("target_vars name \"$v\" not in varnames"))
            push!(idx, i)
        end
    end
    length(idx) == q || throw(ArgumentError("target_vars must have q=$q entries, got $(length(idx))"))
    all(1 .<= idx .<= N) || throw(ArgumentError("target_vars indices must be in 1:$N"))
    idx
end

# =============================================================================
# Sign-restriction helpers (panel or factor space)
# =============================================================================

"""
    varindex(m::StructuralDFM, name) -> Int

1-based index of panel variable `name` in `m.varnames`. Use after estimation,
or to interpret `irf` rows. Throws `ArgumentError` if `name` is absent.

Declarative `sign_restrictions` already resolve names; a closure can use the
same index if the names are known at call time:
`irf -> irf[1, findfirst(==("INDPRO"), names), 1] > 0`.
"""
function varindex(m::StructuralDFM, name::AbstractString)
    i = findfirst(==(String(name)), m.varnames)
    i === nothing && throw(ArgumentError(
        "variable \"$name\" not found among panel varnames $(m.varnames)"))
    i
end

function _sdfm_sign_sym(s)
    if s === :positive || s === :+ || s === :plus
        return 1
    elseif s === :negative || s === :- || s === :minus
        return -1
    elseif s isa Integer
        sg = Int(sign(s))
        sg == 0 && throw(ArgumentError("sign must be :positive or :negative, got $s"))
        return sg
    end
    throw(ArgumentError("sign must be :positive or :negative, got $s"))
end

function _sdfm_resolve_var(var, names::Vector{String}, n::Int)
    if var isa Integer
        (1 <= var <= n) || throw(ArgumentError("variable index $var not in 1:$n"))
        return Int(var), names[Int(var)]
    end
    s = String(var)
    i = findfirst(==(s), names)
    i === nothing && throw(ArgumentError("variable \"$s\" not found among $(names)"))
    i, s
end

"""Parse `SVARRestrictions` or `(variable, shock, horizons, sign)` tuples into rules.

`SignRestriction.horizon` is 0-based (impact = 0 → IRF row 1), matching the rest
of the package. Tuple `horizons` remain 1-based IRF indices. Non-sign entries of
an `SVARRestrictions` container are skipped (`:arias`/`:uhlig` enforce them).
"""
function _sdfm_parse_sign_rules(spec, names::Vector{String}, n::Int, q::Int, H::Int)
    spec === nothing && return NamedTuple{(:variable, :shock, :horizons, :sign, :name),
        Tuple{Int,Int,UnitRange{Int},Int,String}}[]
    rules = NamedTuple{(:variable, :shock, :horizons, :sign, :name),
        Tuple{Int,Int,UnitRange{Int},Int,String}}[]
    if spec isa SVARRestrictions
        for z in spec.signs
            z isa SignRestriction || continue
            h_idx = z.horizon + 1
            (1 <= h_idx <= H) || throw(ArgumentError(
                "restriction horizon $(z.horizon) (0-based) maps to IRF row $h_idx, not in 1:$H"))
            (1 <= z.variable <= n) || throw(ArgumentError(
                "restriction variable $(z.variable) not in 1:$n"))
            (1 <= z.shock <= q) || throw(ArgumentError(
                "restriction shock $(z.shock) not in 1:$q"))
            push!(rules, (variable=z.variable, shock=z.shock, horizons=h_idx:h_idx,
                          sign=Int(z.sign), name=names[z.variable]))
        end
        return rules
    end
    for item in spec
        length(item) == 4 || throw(ArgumentError(
            "each sign restriction is (variable, shock, horizons, sign), got $item"))
        var, shock, horizons, sgn = item
        v, nm = _sdfm_resolve_var(var, names, n)
        sh = Int(shock)
        (1 <= sh <= q) || throw(ArgumentError("restriction shock $sh not in 1:$q"))
        hs = horizons isa Integer ? (Int(horizons):Int(horizons)) : UnitRange{Int}(horizons)
        (first(hs) >= 1 && last(hs) <= H) || throw(ArgumentError(
            "restriction horizons $hs not in 1:$H"))
        push!(rules, (variable=v, shock=sh, horizons=hs, sign=_sdfm_sign_sym(sgn), name=nm))
    end
    rules
end

function _sdfm_rules_hold(irf, rules)
    @inbounds for rule in rules
        sng = rule.sign
        v, j = rule.variable, rule.shock
        for h in rule.horizons
            val = irf[h, v, j]
            (sng > 0 ? val > 0 : val < 0) || return false
        end
    end
    true
end

function _sdfm_tally_rules!(irf, rules, pass::Vector{Int})
    ok = true
    @inbounds for (k, rule) in enumerate(rules)
        holds = true
        sng = rule.sign
        v, j = rule.variable, rule.shock
        for h in rule.horizons
            val = irf[h, v, j]
            if !(sng > 0 ? val > 0 : val < 0)
                holds = false
                break
            end
        end
        holds && (pass[k] += 1)
        ok &= holds
    end
    ok
end

"""Haar search. Returns (Q, identified_set, acceptance_rate)."""
function _sdfm_sign_search(irf_from_Q, q::Int, H::Int, n_var::Int, ::Type{T},
    sign_check, rules, store_all::Bool, max_draws::Int, rng::AbstractRNG,
    variables::Vector{String}, shocks::Vector{String}) where {T<:AbstractFloat}

    n_rules = length(rules)
    pass = zeros(Int, n_rules)
    accepted_Q = Matrix{T}[]
    accepted_irf = Array{T,3}[]
    first_Q = Matrix{T}(undef, 0, 0)
    found = false

    for d in 1:max_draws
        Htry = generate_Q(q, T; rng=rng)
        irf = irf_from_Q(Htry)
        ok = n_rules > 0 ? _sdfm_tally_rules!(irf, rules, pass) : true
        if sign_check !== nothing
            ok = ok && sign_check(irf)
        end
        ok || continue
        if !found
            first_Q = copy(Htry)
            found = true
            store_all || return first_Q, nothing, T(1) / T(d)
        end
        if store_all
            push!(accepted_Q, copy(Htry))
            push!(accepted_irf, copy(irf))
        end
    end

    if !found
        if n_rules > 0
            worst = argmin(pass)
            throw(IdentificationError(
                "No rotation satisfied the sign restrictions in $max_draws draws " *
                "(lowest pass rate: $(rules[worst].name))"))
        end
        throw(IdentificationError(
            "No rotation satisfied the sign restrictions in $max_draws draws"))
    end

    n_acc = length(accepted_Q)
    draws = zeros(T, n_acc, H, n_var, q)
    for (i, a) in enumerate(accepted_irf)
        draws[i, :, :, :] = a
    end
    rate = T(n_acc) / T(max_draws)
    idset = SignIdentifiedSet{T}(accepted_Q, draws, n_acc, max_draws, rate, copy(variables), copy(shocks))
    first_Q, idset, rate
end



"""Panel IRFs Λ Ψ_h B0, optionally rescaled to original units."""
function _fglr_panel_irf(factor_var::VARModel{T}, Lambda::AbstractMatrix{T},
    B0::AbstractMatrix{T}, H::Int; X=nothing, standardized::Bool=false,
    units::Symbol=:raw) where {T<:AbstractFloat}

    r, q = size(B0)
    N = size(Lambda, 1)
    Psi = _ma_array(factor_var, H)
    panel = zeros(T, H, N, q)
    @inbounds for h in 1:H
        fac = @view(Psi[h, :, :]) * B0     # r × q
        panel[h, :, :] = Lambda * fac
    end
    if standardized && units === :raw && X !== nothing
        σ = vec(std(X; dims=1))
        σ = max.(σ, T(1e-10))
        panel .*= reshape(σ, 1, N, 1)
    end
    panel
end

# =============================================================================
# Legacy two-sided GDFM-factor VAR
# =============================================================================

function _estimate_sdfm_gdfm_var(gdfm::GeneralizedDynamicFactorModel{T}, q::Int,
    p, p_max::Int, check_stability::Bool, H::Int, identification::Symbol, sign_check, sign_restrictions,
    restriction_space::Symbol, store_all::Bool, max_draws::Int,
    narrative_check, target_vars, restrictions, transition_var, regime_indicator,
    vn::Vector{String}, sn::Vector{String}, rng::AbstractRNG,
    instrument=nothing, normalize=nothing) where {T<:AbstractFloat}

    F = gdfm.factors
    T_obs, N = size(gdfm.X)
    p_use, lag_crit = _resolve_factor_lag(F, p, p_max)
    T_obs > p_use + q || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for VAR($p_use) with $q factors"))

    factor_varnames = ["Factor $i" for i in 1:q]
    factor_var = estimate_var(F, p_use; check_stability=false, varnames=factor_varnames)

    FtF_inv = Matrix{T}(robust_inv(F' * F))
    Lambda = (FtF_inv * (F' * gdfm.X))'  # N × q

    idset = nothing
    rate = one(T)
    z_store = nothing
    Fstat = T(NaN)
    if identification === :cholesky
        Q = Matrix{T}(I, q, q)
    elseif identification === :proxy
        Idim = Matrix{T}(I, q, q)
        Hpr, z_store, Fstat = _sdfm_proxy_H(Lambda, Idim, factor_var, instrument, normalize, vn)
        Q = Matrix{T}(safe_cholesky(factor_var.Sigma) \ Hpr)
    elseif identification === :long_run
        Q = identify_long_run(factor_var)
    elseif identification === :sign
        if restriction_space === :factor && !store_all && sign_restrictions === nothing
            Q, _ = identify_sign(factor_var, H, sign_check; max_draws=max_draws, rng=rng)
        else
            n_check = restriction_space === :panel ? N : q
            names_check = restriction_space === :panel ? vn : factor_varnames
            rules = _sdfm_parse_sign_rules(sign_restrictions, names_check, n_check, q, H)
            buf = zeros(T, H, n_check, q)
            function irf_from_Q(Htry)
                firf = compute_irf(factor_var, Htry, H)
                if restriction_space === :panel
                    @inbounds for h in 1:H, j in 1:q
                        buf[h, :, j] = Lambda * @view(firf[h, :, j])
                    end
                    return buf
                end
                return firf
            end
            Q, idset, rate = _sdfm_sign_search(irf_from_Q, q, H, n_check, T, sign_check, rules,
                                               store_all, max_draws, rng, names_check, sn)
        end
    elseif identification === :arias
        rest = something(restrictions, sign_restrictions)
        rest isa SVARRestrictions || throw(ArgumentError(":arias requires restrictions::SVARRestrictions"))
        res = identify_arias(factor_var, rest, H; n_draws=1, n_rotations=max_draws, rng=rng)
        Q = res.Q_draws[1]
        rate = T(res.acceptance_rate)
    elseif identification === :uhlig
        rest = something(restrictions, sign_restrictions)
        rest isa SVARRestrictions || throw(ArgumentError(":uhlig requires restrictions::SVARRestrictions"))
        Q = identify_uhlig(factor_var, rest, H; rng=rng).Q
    else
        Q = compute_Q(factor_var, identification; horizon=H, check_func=sign_check,
                      narrative_check=narrative_check, max_draws=max_draws,
                      transition_var=transition_var, regime_indicator=regime_indicator, rng=rng)
    end

    factor_irf = compute_irf(factor_var, Q, H)
    structural_irf = zeros(T, H, N, q)
    for h in 1:H
        for j in 1:q
            structural_irf[h, :, j] = Lambda * @view(factor_irf[h, :, j])
        end
    end

    Lchol = safe_cholesky(factor_var.Sigma)
    B0 = Matrix{T}(Lchol * Q)
    K = Matrix{T}(Lchol)

    modu = _factor_var_modulus(factor_var)
    if check_stability && modu >= one(T)
        @warn "Estimated factor VAR is non-stationary (max eigenvalue modulus = $(round(modu, digits=4))); IRFs may be explosive. Pass check_stability=false to silence."
    end
    StructuralDFM{T}(gdfm, factor_var, B0, Q, identification,
        structural_irf, Lambda, p_use, sn, vn,
        K, q, :gdfm_var, F, Lambda, one(T), :raw, idset, rate,
        collect(1:q), lag_crit, T(modu), z_store, Fstat)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.nobs(m::StructuralDFM) = size(m.gdfm.X, 1)

# Factor-VAR parameters + N·r loadings + q·q rotation (not the spectral GDFM dof).
StatsAPI.dof(m::StructuralDFM) =
    dof(m.factor_var) + size(m.gdfm.X, 2) * m.r + m.gdfm.q * m.gdfm.q

function StatsAPI.r2(m::StructuralDFM{T}) where {T}
    r2(m.gdfm)
end

"""Predicted values: T×N common component from the underlying GDFM."""
StatsAPI.predict(m::StructuralDFM) = predict(m.gdfm)

"""Residuals: idiosyncratic component (panel minus common component)."""
StatsAPI.residuals(m::StructuralDFM) = residuals(m.gdfm)

"""Loadings used for panel projection (`N×r` FGLR, `N×q` legacy)."""
loadings(m::StructuralDFM) = m.method === :fglr ? m.loadings_static : m.loadings_td

"""Factors feeding the factor VAR (`T×r` FGLR static PCA, `T×q` GDFM)."""
factors(m::StructuralDFM) = m.method === :fglr ? m.static_factors : m.gdfm.factors

"""Factor-VAR coefficient matrix."""
StatsAPI.coef(m::StructuralDFM) = coef(m.factor_var)

StatsAPI.aic(m::StructuralDFM) = m.factor_var.aic
StatsAPI.bic(m::StructuralDFM) = m.factor_var.bic

"""Companion-matrix spectral radius of the factor VAR is strictly less than one."""
is_stable(m::StructuralDFM) = m.max_eigenvalue_modulus < one(eltype(m.B0))

"""First-stage F-statistic of the external instrument (`NaN` unless `identification=:proxy`)."""
first_stage_F(m::StructuralDFM) = m.first_stage_F

"""
    structural_shocks(sdfm::StructuralDFM) -> Matrix

Estimated structural shock series (`T_eff × q`) from the stored rotation:
`ε̂_t = B0⁺ û_t`. Under Cholesky with `r = q` the sample covariance is near `I_q`.
"""
structural_shocks(sdfm::StructuralDFM) = _sdfm_structural_shocks(sdfm)

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::StructuralDFM{T}) where {T}
    T_obs, N = size(m.gdfm.X)
    q = m.gdfm.q
    H = size(m.structural_irf, 1)

    spec = Any[
        "Dynamic factors (q)"   q;
        "Static factors (r)"    m.r;
        "Panel variables (N)"   N;
        "Observations (T)"      T_obs;
        "VAR lags (p)"          m.p_var;
        "IRF horizon (H)"       H;
        "Method"                string(m.method);
        "Identification"        string(m.identification);
        "Kernel"                string(m.gdfm.kernel);
        "Bandwidth"             m.gdfm.bandwidth
    ]
    _pretty_table(io, spec;
        title = "Structural DFM (q=$q, r=$(m.r), p=$(m.p_var), $(_id_label(m.identification)))",
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

    fv = m.factor_var
    var_spec = Any[
        "Lags (p)"                  m.p_var;
        "Lag criterion"             string(m.lag_criterion);
        "Variables"                 nvars(fv);
        "AIC"                       _fmt(fv.aic);
        "BIC"                       _fmt(fv.bic);
        "Max eigenvalue modulus"    _fmt(m.max_eigenvalue_modulus)
    ]
    _pretty_table(io, var_spec;
        title = "Factor VAR",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    id_spec = Any[
        "Method"       _id_label(m.identification);
        "Shocks"       join(m.shock_names, ", ");
        "Shock share"  _fmt_pct(m.shock_variance_share)
    ]
    if m.identification === :sign
        id_spec = vcat(id_spec, Any["Acceptance rate" _fmt_pct(m.acceptance_rate)])
        if m.identified_set !== nothing
            id_spec = vcat(id_spec, Any[
                "Accepted draws" "$(m.identified_set.n_accepted) / $(m.identified_set.n_total)"
            ])
        end
    elseif m.identification === :proxy && isfinite(m.first_stage_F)
        id_spec = vcat(id_spec, Any["First-stage F" _fmt(m.first_stage_F)])
    end
    _pretty_table(io, id_spec;
        title = "Identification",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # Show impact matrix B0 (r × q under FGLR)
    rshow = size(m.B0, 1)
    qshow = size(m.B0, 2)
    if rshow <= 8 && qshow <= 6
        b0_data = Matrix{Any}(undef, rshow, qshow + 1)
        rowlab = m.method === :fglr ? "Static factor" : "Factor"
        for i in 1:rshow
            b0_data[i, 1] = "$rowlab $i"
            for j in 1:qshow
                b0_data[i, j + 1] = _fmt(m.B0[i, j])
            end
        end
        _pretty_table(io, b0_data;
            title = "Impact Matrix B0",
            column_labels = vcat([""], m.shock_names),
            alignment = vcat([:l], fill(:r, qshow)),
        )
    end
end

"""Label for identification method in display."""
_id_label(s::Symbol) = s == :cholesky ? "Cholesky" : s == :sign ? "Sign" :
    s == :proxy ? "Proxy / IV" : string(s)
