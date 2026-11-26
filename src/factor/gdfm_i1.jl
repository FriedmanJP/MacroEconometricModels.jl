# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Non-stationary I(1) Cointegrated Generalized Dynamic Factor Model (BLL 2021).

Implements the Barigozzi, Lippi & Luciani (2021) one-sided infinite-dimensional GDFM
for cointegrated I(1) panels via random-permutation blockwise VECMs, level MA
inversion, and permanent/transitory or structural identification.

References:
- Barigozzi, M., Lippi, M., & Luciani, M. (2021). Large-Dimensional Dynamic Factor
  Models: Estimation of Impulse-Response Functions With I(1) Cointegrated Factors.
  *Journal of Business & Economic Statistics*, 39(1), 323–333.
"""

using LinearAlgebra, Statistics, Random, StatsAPI

# =============================================================================
# CointegratedGDFM Type
# =============================================================================

"""
    CointegratedGDFM{T<:AbstractFloat} <: AbstractFactorModel

Non-stationary ``I(1)`` cointegrated Generalized Dynamic Factor Model estimated via
the Barigozzi, Lippi & Luciani (2021) random-permutation blockwise VECM procedure:

```math
X_t = \\tau_t + \\chi_t + \\xi_t, \\qquad \\chi_t = \\sum_{h=0}^{\\infty} C_h \\eta_{t-h}
```

where:
- ``X_t`` is the ``N \\times 1`` observed panel in levels at time ``t``
- ``\\tau_t`` is the deterministic component (`:none`, `:constant`, or `:linear`)
- ``\\chi_t`` is the ``I(1)`` common component in levels driven by ``q`` common shocks
- ``\\xi_t`` is the ``I(0)`` idiosyncratic component
- ``C_h`` is the ``N \\times q`` level impulse response matrix at horizon ``h``, converging to ``C(1)``
- ``\\eta_t`` is the ``q \\times 1`` vector of identified structural common shocks

# Fields
- `X::Matrix{T}`: Original panel data in levels (`T_obs × N`)
- `common_levels::Matrix{T}`: Estimated common component in levels (`T_obs × N`, including deterministic trend)
- `idiosyncratic::Matrix{T}`: Idiosyncratic component `X - common_levels` (`T_obs × N`)
- `irf_matrices::Array{T,3}`: Identified level impulse response matrices ``C_0, \\dots, C_H`` (`(irf_horizon + 1) × N × q`)
- `long_run_impact::Matrix{T}`: Long-run level impact matrix ``C(1)`` (`N × q`, rank ``n_{\\text{permanent}}``)
- `B0::Matrix{T}`: Structural identification rotation matrix (`q × q`) mapping reduced-form innovations to structural shocks
- `shocks::Matrix{T}`: Identified structural shocks ``\\hat{\\eta}_t`` (`T_eff × q`)
- `trend_coef::Matrix{T}`: Deterministic trend coefficients `[μ'; δ']` (`2 × N`)
- `p_blocks::Matrix{Int}`: Selected block VECM VAR lag orders (`n_perm × n_blocks`)
- `q::Int`: Number of dynamic common shocks
- `n_permanent::Int`: Number of permanent stochastic trends (`0 ≤ n_permanent ≤ q`)
- `n_perm::Int`: Number of cross-sectional permutations averaged
- `bandwidth::Int`: Spectral lag-window truncation parameter `M`
- `kernel::Symbol`: Spectral lag-window kernel (`:bartlett`, `:parzen`, or `:tukey`)
- `det_trend::Symbol`: Deterministic trend specification (`:none`, `:constant`, or `:linear`)
- `identification::Symbol`: Structural identification scheme (`:permanent_transitory`, `:cholesky`, `:long_run`, `:sign`, or `:proxy`)
- `irf_horizon::Int`: Maximum stored level IRF horizon `H`
- `varnames::Vector{String}`: Panel variable names (length `N`)
- `shock_names::Vector{String}`: Structural shock names (length `q`)
- `manifest::Union{ReproManifest,Nothing}`: Reproducibility manifest
"""
struct CointegratedGDFM{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    common_levels::Matrix{T}
    idiosyncratic::Matrix{T}
    irf_matrices::Array{T,3}
    long_run_impact::Matrix{T}
    B0::Matrix{T}
    shocks::Matrix{T}
    trend_coef::Matrix{T}
    p_blocks::Matrix{Int}
    q::Int
    n_permanent::Int
    n_perm::Int
    bandwidth::Int
    kernel::Symbol
    det_trend::Symbol
    identification::Symbol
    irf_horizon::Int
    varnames::Vector{String}
    shock_names::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

# =============================================================================
# Public Estimator
# =============================================================================

"""
    estimate_gdfm_i1(X_levels, q; kwargs...) -> CointegratedGDFM

Estimate a non-stationary ``I(1)`` cointegrated Generalized Dynamic Factor Model via the
Barigozzi, Lippi & Luciani (2021) blockwise VECM estimator.

The estimator proceeds in four stages:
1. Detrend `X_levels` according to `det_trend` (`:none`, `:constant`, or `:linear`),
   form stationary first differences ``\\Delta X_t``, estimate the lag-window spectral
   density ``\\hat{\\Sigma}_{\\Delta X}(\\theta)`` and dynamic PCA common spectrum, and
   reconstruct the initial level common component ``\\hat{\\chi}_t``.
2. Partition `1:N` across `n_perm` seeded permutations into ``(q + 1)``-blocks and
   estimate a reduced-rank VECM on each block with cointegrating rank
   ``c_b = n_b - n_{\\text{permanent}}`` via concentrated Johansen moments.
3. Filter the detrended levels through the blockwise polynomial ``A^{(m)}(L)``, extract
   the ``q``-dimensional orthonormal innovations ``\\hat{u}_t^{(m)}``, invert each block
   VECM in levels to obtain level MA impulse responses ``B_h^{(m)}`` and long-run
   impacts via the Granger representation, and average across Procrustes-aligned
   permutations.
4. Identify the ``q \\times q`` structural matrix ``B_0`` (`:permanent_transitory`,
   `:cholesky`, `:long_run`, `:sign`, or `:proxy`) and recover structural shocks
   ``\\hat{\\eta}_t``.

# Arguments
- `X_levels::AbstractMatrix`: Panel data in levels (`T_obs × N`)
- `q::Int`: Number of dynamic common shocks (`1 ≤ q ≤ min(T_obs - 1, N)`)

# Keyword Arguments
- `p_max::Int=4`: Maximum block VECM VAR order in levels (`≥ 1`)
- `n_perm::Int=25`: Number of random cross-sectional permutations to average (`≥ 1`)
- `bandwidth::Int=0`: Lag-window truncation `M` (`0` selects `max(3, round(0.5√T))`)
- `kernel::Symbol=:bartlett`: Lag-window kernel (`:bartlett`, `:parzen`, or `:tukey`)
- `det_trend::Symbol=:constant`: Deterministic trend (`:none`, `:constant`, or `:linear`)
- `irf_horizon::Int=40`: Maximum level IRF horizon `H` (`irf_matrices` stores `h = 0..H`)
- `identification::Symbol=:permanent_transitory`: Identification (`:permanent_transitory`,
  `:cholesky`, `:long_run`, `:sign`, or `:proxy`)
- `sign_check::Union{Nothing,Function}=nothing`: Predicate on `(irf_horizon+1) × N × q`
  IRF array required when `identification=:sign`
- `n_permanent::Union{Nothing,Int}=nothing`: Number of permanent stochastic trends (`0..q`);
  `nothing` selects automatically via Johansen cointegration rank on level PCs
- `order::Union{Nothing,Vector{Int}}=nothing`: Observable indices for Cholesky/permanent ordering
- `target_vars::Union{Nothing,Vector{Int},Vector{<:AbstractString}}=nothing`: Target series for `:long_run`
- `instrument::Union{Nothing,AbstractVector}=nothing`: External instrument for `:proxy`
- `normalize::Union{Nothing,Tuple}=nothing`: Normalization `(var, val)` for `:proxy`
- `max_draws::Int=1000`: Maximum Haar draws for `:sign` search
- `lag_criterion::Symbol=:bic`: Block VECM lag selection criterion (`:bic` or `:aic`)
- `standardize::Bool=true`: Scale each series by first-difference standard deviation prior to PCA
- `threaded::Bool=true`: Parallelize across permutations using `Threads.@threads`
- `varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing`: Panel variable names (length `N`)
- `shock_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing`: Structural shock names (length `q`)
- `rng::AbstractRNG=Random.default_rng()`: RNG for permutation and sign draws
- `seed::Union{Integer,Nothing}=nothing`: Optional seed recorded in `manifest`

# Returns
`CointegratedGDFM{T}` containing level common/idiosyncratic components, level IRFs,
long-run impact matrix, structural shocks, and diagnostics.

# References
- Barigozzi, M., Lippi, M., & Luciani, M. (2021). Large-Dimensional Dynamic Factor
  Models: Estimation of Impulse-Response Functions With `I(1)` Cointegrated Factors.
  *Journal of Business & Economic Statistics*, 39(1), 323–333.
"""
function estimate_gdfm_i1(
    X_levels::AbstractMatrix{T},
    q::Int;
    p_max::Int=4,
    n_perm::Int=25,
    bandwidth::Int=0,
    kernel::Symbol=:bartlett,
    det_trend::Symbol=:constant,
    irf_horizon::Int=40,
    identification::Symbol=:permanent_transitory,
    sign_check::Union{Nothing,Function}=nothing,
    n_permanent::Union{Int,Nothing}=nothing,
    order::Union{Nothing,Vector{Int}}=nothing,
    target_vars::Union{Nothing,Vector{Int},Vector{<:AbstractString}}=nothing,
    instrument::Union{Nothing,AbstractVector}=nothing,
    normalize::Union{Nothing,Tuple}=nothing,
    max_draws::Int=1000,
    lag_criterion::Symbol=:bic,
    standardize::Bool=true,
    threaded::Bool=true,
    varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
    shock_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
) where {T<:AbstractFloat}
    _validate_data(X_levels, "X_levels")
    T_obs, N = size(X_levels)
    T_obs >= 6 || throw(ArgumentError("Need at least 6 observations, got $T_obs"))
    validate_factor_inputs(T_obs - 1, N, q; context="dynamic factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))
    validate_option(det_trend, "det_trend", (:none, :constant, :linear))
    validate_option(identification, "identification",
        (:permanent_transitory, :cholesky, :long_run, :sign, :proxy))
    validate_option(lag_criterion, "lag_criterion", (:bic, :aic))

    p_max >= 1 || throw(ArgumentError("p_max must be >= 1, got $p_max"))
    n_perm >= 1 || throw(ArgumentError("n_perm must be >= 1, got $n_perm"))
    irf_horizon >= 1 || throw(ArgumentError("irf_horizon must be >= 1, got $irf_horizon"))
    max_draws >= 1 || throw(ArgumentError("max_draws must be >= 1, got $max_draws"))

    if n_permanent !== nothing
        (0 <= n_permanent <= q) || throw(ArgumentError(
            "n_permanent must be in 0:$q, got $n_permanent"))
    end
    if identification === :sign && sign_check === nothing
        throw(ArgumentError("sign_check is required for :sign identification"))
    end
    if identification === :proxy && instrument === nothing
        throw(ArgumentError("identification=:proxy requires `instrument`"))
    end

    vn = varnames === nothing ? ["Var $i" for i in 1:N] : String.(collect(varnames))
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X_levels has $N columns"))

    rng = _resolve_repro_rng(rng, seed)

    p_max_use = min(p_max, max(1, T_obs - q - 4))
    T_eff = T_obs - p_max_use
    T_eff >= max(2, q) || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for p_max=$p_max_use and q=$q"))

    X_orig = Matrix{T}(X_levels)

    # Step 1: Detrend levels, scale by first-difference SD, dynamic PCA on ΔX, reconstruct χ̂
    X_d, trend_coef, tau_hat = _gdfmi1_detrend(X_orig, det_trend)
    dX_raw = diff(X_d; dims=1)
    σ_Δ = standardize ? max.(vec(std(dX_raw; dims=1)), T(1e-10)) : ones(T, N)
    X_proc = X_d ./ σ_Δ'
    dX_proc = dX_raw ./ σ_Δ'
    dX_c = dX_proc .- mean(dX_proc; dims=1)

    M_bw = bandwidth > 0 ? bandwidth : _select_lag_window(T_obs)
    frequencies, spectral_dX = _fhlz_lag_window_spectrum(dX_c, M_bw, kernel, p_max_use)
    evals_dX, evecs_dX = _spectral_eigendecomposition(spectral_dX)
    loadings_dX = evecs_dX[:, 1:q, :]

    chi_proc = _reconstruct_time_domain(loadings_dX, X_proc, frequencies;
        spectral=:lag_window, bandwidth=M_bw)

    n_perm_use = n_permanent === nothing ?
        _gdfmi1_detect_n_permanent(X_proc, spectral_dX, evals_dX, q, p_max_use) :
        Int(n_permanent)

    vecm_det = det_trend === :none ? :none : :constant

    # Step 2: Pre-generate per-permutation seeds sequentially (#243) and run blockwise VECMs
    seeds = rand(rng, UInt64, n_perm)
    id_seed = rand(rng, UInt64)
    n_blocks = max(1, div(N, q + 1))

    PermResT = NamedTuple{
        (:p_blocks_m, :U_m, :B_m, :C_inf_m),
        Tuple{Vector{Int},Matrix{T},Array{T,3},Matrix{T}},
    }
    perm_results = Vector{PermResT}(undef, n_perm)

    if threaded && Threads.nthreads() > 1 && n_perm > 1
        Threads.@threads for m in 1:n_perm
            perm_results[m] = _gdfmi1_run_permutation(
                X_proc, chi_proc, q, n_perm_use, p_max_use,
                vecm_det, lag_criterion, irf_horizon, seeds[m])
        end
    else
        for m in 1:n_perm
            perm_results[m] = _gdfmi1_run_permutation(
                X_proc, chi_proc, q, n_perm_use, p_max_use,
                vecm_det, lag_criterion, irf_horizon, seeds[m])
        end
    end

    # Step 3: Procrustes alignment to permutation 1, averaging, re-whitening, unscaling
    p_blocks = Matrix{Int}(undef, n_perm, n_blocks)
    U_ref = perm_results[1].U_m
    U_acc = copy(U_ref)
    B_acc = copy(perm_results[1].B_m)
    C_inf_acc = copy(perm_results[1].C_inf_m)
    p_blocks[1, :] .= perm_results[1].p_blocks_m

    H_len = irf_horizon + 1
    @inbounds for m in 2:n_perm
        res_m = perm_results[m]
        p_blocks[m, :] .= res_m.p_blocks_m
        sv = svd(res_m.U_m' * U_ref)
        O_m = sv.U * sv.Vt
        mul!(U_acc, res_m.U_m, O_m, one(T), one(T))
        for h in 1:H_len
            mul!(@view(B_acc[h, :, :]), @view(res_m.B_m[h, :, :]), O_m, one(T), one(T))
        end
        mul!(C_inf_acc, res_m.C_inf_m, O_m, one(T), one(T))
    end

    inv_perm = one(T) / T(n_perm)
    U_bar = U_acc .* inv_perm
    B_bar = B_acc .* inv_perm
    C_inf_bar = C_inf_acc .* inv_perm

    L_bar, L_bar_inv = _fhlz_symmetric_whitening(U_bar)
    U_final = U_bar * L_bar_inv
    B_final = similar(B_bar)
    @inbounds for h in 1:H_len
        B_final[h, :, :] = @view(B_bar[h, :, :]) * L_bar
    end
    C_inf_final = C_inf_bar * L_bar

    if standardize
        @inbounds for h in 1:H_len
            B_final[h, :, :] .*= σ_Δ
        end
        C_inf_final .*= σ_Δ
    end

    common_levels = chi_proc .* σ_Δ' .+ tau_hat
    idiosyncratic = X_orig - common_levels

    # Step 4: Structural identification & shock recovery
    rng_id = Random.Xoshiro(id_seed)
    B0 = _gdfmi1_identify_B0(
        B_final, C_inf_final, U_final, identification, n_perm_use,
        order, target_vars, sign_check, instrument, normalize,
        max_draws, T_obs, p_max_use, vn, rng_id)

    irf_matrices = similar(B_final)
    @inbounds for h in 1:H_len
        irf_matrices[h, :, :] = @view(B_final[h, :, :]) * B0
    end
    long_run_impact = C_inf_final * B0
    if identification === :permanent_transitory && n_perm_use < q
        long_run_impact[:, (n_perm_use + 1):q] .= zero(T)
    end

    shocks = if _q_is_orthogonal(B0)
        Matrix{T}(U_final * B0)
    else
        Matrix{T}((B0 \ U_final')')
    end

    default_sn = if shock_names !== nothing
        String.(collect(shock_names))
    elseif identification === :permanent_transitory
        _svec_default_shock_names(n_perm_use, q - n_perm_use)
    elseif identification === :proxy
        vcat(["Proxy"], ["Unidentified $i" for i in 2:q])
    else
        ["Shock $i" for i in 1:q]
    end
    length(default_sn) == q || throw(ArgumentError(
        "shock_names has $(length(default_sn)) entries but q=$q"))

    model = CointegratedGDFM{T}(
        X_orig, common_levels, idiosyncratic, irf_matrices, long_run_impact,
        B0, shocks, trend_coef, p_blocks, q, n_perm_use, n_perm, M_bw,
        kernel, det_trend, identification, irf_horizon, vn, default_sn, nothing,
    )
    return _with_manifest(model, capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "q" => q,
            "n_permanent" => n_perm_use,
            "p_max" => p_max_use,
            "n_perm" => n_perm,
            "bandwidth" => M_bw,
            "kernel" => String(kernel),
            "det_trend" => String(det_trend),
            "irf_horizon" => irf_horizon,
            "identification" => String(identification),
            "lag_criterion" => String(lag_criterion),
            "standardize" => standardize,
            "threaded" => threaded,
        )))
end

@float_fallback estimate_gdfm_i1 X_levels

# =============================================================================
# Internal Helpers (_gdfmi1_*)
# =============================================================================

"""Detrend levels matrix (`:none`, `:constant`, or `:linear`); returns `(X_detrended, trend_coef, tau_hat)`."""
function _gdfmi1_detrend(X::Matrix{T}, det_trend::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X)
    trend_coef = zeros(T, 2, N)
    if det_trend === :none
        return copy(X), trend_coef, zeros(T, T_obs, N)
    elseif det_trend === :constant
        μ = vec(mean(X; dims=1))
        trend_coef[1, :] .= μ
        tau = repeat(μ', T_obs, 1)
        return X .- tau, trend_coef, tau
    else  # :linear
        D = hcat(ones(T, T_obs), T.(1:T_obs))
        beta_det = D \ X
        trend_coef .= beta_det
        tau = D * beta_det
        return X - tau, trend_coef, tau
    end
end

"""Detect the number of permanent stochastic trends `n_permanent ∈ 1:q` from level PCs and zero-frequency spectrum."""
function _gdfmi1_detect_n_permanent(
    X_proc::Matrix{T},
    spectral_dX::Array{Complex{T},3},
    evals_dX::Matrix{T},
    q::Int,
    p_max_use::Int,
) where {T<:AbstractFloat}
    q == 1 && return 1
    T_obs, _ = size(X_proc)
    Σ_lev = (X_proc' * X_proc) / T(T_obs)
    E_lev = eigen(Symmetric((Σ_lev + Σ_lev') / T(2)))
    idx_lev = sortperm(E_lev.values; rev=true)
    F_lev = X_proc * E_lev.vectors[:, idx_lev[1:q]]
    dF = diff(F_lev; dims=1)
    sd_dF = max.(vec(std(dF; dims=1)), T(1e-10))
    F_scaled = F_lev ./ sd_dF'

    p_joh = clamp(p_max_use, 1, 2)
    if T_obs >= q + p_joh + 12
        try
            joh = johansen_test(F_scaled, p_joh; deterministic=:constant, significance=0.05)
            return clamp(q - joh.rank, 1, q)
        catch
        end
    end

    # Fallback: ratio of zero-frequency eigenvalue to peak across frequencies
    ratios = [evals_dX[j, 1] / max(maximum(@view(evals_dX[j, :])), T(1e-12)) for j in 1:q]
    n_p = count(>=(T(0.25)), ratios)
    clamp(n_p, 1, q)
end

"""Concentrated reduced-rank VECM estimation on block `Y_b` (`T_obs × n_b`) with cointegration rank `c_b`."""
function _gdfmi1_fit_block_vecm(
    Y_b::AbstractMatrix{T},
    q::Int,
    n_perm_use::Int,
    p_max_use::Int,
    vecm_det::Symbol,
    lag_criterion::Symbol,
) where {T<:AbstractFloat}
    T_obs, n_b = size(Y_b)
    c_b = clamp(n_b - n_perm_use, 0, n_b)
    n_perm_block = n_b - c_b

    pen = lag_criterion === :bic ? log(T(max(T_obs - p_max_use, 2))) : T(2)
    k_top = min(q, n_b)

    best_score = T(Inf)
    best_p = 1
    best_alpha = zeros(T, n_b, c_b)
    best_beta = zeros(T, n_b, c_b)
    best_Pi = zeros(T, n_b, n_b)
    best_Gamma = Matrix{T}[]

    for p_c in 1:p_max_use
        mom = _johansen_moments(Y_b, p_c, vecm_det)
        T_eff_c = mom.T_eff
        T_eff_c >= max(2, n_b + 1) || continue

        tr0 = max(tr(mom.S00) / T(n_b), T(1e-10))
        reg0 = max(T(1e-5), T(100) * eps(T)) * tr0
        eig_floor = max(T(1e-12), T(10) * eps(T)) * tr0

        local alpha_c::Matrix{T}
        local beta_c::Matrix{T}
        local Pi_c::Matrix{T}
        local Gamma_c::Vector{Matrix{T}}
        local U_c::Matrix{T}

        if c_b == 0
            alpha_c = zeros(T, n_b, 0)
            beta_c = zeros(T, n_b, 0)
            Pi_c = zeros(T, n_b, n_b)
            if size(mom.Z, 2) > 0
                Z = mom.Z
                ZtZ = Matrix{T}((Z' * Z) / T(T_eff_c))
                tr_z = max(tr(ZtZ) / T(size(Z, 2)), T(1e-10))
                reg_z = max(T(1e-5), T(100) * eps(T)) * tr_z
                B_sr = robust_inv(ZtZ + reg_z * I; silent=true) * ((Z' * mom.dY_eff) / T(T_eff_c))
                U_c = mom.dY_eff - Z * B_sr
                Gamma_c = [Matrix{T}(B_sr[((k - 1) * n_b + 1):(k * n_b), :]') for k in 1:(p_c - 1)]
            else
                U_c = copy(mom.dY_eff)
                Gamma_c = Matrix{T}[]
            end
        elseif c_b < n_b
            S00_r = Matrix{T}((mom.S00 + mom.S00') / T(2)) + reg0 * I
            tr1 = max(tr(mom.S11) / T(size(mom.S11, 1)), T(1e-10))
            reg1 = max(T(1e-5), T(100) * eps(T)) * tr1
            S11_r = Matrix{T}((mom.S11 + mom.S11') / T(2)) + reg1 * I

            L1 = safe_cholesky(S11_r; silent=true)
            L1_inv = robust_inv(Matrix{T}(L1); silent=true)
            S00_inv = robust_inv(S00_r; silent=true)
            S10 = Matrix{T}(mom.S01')
            W = L1_inv * (S10 * S00_inv * mom.S01) * L1_inv'
            W = Matrix{T}((W + W') / T(2))
            eig = eigen(Symmetric(W))
            idx = sortperm(eig.values; rev=true)
            beta_star = Matrix{T}(L1_inv' * eig.vectors[:, idx[1:c_b]])
            beta_c = Matrix{T}(beta_star[1:n_b, :])
            norm_b = Matrix{T}(beta_star' * S11_r * beta_star)
            alpha_c = Matrix{T}(mom.S01 * beta_star * robust_inv(norm_b; silent=true))
            Pi_c = alpha_c * beta_c'

            ecm = mom.Y_lag_aug * beta_star
            target = mom.dY_eff - ecm * alpha_c'
            if size(mom.Z, 2) > 0
                Z = mom.Z
                ZtZ = Matrix{T}((Z' * Z) / T(T_eff_c))
                tr_z = max(tr(ZtZ) / T(size(Z, 2)), T(1e-10))
                reg_z = max(T(1e-5), T(100) * eps(T)) * tr_z
                B_sr = robust_inv(ZtZ + reg_z * I; silent=true) * ((Z' * target) / T(T_eff_c))
                U_c = target - Z * B_sr
                Gamma_c = [Matrix{T}(B_sr[((k - 1) * n_b + 1):(k * n_b), :]') for k in 1:(p_c - 1)]
            else
                U_c = target
                Gamma_c = Matrix{T}[]
            end
        else
            # All components stationary (n_perm_use == 0): levels VAR(p_c) converted to VECM form
            beta_c = Matrix{T}(I, n_b, n_b)
            Y_eff_v, X_v = construct_var_matrices(Y_b, p_c)
            Xv_use = vecm_det === :none ? @view(X_v[:, 2:end]) : X_v
            XtX = Matrix{T}((Xv_use' * Xv_use) / T(size(Y_eff_v, 1)))
            tr_v = max(tr(XtX) / T(max(size(Xv_use, 2), 1)), T(1e-10))
            reg_v = max(T(1e-5), T(100) * eps(T)) * tr_v
            B_v = robust_inv(XtX + reg_v * I; silent=true) * ((Xv_use' * Y_eff_v) / T(size(Y_eff_v, 1)))
            U_c = Y_eff_v - Xv_use * B_v
            off = vecm_det === :none ? 0 : 1
            A_v = [Matrix{T}(B_v[(off + 1 + (k - 1) * n_b):(off + k * n_b), :]') for k in 1:p_c]
            Pi_c = sum(A_v) - Matrix{T}(I, n_b, n_b)
            alpha_c = copy(Pi_c)
            Gamma_c = Vector{Matrix{T}}(undef, max(0, p_c - 1))
            if p_c > 1
                Gamma_c[p_c - 1] = -A_v[p_c]
                for k in (p_c - 2):-1:1
                    Gamma_c[k] = Gamma_c[k + 1] - A_v[k + 1]
                end
            end
        end

        Σ_r_raw = (U_c' * U_c) / T(size(U_c, 1))
        Σ_r = Symmetric(Matrix{T}((Σ_r_raw + Σ_r_raw') / T(2)))
        evals_r = sort(real.(eigvals(Σ_r)); rev=true)
        λ_top = max.(evals_r[1:k_top], eig_floor)
        n_parms = c_b * n_b + (p_c - 1) * n_b * n_b
        score = sum(log, λ_top) + T(n_parms) * pen / T(size(U_c, 1))

        if score < best_score || p_c == 1
            best_score = score
            best_p = p_c
            best_alpha = alpha_c
            best_beta = beta_c
            best_Pi = Pi_c
            best_Gamma = Gamma_c
        end
    end

    A_b = _gdfmi1_vecm_to_levels_ar(best_Pi, best_Gamma, n_b, best_p)
    best_Gamma, A_b = _gdfmi1_stabilize_block_vecm!(best_Pi, best_Gamma, A_b, n_b, best_p, n_perm_block)
    Xi_b = _gdfmi1_block_long_run(best_alpha, best_beta, best_Gamma, n_b, n_perm_block)

    best_p, A_b, Xi_b
end

"""Convert VECM `(Pi, Gamma)` to levels VAR matrices `A_1, ..., A_p` (matching `to_var`)."""
function _gdfmi1_vecm_to_levels_ar(
    Pi::Matrix{T},
    Gamma::Vector{Matrix{T}},
    n_b::Int,
    p_b::Int,
) where {T<:AbstractFloat}
    In = Matrix{T}(I, n_b, n_b)
    A = Vector{Matrix{T}}(undef, p_b)
    if p_b == 1
        A[1] = Pi + In
    else
        A[1] = Pi + In + Gamma[1]
        for i in 2:(p_b - 1)
            A[i] = Gamma[i] - Gamma[i - 1]
        end
        A[p_b] = -Gamma[p_b - 1]
    end
    A
end

"""Companion eigenvalue moduli of a block levels VAR `[A_1, ..., A_p]`."""
function _gdfmi1_companion_moduli(A_b::Vector{Matrix{T}}, n_b::Int, p_b::Int) where {T<:AbstractFloat}
    dim = n_b * p_b
    F = zeros(T, dim, dim)
    @inbounds for k in 1:p_b
        F[1:n_b, ((k - 1) * n_b + 1):(k * n_b)] .= A_b[k]
    end
    if p_b > 1
        @inbounds for i in 1:(p_b - 1)
            F[(i * n_b + 1):((i + 1) * n_b), ((i - 1) * n_b + 1):(i * n_b)] .= Matrix{T}(I, n_b, n_b)
        end
    end
    vals = eigvals(F)
    all(isfinite, vals) || return fill(T(Inf), dim)
    sort(abs.(vals); rev=true)
end

"""Damp short-run `Gamma` (preserving `Pi` and unit roots) until stationary roots lie inside unit circle."""
function _gdfmi1_stabilize_block_vecm!(
    Pi_b::Matrix{T},
    Gamma_b::Vector{Matrix{T}},
    A_b::Vector{Matrix{T}},
    n_b::Int,
    p_b::Int,
    n_perm_block::Int,
) where {T<:AbstractFloat}
    mods = _gdfmi1_companion_moduli(A_b, n_b, p_b)
    max_mod = mods[1]
    stat_idx = n_perm_block + 1
    stat_mod = stat_idx <= length(mods) ? mods[stat_idx] : zero(T)

    if isfinite(max_mod) && max_mod <= T(1.005) && stat_mod < T(0.99)
        return Gamma_b, A_b
    end

    # Damping short-run Γ preserves Π = αβ' and therefore preserves exact cointegrating space
    if !isempty(Gamma_b)
        for factor in (T(0.8), T(0.6), T(0.4), T(0.2), zero(T))
            G_try = [factor^k .* Gamma_b[k] for k in eachindex(Gamma_b)]
            A_try = _gdfmi1_vecm_to_levels_ar(Pi_b, G_try, n_b, p_b)
            m_try = _gdfmi1_companion_moduli(A_try, n_b, p_b)
            s_try = stat_idx <= length(m_try) ? m_try[stat_idx] : zero(T)
            if isfinite(m_try[1]) && m_try[1] <= T(1.005) && s_try < T(0.99)
                return G_try, A_try
            end
        end
    end

    # Fallback: damp Π toward stationary contraction in cointegrating directions
    G_zero = [zeros(T, n_b, n_b) for _ in 1:(p_b - 1)]
    for scale in (T(0.8), T(0.5), T(0.25), T(0.1))
        Pi_try = scale .* Pi_b
        A_try = _gdfmi1_vecm_to_levels_ar(Pi_try, G_zero, n_b, p_b)
        m_try = _gdfmi1_companion_moduli(A_try, n_b, p_b)
        s_try = stat_idx <= length(m_try) ? m_try[stat_idx] : zero(T)
        if isfinite(m_try[1]) && m_try[1] <= T(1.005) && s_try < T(0.99)
            Pi_b .= Pi_try
            return G_zero, A_try
        end
    end
    Pi_b .= zero(T)
    G_zero, _gdfmi1_vecm_to_levels_ar(Pi_b, G_zero, n_b, p_b)
end

"""Granger representation long-run impact matrix `Ξ_b = β⊥ (α⊥' Ψ_b β⊥)⁻¹ α⊥'` for a block."""
function _gdfmi1_block_long_run(
    alpha_b::Matrix{T},
    beta_b::Matrix{T},
    Gamma_b::Vector{Matrix{T}},
    n_b::Int,
    n_perm_block::Int,
) where {T<:AbstractFloat}
    n_perm_block == 0 && return zeros(T, n_b, n_b)
    Psi_b = Matrix{T}(I, n_b, n_b)
    @inbounds for G in Gamma_b
        Psi_b .-= G
    end
    c_b = size(beta_b, 2)
    c_b == 0 && return Matrix{T}(robust_inv(Psi_b; silent=true))

    sv_a = svd(alpha_b; full=true)
    sv_b = svd(beta_b; full=true)
    aperp = sv_a.U[:, (n_b - n_perm_block + 1):n_b]
    bperp = sv_b.U[:, (n_b - n_perm_block + 1):n_b]
    Mid = aperp' * Psi_b * bperp
    Matrix{T}(bperp * (robust_inv(Matrix{T}(Mid); silent=true) * aperp'))
end

"""Apply blockwise levels-VAR polynomial `A^{(m)}(L)` to `M` (`T_obs × N`) yielding `T_eff × N` filtered matrix."""
function _gdfmi1_apply_blocks(
    M::AbstractMatrix{T},
    blocks::Vector{Vector{Int}},
    A_blocks::Vector{Vector{Matrix{T}}},
    p_max_use::Int,
) where {T<:AbstractFloat}
    T_obs, N = size(M)
    T_eff = T_obs - p_max_use
    t_idx = (p_max_use + 1):T_obs
    Z = Matrix{T}(undef, T_eff, N)
    @inbounds for b in eachindex(blocks)
        idx_b = blocks[b]
        A_b = A_blocks[b]
        p_b = length(A_b)
        Zb = copy(@view M[t_idx, idx_b])
        for k in 1:p_b
            mul!(Zb, @view(M[t_idx .- k, idx_b]), transpose(A_b[k]), -one(T), one(T))
        end
        Z[:, idx_b] .= Zb
    end
    Z .- mean(Z; dims=1)
end

"""Recursively invert `A_b(L) B_b(L) = R_b^{(m)}` in levels for `h = 0..irf_horizon`."""
function _gdfmi1_invert_levels(
    blocks::Vector{Vector{Int}},
    A_blocks::Vector{Vector{Matrix{T}}},
    Xi_blocks::Vector{Matrix{T}},
    R_m::AbstractMatrix{T},
    irf_horizon::Int,
    N::Int,
    q::Int,
) where {T<:AbstractFloat}
    H_len = irf_horizon + 1
    B_m = zeros(T, H_len, N, q)
    C_inf_m = zeros(T, N, q)
    @inbounds for b in eachindex(blocks)
        idx_b = blocks[b]
        n_b = length(idx_b)
        A_b = A_blocks[b]
        p_b = length(A_b)
        R_b = @view R_m[idx_b, :]
        B_m[1, idx_b, :] .= R_b
        acc = Matrix{T}(undef, n_b, q)
        for h in 2:H_len
            fill!(acc, zero(T))
            for k in 1:min(h - 1, p_b)
                mul!(acc, A_b[k], @view(B_m[h - k, idx_b, :]), one(T), one(T))
            end
            B_m[h, idx_b, :] .= acc
        end
        C_inf_m[idx_b, :] .= Xi_blocks[b] * R_b
    end
    B_m, C_inf_m
end

"""Execute one permutation of the BLL (2021) blockwise VECM estimator."""
function _gdfmi1_run_permutation(
    X_proc::Matrix{T},
    chi_proc::Matrix{T},
    q::Int,
    n_perm_use::Int,
    p_max_use::Int,
    vecm_det::Symbol,
    lag_criterion::Symbol,
    irf_horizon::Int,
    seed_m::UInt64,
) where {T<:AbstractFloat}
    N = size(X_proc, 2)
    perm = randperm(Random.Xoshiro(seed_m), N)
    blocks = _fhlz_make_blocks(perm, q)
    n_blocks = length(blocks)

    p_blocks_m = Vector{Int}(undef, n_blocks)
    A_blocks = Vector{Vector{Matrix{T}}}(undef, n_blocks)
    Xi_blocks = Vector{Matrix{T}}(undef, n_blocks)

    for b in 1:n_blocks
        idx_b = blocks[b]
        Y_b = @view chi_proc[:, idx_b]
        p_b, A_b, Xi_b = _gdfmi1_fit_block_vecm(
            Y_b, q, n_perm_use, p_max_use, vecm_det, lag_criterion)
        p_blocks_m[b] = p_b
        A_blocks[b] = A_b
        Xi_blocks[b] = Xi_b
    end

    # Extract q-dimensional innovation space from common filtered levels Z_χ = A^{(m)}(L) χ̂
    Z_chi = _gdfmi1_apply_blocks(chi_proc, blocks, A_blocks, p_max_use)
    T_eff = size(Z_chi, 1)
    Γ_Zχ = Matrix{T}((Z_chi' * Z_chi) / T(T_eff))
    Γ_Zχ = Matrix{T}((Γ_Zχ + Γ_Zχ') / T(2))
    Ez = eigen(Symmetric(Γ_Zχ))
    idx_z = sortperm(Ez.values; rev=true)
    V_q = Matrix{T}(Ez.vectors[:, idx_z[1:q]])
    @inbounds for j in 1:q
        k_max = argmax(abs.(view(V_q, :, j)))
        V_q[k_max, j] < zero(T) && (V_q[:, j] .*= -one(T))
    end

    # Filter observed levels X_proc into stationary common innovation space
    Z_X = _gdfmi1_apply_blocks(X_proc, blocks, A_blocks, p_max_use)
    U_raw = Z_X * V_q
    L_whiten_T, W_whiten = _fhlz_symmetric_whitening(U_raw)
    R_m = V_q * L_whiten_T
    U_m = U_raw * W_whiten

    B_m, C_inf_m = _gdfmi1_invert_levels(
        blocks, A_blocks, Xi_blocks, R_m, irf_horizon, N, q)

    (p_blocks_m=p_blocks_m, U_m=U_m, B_m=B_m, C_inf_m=C_inf_m)
end

"""Identify structural rotation `B0` (`q × q`) for `CointegratedGDFM`."""
function _gdfmi1_identify_B0(
    B_final::Array{T,3},
    C_inf_final::Matrix{T},
    U_final::Matrix{T},
    identification::Symbol,
    n_perm_use::Int,
    order::Union{Nothing,Vector{Int}},
    target_vars,
    sign_check::Union{Nothing,Function},
    instrument,
    normalize,
    max_draws::Int,
    T_obs::Int,
    p_max_use::Int,
    vn::Vector{String},
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    N = size(C_inf_final, 1)
    q = size(C_inf_final, 2)
    ord = something(order, collect(1:q))
    length(ord) == q || throw(ArgumentError(
        "order has $(length(ord)) entries but q=$q"))
    all(1 .<= ord .<= N) || throw(ArgumentError(
        "order indices must be in 1:$N"))

    if identification === :permanent_transitory
        sv = svd(C_inf_final)
        V = Matrix{T}(sv.Vt')
        k = n_perm_use
        d = q - k
        Q_pt = copy(V)

        if k >= 2
            C_perm = C_inf_final * V[:, 1:k]
            M = C_perm[ord[1:k], :]
            qrM = qr(M')
            Qrot = Matrix{T}(qrM.Q)
            Rlow = M * Qrot
            @inbounds for j in 1:k
                Rlow[j, j] < zero(T) && (Qrot[:, j] .*= -one(T))
            end
            Q_pt[:, 1:k] = V[:, 1:k] * Qrot
        elseif k == 1
            c1 = C_inf_final * @view(Q_pt[:, 1])
            ref = order === nothing ? c1[argmax(abs.(c1))] : c1[ord[1]]
            ref < zero(T) && (Q_pt[:, 1] .*= -one(T))
        end

        if d >= 2
            B0_trans = @view(B_final[1, :, :]) * V[:, (k + 1):q]
            Mt = B0_trans[ord[(k + 1):q], :]
            qrT = qr(Mt')
            Qrot_t = Matrix{T}(qrT.Q)
            Rt = Mt * Qrot_t
            @inbounds for j in 1:d
                Rt[j, j] < zero(T) && (Qrot_t[:, j] .*= -one(T))
            end
            Q_pt[:, (k + 1):q] = V[:, (k + 1):q] * Qrot_t
        elseif d == 1
            bt = @view(B_final[1, :, :]) * @view(Q_pt[:, q])
            ref = order === nothing ? bt[argmax(abs.(bt))] : bt[ord[q]]
            ref < zero(T) && (Q_pt[:, q] .*= -one(T))
        end
        return Matrix{T}(Q_pt)
    elseif identification === :cholesky
        P = B_final[1, ord, :]
        L = safe_cholesky(Matrix{T}(P * P'); silent=true)
        return Matrix{T}(P \ L)
    elseif identification === :long_run
        idx = _sdfm_resolve_targets(target_vars, vn, N, q)
        P = C_inf_final[idx, :]
        L = safe_cholesky(Matrix{T}(P * P'); silent=true)
        H_lr = Matrix{T}(P \ L)
        (C_inf_final * H_lr)[idx[1], 1] < zero(T) && (H_lr[:, 1] .*= -one(T))
        return H_lr
    elseif identification === :sign
        H_len = size(B_final, 1)
        irf_buf = zeros(T, H_len, N, q)
        for _ in 1:max_draws
            Htry = generate_Q(q, T; rng=rng)
            @inbounds for h in 1:H_len
                irf_buf[h, :, :] = @view(B_final[h, :, :]) * Htry
            end
            sign_check(irf_buf) && return Matrix{T}(Htry)
        end
        throw(IdentificationError(
            "No rotation satisfied the sign restrictions in $max_draws draws"))
    else  # :proxy
        T_eff = size(U_final, 1)
        Sigma_u = Matrix{T}((U_final' * U_final) / T(T_eff))
        Sigma_u = Matrix{T}((Sigma_u + Sigma_u') / T(2))
        Y_u = vcat(zeros(T, p_max_use, q), U_final)
        B_u = zeros(T, 1 + q * p_max_use, q)
        u_varnames = ["Dynamic shock $i" for i in 1:q]
        u_var = VARModel(Y_u, p_max_use, B_u, U_final, Sigma_u, zero(T), zero(T), zero(T), u_varnames)
        H_px, _, _ = _sdfm_proxy_H(
            Matrix{T}(B_final[1, :, :]), Matrix{T}(I, q, q), u_var, instrument, normalize, vn)
        return Matrix{T}(H_px)
    end
end

"""Extend level IRF array to arbitrary `horizon` using `long_run_impact` beyond `irf_horizon + 1`."""
function _gdfmi1_irf_values(m::CointegratedGDFM{T}, horizon::Int) where {T<:AbstractFloat}
    H_stored, N, q = size(m.irf_matrices)
    horizon <= H_stored && return copy(m.irf_matrices[1:horizon, :, :])
    vals = Array{T,3}(undef, horizon, N, q)
    vals[1:H_stored, :, :] .= m.irf_matrices
    @inbounds for h in (H_stored + 1):horizon
        vals[h, :, :] .= m.long_run_impact
    end
    vals
end

# =============================================================================
# Analysis, Forecasting & StatsAPI Dispatch
# =============================================================================

"""
    irf(m::CointegratedGDFM, horizon=m.irf_horizon + 1) -> ImpulseResponse

Extract level structural impulse response functions from an estimated
[`CointegratedGDFM`](@ref). Row `h + 1` contains the level response at horizon `h`
(`h = 0` is impact).
"""
function irf(m::CointegratedGDFM{T}, horizon::Int=m.irf_horizon + 1) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
    N = nvars(m)
    q = m.q
    vals = _gdfmi1_irf_values(m, horizon)
    ImpulseResponse{T}(vals, zeros(T, horizon, N, q), zeros(T, horizon, N, q),
        horizon, copy(m.varnames), copy(m.shock_names), :none, nothing, zero(T))
end

"""
    fevd(m::CointegratedGDFM, horizon=m.irf_horizon + 1; include_idiosyncratic=false) -> FEVD

Compute panel-wide forecast error variance decomposition from the level structural IRFs
of a [`CointegratedGDFM`](@ref). Optionally appends an `"Idiosyncratic"` column.
"""
function fevd(
    m::CointegratedGDFM{T},
    horizon::Int=m.irf_horizon + 1;
    include_idiosyncratic::Bool=false,
) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
    N = nvars(m)
    q = m.q
    vals = _gdfmi1_irf_values(m, horizon)
    decomp_c, props_c = _compute_fevd_rect(vals, N, q, horizon)
    if !include_idiosyncratic
        return FEVD{T}(decomp_c, props_c, copy(m.varnames), copy(m.shock_names))
    end
    idio_var = vec(var(m.idiosyncratic; dims=1))
    n_s = q + 1
    decomp = zeros(T, N, n_s, horizon)
    props = zeros(T, N, n_s, horizon)
    @inbounds for h in 1:horizon, i in 1:N
        decomp[i, 1:q, h] .= @view(decomp_c[i, :, h])
        decomp[i, n_s, h] = idio_var[i]
        tot = sum(@view(decomp[i, :, h]))
        tot > zero(T) && (props[i, :, h] .= @view(decomp[i, :, h]) ./ tot)
    end
    FEVD{T}(decomp, props, copy(m.varnames), vcat(m.shock_names, ["Idiosyncratic"]))
end

"""
    historical_decomposition(m::CointegratedGDFM, horizon=effective_nobs(m); include_idiosyncratic=true) -> HistoricalDecomposition

Decompose observed panel levels into structural common shock contributions, the
idiosyncratic remainder (when `include_idiosyncratic=true`), and initial conditions
plus deterministic trends.
"""
function historical_decomposition(
    m::CointegratedGDFM{T},
    horizon::Int=effective_nobs(m);
    include_idiosyncratic::Bool=true,
) where {T<:AbstractFloat}
    T_eff = effective_nobs(m)
    T_obs = nobs(m)
    offset = T_obs - T_eff
    horizon = min(horizon, T_eff)
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))

    N = nvars(m)
    q = m.q
    shocks_h = m.shocks[1:horizon, :]
    C_irf = _gdfmi1_irf_values(m, horizon)
    Theta_p = [Matrix{T}(C_irf[s, :, :]) for s in 1:horizon]
    contrib_c = _compute_hd_contributions(shocks_h, Theta_p)
    actual = m.X[(offset + 1):(offset + horizon), :]
    sn = copy(m.shock_names)
    if include_idiosyncratic
        n_s = q + 1
        contrib = zeros(T, horizon, N, n_s)
        contrib[:, :, 1:q] .= contrib_c
        contrib[:, :, n_s] .= @view(m.idiosyncratic[(offset + 1):(offset + horizon), :])
        push!(sn, "Idiosyncratic")
    else
        contrib = contrib_c
    end
    init = actual - dropdims(sum(contrib; dims=3); dims=3)
    HistoricalDecomposition{T}(contrib, init, actual, shocks_h, horizon,
        copy(m.varnames), sn, m.identification)
end

"""
    forecast(m::CointegratedGDFM, h; ci_method=:none, reps=200, conf_level=DEFAULT_CI_MACRO, rng=Random.default_rng(), seed=nothing) -> FactorForecast

Forecast panel variables `h` steps ahead in levels from a [`CointegratedGDFM`](@ref),
preserving long-run cointegrating relationships by iterating differenced level MA
revisions from the terminal common level ``\\hat{\\chi}_T`` and extrapolating the
deterministic trend (default 68% CIs).
"""
function forecast(
    m::CointegratedGDFM{T},
    h::Int;
    ci_method::Symbol=:none,
    reps::Int=200,
    conf_level::Real=DEFAULT_CI_MACRO,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
) where {T<:AbstractFloat}
    h >= 1 || throw(ArgumentError("forecast horizon h must be >= 1, got $h"))
    _validate_conf_level(conf_level)
    ci_method in (:none, :theoretical, :bootstrap) || throw(ArgumentError(
        "ci_method must be :none, :theoretical, or :bootstrap, got :$ci_method"))
    rng = _resolve_repro_rng(rng, seed)

    T_obs = nobs(m)
    N = nvars(m)
    q = m.q
    T_eff = effective_nobs(m)
    H_stored = size(m.irf_matrices, 1)

    # Terminal deterministic trend & detrended common level at t = T_obs
    tau_T = vec(m.trend_coef[1, :]) .+ T(T_obs) .* vec(m.trend_coef[2, :])
    chi_T = vec(m.common_levels[end, :]) .- tau_T

    # Differenced level MA coefficients ΔC_s = C_s - C_{s-1} (decay to zero for all shocks)
    dC = Array{T,3}(undef, H_stored, N, q)
    dC[1, :, :] .= @view(m.irf_matrices[1, :, :])
    @inbounds for s in 2:H_stored
        dC[s, :, :] .= @view(m.irf_matrices[s, :, :]) .- @view(m.irf_matrices[s - 1, :, :])
    end

    F_fc = zeros(T, h, q)
    X_fc = Matrix{T}(undef, h, N)
    chi_cur = copy(chi_T)
    @inbounds for step in 1:h
        dchi = zeros(T, N)
        for s in step:(H_stored - 1)
            t_src = T_eff + step - s
            t_src < 1 && break
            mul!(dchi, @view(dC[s + 1, :, :]), @view(m.shocks[t_src, :]), one(T), one(T))
        end
        chi_cur .+= dchi
        tau_future = vec(m.trend_coef[1, :]) .+ T(T_obs + step) .* vec(m.trend_coef[2, :])
        X_fc[step, :] .= chi_cur .+ tau_future
    end

    F_lo = zeros(T, h, q)
    F_hi = zeros(T, h, q)
    X_lo = zeros(T, h, N)
    X_hi = zeros(T, h, N)
    F_se = zeros(T, h, q)
    X_se = zeros(T, h, N)
    conf_T = T(conf_level)

    if ci_method === :none
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :none)
    end

    idio_var = max.(vec(var(m.idiosyncratic; dims=1)), zero(T))
    C_full = _gdfmi1_irf_values(m, h)

    if ci_method === :theoretical
        z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
        cum_var = zeros(T, N)
        @inbounds for step in 1:h
            C_s = @view(C_full[step, :, :])
            for i in 1:N
                cum_var[i] += sum(abs2, @view(C_s[i, :]))
                X_se[step, i] = sqrt(max(cum_var[i] + idio_var[i], zero(T)))
            end
            F_se[step, :] .= one(T)
        end
        F_lo .= F_fc .- z_val .* F_se
        F_hi .= F_fc .+ z_val .* F_se
        X_lo .= X_fc .- z_val .* X_se
        X_hi .= X_fc .+ z_val .* X_se
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :theoretical)
    end

    # :bootstrap — resample structural shocks and idiosyncratic residuals
    reps >= 1 || throw(ArgumentError("reps must be >= 1, got $reps"))
    shocks_c = m.shocks .- mean(m.shocks; dims=1)
    idio_c = m.idiosyncratic .- mean(m.idiosyncratic; dims=1)
    F_boot = zeros(T, reps, h, q)
    X_boot = zeros(T, reps, h, N)
    eta_path = Matrix{T}(undef, h, q)
    @inbounds for b in 1:reps
        for step in 1:h
            idx_s = rand(rng, 1:T_eff)
            eta_path[step, :] .= @view(shocks_c[idx_s, :])
            F_boot[b, step, :] .= @view(eta_path[step, :])
            δX = copy(@view(idio_c[rand(rng, 1:T_obs), :]))
            for s in 0:(step - 1)
                mul!(δX, @view(C_full[s + 1, :, :]), @view(eta_path[step - s, :]), one(T), one(T))
            end
            X_boot[b, step, :] .= @view(X_fc[step, :]) .+ δX
        end
    end
    alpha = (one(T) - conf_T) / T(2)
    @inbounds for hi in 1:h, j in 1:q
        d = @view F_boot[:, hi, j]
        F_lo[hi, j] = quantile(d, alpha)
        F_hi[hi, j] = quantile(d, one(T) - alpha)
        F_se[hi, j] = std(d)
    end
    @inbounds for hi in 1:h, j in 1:N
        d = @view X_boot[:, hi, j]
        X_lo[hi, j] = quantile(d, alpha)
        X_hi[hi, j] = quantile(d, one(T) - alpha)
        X_se[hi, j] = std(d)
    end
    _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :bootstrap)
end

# Accessors & StatsAPI
nvars(m::CointegratedGDFM) = size(m.X, 2)
varnames(m::CointegratedGDFM) = m.varnames
effective_nobs(m::CointegratedGDFM) = size(m.shocks, 1)
StatsAPI.predict(m::CointegratedGDFM) = m.common_levels
StatsAPI.residuals(m::CointegratedGDFM) = m.idiosyncratic
StatsAPI.nobs(m::CointegratedGDFM) = size(m.X, 1)

function StatsAPI.dof(m::CointegratedGDFM)
    N = nvars(m)
    T_eff = effective_nobs(m)
    n_blocks = size(m.p_blocks, 2)
    avg_nb = N / max(n_blocks, 1)
    var_params = round(Int, mean(sum(m.p_blocks; dims=2)) * avg_nb^2)
    var_params + N * m.q + T_eff * m.q
end

function StatsAPI.r2(m::CointegratedGDFM{T}) where {T}
    N = nvars(m)
    [one(T) - var(@view(m.idiosyncratic[:, i])) / max(var(@view(m.X[:, i])), T(1e-10)) for i in 1:N]
end

"""Fraction of each variable's level variance explained by the common component."""
function common_variance_share(m::CointegratedGDFM{T}) where {T}
    N = nvars(m)
    [var(@view(m.common_levels[:, i])) / max(var(@view(m.X[:, i])), T(1e-10)) for i in 1:N]
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::CointegratedGDFM{T}) where {T}
    T_obs, N = size(m.X)
    T_eff = effective_nobs(m)
    mean_p = round(mean(m.p_blocks); digits=2)
    max_p = maximum(m.p_blocks)
    spec = Any[
        "Dynamic shocks (q)"     m.q;
        "Permanent trends"       m.n_permanent;
        "Transitory shocks"      m.q - m.n_permanent;
        "Variables (N)"          N;
        "Observations (T)"       T_obs;
        "Effective obs"          T_eff;
        "Permutations"           m.n_perm;
        "Block lag (mean/max)"   "$mean_p / $max_p";
        "Deterministic trend"    string(m.det_trend);
        "Identification"         string(m.identification);
        "Kernel"                 string(m.kernel);
        "Bandwidth"              m.bandwidth;
        "IRF horizon"            m.irf_horizon
    ]
    _pretty_table(io, spec;
        title = "Cointegrated I(1) GDFM (q=$(m.q), permanent=$(m.n_permanent), $(m.identification))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    lr_norms = [norm(@view(m.long_run_impact[:, j])) for j in 1:m.q]
    impact_norms = [norm(@view(m.irf_matrices[1, :, j])) for j in 1:m.q]
    shock_data = Matrix{Any}(undef, m.q, 3)
    for j in 1:m.q
        shock_data[j, 1] = m.shock_names[j]
        shock_data[j, 2] = _fmt(impact_norms[j])
        shock_data[j, 3] = _fmt(lr_norms[j])
    end
    _pretty_table(io, shock_data;
        title = "Shock Norms (Impact C₀ vs Long-Run C(1))",
        column_labels = ["Shock", "‖C₀[:,j]‖", "‖C(1)[:,j]‖"],
        alignment = [:l, :r, :r],
    )
    _manifest_footer(io, m.manifest)
end
