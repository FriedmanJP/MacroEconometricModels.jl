# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Quantile Vector Autoregression (QVAR) and Quantile Impulse Response Functions
following Chavleishvili & Manganelli (2024).

Models each equation `i` in a recursive structural ordering at quantile `τ` as
a linear function of contemporaneously ordered predecessors `y_{ordering[1:(i-1)], t}`
and `p` lags of all variables, estimated equation-by-equation via interior-point
quantile regression (`_qreg_ipm!`).

References:
- Chavleishvili, S., & Manganelli, S. (2024). Forecasting and stress testing with
  quantile vector autoregression. *Journal of Applied Econometrics*, 39(1), 66-85.
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
"""

# =============================================================================
# Model & IRF Types
# =============================================================================

"""
    QuantileVARModel{T<:AbstractFloat} <: AbstractVARModel

Recursive structural Quantile VAR(p) model (Chavleishvili & Manganelli 2024)
estimated across quantile levels `taus`:

```math
A_0(\\tau) \\, y_t = c(\\tau) + \\sum_{\\ell=1}^{p} A_{\\ell}(\\tau) \\, y_{t-\\ell} + u_t(\\tau)
```

where:
- ``y_t`` is the ``n \\times 1`` vector of endogenous variables at time ``t``
- ``\\tau \\in (0, 1)`` is the quantile level
- ``A_0(\\tau)`` is the ``n \\times n`` unit-diagonal recursive structural impact matrix in `ordering` coordinates
- ``c(\\tau)`` is the ``n \\times 1`` structural intercept vector
- ``A_{\\ell}(\\tau)`` are the ``n \\times n`` structural lag coefficient matrices for ``\\ell = 1, \\dots, p``
- ``u_t(\\tau)`` is the ``n \\times 1`` structural quantile residual vector satisfying ``Q_\\tau(u_{i,t}(\\tau) \\mid \\mathcal{F}_{i,t}) = 0``

# Fields
- `A0_list::Vector{Matrix{T}}` — per-quantile ``n \\times n`` structural impact matrices (in original variable coordinates)
- `B_list::Vector{Matrix{T}}` — per-quantile ``(1 + n p) \\times n`` structural intercept and lag coefficient matrices
- `resid_list::Vector{Matrix{T}}` — per-quantile ``T_{\\text{eff}} \\times n`` structural quantile residual matrices
- `se_list::Vector{Matrix{T}}` — per-quantile ``(1 + n p) \\times n`` standard error matrices for `B_list`
- `taus::Vector{T}` — quantile levels strictly in ``(0, 1)``
- `ordering::Vector{Int}` — recursive Cholesky variable ordering permutation of ``1:n``
- `Y::Matrix{T}` — original ``T_{\\text{obs}} \\times n`` data matrix
- `p::Int` — VAR lag order
- `varnames::Vector{String}` — variable names of length ``n``
- `manifest::Union{ReproManifest,Nothing}` — reproducibility metadata
"""
struct QuantileVARModel{T<:AbstractFloat} <: AbstractVARModel
    A0_list::Vector{Matrix{T}}
    B_list::Vector{Matrix{T}}
    resid_list::Vector{Matrix{T}}
    se_list::Vector{Matrix{T}}
    taus::Vector{T}
    ordering::Vector{Int}
    Y::Matrix{T}
    p::Int
    varnames::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

QuantileVARModel{T}(A0_list, B_list, resid_list, se_list, taus, ordering, Y, p, varnames;
                    manifest=nothing) where {T<:AbstractFloat} =
    QuantileVARModel{T}(A0_list, B_list, resid_list, se_list, taus, ordering, Y, p, varnames, manifest)

"""
    QuantileImpulseResponse{T<:AbstractFloat} <: AbstractImpulseResponse

Impulse response functions from a Quantile VAR or Quantile FAVAR model across
horizons ``h = 0, \\dots, H - 1`` (stored at indices ``1:H``).

# Fields
- `values::Array{T,3}` — ``H \\times n_{\\text{vars}} \\times n_{\\text{shocks}}`` point responses (`values[1, :, :]` is impact ``h = 0``)
- `ci_lower::Array{T,3}` — ``H \\times n_{\\text{vars}} \\times n_{\\text{shocks}}`` lower interval bounds (zeros when `ci_type == :none`)
- `ci_upper::Array{T,3}` — ``H \\times n_{\\text{vars}} \\times n_{\\text{shocks}}`` upper interval bounds (zeros when `ci_type == :none`)
- `horizon::Int` — number of horizons ``H``
- `tau_profile::Vector{T}` — horizon-by-horizon quantile levels of length ``H``
- `baseline_tau::T` — baseline reference quantile level
- `shock_size::T` — structural shock magnitude
- `method::Symbol` — propagation method (`:analytical`, `:simulation`, or `:vb`)
- `ci_type::Symbol` — uncertainty interval type (`:none`, `:bootstrap`, or `:credible`)
- `conf_level::T` — confidence or credible interval level
- `variables::Vector{String}` — response variable names
- `shocks::Vector{String}` — shock names
"""
struct QuantileImpulseResponse{T<:AbstractFloat} <: AbstractImpulseResponse
    values::Array{T,3}
    ci_lower::Array{T,3}
    ci_upper::Array{T,3}
    horizon::Int
    tau_profile::Vector{T}
    baseline_tau::T
    shock_size::T
    method::Symbol
    ci_type::Symbol
    conf_level::T
    variables::Vector{String}
    shocks::Vector{String}
end

# =============================================================================
# Internal QVAR Estimation & Interpolation Helpers
# =============================================================================

"""Fit a single quantile `tau` of recursive structural QVAR given prebuilt `(Y_eff, X_lag)`."""
function _qvar_fit_single(Y_eff::Matrix{T}, X_lag::Matrix{T}, n::Int, p::Int,
                          tau::T, ordering::Vector{Int}, se::Symbol,
                          n_boot::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    T_eff = size(Y_eff, 1)
    k_lag = 1 + n * p  # intercept + p lags of all n variables

    A0 = Matrix{T}(I, n, n)
    B_struct = zeros(T, k_lag, n)
    resid_mat = Matrix{T}(undef, T_eff, n)
    se_mat = zeros(T, k_lag, n)

    for (pos, eq) in enumerate(ordering)
        n_contemp = pos - 1
        k_eq = k_lag + n_contemp
        Z_eq = Matrix{T}(undef, T_eff, k_eq)
        @views Z_eq[:, 1] .= one(T)
        if n_contemp > 0
            @inbounds for j in 1:n_contemp
                @views Z_eq[:, 1 + j] .= Y_eff[:, ordering[j]]
            end
        end
        @views Z_eq[:, (1 + n_contemp + 1):k_eq] .= X_lag[:, 2:k_lag]

        y_eq = Vector{T}(@view Y_eff[:, eq])
        theta = Vector{T}(undef, k_eq)
        ws = _QRegWorkspace{T}(T_eff, k_eq)
        _qreg_ipm!(theta, Z_eq, y_eq, tau, ws)

        B_struct[1, eq] = theta[1]
        for j in 1:n_contemp
            A0[eq, ordering[j]] = -theta[1 + j]
        end
        @views B_struct[2:k_lag, eq] .= theta[(1 + n_contemp + 1):k_eq]

        u_eq = y_eq - Z_eq * theta
        @views resid_mat[:, eq] .= u_eq

        if se === :asymptotic
            sd_u = Statistics.std(u_eq)
            if isfinite(sd_u) && sd_u > max(T(1e-10), eps(T))
                V_eq = _qreg_vcov(Z_eq, u_eq, tau, :robust)
                d_v = diag(V_eq)
                se_mat[1, eq] = sqrt(max(d_v[1], zero(T)))
                for lag_i in 2:k_lag
                    se_mat[lag_i, eq] = sqrt(max(d_v[n_contemp + lag_i], zero(T)))
                end
            end
        end
    end

    if se === :bootstrap
        ell = clamp(ceil(Int, cbrt(T_eff)), 1, T_eff)
        last_start = max(T_eff - ell + 1, 1)
        idx = Vector{Int}(undef, T_eff)
        boot_B = zeros(T, k_lag, n, n_boot)
        Yb = Matrix{T}(undef, T_eff, n)
        Xb = Matrix{T}(undef, T_eff, k_lag)
        kept = 0

        for _ in 1:n_boot
            pos_idx = 1
            while pos_idx <= T_eff
                s = rand(rng, 1:last_start)
                take = min(ell, T_eff - pos_idx + 1)
                @inbounds for j in 0:(take - 1)
                    idx[pos_idx + j] = s + j
                end
                pos_idx += take
            end
            @inbounds for col in 1:n, row in 1:T_eff
                Yb[row, col] = Y_eff[idx[row], col]
            end
            @inbounds for col in 1:k_lag, row in 1:T_eff
                Xb[row, col] = X_lag[idx[row], col]
            end
            A0_b, B_b, _, _ = _qvar_fit_single_point(Yb, Xb, n, p, tau, ordering)
            if all(isfinite, B_b) && all(isfinite, A0_b)
                kept += 1
                @views boot_B[:, :, kept] .= B_b
            end
        end
        if kept >= 2
            D = @view boot_B[:, :, 1:kept]
            @inbounds for col in 1:n, row in 1:k_lag
                se_mat[row, col] = T(Statistics.std(@view D[row, col, :]))
            end
        end
    end

    return A0, B_struct, resid_mat, se_mat
end

"""Fast point-only single-quantile QVAR fit (no SE allocation)."""
function _qvar_fit_single_point(Y_eff::Matrix{T}, X_lag::Matrix{T}, n::Int, p::Int,
                                tau::T, ordering::Vector{Int}) where {T<:AbstractFloat}
    T_eff = size(Y_eff, 1)
    k_lag = 1 + n * p
    A0 = Matrix{T}(I, n, n)
    B_struct = zeros(T, k_lag, n)
    resid_mat = Matrix{T}(undef, T_eff, n)

    for (pos, eq) in enumerate(ordering)
        n_contemp = pos - 1
        k_eq = k_lag + n_contemp
        Z_eq = Matrix{T}(undef, T_eff, k_eq)
        @views Z_eq[:, 1] .= one(T)
        if n_contemp > 0
            @inbounds for j in 1:n_contemp
                @views Z_eq[:, 1 + j] .= Y_eff[:, ordering[j]]
            end
        end
        @views Z_eq[:, (1 + n_contemp + 1):k_eq] .= X_lag[:, 2:k_lag]
        y_eq = Vector{T}(@view Y_eff[:, eq])
        theta = Vector{T}(undef, k_eq)
        ws = _QRegWorkspace{T}(T_eff, k_eq)
        _qreg_ipm!(theta, Z_eq, y_eq, tau, ws)

        B_struct[1, eq] = theta[1]
        for j in 1:n_contemp
            A0[eq, ordering[j]] = -theta[1 + j]
        end
        @views B_struct[2:k_lag, eq] .= theta[(1 + n_contemp + 1):k_eq]
        @views resid_mat[:, eq] .= y_eq .- Z_eq * theta
    end
    return A0, B_struct, resid_mat, zeros(T, k_lag, n)
end

"""Interpolate `(A0, B_struct)` at arbitrary `tau` across the model's sorted quantile grid."""
function _qvar_interp_params(A0_list::Vector{Matrix{T}}, B_list::Vector{Matrix{T}},
                             taus::Vector{T}, tau::T) where {T<:AbstractFloat}
    M = length(taus)
    M == 1 && return A0_list[1], B_list[1]
    perm = sortperm(taus)
    ts = @view taus[perm]

    tol_match = max(T(1e-8), T(10) * eps(T))
    for (k, idx) in enumerate(perm)
        if abs(taus[idx] - tau) <= tol_match
            return A0_list[idx], B_list[idx]
        end
    end

    if tau <= ts[1]
        return A0_list[perm[1]], B_list[perm[1]]
    elseif tau >= ts[M]
        return A0_list[perm[M]], B_list[perm[M]]
    end

    hi = searchsortedfirst(ts, tau)
    lo = hi - 1
    w = (tau - ts[lo]) / (ts[hi] - ts[lo])
    ilo, ihi = perm[lo], perm[hi]
    A0_interp = (one(T) - w) .* A0_list[ilo] .+ w .* A0_list[ihi]
    B_interp = (one(T) - w) .* B_list[ilo] .+ w .* B_list[ihi]
    return A0_interp, B_interp
end

"""Interpolate residual quantile shift outside `[τ_min, τ_max]` for simulation draws."""
function _qvar_draw_step_params(qvar::QuantileVARModel{T}, tau::T) where {T<:AbstractFloat}
    M = length(qvar.taus)
    perm = sortperm(qvar.taus)
    A0_u, B_u = _qvar_interp_params(qvar.A0_list, qvar.B_list, qvar.taus, tau)
    if M == 1 || tau < qvar.taus[perm[1]] || tau > qvar.taus[perm[M]]
        ref_idx = tau <= qvar.taus[perm[1]] ? perm[1] : perm[M]
        n = nvars(qvar)
        shift = Vector{T}(undef, n)
        R_ref = qvar.resid_list[ref_idx]
        for j in 1:n
            shift[j] = T(Statistics.quantile(@view(R_ref[:, j]), tau))
        end
        return A0_u, B_u, shift
    end
    return A0_u, B_u, zeros(T, nvars(qvar))
end

# =============================================================================
# Public Estimation API
# =============================================================================

"""
    estimate_qvar(Y, p, taus=[0.10, 0.50, 0.90]; ordering=1:size(Y,2),
                  se=:asymptotic, n_boot=199, varnames=nothing,
                  rng=Random.default_rng(), seed=nothing) -> QuantileVARModel

Estimate a recursive structural Quantile VAR(p) model (Chavleishvili & Manganelli 2024)
at quantile levels `taus`.

For each quantile ``\\tau \\in \\text{taus}`` and each variable ``i`` in `ordering`, solves the
check-loss regression of ``y_{\\pi[i], t}`` on an intercept, contemporaneous earlier ordered
variables ``y_{\\pi[1:(i-1)], t}``, and ``p`` lags of all ``n`` variables using interior-point
quantile regression:

```math
Q_\\tau(y_{\\pi[i], t} \\mid y_{\\pi[1:(i-1)], t}, y_{t-1}, \\dots, y_{t-p}) =
c_{\\pi[i]}(\\tau) + \\sum_{k=1}^{i-1} \\gamma_{i,k}(\\tau)\\, y_{\\pi[k], t} +
\\sum_{\\ell=1}^{p} b_{\\pi[i]}^{(\\ell)}(\\tau)' y_{t-\\ell}
```

where:
- ``\\pi`` is the recursive `ordering` permutation of ``1:n``
- ``c_{\\pi[i]}(\\tau)`` is the equation intercept at quantile ``\\tau``
- ``\\gamma_{i,k}(\\tau) = -A_0(\\tau)_{\\pi[i], \\pi[k]}`` are the contemporaneous recursive structural coefficients
- ``b_{\\pi[i]}^{(\\ell)}(\\tau)`` are the ``n \\times 1`` lag-``\\ell`` coefficient vectors at quantile ``\\tau``

# Arguments
- `Y::AbstractMatrix` — endogenous time series matrix (``T_{\\text{obs}} \\times n``)
- `p::Int` — VAR lag order (``p \\ge 1``)
- `taus::AbstractVector{<:Real} = [0.10, 0.50, 0.90]` — quantile levels strictly in ``(0, 1)``

# Keyword Arguments
- `ordering::AbstractVector{<:Integer} = 1:size(Y, 2)` — recursive Cholesky ordering permutation
- `se::Symbol = :asymptotic` — standard error method (`:asymptotic` Powell sandwich, `:bootstrap` moving-block pairs, or `:none`)
- `n_boot::Int = 199` — bootstrap replications when `se == :bootstrap`
- `varnames::Union{Nothing,AbstractVector{<:AbstractString}} = nothing` — variable names (default `"y1"`, …)
- `rng::AbstractRNG = Random.default_rng()` — random number generator for bootstrap inference
- `seed::Union{Integer,Nothing} = nothing` — optional integer seed recorded in `manifest`

# Returns
A [`QuantileVARModel`](@ref) subtyping `AbstractVARModel`.

# Example
```julia
Y = randn(200, 3)
qv = estimate_qvar(Y, 2, [0.10, 0.50, 0.90]; seed=42)
report(qv)
```

# References
- Chavleishvili, S., & Manganelli, S. (2024). Forecasting and stress testing with
  quantile vector autoregression. *Journal of Applied Econometrics*, 39(1), 66-85.
"""
function estimate_qvar(Y::AbstractMatrix{T}, p::Int,
                       taus::AbstractVector{<:Real}=T[T(0.10), T(0.50), T(0.90)];
                       ordering::AbstractVector{<:Integer}=1:size(Y, 2),
                       se::Symbol=:asymptotic,
                       n_boot::Int=199,
                       varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                       rng::AbstractRNG=Random.default_rng(),
                       seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(Y, "Y")
    T_obs, n = size(Y)
    validate_var_inputs(T_obs, n, p)
    T_eff = T_obs - p
    T_eff > n * (p + 1) || throw(ArgumentError(
        "Not enough effective observations (T_eff=$T_eff) for QVAR($p) with $n variables (need > $(n * (p + 1)))"))

    !isempty(taus) || throw(ArgumentError("taus must not be empty"))
    taus_T = Vector{T}(undef, length(taus))
    for (m, tm) in enumerate(taus)
        tm_T = T(tm)
        (isfinite(tm) && zero(T) < tm_T < one(T)) ||
            throw(ArgumentError("every tau must lie strictly in (0, 1), got $tm"))
        taus_T[m] = tm_T
    end

    ord = Int.(collect(ordering))
    length(ord) == n && sort(ord) == collect(1:n) || throw(ArgumentError(
        "ordering must be a permutation of 1:$n, got $ord"))

    validate_option(se, "se", (:asymptotic, :bootstrap, :none))
    if se === :bootstrap
        n_boot >= 2 || throw(ArgumentError("n_boot must be >= 2 when se == :bootstrap, got $n_boot"))
    else
        n_boot >= 1 || throw(ArgumentError("n_boot must be >= 1, got $n_boot"))
    end

    vn = varnames === nothing ? ["y$i" for i in 1:n] : String.(collect(varnames))
    length(vn) == n || throw(ArgumentError(
        "varnames has $(length(vn)) entries but Y has $n columns"))

    rng_use = _resolve_repro_rng(rng, seed)
    Y_mat = Matrix{T}(Y)
    Y_eff, X_lag = construct_var_matrices(Y_mat, p)

    M = length(taus_T)
    A0_list = Vector{Matrix{T}}(undef, M)
    B_list = Vector{Matrix{T}}(undef, M)
    resid_list = Vector{Matrix{T}}(undef, M)
    se_list = Vector{Matrix{T}}(undef, M)

    for m in 1:M
        A0_m, B_m, R_m, se_m = _qvar_fit_single(Y_eff, X_lag, n, p, taus_T[m], ord, se, n_boot, rng_use)
        A0_list[m] = A0_m
        B_list[m] = B_m
        resid_list[m] = R_m
        se_list[m] = se_m
    end

    mani = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "p" => p, "taus" => Float64.(taus_T), "ordering" => ord,
        "se" => String(se), "n_boot" => n_boot))

    return QuantileVARModel{T}(A0_list, B_list, resid_list, se_list, taus_T, ord, Y_mat, p, vn, mani)
end

@float_fallback estimate_qvar Y

# =============================================================================
# StatsAPI & Accessor Interface
# =============================================================================

nvars(m::QuantileVARModel) = size(m.Y, 2)
nlags(m::QuantileVARModel) = m.p
ncoefs(m::QuantileVARModel) = 1 + nvars(m) * m.p
effective_nobs(m::QuantileVARModel) = size(m.Y, 1) - m.p
varnames(m::QuantileVARModel) = m.varnames

StatsAPI.coef(m::QuantileVARModel) = length(m.B_list) == 1 ? m.B_list[1] : cat(m.B_list...; dims=3)
StatsAPI.stderror(m::QuantileVARModel) = length(m.se_list) == 1 ? m.se_list[1] : cat(m.se_list...; dims=3)
StatsAPI.residuals(m::QuantileVARModel) = length(m.resid_list) == 1 ? m.resid_list[1] : cat(m.resid_list...; dims=3)
StatsAPI.nobs(m::QuantileVARModel) = size(m.Y, 1)
StatsAPI.dof(m::QuantileVARModel) = length(m.taus) * (nvars(m) * ncoefs(m) + div(nvars(m) * (nvars(m) - 1), 2))
StatsAPI.dof_residual(m::QuantileVARModel) = max(effective_nobs(m) - (ncoefs(m) + nvars(m) - 1), 1)
StatsAPI.islinear(::QuantileVARModel) = false

"""
    StatsAPI.vcov(m::QuantileVARModel{T})

Diagonal variance-covariance matrix (when `length(m.taus) == 1`) or vector of
diagonal variance-covariance matrices (when `length(m.taus) > 1`) constructed from
the squared per-equation standard errors `m.se_list`.
"""
function StatsAPI.vcov(m::QuantileVARModel{T}) where {T}
    length(m.se_list) == 1 && return LinearAlgebra.diagm(vec(m.se_list[1]) .^ 2)
    return [LinearAlgebra.diagm(vec(se) .^ 2) for se in m.se_list]
end

"""
    StatsAPI.confint(m::QuantileVARModel{T}; level::Real=0.95)

Two-sided Student-`t` confidence intervals (`df = dof_residual(m)`) for the
vectorized coefficients of `m`, returning a `(k * n) × 2` matrix for a single
quantile or a `(k * n) × 2 × M` array across `M` quantiles.
"""
function StatsAPI.confint(m::QuantileVARModel{T}; level::Real=0.95) where {T}
    (!(level isa Bool) && isfinite(level) && zero(T) < T(level) < one(T)) ||
        throw(ArgumentError("level must lie strictly in (0, 1), got $level"))
    df_r = StatsAPI.dof_residual(m)
    crit = T(Distributions.quantile(Distributions.TDist(df_r), one(T) - (one(T) - T(level)) / 2))
    if length(m.B_list) == 1
        b = vec(m.B_list[1])
        s = vec(m.se_list[1])
        return hcat(b .- crit .* s, b .+ crit .* s)
    end
    cis = [hcat(vec(B) .- crit .* vec(se), vec(B) .+ crit .* vec(se))
           for (B, se) in zip(m.B_list, m.se_list)]
    return cat(cis...; dims=3)
end

# Result interface for QuantileImpulseResponse
point_estimate(r::QuantileImpulseResponse) = r.values
has_uncertainty(r::QuantileImpulseResponse) = r.ci_type != :none
uncertainty_bounds(r::QuantileImpulseResponse) = r.ci_type == :none ? nothing : (r.ci_lower, r.ci_upper)

# =============================================================================
# Quantile Impulse Response Functions (qirf)
# =============================================================================

"""Normalize scalar or vector `tau_profile` into a length-`H` vector of quantiles."""
function _qirf_resolve_tau_profile(tau_profile::Union{Real,AbstractVector{<:Real}},
                                   baseline_tau::Real, H::Int, ::Type{T}) where {T<:AbstractFloat}
    base_T = T(baseline_tau)
    (isfinite(baseline_tau) && zero(T) < base_T < one(T)) ||
        throw(ArgumentError("baseline_tau must lie strictly in (0, 1), got $baseline_tau"))

    if tau_profile isa Real
        tp_T = T(tau_profile)
        (isfinite(tau_profile) && zero(T) < tp_T < one(T)) ||
            throw(ArgumentError("tau_profile must lie strictly in (0, 1), got $tau_profile"))
        return fill(tp_T, H), base_T
    else
        !isempty(tau_profile) || throw(ArgumentError("tau_profile must not be empty"))
        length(tau_profile) <= H || throw(ArgumentError(
            "tau_profile length ($(length(tau_profile))) exceeds horizon H=$H"))
        out = Vector{T}(undef, H)
        for i in 1:H
            ti = i <= length(tau_profile) ? T(tau_profile[i]) : base_T
            (isfinite(ti) && zero(T) < ti < one(T)) ||
                throw(ArgumentError("every element of tau_profile must lie strictly in (0, 1)"))
            out[i] = ti
        end
        return out, base_T
    end
end

"""Analytical recursive QIRF propagation across `tau_h` (`H × n × n_shocks`)."""
function _qirf_analytical(qvar::QuantileVARModel{T}, H::Int, tau_h::Vector{T},
                          shock::Union{Int,Nothing}, shock_size::T) where {T<:AbstractFloat}
    n = nvars(qvar)
    p = qvar.p
    n_shocks = shock === nothing ? n : 1
    vals = zeros(T, H, n, n_shocks)

    A0_0, _ = _qvar_interp_params(qvar.A0_list, qvar.B_list, qvar.taus, tau_h[1])
    A0_0_inv = Matrix{T}(robust_inv(A0_0; silent=true))
    if shock === nothing
        @views vals[1, :, :] .= A0_0_inv .* shock_size
    else
        @views vals[1, :, 1] .= A0_0_inv[:, shock] .* shock_size
    end

    for h in 2:H
        A0_h, B_h = _qvar_interp_params(qvar.A0_list, qvar.B_list, qvar.taus, tau_h[h])
        A0_h_inv = Matrix{T}(robust_inv(A0_h; silent=true))
        A_struct = extract_ar_coefficients(B_h, n, p)
        acc = zeros(T, n, n_shocks)
        for lag in 1:min(p, h - 1)
            Phi_hl = A0_h_inv * A_struct[lag]
            mul!(acc, Phi_hl, @view(vals[h - lag, :, :]), one(T), one(T))
        end
        @views vals[h, :, :] .= acc
    end

    return vals
end

"""Simulate one forward path pair `(Y_shocked, Y_base)` using common quantile draws `U_draws`."""
function _qirf_simulate_paths(qvar::QuantileVARModel{T}, H::Int, shock_col::Int,
                              shock_size::T, baseline_tau::T, n_sim::Int,
                              rng::AbstractRNG) where {T<:AbstractFloat}
    n = nvars(qvar)
    p = qvar.p
    ord = qvar.ordering

    # Initialize p-lag history at the sample baseline_tau quantile
    y_init_vec = Vector{T}(undef, n)
    for j in 1:n
        y_init_vec[j] = T(Statistics.quantile(@view(qvar.Y[:, j]), baseline_tau))
    end

    # Pre-draw common random quantile levels U ~ Uniform(0, 1) for (n_sim, H, n)
    U_draws = rand(rng, T, n_sim, H, n)
    Y_base = Array{T,3}(undef, n_sim, H, n)
    Y_shock = Array{T,3}(undef, n_sim, H, n)

    hist_b = Matrix{T}(undef, p, n)
    hist_s = Matrix{T}(undef, p, n)
    yb = Vector{T}(undef, n)
    ys = Vector{T}(undef, n)

    for s in 1:n_sim
        for r_idx in 1:p, j in 1:n
            hist_b[r_idx, j] = y_init_vec[j]
            hist_s[r_idx, j] = y_init_vec[j]
        end

        for h in 1:H
            for (pos, eq) in enumerate(ord)
                u_draw = clamp(U_draws[s, h, pos], T(0.01), T(0.99))
                A0_u, B_u, shift_u = _qvar_draw_step_params(qvar, u_draw)

                val_b = B_u[1, eq] + shift_u[eq]
                val_s = B_u[1, eq] + shift_u[eq]
                for k in 1:(pos - 1)
                    prev = ord[k]
                    gamma = -A0_u[eq, prev]
                    val_b += gamma * yb[prev]
                    val_s += gamma * ys[prev]
                end
                for lag in 1:p
                    hist_row = p - lag + 1
                    row_off = 1 + (lag - 1) * n
                    @inbounds for j in 1:n
                        coef_lj = B_u[row_off + j, eq]
                        val_b += coef_lj * hist_b[hist_row, j]
                        val_s += coef_lj * hist_s[hist_row, j]
                    end
                end
                if h == 1 && eq == shock_col
                    val_s += shock_size
                end
                yb[eq] = val_b
                ys[eq] = val_s
            end

            @views Y_base[s, h, :] .= yb
            @views Y_shock[s, h, :] .= ys

            # Shift lag ring buffers
            for r_idx in 1:(p - 1)
                @views hist_b[r_idx, :] .= hist_b[r_idx + 1, :]
                @views hist_s[r_idx, :] .= hist_s[r_idx + 1, :]
            end
            @views hist_b[p, :] .= yb
            @views hist_s[p, :] .= ys
        end
    end

    return Y_shock, Y_base
end

"""Forward multi-step conditional quantile simulation QIRF (`H × n × n_shocks`)."""
function _qirf_simulation(qvar::QuantileVARModel{T}, H::Int, tau_h::Vector{T},
                          baseline_tau::T, shock::Union{Int,Nothing},
                          shock_size::T, n_sim::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    n = nvars(qvar)
    shock_list = shock === nothing ? collect(1:n) : [shock]
    n_shocks = length(shock_list)
    vals = zeros(T, H, n, n_shocks)

    for (s_idx, sh) in enumerate(shock_list)
        Y_shock, Y_base = _qirf_simulate_paths(qvar, H, sh, shock_size, baseline_tau, n_sim, rng)
        for h in 1:H, v in 1:n
            q_s = T(Statistics.quantile(@view(Y_shock[:, h, v]), tau_h[h]))
            q_b = T(Statistics.quantile(@view(Y_base[:, h, v]), tau_h[h]))
            vals[h, v, s_idx] = q_s - q_b
        end
    end
    return vals
end

"""
    qirf(qvar, H; shock=nothing, shock_size=1.0, tau_profile=0.10, baseline_tau=0.50,
         method=:analytical, ci_type=:none, n_boot=199, conf_level=0.95,
         rng=Random.default_rng(), seed=nothing) -> QuantileImpulseResponse

Compute Quantile Impulse Response Functions (QIRFs) from a [`QuantileVARModel`](@ref)
over horizons ``h = 0, \\dots, H - 1`` (stored at indices ``1:H``).

Under `method=:analytical`, propagates recursive structural impact
``A_0(\\tau_0)^{-1} \\cdot \\delta`` through the quantile-specific reduced-form lag matrices
along `tau_profile`:

```math
\\Theta_0 = A_0(\\tau_0)^{-1} D, \\qquad
\\Theta_h = \\sum_{\\ell=1}^{\\min(p, h)} A_0(\\tau_h)^{-1} A_{\\ell}(\\tau_h) \\, \\Theta_{h-\\ell}
```

where:
- ``\\tau_h`` is the quantile level at horizon index ``h`` from `tau_profile`
- ``A_0(\\tau_h)`` is the structural contemporaneous impact matrix at quantile ``\\tau_h``
- ``A_{\\ell}(\\tau_h)`` is the lag-``\\ell`` structural coefficient matrix at quantile ``\\tau_h``
- ``D`` is the diagonal shock scaling matrix proportional to `shock_size`

Under `method=:simulation`, simulates paired forward conditional quantile trajectories via
common uniform quantile draws across the recursive structural equations and evaluates
``Q_{\\tau_h}(Y_{t+h}^{\\text{shock}}) - Q_{\\tau_h}(Y_{t+h}^{\\text{base}})``.

# Arguments
- `qvar::QuantileVARModel` — fitted recursive structural Quantile VAR model
- `H::Int` — IRF horizon length (``H \\ge 1``)

# Keyword Arguments
- `shock::Union{Int,Nothing} = nothing` — specific structural shock index (`1:n`) or `nothing` for all ``n`` shocks
- `shock_size::Real = 1.0` — structural shock magnitude
- `tau_profile::Union{Real,AbstractVector{<:Real}} = 0.10` — scalar quantile or vector of horizon-specific quantiles
- `baseline_tau::Real = 0.50` — baseline quantile used for initial conditions and tail-padding `tau_profile`
- `method::Symbol = :analytical` — `:analytical` or `:simulation`
- `ci_type::Symbol = :none` — `:none` or `:bootstrap` (moving-block pairs bootstrap)
- `n_boot::Int = 199` — bootstrap replications (also controls simulation draws when `method == :simulation`)
- `conf_level::Real = 0.95` — confidence level strictly in ``(0, 1)``
- `rng::AbstractRNG = Random.default_rng()` — random number generator
- `seed::Union{Integer,Nothing} = nothing` — optional integer seed

# Returns
A [`QuantileImpulseResponse`](@ref) object.

# Example
```julia
qv = estimate_qvar(randn(200, 3), 2, [0.10, 0.50, 0.90]; seed=1)
ir = qirf(qv, 10; tau_profile=0.10)
```
"""
function qirf(qvar::QuantileVARModel{T}, H::Int;
              shock::Union{Int,Nothing}=nothing,
              shock_size::Real=1.0,
              tau_profile::Union{Real,AbstractVector{<:Real}}=0.10,
              baseline_tau::Real=0.50,
              method::Symbol=:analytical,
              ci_type::Symbol=:none,
              n_boot::Int=199,
              conf_level::Real=0.95,
              rng::AbstractRNG=Random.default_rng(),
              seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("horizon H must be >= 1, got $H"))
    n = nvars(qvar)
    p = qvar.p
    if shock !== nothing
        1 <= shock <= n || throw(ArgumentError("shock must be in 1:$n, got $shock"))
    end
    isfinite(shock_size) || throw(ArgumentError("shock_size must be finite, got $shock_size"))
    validate_option(method, "method", (:analytical, :simulation))
    validate_option(ci_type, "ci_type", (:none, :bootstrap))
    conf_T = T(conf_level)
    (isfinite(conf_level) && zero(T) < conf_T < one(T)) ||
        throw(ArgumentError("conf_level must lie strictly in (0, 1), got $conf_level"))
    if ci_type === :bootstrap
        n_boot >= 2 || throw(ArgumentError("n_boot must be >= 2 when ci_type == :bootstrap, got $n_boot"))
    else
        n_boot >= 1 || throw(ArgumentError("n_boot must be >= 1, got $n_boot"))
    end

    tau_h, base_T = _qirf_resolve_tau_profile(tau_profile, baseline_tau, H, T)
    shock_T = T(shock_size)
    rng_use = _resolve_repro_rng(rng, seed)

    n_sim = max(n_boot, 250)
    vals = if method === :analytical
        _qirf_analytical(qvar, H, tau_h, shock, shock_T)
    else
        _qirf_simulation(qvar, H, tau_h, base_T, shock, shock_T, n_sim, rng_use)
    end

    n_shocks = size(vals, 3)
    ci_lo = zeros(T, H, n, n_shocks)
    ci_hi = zeros(T, H, n, n_shocks)

    if ci_type === :bootstrap
        Y_eff, X_lag = construct_var_matrices(qvar.Y, p)
        T_eff = size(Y_eff, 1)
        k_lag = 1 + n * p
        ell = clamp(ceil(Int, cbrt(T_eff)), 1, T_eff)
        last_start = max(T_eff - ell + 1, 1)
        idx = Vector{Int}(undef, T_eff)
        Yb = Matrix{T}(undef, T_eff, n)
        Xb = Matrix{T}(undef, T_eff, k_lag)

        boot_taus = method === :analytical ? unique(vcat(tau_h, base_T)) : copy(qvar.taus)
        M_b = length(boot_taus)
        draws = Array{T,4}(undef, n_boot, H, n, n_shocks)
        kept = 0

        for _ in 1:n_boot
            pos_idx = 1
            while pos_idx <= T_eff
                s = rand(rng_use, 1:last_start)
                take = min(ell, T_eff - pos_idx + 1)
                @inbounds for j in 0:(take - 1)
                    idx[pos_idx + j] = s + j
                end
                pos_idx += take
            end
            @inbounds for col in 1:n, row in 1:T_eff
                Yb[row, col] = Y_eff[idx[row], col]
            end
            @inbounds for col in 1:k_lag, row in 1:T_eff
                Xb[row, col] = X_lag[idx[row], col]
            end

            A0_b_list = Vector{Matrix{T}}(undef, M_b)
            B_b_list = Vector{Matrix{T}}(undef, M_b)
            R_b_list = Vector{Matrix{T}}(undef, M_b)
            se_b_list = Vector{Matrix{T}}(undef, M_b)
            ok = true
            for m in 1:M_b
                A0_m, B_m, R_m, se_m = _qvar_fit_single_point(Yb, Xb, n, p, boot_taus[m], qvar.ordering)
                if !(all(isfinite, A0_m) && all(isfinite, B_m))
                    ok = false
                    break
                end
                A0_b_list[m] = A0_m
                B_b_list[m] = B_m
                R_b_list[m] = R_m
                se_b_list[m] = se_m
            end
            ok || continue

            qvar_b = QuantileVARModel{T}(A0_b_list, B_b_list, R_b_list, se_b_list,
                                         boot_taus, qvar.ordering, qvar.Y, p, qvar.varnames, nothing)
            irf_b = if method === :analytical
                _qirf_analytical(qvar_b, H, tau_h, shock, shock_T)
            else
                _qirf_simulation(qvar_b, H, tau_h, base_T, shock, shock_T, max(50, div(n_sim, 4)), rng_use)
            end
            if all(isfinite, irf_b)
                kept += 1
                @views draws[kept, :, :, :] .= irf_b
            end
        end

        if kept >= 2
            alpha = (one(T) - conf_T) / 2
            d_sub = @view draws[1:kept, :, :, :]
            @inbounds for h in 1:H, v in 1:n, s in 1:n_shocks
                dv = @view d_sub[:, h, v, s]
                ci_lo[h, v, s] = T(Statistics.quantile(dv, alpha))
                ci_hi[h, v, s] = T(Statistics.quantile(dv, one(T) - alpha))
            end
        end
    end

    snames = shock === nothing ? copy(qvar.varnames) : String[qvar.varnames[shock]]
    return QuantileImpulseResponse{T}(
        vals, ci_lo, ci_hi, H, tau_h, base_T, shock_T, method, ci_type,
        ci_type === :none ? zero(T) : conf_T, copy(qvar.varnames), snames
    )
end

# =============================================================================
# Display & Report
# =============================================================================

function Base.show(io::IO, m::QuantileVARModel{T}) where {T}
    n = nvars(m)
    T_eff = effective_nobs(m)
    k = ncoefs(m)
    ts = join([_fmt(t; digits=2) for t in m.taus], ", ")
    ord_str = join(m.varnames[m.ordering], " → ")

    spec = Any[
        "Variables"                n;
        "Lags (p)"                 m.p;
        "Observations (effective)" T_eff;
        "Parameters / eq"          k;
        "Quantiles (τ)"            ts;
        "Recursive ordering"       ord_str
    ]
    _pretty_table(io, spec;
        title = "Quantile VAR($(m.p)) Model",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    coef_names = String[_INTERCEPT_LABEL]
    for lag in 1:m.p, v in 1:n
        push!(coef_names, "$(m.varnames[v]).L$lag")
    end
    dof_r = StatsAPI.dof_residual(m)

    for (mi, tau) in enumerate(m.taus)
        _matrix_table(io, m.A0_list[mi], "Contemporaneous Impact A₀ (τ = $(_fmt(tau; digits=2)))";
            row_labels=m.varnames, col_labels=m.varnames)
        for eq in 1:n
            _coef_table(io, "Eq. $(m.varnames[eq]) (τ = $(_fmt(tau; digits=2)))",
                        coef_names, Vector{T}(m.B_list[mi][:, eq]), Vector{T}(m.se_list[mi][:, eq]);
                        dist=:t, dof_r=dof_r)
        end
    end
    _sig_legend(io)
    _manifest_footer(io, m.manifest)
    return nothing
end

"""
    report(m::QuantileVARModel; repro=false)

Print a summary of the estimated [`QuantileVARModel`](@ref), including quantile
grid, recursive ordering, contemporaneous `A₀(τ)` matrices, and per-equation lag
coefficient tables. Pass `repro=true` to append the reproducibility manifest footer.
"""
report(m::QuantileVARModel; repro::Bool=false) = report(stdout, m; repro=repro)
report(io::IO, m::QuantileVARModel; repro::Bool=false) =
    (show(IOContext(io, :repro => repro), m); nothing)

function Base.show(io::IO, r::QuantileImpulseResponse{T}) where {T}
    n_vars, n_shocks = length(r.variables), length(r.shocks)
    H = r.horizon
    tp_str = length(unique(r.tau_profile)) == 1 ? _fmt(r.tau_profile[1]; digits=2) :
             "[" * join([_fmt(t; digits=2) for t in r.tau_profile[1:min(3, H)]], ", ") * (H > 3 ? ", …]" : "]")
    ci_str = r.ci_type == :none ? "None" : "$(r.ci_type) ($(round(Int, 100 * r.conf_level))%)"

    spec = Any[
        "Variables"    n_vars;
        "Shocks"       n_shocks;
        "Horizon"      H;
        "τ profile"    tp_str;
        "Baseline τ"   _fmt(r.baseline_tau; digits=2);
        "Shock size"   _fmt(r.shock_size; digits=2);
        "Method"       string(r.method);
        "CI"           ci_str
    ]
    _pretty_table(io, spec;
        title = "Quantile Impulse Response Functions",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    hs = _select_horizons(H)
    n_show = min(n_vars, 8)
    for j in 1:n_shocks
        data = Matrix{Any}(undef, n_show, length(hs) + 1)
        for v in 1:n_show
            data[v, 1] = r.variables[v]
            for (hi, h) in enumerate(hs)
                val = r.values[h, v, j]
                if r.ci_type != :none
                    lo, up = r.ci_lower[h, v, j], r.ci_upper[h, v, j]
                    star = (lo > zero(T) || up < zero(T)) ? "*" : ""
                    data[v, hi + 1] = string(_fmt(val), star)
                else
                    data[v, hi + 1] = _fmt(val)
                end
            end
        end
        _pretty_table(io, data;
            title = "Shock: $(r.shocks[j])" * (n_vars > n_show ? " (first $n_show of $n_vars vars)" : ""),
            column_labels = vcat([""], ["h=$(h - 1)" for h in hs]),
            alignment = vcat([:l], fill(:r, length(hs))),
        )
    end
    if r.ci_type != :none
        println(io, "* interval excludes zero")
    end
    return nothing
end

"""
    report(r::QuantileImpulseResponse)

Print a summary table of the [`QuantileImpulseResponse`](@ref) across selected horizons.
"""
report(r::QuantileImpulseResponse) = report(stdout, r)
report(io::IO, r::QuantileImpulseResponse) = (show(io, r); nothing)
