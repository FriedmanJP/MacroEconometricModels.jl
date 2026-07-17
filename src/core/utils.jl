# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Utility functions for MacroEconometricModels.jl - numerical routines, matrix operations, helpers.
"""

using LinearAlgebra

# =============================================================================
# Input Validation
# =============================================================================

"""Validate VAR inputs: p ≥ 1, T > p + min_obs_factor, n ≥ 1."""
function validate_var_inputs(T_obs::Int, n::Int, p::Int; min_obs_factor::Int=1)
    p < 1 && throw(ArgumentError("Number of lags p must be positive, got p=$p"))
    T_obs <= p + min_obs_factor && throw(ArgumentError(
        "Not enough observations (T=$T_obs) for p=$p lags. Need T > $(p + min_obs_factor)."))
    n < 1 && throw(ArgumentError("Number of variables must be positive, got n=$n"))
end

"""Validate factor model inputs: 1 ≤ r ≤ min(T, N)."""
function validate_factor_inputs(T_obs::Int, N::Int, r::Int; context::String="factors")
    r < 1 && throw(ArgumentError("Number of $context r must be at least 1, got r=$r"))
    r > min(T_obs, N) && throw(ArgumentError(
        "Number of $context r must be at most min(T, N) = $(min(T_obs, N)), got r=$r"))
end

"""Validate dynamic factor model inputs."""
function validate_dynamic_factor_inputs(T_obs::Int, N::Int, r::Int, p::Int)
    validate_factor_inputs(T_obs, N, r)
    p < 1 && throw(ArgumentError("Number of lags p must be at least 1, got p=$p"))
    p >= T_obs - r && throw(ArgumentError(
        "Number of lags p must be less than T - r = $(T_obs - r), got p=$p"))
end

"""Validate value > 0."""
validate_positive(value::Real, name::String) =
    value <= 0 && throw(ArgumentError("$name must be positive, got $value"))

"""Validate lo ≤ value ≤ hi."""
validate_in_range(value::Real, name::String, lo::Real, hi::Real) =
    (value < lo || value > hi) && throw(ArgumentError("$name must be in [$lo, $hi], got $value"))

"""Validate value ≥ 0."""
validate_nonnegative(value::Real, name::String) =
    value < 0 && throw(ArgumentError("$name must be non-negative, got $value"))

"""Validate lag order: p ≥ 1."""
validate_lags(p::Int; name::String="p") =
    p < 1 && throw(ArgumentError("Number of lags $name must be ≥ 1, got $p"))

"""Validate horizon: h ≥ min_val (default 1)."""
validate_horizon(h::Int; min_val::Int=1) =
    h < min_val && throw(ArgumentError("Horizon must be ≥ $min_val, got $h"))

"""Validate symbol is in valid_options."""
validate_option(value::Symbol, name::String, valid_options::Tuple) =
    value ∉ valid_options && throw(ArgumentError("$name must be one of $valid_options, got :$value"))

"""Validate data contains no NaN or Inf values."""
function _validate_data(Y::AbstractMatrix, name::String="data")
    nan_count = count(isnan, Y)
    inf_count = count(isinf, Y)
    if nan_count > 0 || inf_count > 0
        nan_rows = isempty(findall(isnan, Y)) ? Int[] : unique(getindex.(findall(isnan, Y), 1))
        inf_rows = isempty(findall(isinf, Y)) ? Int[] : unique(getindex.(findall(isinf, Y), 1))
        bad_rows = sort(unique(vcat(nan_rows, inf_rows)))
        row_info = length(bad_rows) <= 5 ? " in rows $(bad_rows)" : " in $(length(bad_rows)) rows"
        throw(ArgumentError("$name contains $nan_count NaN and $inf_count Inf values$row_info. Use `fix()` or remove missing data before estimation."))
    end
end

function _validate_data(y::AbstractVector, name::String="data")
    nan_count = count(isnan, y)
    inf_count = count(isinf, y)
    if nan_count > 0 || inf_count > 0
        throw(ArgumentError(
            "$name contains $nan_count NaN and $inf_count Inf values. Use `fix()` or remove missing data before estimation."))
    end
    return nothing
end

# =============================================================================
# Type Conversion Macro
# =============================================================================

"""
    @float_fallback func_name arg_name

Generate fallback method converting AbstractMatrix to Float64.
Usage: `@float_fallback estimate_var Y`
"""
macro float_fallback(func_name, arg_name)
    quote
        function $(esc(func_name))($(esc(arg_name))::AbstractMatrix, args...; kwargs...)
            $(esc(func_name))(Float64.($(esc(arg_name))), args...; kwargs...)
        end
    end
end

# =============================================================================
# Warning Suppression / structured logging
# =============================================================================
# `_MinLevelLogger`, `_suppress_warnings`, `with_min_level`, and `set_log_level`
# live in `core/logging.jl` (included immediately after this file). See T249/#348.

# =============================================================================
# Matrix Utilities
# =============================================================================

"""Compute inverse with fallback to pseudo-inverse for singular OR near-singular matrices.

The near-singular guard uses the 1-norm condition estimate `κ = ‖A‖₁‖A⁻¹‖₁` (reusing the
already-computed inverse, O(n²)) and switches to `pinv` when `κ > 1/rcond_tol`, catching
matrices that `inv` factors "successfully" into garbage. Pass `rcond_tol=0` to disable the
guard (hot loops), `silent=true` to suppress the warning. The caught exception set is
narrowed to genuine singularity types (no bare `ErrorException`)."""
function robust_inv(A::AbstractMatrix{T}; silent::Bool=false,
                    rcond_tol::T=sqrt(eps(T))) where {T<:AbstractFloat}
    try
        Ai = inv(A)
        # Near-singularity guard: reuse Ai for a cheap 1-norm condition estimate.
        kappa = opnorm(A, 1) * opnorm(Ai, 1)
        if !isfinite(kappa) || kappa > one(T) / rcond_tol
            silent || @warn "Matrix near-singular (1-norm cond ≈ $(kappa)). Using pseudo-inverse." maxlog=3
            return pinv(A; rtol=rcond_tol)
        end
        return Ai   # preserves the inverse's type (e.g. inv(Hermitian)::Hermitian)
    catch e
        if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException
            silent || @warn "Matrix singular. Using pseudo-inverse." maxlog=3
            return pinv(A; rtol=rcond_tol)
        else
            rethrow(e)
        end
    end
end
robust_inv(A::AbstractMatrix; kwargs...) = robust_inv(float.(A); kwargs...)

"""
    safe_cholesky_jitter(A; rel_jitter=1e-10, silent=false) -> (L, applied_jitter)

Lower Cholesky factor of `A` with **scale-relative** jitter. If `cholesky(Hermitian(A))`
fails, adds `applied·I` where the base jitter is `rel_jitter · tr(A)/n` — proportional to
the matrix's magnitude rather than a fixed absolute constant — escalating by ×10 until it
factors. Returns the factor together with the actual jitter applied (`0` when none was
needed). For a degenerate (non-positive mean-diagonal) input, falls back to an absolute
`rel_jitter` floor. Pass `silent=true` to suppress the warning."""
function safe_cholesky_jitter(A::AbstractMatrix{T}; rel_jitter::T=T(1e-10),
                              silent::Bool=false) where {T<:AbstractFloat}
    try
        return cholesky(Hermitian(A)).L, zero(T)
    catch
        n = size(A, 1)
        s = tr(A) / n                                   # mean diagonal = matrix scale
        base = s > zero(T) ? rel_jitter * s : rel_jitter
        s_ref = s > zero(T) ? s : one(T)
        # Escalate up to 1e5·base so the worst-case ceiling matches the old absolute path
        # (old: 1e-8·[1..1000] = 1e-5 for a scale-1 matrix; new: 1e-10·[1..1e5] = 1e-5).
        for scale in (1, 10, 100, 1_000, 10_000, 100_000)
            applied = T(scale) * base
            try
                L = cholesky(Hermitian(A + applied * I)).L
                silent || @warn "Covariance matrix required jitter (absolute=$(applied), relative=$(applied / s_ref), mean diagonal=$(s)) for Cholesky decomposition. Results may be affected by near-collinearity." maxlog=3
                return L, applied
            catch
                continue
            end
        end
        error("Failed to compute Cholesky decomposition even with scale-relative regularization")
    end
end

"""Cholesky decomposition with automatic scale-relative jitter for numerical stability.
Returns the lower factor `L`; the `jitter` kwarg is the relative coefficient (∝ tr(A)/n).
Use [`safe_cholesky_jitter`](@ref) to also obtain the applied jitter. Pass `silent=true`
to suppress the warning."""
function safe_cholesky(A::AbstractMatrix{T}; jitter::T=T(1e-10), silent::Bool=false) where {T<:AbstractFloat}
    L, _ = safe_cholesky_jitter(A; rel_jitter=jitter, silent=silent)
    return L
end

"""Log determinant with eigenvalue fallback for numerical issues."""
function logdet_safe(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    try
        logdet(A)
    catch
        @warn "logdet_safe: matrix not positive-definite; returning pseudo-logdet (sum of logs of positive eigenvalues only); downstream likelihood values are approximate." maxlog=3
        eigenvals = eigvals(Hermitian(A))
        pos = filter(x -> x > zero(T), eigenvals)
        isempty(pos) ? T(-Inf) : sum(log, pos)
    end
end

# =============================================================================
# VAR Matrix Construction
# =============================================================================

"""
Construct VAR design matrices: Y_eff = X * B + U.
Returns (Y_eff, X) where X = [1, Y_{t-1}, ..., Y_{t-p}].
"""
function construct_var_matrices(Y::AbstractMatrix{T}, p::Int) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    T_obs <= p && throw(ArgumentError("Not enough observations (T=$T_obs) for p=$p lags"))

    T_eff = T_obs - p
    Y_eff = Y[(p+1):end, :]
    X = Matrix{T}(undef, T_eff, 1 + n*p)
    @views X[:, 1] .= one(T)

    @inbounds for lag in 1:p
        cols = (2 + (lag-1)*n):(1 + lag*n)
        rows = (p+1-lag):(T_obs-lag)
        @views X[:, cols] .= Y[rows, :]
    end
    Y_eff, X
end
construct_var_matrices(Y::AbstractMatrix, p::Int) = construct_var_matrices(float.(Y), p)

"""Extract AR coefficient matrices [A₁, ..., Aₚ] from stacked B matrix."""
function extract_ar_coefficients(B::AbstractMatrix{T}, n::Int, p::Int) where {T}
    [Matrix{T}(B[(2 + (i-1)*n):(1 + i*n), :]') for i in 1:p]
end

"""Construct companion matrix F for VAR(p) → VAR(1) representation."""
function companion_matrix(B::AbstractMatrix{T}, n::Int, p::Int) where {T<:AbstractFloat}
    np = n * p
    F = zeros(T, np, np)
    A_coeffs = extract_ar_coefficients(B, n, p)

    @inbounds for i in 1:p
        F[1:n, ((i-1)*n+1):(i*n)] .= A_coeffs[i]
    end
    if p > 1
        @inbounds for i in 1:(p-1)
            F[(i*n+1):((i+1)*n), ((i-1)*n+1):(i*n)] .= I(n)
        end
    end
    F
end

# =============================================================================
# Statistical Utilities
# =============================================================================

"""AR(1) residual standard deviation for Minnesota prior scaling."""
function univariate_ar_variance(y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && return std(y)

    y_lag, y_curr = @view(y[1:end-1]), @view(y[2:end])
    X = hcat(ones(T, n-1), y_lag)
    std(y_curr - X * (X \ y_curr); corrected=true)
end

# =============================================================================
# Compound Validation Helpers
# =============================================================================

"""Resolve variable/shock names to indices, throwing on invalid names."""
function _validate_var_shock_indices(var::String, shock::String,
                                     variables::Vector{String}, shocks::Vector{String})
    vi = findfirst(==(var), variables)
    si = findfirst(==(shock), shocks)
    isnothing(vi) && throw(ArgumentError("Variable '$var' not found"))
    isnothing(si) && throw(ArgumentError("Shock '$shock' not found"))
    (vi, si)
end

"""Validate that narrative method has required data matrix."""
function _validate_narrative_data(method::Symbol, data::AbstractMatrix)
    method == :narrative && isempty(data) &&
        throw(ArgumentError("Narrative method requires data matrix"))
end

# =============================================================================
# Name Generation
# =============================================================================

"""Generate default names: ["prefix 1", "prefix 2", ...]"""
_default_names(n::Int, prefix::String) = ["$prefix $i" for i in 1:n]
default_var_names(n::Int; prefix::String="Var") = _default_names(n, prefix)
default_shock_names(n::Int; prefix::String="Shock") = _default_names(n, prefix)
