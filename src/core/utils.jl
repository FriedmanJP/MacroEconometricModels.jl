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
# Warning Suppression
# =============================================================================

"""Suppress all log messages (warnings, info) within `f()`. Used in bootstrap loops."""
_suppress_warnings(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# =============================================================================
# Matrix Utilities
# =============================================================================

"""Compute inverse with fallback to pseudo-inverse for singular matrices."""
function robust_inv(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    try
        inv(A)
    catch e
        if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || e isa ErrorException
            @warn "Matrix singular or near-singular. Using pseudo-inverse."
            pinv(A)
        else
            rethrow(e)
        end
    end
end
robust_inv(A::AbstractMatrix) = robust_inv(float.(A))

"""Cholesky decomposition with automatic jitter for numerical stability."""
function safe_cholesky(A::AbstractMatrix{T}; jitter::T=T(1e-8)) where {T<:AbstractFloat}
    try
        return cholesky(Hermitian(A)).L
    catch
        for scale in [1, 10, 100, 1000]
            try
                return cholesky(Hermitian(A + scale * jitter * I)).L
            catch; continue; end
        end
        error("Failed to compute Cholesky decomposition even with regularization")
    end
end

"""Log determinant with eigenvalue fallback for numerical issues."""
function logdet_safe(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    try
        logdet(A)
    catch
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
