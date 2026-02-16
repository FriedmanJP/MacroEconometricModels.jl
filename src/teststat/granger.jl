# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
Granger causality tests for VAR models: pairwise (univariate) and block (multivariate).

Tests whether lagged values of one variable (or group of variables) help predict another
variable in a VAR system. Uses the Wald test formulation:

- **Pairwise**: H₀: A₁[i,j] = A₂[i,j] = ... = Aₚ[i,j] = 0 (variable j does not
  Granger-cause variable i)
- **Block**: H₀: All lag coefficients from a group of cause variables to the effect
  variable are zero

Under H₀, the Wald statistic W = θ'V⁻¹θ ~ χ²(df).

# References
- Granger (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods."
- Lütkepohl (2005). New Introduction to Multiple Time Series Analysis. Chapter 3.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Result Type
# =============================================================================

"""
    GrangerCausalityResult{T} <: StatsAPI.HypothesisTest

Result from a Granger causality test in a VAR model.

# Fields
- `statistic::T`: Wald χ² statistic
- `pvalue::T`: p-value from χ²(df) distribution
- `df::Int`: Degrees of freedom (number of restrictions)
- `cause::Vector{Int}`: Indices of causing variable(s)
- `effect::Int`: Index of effect variable
- `n::Int`: Number of variables in VAR
- `p::Int`: Lag order
- `nobs::Int`: Effective number of observations
- `test_type::Symbol`: `:pairwise` or `:block`
"""
struct GrangerCausalityResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    df::Int
    cause::Vector{Int}
    effect::Int
    n::Int
    p::Int
    nobs::Int
    test_type::Symbol
end

StatsAPI.nobs(r::GrangerCausalityResult) = r.nobs
StatsAPI.dof(r::GrangerCausalityResult) = r.df

# =============================================================================
# Pairwise Granger Test
# =============================================================================

"""
    granger_test(model::VARModel, cause::Int, effect::Int) -> GrangerCausalityResult

Test whether variable `cause` Granger-causes variable `effect` in a VAR model.

H₀: A₁[effect, cause] = A₂[effect, cause] = ... = Aₚ[effect, cause] = 0
H₁: At least one lag coefficient is nonzero

Uses a Wald χ² test with df = p (number of lags).

# Arguments
- `model`: Estimated `VARModel`
- `cause`: Index of the causing variable (1-based)
- `effect`: Index of the effect variable (1-based)

# Returns
[`GrangerCausalityResult`](@ref) with Wald statistic, p-value, and test details.

# Example
```julia
Y = randn(200, 3)
m = estimate_var(Y, 2)
g = granger_test(m, 1, 2)  # does variable 1 Granger-cause variable 2?
```
"""
function granger_test(model::VARModel{T}, cause::Int, effect::Int) where {T}
    n = nvars(model)
    p = model.p

    (1 <= cause <= n) || throw(ArgumentError("cause must be in 1:$n, got $cause"))
    (1 <= effect <= n) || throw(ArgumentError("effect must be in 1:$n, got $effect"))
    cause == effect && throw(ArgumentError("cause and effect must be different variables"))

    granger_test(model, [cause], effect)
end

# =============================================================================
# Block (Multivariate) Granger Test
# =============================================================================

"""
    granger_test(model::VARModel, cause::Vector{Int}, effect::Int) -> GrangerCausalityResult

Test whether a group of variables `cause` jointly Granger-cause variable `effect`.

H₀: All lag coefficients from cause variables to the effect equation are zero
H₁: At least one lag coefficient is nonzero

Uses a Wald χ² test with df = p × length(cause).

# Arguments
- `model`: Estimated `VARModel`
- `cause`: Indices of the causing variables (1-based)
- `effect`: Index of the effect variable (1-based)

# Returns
[`GrangerCausalityResult`](@ref) with Wald statistic, p-value, and test details.

# Example
```julia
Y = randn(200, 4)
m = estimate_var(Y, 2)
g = granger_test(m, [1, 2], 3)  # do variables 1 and 2 jointly Granger-cause variable 3?
```
"""
function granger_test(model::VARModel{T}, cause::Vector{Int}, effect::Int) where {T}
    n = nvars(model)
    p = model.p

    isempty(cause) && throw(ArgumentError("cause must be non-empty"))
    for j in cause
        (1 <= j <= n) || throw(ArgumentError("cause indices must be in 1:$n, got $j"))
    end
    (1 <= effect <= n) || throw(ArgumentError("effect must be in 1:$n, got $effect"))
    effect ∈ cause && throw(ArgumentError("effect variable ($effect) cannot be in the cause set"))
    length(unique(cause)) == length(cause) || throw(ArgumentError("cause indices must be unique"))

    test_type = length(cause) == 1 ? :pairwise : :block

    # Reconstruct design matrices
    Y_eff, X = construct_var_matrices(model.Y, p)
    T_eff = size(Y_eff, 1)

    # Covariance of OLS coefficients for `effect` equation:
    # Var(B[:,effect]) = σ_ee * (X'X)⁻¹
    XtX_inv = robust_inv(X'X)
    sigma_ee = model.Sigma[effect, effect]

    # B layout: row 1 = intercept, rows (2+(l-1)*n):(1+l*n) = lag l coefficients
    # For cause variable j at lag l: B row index = 1 + (l-1)*n + j
    # These are the coefficients in the `effect` column of B
    coef_indices = Int[]
    for l in 1:p
        for j in cause
            push!(coef_indices, 1 + (l - 1) * n + j)
        end
    end

    # Extract coefficient values (restrictions under H₀: these are all zero)
    theta = T[model.B[idx, effect] for idx in coef_indices]

    # Covariance of the restricted coefficients
    V = sigma_ee * XtX_inv[coef_indices, coef_indices]

    # Wald statistic: θ' V⁻¹ θ ~ χ²(df)
    df = p * length(cause)
    W = T(theta' * robust_inv(V) * theta)
    W = max(W, zero(T))

    pval = T(ccdf(Chisq(df), W))

    GrangerCausalityResult{T}(W, pval, df, sort(cause), effect, n, p, T_eff, test_type)
end

# =============================================================================
# All-Pairs Granger Test
# =============================================================================

"""
    granger_test_all(model::VARModel) -> Matrix{Union{GrangerCausalityResult, Nothing}}

Compute pairwise Granger causality tests for all variable pairs in a VAR model.

Returns an n×n matrix where entry [i,j] tests whether variable j Granger-causes
variable i. Diagonal entries are `nothing`.

# Arguments
- `model`: Estimated `VARModel`

# Returns
n×n matrix of [`GrangerCausalityResult`](@ref) (or `nothing` on diagonal).

# Example
```julia
Y = randn(200, 3)
m = estimate_var(Y, 2)
results = granger_test_all(m)
results[2, 1]  # does variable 1 Granger-cause variable 2?
```
"""
function granger_test_all(model::VARModel{T}) where {T}
    n = nvars(model)
    results = Matrix{Union{GrangerCausalityResult{T}, Nothing}}(nothing, n, n)

    # Pre-compute shared quantities
    Y_eff, X = construct_var_matrices(model.Y, model.p)
    T_eff = size(Y_eff, 1)
    XtX_inv = robust_inv(X'X)
    p = model.p

    for effect in 1:n
        sigma_ee = model.Sigma[effect, effect]
        for cause in 1:n
            cause == effect && continue

            coef_indices = Int[]
            for l in 1:p
                push!(coef_indices, 1 + (l - 1) * n + cause)
            end

            theta = T[model.B[idx, effect] for idx in coef_indices]
            V = sigma_ee * XtX_inv[coef_indices, coef_indices]

            df = p
            W = T(theta' * robust_inv(V) * theta)
            W = max(W, zero(T))
            pval = T(ccdf(Chisq(df), W))

            results[effect, cause] = GrangerCausalityResult{T}(
                W, pval, df, [cause], effect, n, p, T_eff, :pairwise
            )
        end
    end

    results
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, r::GrangerCausalityResult)
    stars = _significance_stars(r.pvalue)
    type_str = r.test_type == :pairwise ? "Pairwise" : "Block"
    cause_str = length(r.cause) == 1 ? "Variable $(r.cause[1])" : "Variables $(r.cause)"

    spec_data = Any[
        "H₀"          "$cause_str does not Granger-cause Variable $(r.effect)";
        "H₁"          "$cause_str Granger-causes Variable $(r.effect)";
        "Test type"    type_str;
        "Lag order"    r.p;
        "Variables"    r.n;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Granger Causality Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    results_data = Any[
        "Wald statistic"      string(_fmt(r.statistic), " ", stars);
        "Degrees of freedom"  r.df;
        "P-value"             _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )

    conclusion = if r.pvalue < 0.01
        "Reject H₀ at 1% level — $cause_str Granger-causes Variable $(r.effect)"
    elseif r.pvalue < 0.05
        "Reject H₀ at 5% level — $cause_str Granger-causes Variable $(r.effect)"
    elseif r.pvalue < 0.10
        "Reject H₀ at 10% level — $cause_str Granger-causes Variable $(r.effect)"
    else
        "Fail to reject H₀ — no evidence that $cause_str Granger-causes Variable $(r.effect)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

"""Display compact p-value matrix for all-pairs Granger causality results."""
function Base.show(io::IO, ::MIME"text/plain", results::Matrix{<:Union{GrangerCausalityResult, Nothing}})
    n = size(results, 1)
    n != size(results, 2) && return show(io, results)

    # Build p-value matrix with column labels
    col_labels = vcat(["Effect \\ Cause"], ["Var $j" for j in 1:n])
    data = Matrix{Any}(undef, n, n + 1)
    for i in 1:n
        data[i, 1] = "Variable $i"
        for j in 1:n
            if i == j
                data[i, j + 1] = "—"
            else
                r = results[i, j]
                if r === nothing
                    data[i, j + 1] = "—"
                else
                    pstr = _format_pvalue(r.pvalue)
                    stars = _significance_stars(r.pvalue)
                    data[i, j + 1] = isempty(stars) ? pstr : "$pstr $stars"
                end
            end
        end
    end

    _pretty_table(io, data;
        title = "Granger Causality P-values (row ← column)",
        column_labels = col_labels,
        alignment = vcat([:l], fill(:r, n)),
    )
    println(io, "Note: *** p<0.01, ** p<0.05, * p<0.10")
end
