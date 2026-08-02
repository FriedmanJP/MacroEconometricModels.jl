# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Moon-Perron (2004) panel unit root test.

Uses factor-adjusted t-statistics for testing the unit root null in panels
with cross-sectional dependence. Reports two modified t-statistics (t_a^*
and t_b^*) with bias and variance corrections.

References:
- Moon, H. R., & Perron, B. (2004). Testing for a unit root in panels with
  dynamic factors. Journal of Econometrics, 122(1), 81-126.
"""

# =============================================================================
# Moon-Perron Test
# =============================================================================

"""
    moon_perron_test(X::AbstractMatrix{T}; r=:auto) -> MoonPerronResult{T}

Moon-Perron (2004) panel unit root test with factor adjustment.

Projects out common factors from panel data, then constructs modified
pooled t-statistics with bias and variance corrections.

# Arguments
- `X`: Panel data matrix (T x N), time in rows, units in columns

# Keyword Arguments
- `r::Union{Int,Symbol}=:auto`: Number of common factors (`:auto` uses IC criteria)

# Returns
`MoonPerronResult{T}` with two modified t-statistics (t_a^*, t_b^*) and
their p-values (standard normal under H0).

# Example
```julia
X = randn(80, 15)
result = moon_perron_test(X; r=1)
result.pvalue_a  # p-value for t_a^* statistic
result.pvalue_b  # p-value for t_b^* statistic
```

# References
- Moon, H. R., & Perron, B. (2004). Journal of Econometrics, 122(1), 81-126.
"""
function moon_perron_test(X::AbstractMatrix{T};
                          r::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # Validate inputs
    T_obs < 20 && throw(ArgumentError(
        "Time dimension T=$T_obs too small; need at least 20 observations"))

    # Moon–Perron (2004): estimate factors from *first differences*, recumulate,
    # then de-factor the levels and form bias-corrected pooled t-statistics.
    dX = diff(X; dims=1)                   # (T_obs-1) × N
    Td = size(dX, 1)

    n_factors = if r === :auto
        r_max = min(10, min(Td, N) - 1)
        r_max < 1 && throw(ArgumentError(
            "Panel too small for automatic factor selection"))
        ic = ic_criteria(dX, r_max; standardize=true)
        max(1, ic.r_IC2)
    else
        r::Int
        r < 1 && throw(ArgumentError("Number of factors r must be >= 1, got r=$r"))
        r > min(Td, N) - 1 && throw(ArgumentError(
            "Number of factors r=$r too large for differenced panel of size ($Td, $N)"))
        r
    end

    # Step 1: PCA on differences → loadings; de-factor levels via Q_⊥ = I − Λ(Λ'Λ)⁻¹Λ'.
    # standardize=false (#582): Q_⊥ is applied to RAW levels below, so the loadings
    # must span the raw-scale factor space — standardized loadings project out the
    # wrong subspace when series scales differ (panic_test does the same).
    fm = estimate_factors(dX, n_factors; standardize=false)
    Lambda_hat = fm.loadings  # N × r

    LtL = Lambda_hat' * Lambda_hat
    LtL_inv = robust_inv(LtL; silent=true)
    Q_perp = Matrix{T}(I, N, N) - Lambda_hat * LtL_inv * Lambda_hat'

    # De-factored levels
    X_star = X * Q_perp'

    # Step 2: Pooled AR(1) on de-factored data + per-unit long-run variances.
    # Moon–Perron (2004) eqs (8)–(10) / feasible forms after Lemma 4:
    #   ρ̂*_pool = [tr(Z_{-1} Q Z') − N·T·λ̂_e] / tr(Z_{-1} Q Z_{-1}')
    #   t*_a = √N·T·(ρ̂*−1) / √(2 φ̂_e⁴ / ω̂_e⁴)
    #   t*_b = √N·T·(ρ̂*−1) · √(tr(Z_{-1}Q Z_{-1}')/(N T²)) · (ω̂_e / φ̂_e²)
    # Bias correction is INSIDE ρ̂* (the −N T λ̂_e term), not a post-hoc t subtractand.
    numerator_rho = zero(T)   # tr(Z_{-1} Q Z') = Σ_i Σ_t x*_{i,t-1} x*_{i,t}
    denominator_rho = zero(T) # tr(Z_{-1} Q Z_{-1}') = Σ_i Σ_t x*_{i,t-1}²

    sigma2_hat = Vector{T}(undef, N)
    omega2_hat = Vector{T}(undef, N)
    lambda_hat = Vector{T}(undef, N)   # one-sided LR cov λ_{e,i}

    for i in 1:N
        xi = X_star[:, i]
        xi_lag = xi[1:end-1]
        xi_cur = xi[2:end]
        T_eff_i = length(xi_cur)

        numerator_rho += dot(xi_lag, xi_cur)
        denominator_rho += dot(xi_lag, xi_lag)

        # Innovations under the unit-root null (for nuisance estimation)
        ui = xi_cur .- xi_lag
        sigma2_hat[i] = sum(abs2, ui) / T_eff_i
        bw = _nw_bandwidth(ui)
        omega2_hat[i] = max(_long_run_variance(ui, bw), eps(T))
        # λ_e,i = (ω² − σ²)/2  (one-sided long-run covariance)
        lambda_hat[i] = (omega2_hat[i] - sigma2_hat[i]) / 2
    end

    T_eff = T_obs - 1
    lambda_mean = mean(lambda_hat)                 # λ̂_e
    omega2_mean = max(mean(omega2_hat), eps(T))    # ω̂_e²
    # φ̂_e⁴ = (1/N) Σ ω_{e,i}⁴   (Assumption 8 / eq 17)
    phi4_mean = max(mean(ω -> ω^2, omega2_hat), eps(T))

    # Bias-modified pooled estimator (eq 8 / feasible ρ̂*)
    denom = max(denominator_rho, eps(T))
    rho_star = (numerator_rho - T(N) * T(T_eff) * lambda_mean) / denom

    # t*_a (eq after Lemma 4)
    se_a = sqrt(max(2 * phi4_mean / (omega2_mean^2), T(1e-10)))
    t_a_star = sqrt(T(N)) * T(T_eff) * (rho_star - one(T)) / se_a

    # t*_b
    # √(tr /(N T²)) · ω / φ², with φ² := √(φ⁴)
    tr_scale = sqrt(denom / (T(N) * T(T_eff)^2))
    phi2 = sqrt(phi4_mean)
    omega_bar = sqrt(omega2_mean)
    t_b_star = sqrt(T(N)) * T(T_eff) * (rho_star - one(T)) * tr_scale *
               (omega_bar / max(phi2, eps(T)))

    # P-values: left-tailed N(0,1) under H0 (reject for large negative values)
    pvalue_a = T(cdf(Normal(), t_a_star))
    pvalue_b = T(cdf(Normal(), t_b_star))

    MoonPerronResult{T}(
        t_a_star,
        t_b_star,
        pvalue_a,
        pvalue_b,
        n_factors,
        T_obs,
        N
    )
end

# Float64 fallback
moon_perron_test(X::AbstractMatrix; kwargs...) = moon_perron_test(Float64.(X); kwargs...)

# PanelData dispatch
function moon_perron_test(pd::PanelData; kwargs...)
    X = _panel_to_matrix(pd)
    moon_perron_test(X; kwargs...)
end

# =============================================================================
# Show method
# =============================================================================

function Base.show(io::IO, r::MoonPerronResult{T}) where {T}
    spec_data = Any[
        "H0"          "All units have unit roots (panel non-stationary)";
        "H1"          "Some units are stationary";
        "Factors"      r.n_factors;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Moon-Perron (2004) Panel Unit Root Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    stars_a = _significance_stars(r.pvalue_a)
    stars_b = _significance_stars(r.pvalue_b)
    results_data = Any[
        "t*_a statistic" string(round(r.t_a_statistic, digits=4), " ", stars_a) _format_pvalue(r.pvalue_a);
        "t*_b statistic" string(round(r.t_b_statistic, digits=4), " ", stars_b) _format_pvalue(r.pvalue_b)
    ]
    _pretty_table(io, results_data;
        title = "Modified t-Statistics (N(0,1) under H0)",
        column_labels = ["Statistic", "Value", "P-value"],
        alignment = [:l, :r, :r],
    )

    reject_a = r.pvalue_a < 0.05
    reject_b = r.pvalue_b < 0.05
    conclusion = if reject_a && reject_b
        "Both statistics reject H0: strong evidence against panel unit root"
    elseif reject_a || reject_b
        "One statistic rejects H0: moderate evidence against panel unit root"
    else
        "Fail to reject H0: panel appears non-stationary"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Convenience: Panel Unit Root Summary
# =============================================================================

"""
    PanelUnitRootSummary

Container for the panel unit-root test battery, returned by `panel_unit_root_summary`
so the battery can be composed, re-displayed, and inspected — not just printed once.

Holds both the **second-generation** (cross-sectional-dependence-robust) tests —
PANIC, Pesaran CIPS, Moon-Perron — and the **first-generation**
(cross-sectional-independence) tests added in EV-20 (#428): Levin-Lin-Chu, IPS,
Breitung, Fisher, and Hadri. Any sub-test that errors is recorded in `errors` and
skipped. (S3/T167) Note Hadri flips the null to *stationarity* (right-tailed).
"""
struct PanelUnitRootSummary
    panic::Union{PANICResult,Nothing}
    cips::Union{PesaranCIPSResult,Nothing}
    moon_perron::Union{MoonPerronResult,Nothing}
    llc::Union{LLCResult,Nothing}
    ips::Union{IPSResult,Nothing}
    breitung::Union{BreitungPanelResult,Nothing}
    fisher::Union{FisherPanelResult,Nothing}
    hadri::Union{HadriResult,Nothing}
    errors::Vector{String}
end

function _build_panel_unit_root_summary(X::AbstractMatrix; r::Union{Int,Symbol}=:auto,
                                        lags::Union{Int,Symbol}=:auto)
    panic = nothing; cips = nothing; mp = nothing
    llc = nothing; ips = nothing; breit = nothing; fish = nothing; had = nothing
    errors = String[]
    try
        panic = panic_test(X; r=r, method=:pooled)
    catch e
        push!(errors, "PANIC test failed: " * sprint(showerror, e))
    end
    try
        cips = pesaran_cips_test(X; lags=lags, deterministic=:constant)
    catch e
        push!(errors, "Pesaran CIPS test failed: " * sprint(showerror, e))
    end
    try
        mp = moon_perron_test(X; r=r)
    catch e
        push!(errors, "Moon-Perron test failed: " * sprint(showerror, e))
    end
    try
        llc = llc_test(X; lags=lags, deterministic=:constant)
    catch e
        push!(errors, "Levin-Lin-Chu test failed: " * sprint(showerror, e))
    end
    try
        ips = ips_test(X; lags=lags, deterministic=:constant)
    catch e
        push!(errors, "Im-Pesaran-Shin test failed: " * sprint(showerror, e))
    end
    try
        breit = breitung_panel_test(X; deterministic=:constant)
    catch e
        push!(errors, "Breitung test failed: " * sprint(showerror, e))
    end
    try
        fish = fisher_panel_test(X; base=:adf, combine=:mw, lags=lags, deterministic=:constant)
    catch e
        push!(errors, "Fisher test failed: " * sprint(showerror, e))
    end
    try
        had = hadri_test(X; deterministic=:constant)
    catch e
        push!(errors, "Hadri test failed: " * sprint(showerror, e))
    end
    return PanelUnitRootSummary(panic, cips, mp, llc, ips, breit, fish, had, errors)
end

"""
    panel_unit_root_summary(X; r=:auto, lags=:auto) -> PanelUnitRootSummary

Run the full panel unit-root battery (eight tests since EV-20): first-generation
LLC, IPS, Breitung, Fisher, Hadri; second-generation PANIC (Bai-Ng 2004),
Pesaran CIPS (2007), and Moon-Perron (2004). Returns a
[`PanelUnitRootSummary`](@ref). Pass an `io` first argument to also print the battery.

# Arguments
- `X::AbstractMatrix`: Panel data (T × N)
- `r`: Number of factors for PANIC and Moon-Perron (`:auto` for IC selection)
- `lags`: Number of lags for CIPS (`:auto` for T^{1/3} rule)

# Example
```julia
X = randn(100, 20)
s = panel_unit_root_summary(X; r=1)
```
"""
function panel_unit_root_summary(X::AbstractMatrix; r::Union{Int,Symbol}=:auto,
                                  lags::Union{Int,Symbol}=:auto)
    return _build_panel_unit_root_summary(X; r=r, lags=lags)
end

function panel_unit_root_summary(io::IO, X::AbstractMatrix; r::Union{Int,Symbol}=:auto,
                                  lags::Union{Int,Symbol}=:auto)
    s = _build_panel_unit_root_summary(X; r=r, lags=lags)
    show(io, s)
    return s
end

function Base.show(io::IO, s::PanelUnitRootSummary)
    # Borderless title (dropped the ='^60 ASCII banner for dialect consistency). (S5/T169)
    println(io, "\n  Panel Unit Root Test Battery\n")
    # First-generation (cross-sectional independence).
    s.llc !== nothing && (show(io, s.llc); println(io))
    s.ips !== nothing && (show(io, s.ips); println(io))
    s.breitung !== nothing && (show(io, s.breitung); println(io))
    s.fisher !== nothing && (show(io, s.fisher); println(io))
    s.hadri !== nothing && (show(io, s.hadri); println(io))
    # Second-generation (cross-sectional dependence robust).
    s.panic !== nothing && (show(io, s.panic); println(io))
    s.cips !== nothing && (show(io, s.cips); println(io))
    s.moon_perron !== nothing && (show(io, s.moon_perron); println(io))
    for e in s.errors
        println(io, e)
    end
    return nothing
end

# PanelData dispatch
function panel_unit_root_summary(pd::PanelData; kwargs...)
    X = _panel_to_matrix(pd)
    panel_unit_root_summary(X; kwargs...)
end
