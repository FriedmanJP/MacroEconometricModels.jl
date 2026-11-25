# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Multivariate fundamentalness and informational sufficiency tests (NGID-07, #907):

1. Chen, Choi & Escanciano (2017, *Quantitative Economics*): higher-moment martingale-difference
   orthogonality test for fundamentalness with first-step estimation uncertainty adjustment.
2. Forni & Gambetti (2014, *Journal of Monetary Economics*): large-information Granger-orthogonality
   test against extracted macroeconomic factors.
"""

using LinearAlgebra, Statistics, Distributions, Random, StatsAPI

# =============================================================================
# Result Types
# =============================================================================

"""
    FundamentalnessTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest

Result from the Chen, Choi & Escanciano (2017) higher-moment test for fundamentalness
of VAR / ARMA / SSVARMA representations.

Under ``H_0`` (fundamental representation with independent structural innovations),
Wold innovations ``\\hat{u}_t`` satisfy martingale-difference orthogonality in higher
moments relative to past observables ``Y_{t-1}, \\dots, Y_{t-m}``:

```math
\\operatorname{Cov}\\bigl(\\phi(\\hat{u}_{i,t}),\\, h(Y_{j,t-k})\\bigr) = 0,
\\qquad k = 1, \\dots, m_{\\max},
```

where ``\\phi`` spans squared/cubic/heavy-tailed transformations (``\\text{:skew}``,
``\\text{:kurt}``). Multiplier bootstrap ``p``-values account for first-step VAR/AR
estimation uncertainty (Chen, Choi & Escanciano 2017, Theorem 2).

# Fields
- `statistic::T`: aggregate lag-weighted portmanteau statistic
- `pvalue::T`: wild/multiplier bootstrap ``p``-value
- `lag_statistics::Vector{T}`: per-lag higher-moment statistics (length `max_lag`)
- `lag_pvalues::Vector{T}`: per-lag multiplier bootstrap ``p``-values
- `moment_statistics::Dict{Symbol,T}`: statistics decomposed by moment category
- `moment_pvalues::Dict{Symbol,T}`: multiplier bootstrap ``p``-values by moment category
- `max_lag::Int`: maximum lag tested
- `moments::Vector{Symbol}`: higher-moment classes tested (`:skew`, `:kurt`)
- `n_boot::Int`: number of multiplier bootstrap replications
- `n::Int`: number of observable series
- `p::Int`: autoregressive order of the fitted model
- `nobs::Int`: effective sample size used in the test
- `model_type::Symbol`: model family identifier (`:var`, `:svarma`, `:ar`, `:ma`, `:arma`, `:arima`)
"""
struct FundamentalnessTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    lag_statistics::Vector{T}
    lag_pvalues::Vector{T}
    moment_statistics::Dict{Symbol,T}
    moment_pvalues::Dict{Symbol,T}
    max_lag::Int
    moments::Vector{Symbol}
    n_boot::Int
    n::Int
    p::Int
    nobs::Int
    model_type::Symbol
end

StatsAPI.nobs(r::FundamentalnessTestResult) = r.nobs
StatsAPI.dof(r::FundamentalnessTestResult) = r.max_lag * length(r.moments) * r.n * r.n
StatsAPI.pvalue(r::FundamentalnessTestResult) = r.pvalue

"""
    SufficientInformationTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest

Result from the Forni & Gambetti (2014) test for informational sufficiency (fundamentalness)
in structural VARs.

Tests whether reduced-form VAR residuals ``\\hat{u}_t`` (or identified structural shocks
``\\hat{\\varepsilon}_t``) are orthogonal to ``p_f`` lags of the top ``r`` principal
components ``\\hat{F}_t`` extracted from a large macroeconomic panel ``X_t``:

```math
\\hat{u}_t = c + \\sum_{\\ell=1}^{p_f} \\Gamma_\\ell \\hat{F}_{t-\\ell}^{(1:r)} + e_t,
\\qquad H_0\\!: \\Gamma_1 = \\dots = \\Gamma_{p_f} = 0.
```

Reports both system ``F``-test and Newey–West HAC Wald ``\\chi^2`` statistics across
``r = 1, \\dots, r_{\\max}``.

# Fields
- `statistic::T`: headline ``F``-statistic across ``r \\in 1\\dots r_{\\max}``
- `pvalue::T`: Holm-adjusted minimum ``F``-test ``p``-value
- `wald_statistic::T`: headline HAC Wald ``\\chi^2`` statistic
- `wald_pvalue::T`: Holm-adjusted minimum HAC Wald ``p``-value
- `r_max::Int`: maximum factor count evaluated
- `p_f::Int`: number of factor lags
- `n::Int`: number of VAR residuals / shocks tested
- `nobs::Int`: effective observations in orthogonality regressions
- `df::Int`: numerator degrees of freedom ``n \\cdot r_{\\text{sel}} \\cdot p_f``
- `r_grid::Vector{Int}`: grid of factor counts ``1:r_{\\max}``
- `f_statistics::Vector{T}`: ``F``-statistic at each ``r``
- `f_pvalues::Vector{T}`: ``F``-test ``p``-value at each ``r``
- `wald_statistics::Vector{T}`: HAC Wald ``\\chi^2`` statistic at each ``r``
- `wald_pvalues::Vector{T}`: HAC Wald ``p``-value at each ``r``
"""
struct SufficientInformationTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    wald_statistic::T
    wald_pvalue::T
    r_max::Int
    p_f::Int
    n::Int
    nobs::Int
    df::Int
    r_grid::Vector{Int}
    f_statistics::Vector{T}
    f_pvalues::Vector{T}
    wald_statistics::Vector{T}
    wald_pvalues::Vector{T}
end

StatsAPI.nobs(r::SufficientInformationTestResult) = r.nobs
StatsAPI.dof(r::SufficientInformationTestResult) = r.df
StatsAPI.pvalue(r::SufficientInformationTestResult) = r.pvalue

# =============================================================================
# Pretty-Print & Report Methods
# =============================================================================

function Base.show(io::IO, r::FundamentalnessTestResult{T}) where {T}
    spec = Any[
        "H₀"             "Representation is fundamental (higher-moment MDD)";
        "H₁"             "Non-fundamental / missing higher-moment information";
        "Model"          uppercase(string(r.model_type));
        "Variables / AR" string(r.n, " / ", r.p);
        "Max Lag"        r.max_lag;
        "Moments"        join(string.(r.moments), ", ");
        "Bootstrap Reps" r.n_boot;
        "Observations"   r.nobs;
    ]
    _pretty_table(io, spec;
        title = "Chen–Choi–Escanciano (2017) Fundamentalness Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    stars = _significance_stars(r.pvalue)
    res_tbl = Any[
        "Aggregate Statistic" string(_fmt(r.statistic; digits=4), " ", stars);
        "Bootstrap P-value"   _format_pvalue(r.pvalue);
    ]
    for m in r.moments
        if haskey(r.moment_statistics, m)
            mp = r.moment_pvalues[m]
            ms = _significance_stars(mp)
            res_tbl = vcat(res_tbl, Any[
                string("  Moment :", m) string(_fmt(r.moment_statistics[m]; digits=4), " ", ms,
                                               " (p = ", _format_pvalue(mp), ")")
            ])
        end
    end
    _pretty_table(io, res_tbl;
        title = "Test Results",
        column_labels = ["Statistic", "Value"],
        alignment = [:l, :r],
    )

    lag_rows = Matrix{Any}(undef, r.max_lag, 3)
    for k in 1:r.max_lag
        lag_rows[k, 1] = k
        lag_rows[k, 2] = _fmt(r.lag_statistics[k]; digits=4)
        lag_rows[k, 3] = string(_format_pvalue(r.lag_pvalues[k]), " ", _significance_stars(r.lag_pvalues[k]))
    end
    _pretty_table(io, lag_rows;
        title = "By Lag k",
        column_labels = ["Lag k", "Stat", "P-value"],
        alignment = [:r, :r, :r],
    )

    concl = r.pvalue < T(0.01) ? "Reject H₀ at 1% (evidence of non-fundamentalness)" :
            r.pvalue < T(0.05) ? "Reject H₀ at 5% (evidence of non-fundamentalness)" :
            r.pvalue < T(0.10) ? "Reject H₀ at 10% (weak evidence of non-fundamentalness)" :
                                 "Fail to reject H₀ (no evidence against fundamentalness)"
    _pretty_table(io, Any["Conclusion" concl; "Note" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels = ["", ""], alignment = [:l, :l])
end

report(r::FundamentalnessTestResult) = report(stdout, r)
report(io::IO, r::FundamentalnessTestResult) = (show(io, r); nothing)

function Base.show(io::IO, r::SufficientInformationTestResult{T}) where {T}
    spec = Any[
        "H₀"                "Residuals orthogonal to lagged panel factors (sufficient info)";
        "H₁"                "Panel factors Granger-cause residuals (informationally deficient)";
        "Series / Lag p_f"  string(r.n, " / ", r.p_f);
        "Max Factors r_max" r.r_max;
        "Observations"      r.nobs;
        "Headline F-stat"   string(_fmt(r.statistic; digits=4), " ", _significance_stars(r.pvalue));
        "Headline F p-val"  _format_pvalue(r.pvalue);
        "Headline Wald χ²"  string(_fmt(r.wald_statistic; digits=4), " ", _significance_stars(r.wald_pvalue));
        "Headline Wald p"   _format_pvalue(r.wald_pvalue);
    ]
    _pretty_table(io, spec;
        title = "Forni–Gambetti (2014) Informational Sufficiency Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    tbl = Matrix{Any}(undef, length(r.r_grid), 5)
    for (idx, rf) in enumerate(r.r_grid)
        tbl[idx, 1] = rf
        tbl[idx, 2] = _fmt(r.f_statistics[idx]; digits=4)
        tbl[idx, 3] = string(_format_pvalue(r.f_pvalues[idx]), " ", _significance_stars(r.f_pvalues[idx]))
        tbl[idx, 4] = _fmt(r.wald_statistics[idx]; digits=4)
        tbl[idx, 5] = string(_format_pvalue(r.wald_pvalues[idx]), " ", _significance_stars(r.wald_pvalues[idx]))
    end
    _pretty_table(io, tbl;
        title = "Orthogonality Tests Across Factor Counts",
        column_labels = ["r", "F-stat", "F P-value", "HAC Wald", "Wald P-value"],
        alignment = [:r, :r, :r, :r, :r],
    )

    p_min = min(r.pvalue, r.wald_pvalue)
    concl = p_min < T(0.01) ? "Reject H₀ at 1% (VAR is informationally deficient)" :
            p_min < T(0.05) ? "Reject H₀ at 5% (VAR is informationally deficient)" :
            p_min < T(0.10) ? "Reject H₀ at 10% (weak evidence of deficiency)" :
                              "Fail to reject H₀ (VAR appears informationally sufficient)"
    _pretty_table(io, Any["Conclusion" concl; "Note" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels = ["", ""], alignment = [:l, :l])
end

report(r::SufficientInformationTestResult) = report(stdout, r)
report(io::IO, r::SufficientInformationTestResult) = (show(io, r); nothing)

# =============================================================================
# CCE (2017) Higher-Moment Fundamentalness Test Helpers
# =============================================================================

function _svarma_tst_extract_inputs(model::VARModel{T}) where {T<:AbstractFloat}
    _, W_reg = construct_var_matrices(model.Y, model.p)
    u_mat = Matrix{T}(model.U)
    Y_hist = Matrix{T}(model.Y)
    return u_mat, Y_hist, model.p, W_reg, :var
end

function _svarma_tst_extract_inputs(model::SVARMAModel{T}) where {T<:AbstractFloat}
    u_mat = Matrix{T}(model.shocks)
    Y_hist = Matrix{T}(model.Y)
    p_eff = model.p
    if model.q == 0 && p_eff >= 1
        _, W_reg = construct_var_matrices(model.Y, p_eff)
    elseif model.q == 0
        W_reg = ones(T, size(u_mat, 1), 1)
    else
        W_reg = zeros(T, size(u_mat, 1), 0)
    end
    return u_mat, Y_hist, p_eff, W_reg, :svarma
end

function _svarma_tst_extract_inputs(model::AbstractARIMAModel)
    y_raw = model.y
    res_raw = model.residuals
    T_el = float(promote_type(eltype(y_raw), eltype(res_raw)))
    T_el <: AbstractFloat || (T_el = Float64)
    res_v = T_el.(res_raw)
    T_u = length(res_v)
    d_ord = hasproperty(model, :d) ? Int(getfield(model, :d)) : 0
    y_work = T_el.(y_raw)
    for _ in 1:d_ord
        y_work = diff(y_work)
    end
    length(y_work) >= T_u || throw(ArgumentError("ARIMA residual length exceeds differenced series length"))
    y_aligned = length(y_work) == T_u ? y_work : y_work[(end - T_u + 1):end]
    p_ord = hasproperty(model, :p) ? Int(getfield(model, :p)) : 0
    q_ord = hasproperty(model, :q) ? Int(getfield(model, :q)) : 0

    mtype = model isa ARModel ? :ar :
            model isa MAModel ? :ma :
            model isa ARMAModel ? :arma : :arima

    if q_ord == 0 && p_ord >= 1 && length(y_work) > p_ord
        # Align W_reg when pure AR residuals match construct_var_matrices
        T_ar = length(y_work) - p_ord
        if T_ar == T_u
            _, W_reg = construct_var_matrices(reshape(y_work, :, 1), p_ord)
            return reshape(res_v, T_u, 1), reshape(y_work, :, 1), p_ord, W_reg, mtype
        end
    end
    W_reg = zeros(T_el, T_u, 0)
    return reshape(res_v, T_u, 1), reshape(y_aligned, T_u, 1), 0, W_reg, mtype
end

@inline _svarma_tst_bsq(x::T) where {T<:AbstractFloat} = (x * x) / (one(T) + T(0.08) * x * x)
@inline _svarma_tst_dbsq(x::T) where {T<:AbstractFloat} = (T(2) * x) / (one(T) + T(0.08) * x * x)^2

function _svarma_tst_cce_run(u_mat::Matrix{T},
                             Y_hist::Matrix{T},
                             p_eff::Int,
                             W_reg::Matrix{T},
                             mtype::Symbol,
                             max_lag::Int,
                             moments::Vector{Symbol},
                             n_boot::Int,
                             rng_eff::AbstractRNG) where {T<:AbstractFloat}
    T_u, n = size(u_mat)
    start_tau = max_lag + 1
    T_c = T_u - start_tau + 1
    T_c >= max(16, size(W_reg, 2) + 4) || throw(ArgumentError(
        "Insufficient effective observations (T_eff=$T_c) for max_lag=$max_lag"))

    # Symmetric PCA whitening on residuals (exact identity for n == 1)
    Sigma_u = Matrix{T}(Symmetric((transpose(u_mat) * u_mat) / T(T_u)))
    F_u = eigen(Symmetric(Sigma_u))
    R_wh = Matrix{T}(F_u.vectors * Diagonal(T[one(T) / sqrt(max(v, sqrt(eps(T)))) for v in F_u.values]) * transpose(F_u.vectors))
    Z = u_mat * R_wh

    mu_Y = vec(mean(Y_hist; dims=1))
    sig_Y = T[max(std(@view(Y_hist[:, j])), sqrt(eps(T))) for j in 1:n]
    ys_all = Vector{T}[T[(Y_hist[t, j] - mu_Y[j]) / sig_Y[j] for t in 1:size(Y_hist, 1)] for j in 1:n]

    has_var_est = size(W_reg, 2) > 0
    if has_var_est
        S_WW = Matrix{T}(Symmetric((transpose(W_reg) * W_reg) / T(T_u)))
        S_WW_inv = Matrix{T}(robust_inv(S_WW; silent=true))
        W_sub = Matrix{T}(@view W_reg[start_tau:T_u, :])
        H_W = S_WW_inv * transpose(W_sub)
    else
        W_sub = zeros(T, T_c, 0)
        H_W = zeros(T, 0, T_c)
    end
    U_sub = Matrix{T}(@view u_mat[start_tau:T_u, :])

    # P cols: 1..n = bsq(Z_i) [:kurt], (n+1)..2n = Z_i [:skew]
    P_cols = Vector{T}[]
    dP_cols = Vector{Matrix{T}}()
    for i in 1:n
        v = T[_svarma_tst_bsq(Z[t, i]) for t in start_tau:T_u]
        push!(P_cols, v .- mean(v))
        dv = T[_svarma_tst_dbsq(Z[t, i]) for t in start_tau:T_u]
        push!(dP_cols, dv * transpose(@view(R_wh[:, i])))
    end
    for i in 1:n
        v = Vector{T}(@view Z[start_tau:T_u, i])
        push!(P_cols, v .- mean(v))
        push!(dP_cols, repeat(transpose(@view(R_wh[:, i])), T_c, 1))
    end

    H_cols = Vector{T}[]
    Adj_cols = Vector{T}[]
    pair_meta = Tuple{Int,Int,Vector{Int}}[]
    meta_lag = Int[]
    meta_mom = Symbol[]
    wk_vec = T[]
    obs_raw = T[]

    function register_pair!(k::Int, ms::Symbol, wk::T, pc::Int, hc::Int)
        hv = H_cols[hc]
        ph = P_cols[pc]
        dP = dP_cols[pc]
        ca_vec = Int[]
        psi_obs = dot(ph, hv)
        nv_core = mean(abs2, ph) * mean(abs2, hv)
        nv_adj = zero(T)
        if has_var_est
            J_mat = -(transpose(W_sub) * (dP .* hv)) ./ T(T_c)
            adj_mat = transpose(H_W) * J_mat
            for r in 1:n
                ar = vec(@view(adj_mat[:, r]))
                push!(Adj_cols, ar)
                push!(ca_vec, length(Adj_cols))
                ur = @view U_sub[:, r]
                psi_obs += dot(ur, ar)
                nv_adj += mean(abs2, ur) * mean(abs2, ar) +
                          T(2) * mean(ph .* ur) * mean(hv .* ar)
            end
        end
        nv = max(T(T_c) * max(nv_core + nv_adj, nv_core * T(0.25)), eps(T))
        push!(obs_raw, (psi_obs * psi_obs) / nv)
        push!(wk_vec, wk / nv)
        push!(meta_lag, k)
        push!(meta_mom, ms)
        push!(pair_meta, (pc, hc, ca_vec))
        return nothing
    end

    do_kurt = :kurt in moments
    do_skew = :skew in moments

    for k in 1:max_lag
        wk = one(T) / T(k)^4
        for j in 1:n
            hz_e = T[_svarma_tst_bsq(Z[t, j]) for t in (start_tau - k):(T_u - k)]
            hz_e .-= mean(hz_e)
            push!(H_cols, hz_e)
            hze_c = length(H_cols)

            hy_e = T[_svarma_tst_bsq(ys_all[j][p_eff + (start_tau + c - 1) - k]) for c in 1:T_c]
            hy_e .-= mean(hy_e)
            push!(H_cols, hy_e)
            hye_c = length(H_cols)

            hzo_c = 0
            hyo_c = 0
            if do_skew
                hz_o = Vector{T}(@view Z[(start_tau - k):(T_u - k), j])
                hz_o .-= mean(hz_o)
                push!(H_cols, hz_o)
                hzo_c = length(H_cols)

                hy_o = T[ys_all[j][p_eff + (start_tau + c - 1) - k] for c in 1:T_c]
                hy_o .-= mean(hy_o)
                push!(H_cols, hy_o)
                hyo_c = length(H_cols)
            end

            for i in 1:n
                if do_kurt
                    register_pair!(k, :kurt, wk, i, hze_c)
                    register_pair!(k, :kurt, wk, i, hye_c)
                end
                if do_skew
                    register_pair!(k, :skew, wk, n + i, hze_c)
                    register_pair!(k, :skew, wk, n + i, hye_c)
                    register_pair!(k, :skew, wk, i, hzo_c)
                    register_pair!(k, :skew, wk, i, hyo_c)
                end
            end
        end
    end

    P_mat = reduce(hcat, P_cols)
    H_mat = reduce(hcat, H_cols)
    Adj_mat = isempty(Adj_cols) ? zeros(T, T_c, 0) : reduce(hcat, Adj_cols)

    # Observed statistics per lag & per moment
    lag_stats = zeros(T, max_lag)
    mom_stats = Dict{Symbol,T}(m => zero(T) for m in moments)
    M_tot = length(wk_vec)
    lag_w = T[one(T) / T(k)^4 for k in 1:max_lag]
    @inbounds for m_i in 1:M_tot
        k = meta_lag[m_i]
        ms = meta_mom[m_i]
        lag_stats[k] += obs_raw[m_i]
        mom_stats[ms] += lag_w[k] * obs_raw[m_i]
    end

    # Joint permutation-multiplier bootstrap preserving exact CCE Theorem 2 first-step uncertainty
    B_lag = zeros(T, n_boot, max_lag)
    B_kurt = do_kurt ? Vector{T}(undef, n_boot) : T[]
    B_skew = do_skew ? Vector{T}(undef, n_boot) : T[]
    perm = collect(1:T_c)
    inv_wk = T[one(T) / lag_w[meta_lag[m_i]] for m_i in 1:M_tot]

    for b in 1:n_boot
        shuffle!(rng_eff, perm)
        Cp = transpose(@view(P_mat[perm, :])) * H_mat
        Ca = has_var_est ? (transpose(@view(U_sub[perm, :])) * Adj_mat) : zeros(T, n, 0)
        sk = zero(T)
        ss = zero(T)
        @inbounds for m_i in 1:M_tot
            cp, ch, cas = pair_meta[m_i]
            tv = Cp[cp, ch]
            if has_var_est
                for r in 1:n
                    tv += Ca[r, cas[r]]
                end
            end
            wval = wk_vec[m_i] * tv * tv
            B_lag[b, meta_lag[m_i]] += wval * inv_wk[m_i]
            if meta_mom[m_i] === :kurt
                sk += wval
            else
                ss += wval
            end
        end
        do_kurt && (B_kurt[b] = sk)
        do_skew && (B_skew[b] = ss)
    end

    lag_pvals = T[T((1 + count(b -> B_lag[b, k] >= lag_stats[k], 1:n_boot)) / (1 + n_boot)) for k in 1:max_lag]
    mom_pvals = Dict{Symbol,T}()
    do_kurt && (mom_pvals[:kurt] = T((1 + count(b -> B_kurt[b] >= mom_stats[:kurt], 1:n_boot)) / (1 + n_boot)))
    do_skew && (mom_pvals[:skew] = T((1 + count(b -> B_skew[b] >= mom_stats[:skew], 1:n_boot)) / (1 + n_boot)))

    # Omnibus non-negative union-intersection statistic across selected moment families
    if do_kurt && do_skew
        mk = mean(B_kurt); sdk = max(std(B_kurt), sqrt(eps(T)))
        ms_mean = mean(B_skew); sds = max(std(B_skew), sqrt(eps(T)))
        zk_obs = max(zero(T), (mom_stats[:kurt] - mk) / sdk)
        zs_obs = max(zero(T), (mom_stats[:skew] - ms_mean) / sds)
        agg_stat = max(zk_obs, zs_obs)^2
        agg_exceed = 0
        @inbounds for b in 1:n_boot
            zk_b = max(zero(T), (B_kurt[b] - mk) / sdk)
            zs_b = max(zero(T), (B_skew[b] - ms_mean) / sds)
            max(zk_b, zs_b)^2 >= agg_stat && (agg_exceed += 1)
        end
        pval = T((1 + agg_exceed) / (1 + n_boot))
    elseif do_kurt
        agg_stat = mom_stats[:kurt]
        pval = mom_pvals[:kurt]
    else
        agg_stat = mom_stats[:skew]
        pval = mom_pvals[:skew]
    end

    return FundamentalnessTestResult{T}(
        T(agg_stat), pval, lag_stats, lag_pvals, mom_stats, mom_pvals,
        max_lag, moments, n_boot, n, p_eff, T_c, mtype,
    )
end

"""
    test_fundamentalness(model; max_lag::Int=6, moments::Vector{Symbol}=[:skew, :kurt],
                         n_boot::Int=499, rng::AbstractRNG=Random.default_rng(),
                         seed::Union{Integer,Nothing}=nothing) -> FundamentalnessTestResult{T}

Chen, Choi & Escanciano (2017) higher-moment martingale-difference test for fundamentalness
of estimated `VARModel`, `SVARMAModel`, or `AbstractARIMAModel` innovations.

Under the null hypothesis that the Wold/structural innovations ``\\hat{u}_t`` are
fundamental and serially independent, higher-moment transformations of ``\\hat{u}_t``
are orthogonal to past innovations and past observables ``Y_{t-1}, \\dots, Y_{t-m_{\\max}}``:

```math
H_0\\!: \\operatorname{Cov}\\bigl(\\phi(\\hat{u}_{i,t}),\\, h(Y_{j,t-k})\\bigr) = 0
\\quad \\forall\\, 1 \\le k \\le m_{\\max}.
```

Bootstrap ``p``-values account for first-step VAR/AR estimation uncertainty using the
Theorem 2 Jacobian expansion of Chen, Choi & Escanciano (2017):

```math
\\hat{\\psi}_{m,k,t} = \\bigl(\\phi(\\hat{u}_{i,t}) - \\bar{\\phi}_i\\bigr) \\tilde{h}_{j,t-k}
+ \\hat{J}_{m,k} \\hat{\\Sigma}_{WW}^{-1} W_t \\hat{u}_t,
```

where:
- ``\\phi`` includes odd/quadratic bounded transformations selected by `moments` (`:skew`, `:kurt`),
- ``\\hat{J}_{m,k}`` is the sample Jacobian of the moment wrt the first-step VAR/AR parameters,
- ``W_t`` stacks the intercept and lagged regressors from the first-step estimation,
- ``\\hat{\\Sigma}_{WW}^{-1} W_t \\hat{u}_t`` captures first-step estimation uncertainty.

# Arguments
- `model`: estimated `VARModel`, `SVARMAModel`, or `AbstractARIMAModel`
- `max_lag::Int=6`: maximum past lag ``k`` tested
- `moments::Vector{Symbol}=[:skew, :kurt]`: subset of `[:skew, :kurt]`
- `n_boot::Int=499`: number of bootstrap replications
- `rng::AbstractRNG=Random.default_rng()`: random number generator
- `seed::Union{Integer,Nothing}=nothing`: optional random seed

# Example
```julia
sim = dgp_ssvarma(; T=400, Theta=[-0.5 0.0; 1.0 -2.0], dist=:mixture_normal_asym, seed=1)
var_fit = estimate_var(sim.Y, 4)
res = test_fundamentalness(var_fit; max_lag=6, n_boot=499, seed=1)
```
"""
function test_fundamentalness(model::Union{VARModel,SVARMAModel,AbstractARIMAModel};
                              max_lag::Int=6,
                              moments::Vector{Symbol}=Symbol[:skew, :kurt],
                              n_boot::Int=499,
                              rng::AbstractRNG=Random.default_rng(),
                              seed::Union{Integer,Nothing}=nothing)
    max_lag < 1 && throw(ArgumentError("max_lag must be ≥ 1, got $max_lag"))
    n_boot < 1 && throw(ArgumentError("n_boot must be ≥ 1, got $n_boot"))
    isempty(moments) && throw(ArgumentError("moments must be non-empty"))
    for m in moments
        m in (:skew, :kurt) || throw(ArgumentError("Invalid moment :$m; expected :skew or :kurt"))
    end
    mom_unique = unique(moments)
    rng_eff = _resolve_repro_rng(rng, seed)

    u_mat, Y_hist, p_eff, W_reg, mtype = _svarma_tst_extract_inputs(model)
    return _svarma_tst_cce_run(u_mat, Y_hist, p_eff, W_reg, mtype, max_lag, mom_unique, n_boot, rng_eff)
end

# =============================================================================
# Forni & Gambetti (2014) Informational Sufficiency Test
# =============================================================================

function _svarma_tst_sufficient_core(U_mat::AbstractMatrix{T},
                                     F_all::AbstractMatrix{T},
                                     u_row_offset::Int;
                                     r_max::Int=4,
                                     p_f::Int=4) where {T<:AbstractFloat}
    T_u, n = size(U_mat)
    T_f, r_avail = size(F_all)
    T_f >= u_row_offset + T_u || throw(ArgumentError(
        "Factor matrix has $T_f rows, fewer than required $(u_row_offset + T_u) rows"))

    r_eff = min(r_max, r_avail)
    r_eff >= 1 || throw(ArgumentError("r_max must be at least 1, got $r_max"))
    p_f >= 1 || throw(ArgumentError("p_f must be at least 1, got $p_f"))

    # Effective sample in U_mat such that F_all[(u_row_offset + tau - p_f)] is valid
    tau_start = max(1, p_f + 1 - u_row_offset)
    T_reg = T_u - tau_start + 1
    T_reg > r_eff * p_f + 2 || throw(ArgumentError(
        "Insufficient observations (T_reg=$T_reg) for r_max=$r_eff, p_f=$p_f lags"))

    U_target = Matrix{T}(@view U_mat[tau_start:T_u, :])
    U_c = U_target .- mean(U_target; dims=1)

    r_grid = collect(1:r_eff)
    f_stats = Vector{T}(undef, r_eff)
    f_pvals = Vector{T}(undef, r_eff)
    w_stats = Vector{T}(undef, r_eff)
    w_pvals = Vector{T}(undef, r_eff)
    dof_grid = Int[n * rf * p_f for rf in r_grid]

    for (idx, rf) in enumerate(r_grid)
        K_r = rf * p_f
        Z_r = Matrix{T}(undef, T_reg, K_r)
        @inbounds for ell in 1:p_f
            col0 = (ell - 1) * rf
            for t in 1:T_reg
                f_idx = u_row_offset + (tau_start + t - 1) - ell
                for j in 1:rf
                    Z_r[t, col0 + j] = F_all[f_idx, j]
                end
            end
        end
        Z_c = Z_r .- mean(Z_r; dims=1)

        S_ZZ = Matrix{T}(Symmetric((transpose(Z_c) * Z_c) / T(T_reg)))
        S_ZZ_inv = Matrix{T}(robust_inv(S_ZZ; silent=true))
        S_ZU = (transpose(Z_c) * U_c) / T(T_reg)
        B_hat = S_ZZ_inv * S_ZU
        E_hat = U_c - Z_c * B_hat
        Fit_Z = Z_c * B_hat

        df2_eq = max(T_reg - 1 - K_r, 1)
        Sigma_E = Matrix{T}(Symmetric((transpose(E_hat) * E_hat) / T(df2_eq)))
        Sigma_E_inv = Matrix{T}(robust_inv(Sigma_E; silent=true))

        # System F-statistic matching exact F(n*K_r, n*df2_eq) under Gaussianity
        tr_num = tr(transpose(Fit_Z) * Fit_Z * Sigma_E_inv)
        F_r = max(zero(T), T(tr_num) / T(n * K_r))
        pf_r = T(ccdf(FDist(n * K_r, n * df2_eq), Float64(F_r)))

        # HAC Wald chi-square on normal equation score g_t = E_hat_t ⊗ Z_c_t
        M_score = Matrix{T}(undef, T_reg, n * K_r)
        @inbounds for t in 1:T_reg
            for i in 1:n
                ei = E_hat[t, i]
                off = (i - 1) * K_r
                for k in 1:K_r
                    M_score[t, off + k] = ei * Z_c[t, k]
                end
            end
        end
        bw_cap = max(0, min(floor(Int, T_reg^(1 / 3)), p_f))
        Omega_hac = bw_cap == 0 ?
            Matrix{T}(Symmetric((transpose(M_score) * M_score) / T(T_reg))) :
            Matrix{T}(lrcov(M_score; kernel=:bartlett, bandwidth=bw_cap, demean=true))
        Omega_inv = _spd_inv(Omega_hac)
        g_bar = vec(S_ZU)
        W_r = max(zero(T), T(T_reg) * T(dot(g_bar, Omega_inv * g_bar)))
        pw_r = T(ccdf(Chisq(n * K_r), Float64(W_r)))

        f_stats[idx] = F_r
        f_pvals[idx] = clamp(pf_r, zero(T), one(T))
        w_stats[idx] = W_r
        w_pvals[idx] = clamp(pw_r, zero(T), one(T))
    end

    # Holm-adjusted familywise p-values across nested r = 1..r_max
    adj_f = _holm_adjust(f_pvals)
    adj_w = _holm_adjust(w_pvals)
    idx_f = argmin(adj_f)
    idx_w = argmin(adj_w)

    return SufficientInformationTestResult{T}(
        f_stats[idx_f],
        adj_f[idx_f],
        w_stats[idx_w],
        adj_w[idx_w],
        r_eff,
        p_f,
        n,
        T_reg,
        dof_grid[idx_f],
        r_grid,
        f_stats,
        f_pvals,
        w_stats,
        w_pvals,
    )
end

"""
    test_sufficient_information(model_or_shocks, X_panel; r_max::Int=4, p_f::Int=4)
        -> SufficientInformationTestResult{T}

Forni & Gambetti (2014) test for informational sufficiency (fundamentalness) in
structural VARs.

Given a reduced-form `VARModel`, an `SVARMAModel`, an identified `AbstractNonGaussianSVAR`,
or a raw matrix of innovations/shocks, extracts the top ``r \\in 1\\dots r_{\\max}`` principal
components ``\\hat{F}_t`` from a large macroeconomic panel `X_panel` (or accepts an
already-estimated `FactorModel`) and tests Granger-orthogonality of residuals/shocks
``\\hat{u}_t`` onto ``p_f`` lags of ``\\hat{F}_t``:

```math
\\hat{u}_t = c + \\sum_{\\ell=1}^{p_f} \\Gamma_\\ell \\hat{F}_{t-\\ell}^{(1:r)} + e_t,
\\qquad H_0\\!: \\Gamma_1 = \\dots = \\Gamma_{p_f} = 0,
```

where:
- ``\\hat{F}_t^{(1:r)}`` are standardized static principal components from [`estimate_factors`](@ref),
- ``p_f`` is the number of factor lags included in each orthogonality regression,
- Rejection (``p < 0.05``) implies the VAR omits state information needed for fundamentalness.

# Arguments
- `model_or_shocks`: `VARModel`, `SVARMAModel`, `AbstractNonGaussianSVAR`, or `AbstractMatrix`
- `X_panel`: `T × N` macroeconomic panel matrix or pre-estimated `FactorModel`
- `r_max::Int=4`: maximum number of principal components to test
- `p_f::Int=4`: number of factor lags in the orthogonality regression

# Example
```julia
sim = dgp_var(; T=250, n=2, seed=1)
m = estimate_var(sim.Y, 2)
X_panel = hcat(sim.Y, randn(250, 15))
res = test_sufficient_information(m, X_panel; r_max=4, p_f=4)
```
"""
function test_sufficient_information(model::VARModel{T},
                                     X_panel::AbstractMatrix;
                                     r_max::Int=4,
                                     p_f::Int=4) where {T<:AbstractFloat}
    _validate_data(X_panel, "X_panel")
    T_x, N_x = size(X_panel)
    T_u = size(model.U, 1)
    T_y = size(model.Y, 1)
    if T_x == T_y
        u_offset = model.p
    elseif T_x == T_u
        u_offset = 0
    else
        throw(ArgumentError(
            "X_panel row count ($T_x) must equal VAR observations ($T_y) or residuals ($T_u)"))
    end
    r_extract = min(r_max, min(T_x, N_x))
    r_extract >= 1 || throw(ArgumentError("r_max must be ≥ 1 and panel non-empty"))
    fm = estimate_factors(Matrix{T}(X_panel), r_extract; standardize=true)
    return _svarma_tst_sufficient_core(model.U, fm.factors, u_offset; r_max=r_extract, p_f=p_f)
end

function test_sufficient_information(model::VARModel{T},
                                     factors::FactorModel{Tf};
                                     r_max::Int=4,
                                     p_f::Int=4) where {T<:AbstractFloat,Tf<:AbstractFloat}
    F_mat = Matrix{T}(factors.factors)
    T_f = size(F_mat, 1)
    T_u = size(model.U, 1)
    T_y = size(model.Y, 1)
    u_offset = if T_f == T_y
        model.p
    elseif T_f == T_u
        0
    else
        throw(ArgumentError(
            "Factor row count ($T_f) must equal VAR observations ($T_y) or residuals ($T_u)"))
    end
    return _svarma_tst_sufficient_core(model.U, F_mat, u_offset; r_max=r_max, p_f=p_f)
end

function test_sufficient_information(shocks::AbstractMatrix,
                                     X_panel::AbstractMatrix;
                                     r_max::Int=4,
                                     p_f::Int=4)
    _validate_data(shocks, "shocks")
    _validate_data(X_panel, "X_panel")
    T_el = float(promote_type(eltype(shocks), eltype(X_panel)))
    T_el <: AbstractFloat || (T_el = Float64)
    S_mat = Matrix{T_el}(shocks)
    T_s = size(S_mat, 1)
    T_x, N_x = size(X_panel)
    T_x >= T_s || throw(ArgumentError("X_panel rows ($T_x) must be ≥ shocks rows ($T_s)"))
    u_offset = T_x - T_s
    r_extract = min(r_max, min(T_x, N_x))
    r_extract >= 1 || throw(ArgumentError("r_max must be ≥ 1 and panel non-empty"))
    fm = estimate_factors(Matrix{T_el}(X_panel), r_extract; standardize=true)
    return _svarma_tst_sufficient_core(S_mat, fm.factors, u_offset; r_max=r_extract, p_f=p_f)
end

function test_sufficient_information(shocks::AbstractMatrix,
                                     factors::FactorModel;
                                     r_max::Int=4,
                                     p_f::Int=4)
    _validate_data(shocks, "shocks")
    T_el = float(promote_type(eltype(shocks), eltype(factors.factors)))
    T_el <: AbstractFloat || (T_el = Float64)
    S_mat = Matrix{T_el}(shocks)
    F_mat = Matrix{T_el}(factors.factors)
    T_s = size(S_mat, 1)
    T_f = size(F_mat, 1)
    T_f >= T_s || throw(ArgumentError("Factor rows ($T_f) must be ≥ shocks rows ($T_s)"))
    u_offset = T_f - T_s
    return _svarma_tst_sufficient_core(S_mat, F_mat, u_offset; r_max=r_max, p_f=p_f)
end

function test_sufficient_information(res::AbstractNonGaussianSVAR,
                                     X_panel::Union{AbstractMatrix,FactorModel};
                                     r_max::Int=4,
                                     p_f::Int=4)
    return test_sufficient_information(_result_shocks(res), X_panel; r_max=r_max, p_f=p_f)
end

function test_sufficient_information(model::SVARMAModel{T},
                                     X_panel::Union{AbstractMatrix,FactorModel};
                                     r_max::Int=4,
                                     p_f::Int=4) where {T<:AbstractFloat}
    return test_sufficient_information(model.shocks, X_panel; r_max=r_max, p_f=p_f)
end

function test_sufficient_information(shocks::AbstractVector,
                                     X_panel::Union{AbstractMatrix,FactorModel};
                                     r_max::Int=4,
                                     p_f::Int=4)
    return test_sufficient_information(reshape(shocks, :, 1), X_panel; r_max=r_max, p_f=p_f)
end
