# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel specification tests: Hausman, Breusch-Pagan LM, F-test for FE,
Pesaran CD, Wooldridge serial correlation, Modified Wald groupwise heteroskedasticity.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# 1. Hausman Test: FE vs RE
# =============================================================================

"""
    hausman_test(fe::PanelRegModel, re::PanelRegModel) -> PanelTestResult{T}

Hausman specification test for fixed vs random effects.

H0: Random effects is consistent (both FE and RE consistent under H0).
H1: Random effects is inconsistent (only FE consistent).

Computes chi2 = (beta_FE - beta_RE)' (V_FE - V_RE)^{-1} (beta_FE - beta_RE)
on the common slope coefficients (intercept excluded from RE).

# Arguments
- `fe::PanelRegModel` — fixed effects model
- `re::PanelRegModel` — random effects model

# Returns
`PanelTestResult{T}` with chi-squared statistic and p-value.

# References
- Hausman, J. A. (1978). *Econometrica* 46(6), 1251-1271.
"""
function hausman_test(fe::PanelRegModel{T}, re::PanelRegModel{T}) where {T}
    fe.method == :fe || throw(ArgumentError("First argument must be a FE model (got :$(fe.method))"))
    re.method == :re || throw(ArgumentError("Second argument must be a RE model (got :$(re.method))"))

    pd = fe.data
    groups = pd.group_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)
    n = length(fe.y)
    y = fe.y
    X = fe.X
    k = size(X, 2)

    # ---- Compute FE (within) estimator and its textbook variance ----
    y_dm, y_gm = _within_demean(y, groups, unique_groups)
    X_dm, X_gm = _within_demean_matrix(X, groups, unique_groups)
    XtX_w = X_dm' * X_dm
    XtXinv_w = robust_inv(XtX_w; silent=true)
    b_fe = XtXinv_w * (X_dm' * y_dm)
    resid_w = y_dm .- X_dm * b_fe
    sigma_e2 = dot(resid_w, resid_w) / T(max(n - N - k, 1))
    V_fe = sigma_e2 .* XtXinv_w

    # ---- Compute RE (GLS) estimator and its textbook variance ----
    # Swamy-Arora variance components
    _, sigma2_between, _, _, _, _, _ = _between_regression(y, X, groups, unique_groups)
    T_i = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        T_i[j] = T(count(==(g), groups))
    end
    T_bar_harmonic = T(N) / sum(one(T) ./ T_i)
    sigma_u2 = max(sigma2_between - sigma_e2 / T_bar_harmonic, zero(T))

    # Theta per group
    theta_vec = zeros(T, N)
    for j in 1:N
        denom = T_i[j] * sigma_u2 + sigma_e2
        if denom > zero(T)
            theta_vec[j] = one(T) - sqrt(sigma_e2 / denom)
        end
    end

    # Quasi-demean
    y_qd = copy(y)
    X_qd = copy(X)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        ym = y_gm[g]
        xm = X_gm[g]
        for i in idx
            y_qd[i] = y[i] - th * ym
            for p in 1:k
                X_qd[i, p] = X[i, p] - th * xm[p]
            end
        end
    end
    X_qd_c = hcat(ones(T, n), X_qd)
    XtX_re = X_qd_c' * X_qd_c
    XtXinv_re = robust_inv(XtX_re; silent=true)
    b_re_full = XtXinv_re * (X_qd_c' * y_qd)
    b_re = b_re_full[2:end]
    resid_re = y_qd .- X_qd_c * b_re_full
    sigma_re2 = dot(resid_re, resid_re) / T(max(n - k - 1, 1))
    V_re_full = sigma_re2 .* XtXinv_re
    V_re = V_re_full[2:end, 2:end]

    # ---- Hausman statistic ----
    # Under H0, Var(b_FE - b_RE) = Var(b_FE) - Var(b_RE)
    # is PSD. But numerically it can fail, so we use |chi2|.
    db = b_fe .- b_re
    dV = V_fe .- V_re
    dV_inv = Matrix{T}(robust_inv(dV; silent=true))
    chi2 = abs(dot(db, dV_inv * db))
    df = k
    pval = T(1 - cdf(Chisq(df), chi2))

    desc = pval < T(0.05) ?
        "Reject H0: RE inconsistent, use FE (p=$(round(pval; digits=4)))" :
        "Fail to reject H0: RE is consistent (p=$(round(pval; digits=4)))"

    PanelTestResult{T}("Hausman test", chi2, pval, df, desc)
end

# =============================================================================
# 2. Breusch-Pagan LM Test: Pooled OLS vs RE
# =============================================================================

"""
    breusch_pagan_test(re::PanelRegModel) -> PanelTestResult{T}

Breusch-Pagan Lagrange multiplier test for random effects.

H0: sigma_u^2 = 0 (pooled OLS is adequate).
H1: sigma_u^2 > 0 (random effects needed).

LM = (nT / (2(T-1))) * [sum_i (sum_t e_it)^2 / (sum_i sum_t e_it^2) - 1]^2

# Arguments
- `re::PanelRegModel` — random effects model (pooled OLS residuals computed internally)

# Returns
`PanelTestResult{T}` with chi-squared(1) statistic and p-value.

# References
- Breusch, T. S. & Pagan, A. R. (1980). *Review of Economic Studies* 47(1), 239-253.
"""
function breusch_pagan_test(re::PanelRegModel{T}) where {T}
    re.method == :re || throw(ArgumentError("Model must be RE (got :$(re.method))"))

    pd = re.data
    groups = pd.group_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)
    n = length(re.y)

    # Run pooled OLS internally to get residuals
    X_c = hcat(ones(T, n), re.X)
    XtXinv = robust_inv(X_c' * X_c; silent=true)
    beta_ols = XtXinv * (X_c' * re.y)
    e_ols = re.y .- X_c * beta_ols

    # Compute LM statistic
    sum_e2 = dot(e_ols, e_ols)
    sum_e2 = max(sum_e2, T(1e-300))

    sum_gi2 = zero(T)
    for g in unique_groups
        idx = findall(==(g), groups)
        sum_gi2 += sum(@view e_ols[idx])^2
    end

    ratio = sum_gi2 / sum_e2 - one(T)
    # Average T per group
    T_avg = T(n) / T(N)
    LM = T(n) / (2 * (T_avg - one(T))) * ratio^2

    # Safeguard for very small panels
    LM = max(LM, zero(T))
    pval = T(1 - cdf(Chisq(1), LM))

    desc = pval < T(0.05) ?
        "Reject H0: Random effects needed (p=$(round(pval; digits=4)))" :
        "Fail to reject H0: Pooled OLS adequate (p=$(round(pval; digits=4)))"

    PanelTestResult{T}("Breusch-Pagan LM test", LM, pval, 1, desc)
end

# =============================================================================
# 3. F-test for Fixed Effects
# =============================================================================

"""
    f_test_fe(fe::PanelRegModel) -> PanelTestResult{T}

F-test for joint significance of all entity fixed effects.

H0: all alpha_i = 0 (pooled OLS is adequate).
H1: at least one alpha_i != 0 (entity effects present).

F = ((SSR_pooled - SSR_FE) / (N-1)) / (SSR_FE / (NT - N - K))

# Arguments
- `fe::PanelRegModel` — fixed effects model

# Returns
`PanelTestResult{T}` with F(N-1, NT-N-K) statistic and p-value.

# References
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer. Ch. 4.
"""
function f_test_fe(fe::PanelRegModel{T}) where {T}
    fe.method == :fe || throw(ArgumentError("Model must be FE (got :$(fe.method))"))

    pd = fe.data
    groups = pd.group_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)
    n = length(fe.y)
    k = length(fe.beta)

    # Run pooled OLS internally
    X_c = hcat(ones(T, n), fe.X)
    XtXinv = robust_inv(X_c' * X_c; silent=true)
    beta_ols = XtXinv * (X_c' * fe.y)
    e_ols = fe.y .- X_c * beta_ols
    ssr_pooled = dot(e_ols, e_ols)

    # SSR from FE (within residuals)
    y_dm, _ = _within_demean(fe.y, groups, unique_groups)
    X_dm, _ = _within_demean_matrix(fe.X, groups, unique_groups)
    resid_dm = y_dm .- X_dm * fe.beta
    ssr_fe = dot(resid_dm, resid_dm)

    df1 = N - 1
    df2 = n - N - k
    df2 = max(df2, 1)

    f_stat = ((ssr_pooled - ssr_fe) / T(df1)) / (ssr_fe / T(df2))
    f_stat = max(f_stat, zero(T))
    pval = T(1 - cdf(FDist(df1, df2), f_stat))

    desc = pval < T(0.05) ?
        "Reject H0: Entity effects significant, use FE (p=$(round(pval; digits=4)))" :
        "Fail to reject H0: Pooled OLS adequate (p=$(round(pval; digits=4)))"

    PanelTestResult{T}("F-test for fixed effects", f_stat, pval, (df1, df2), desc)
end

# =============================================================================
# 4. Pesaran CD Test for Cross-Sectional Dependence
# =============================================================================

"""
    pesaran_cd_test(m::PanelRegModel) -> PanelTestResult{T}

Pesaran CD test for cross-sectional dependence in panel residuals.

H0: No cross-sectional dependence.
H1: Cross-sectional dependence present.

CD = sqrt(2 / (N(N-1))) * sum_{i<j} sqrt(T_ij) * rho_ij

where rho_ij is the pairwise residual correlation for groups i, j over
their overlapping time periods.

# Arguments
- `m::PanelRegModel` — any panel regression model (FE, RE, etc.)

# Returns
`PanelTestResult{T}` with standard normal test statistic and p-value.

# References
- Pesaran, M. H. (2004). *Cambridge Working Papers in Economics* No. 0435.
- Pesaran, M. H. (2015). *Econometrica* 83(4), 1481-1507.
"""
function pesaran_cd_test(m::PanelRegModel{T}) where {T}
    pd = m.data
    groups = pd.group_id
    time_ids = pd.time_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)
    N >= 2 || throw(ArgumentError("Need at least 2 groups for Pesaran CD test"))

    # Build residual vectors by group: Dict{group => Dict{time => residual}}
    resid_map = Dict{Int,Dict{Int,T}}()
    for (i, g) in enumerate(groups)
        t = time_ids[i]
        if !haskey(resid_map, g)
            resid_map[g] = Dict{Int,T}()
        end
        resid_map[g][t] = m.residuals[i]
    end

    # Compute CD statistic
    cd_sum = zero(T)
    for i in 1:(N-1)
        gi = unique_groups[i]
        ri = resid_map[gi]
        for j in (i+1):N
            gj = unique_groups[j]
            rj = resid_map[gj]

            # Overlapping time periods
            common_t = intersect(keys(ri), keys(rj))
            Tij = length(common_t)
            Tij < 2 && continue

            # Compute correlation
            ei = T[ri[t] for t in common_t]
            ej = T[rj[t] for t in common_t]
            rho_ij = cor(ei, ej)
            isfinite(rho_ij) || continue

            cd_sum += sqrt(T(Tij)) * rho_ij
        end
    end

    CD = sqrt(T(2) / T(N * (N - 1))) * cd_sum
    # Two-sided test: |CD| ~ N(0,1)
    pval = T(2 * (1 - cdf(Normal(), abs(CD))))

    desc = pval < T(0.05) ?
        "Reject H0: Cross-sectional dependence detected (p=$(round(pval; digits=4)))" :
        "Fail to reject H0: No cross-sectional dependence (p=$(round(pval; digits=4)))"

    PanelTestResult{T}("Pesaran CD test", CD, pval, 1, desc)
end

# =============================================================================
# 5. Wooldridge AR(1) Test for Serial Correlation
# =============================================================================

"""
    wooldridge_ar_test(fe::PanelRegModel) -> PanelTestResult{T}

Wooldridge test for first-order serial correlation in FE panel residuals.

H0: No serial correlation in idiosyncratic errors.
H1: AR(1) serial correlation present.

Procedure:
1. Compute first-differences of FE residuals within each group
2. Regress Delta_e_it on Delta_e_i,t-1 (pooled OLS)
3. Test H0: coefficient = -0.5 (under no serial correlation,
   first-differenced iid errors have autocorrelation -0.5)

# Arguments
- `fe::PanelRegModel` — fixed effects model

# Returns
`PanelTestResult{T}` with F(1, N-1) statistic and p-value.

# References
- Wooldridge, J. M. (2002). *Econometric Analysis of Cross Section and Panel Data*. MIT Press. Ch. 10.
- Drukker, D. M. (2003). *Stata Journal* 3(2), 168-177.
"""
function wooldridge_ar_test(fe::PanelRegModel{T}) where {T}
    fe.method == :fe || throw(ArgumentError("Model must be FE (got :$(fe.method))"))

    pd = fe.data
    groups = pd.group_id
    time_ids = pd.time_id
    unique_groups = sort(unique(groups))

    # Compute first-differences of residuals within each group
    de_curr = T[]   # Delta e_it
    de_lag = T[]    # Delta e_i,t-1
    de_groups = Int[]

    for g in unique_groups
        idx = findall(==(g), groups)
        t_g = time_ids[idx]
        perm = sortperm(t_g)
        idx_sorted = idx[perm]
        t_sorted = t_g[perm]

        # First-differences of residuals (consecutive periods only)
        de_g = T[]
        for j in 2:length(idx_sorted)
            if t_sorted[j] == t_sorted[j-1] + 1
                push!(de_g, fe.residuals[idx_sorted[j]] - fe.residuals[idx_sorted[j-1]])
            end
        end

        # Now pair (Delta e_t, Delta e_{t-1})
        for j in 2:length(de_g)
            push!(de_curr, de_g[j])
            push!(de_lag, de_g[j-1])
            push!(de_groups, g)
        end
    end

    n_obs = length(de_curr)
    n_obs > 2 || throw(ArgumentError("Too few observations for Wooldridge AR test (need at least 3 consecutive periods)"))

    # Regress Delta_e_it on Delta_e_i,t-1 with intercept
    X_w = hcat(ones(T, n_obs), de_lag)
    XtXinv = robust_inv(X_w' * X_w; silent=true)
    beta_w = XtXinv * (X_w' * de_curr)
    resid_w = de_curr .- X_w * beta_w

    # Cluster-robust variance (by group)
    unique_wgroups = sort(unique(de_groups))
    N_g = length(unique_wgroups)
    k_w = 2  # intercept + slope

    # Cluster-robust sandwich
    meat = zeros(T, k_w, k_w)
    for g in unique_wgroups
        idx = findall(==(g), de_groups)
        Xi = @view X_w[idx, :]
        ei = @view resid_w[idx]
        score_g = Xi' * ei  # k_w x 1
        meat .+= score_g * score_g'
    end
    scale = T(N_g) / T(N_g - 1) * T(n_obs - 1) / T(n_obs - k_w)
    V_cluster = XtXinv * (scale .* meat) * XtXinv

    # Test H0: slope coefficient = -0.5
    rho_hat = beta_w[2]
    se_rho = sqrt(max(V_cluster[2, 2], T(1e-300)))
    f_stat = ((rho_hat - T(-0.5)) / se_rho)^2
    f_stat = max(f_stat, zero(T))

    df1 = 1
    df2 = max(N_g - 1, 1)
    pval = T(1 - cdf(FDist(df1, df2), f_stat))

    desc = pval < T(0.05) ?
        "Reject H0: Serial correlation detected (p=$(round(pval; digits=4)))" :
        "Fail to reject H0: No serial correlation (p=$(round(pval; digits=4)))"

    PanelTestResult{T}("Wooldridge AR(1) test", f_stat, pval, (df1, df2), desc)
end

# =============================================================================
# 6. Modified Wald Test for Groupwise Heteroskedasticity
# =============================================================================

"""
    modified_wald_test(fe::PanelRegModel) -> PanelTestResult{T}

Modified Wald test for groupwise heteroskedasticity in FE panel residuals.

H0: sigma_i^2 = sigma^2 for all i (homoskedasticity across groups).
H1: At least one group has different error variance.

W = sum_i (sigma_i^2 - sigma^2)^2 / (2 * sigma_i^4 / T_i)

# Arguments
- `fe::PanelRegModel` — fixed effects model

# Returns
`PanelTestResult{T}` with chi-squared(N) statistic and p-value.

# References
- Greene, W. H. (2018). *Econometric Analysis*. 8th ed. Pearson. Ch. 9.
- Baum, C. F. (2001). *Stata Journal* 1(1), 101-104.
"""
function modified_wald_test(fe::PanelRegModel{T}) where {T}
    fe.method == :fe || throw(ArgumentError("Model must be FE (got :$(fe.method))"))

    pd = fe.data
    groups = pd.group_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)

    # Compute group-specific residual variances
    sigma2_i = zeros(T, N)
    T_i = zeros(Int, N)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        T_i[j] = length(idx)
        ei = @view fe.residuals[idx]
        sigma2_i[j] = dot(ei, ei) / T(T_i[j])
    end

    # Overall sigma^2 (pooled)
    sigma2_bar = fe.sigma_e^2

    # Modified Wald statistic
    W = zero(T)
    for j in 1:N
        s2i = max(sigma2_i[j], T(1e-300))
        var_s2i = 2 * s2i^2 / T(T_i[j])
        W += (sigma2_i[j] - sigma2_bar)^2 / var_s2i
    end

    pval = T(1 - cdf(Chisq(N), W))

    desc = pval < T(0.05) ?
        "Reject H0: Groupwise heteroskedasticity detected (p=$(round(pval; digits=4)))" :
        "Fail to reject H0: Homoskedasticity across groups (p=$(round(pval; digits=4)))"

    PanelTestResult{T}("Modified Wald test", W, pval, N, desc)
end
