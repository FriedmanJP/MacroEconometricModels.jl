# [Model Diagnostics](@id tests_diagnostics_page)

Post-estimation specification testing validates the assumptions that inference rests on. This page covers the diagnostics that take a *fitted model* or a *raw series* as input: VAR stability, Granger causality, multivariate normality, ARCH effects, nonlinear dependence, distributional fit, the random-walk hypothesis, group comparisons, and nested-model comparison. For the full test battery and the tables that route a question to a test, see [Hypothesis Tests](@ref tests_page).

!!! note "Where the other diagnostics live"
    Two families of specification tests are documented elsewhere because they belong to the estimator they diagnose. The **regression diagnostics** --- `chow_test`, `cusum_test`, `cusumsq_test`, `breusch_godfrey_test`, `reset_test`, `white_test`, and `breusch_pagan_test` --- are on [Linear Regression](@ref regression_page), since each takes a fitted `RegModel`. The **portmanteau tests** --- `ljung_box_test`, `box_pierce_test`, and `durbin_watson_test` --- are on [Spectral Analysis](@ref spectral_page), where they sit next to their frequency-domain counterparts. Panel VAR GMM diagnostics (`pvar_hansen_j`, `pvar_mmsc`, `pvar_lag_selection`) are on [Panel Tests](@ref tests_panel_page).

- **VAR stationarity**: Companion matrix eigenvalue check for stable dynamics
- **Granger causality**: Pairwise and block Wald tests for predictive causality (Granger 1969)
- **Normality**: Jarque-Bera, Mardia, Doornik-Hansen, and Henze-Zirkler tests plus a seven-test suite
- **ARCH diagnostics**: Engle (1982) ARCH-LM and Ljung-Box on squared residuals
- **BDS independence**: Brock, Dechert, Scheinkman & LeBaron (1996) test against *any* departure from i.i.d.
- **EDF goodness of fit**: Kolmogorov-Smirnov, Lilliefors, Cramér-von Mises, Anderson-Darling, and Watson against seven parametric families
- **Variance ratios**: Lo-MacKinlay (1988), Chow-Denning (1993), Wright (2000) rank/sign, and Kim (2006) wild bootstrap
- **Equality and correlation**: One-way location and scale tests by classifier, plus Pearson, Spearman, and Kendall correlation tests
- **Model comparison**: Likelihood ratio (Wilks 1938) and Lagrange multiplier (Rao 1948) tests for nested models

```@setup test_diag
using MacroEconometricModels, Random, Distributions
Random.seed!(42)
```

## Quick Start

The VAR examples use three FRED-MD series in their recommended transformations: industrial production growth, the change in CPI inflation, and the federal funds rate. Blocks that simulate data seed their own generator, so those numbers reproduce exactly in a fresh session.

**Recipe 1: Fit a VAR and check stability**

```@example test_diag
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]           # drop rows lost to differencing
m = estimate_var(Y, 2)

show(stdout, is_stationary(m))
```

**Recipe 2: Granger causality**

```@example test_diag
report(granger_test(m, 3, 1))                  # does the funds rate cause output?
```

**Recipe 3: Normality suite on the residuals**

```@example test_diag
report(normality_test_suite(m))
```

**Recipe 4: ARCH-LM test on a raw series**

```@example test_diag
Random.seed!(7)
arch_iid = arch_lm_test(randn(500), 5)
(statistic = round(arch_iid.statistic, digits=4),
 pvalue = round(arch_iid.pvalue, digits=4),
 q = arch_iid.q)
```

**Recipe 5: Likelihood ratio test for lag order**

```@example test_diag
report(lr_test(estimate_var(Y, 1), m))
```

---

## VAR Stationarity

A VAR(``p``) process is covariance-stationary if and only if every eigenvalue of its companion matrix lies strictly inside the unit circle. Stationarity is what makes impulse responses decay, forecasts converge to the unconditional mean, and the asymptotic theory behind coefficient standard errors and Granger tests valid.

The companion form stacks the VAR(``p``) system into a first-order representation:

```math
\xi_t = F \, \xi_{t-1} + v_t
```

where the ``np \times np`` companion matrix is:

```math
F = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \\ I_n & 0 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & I_n & 0 \end{bmatrix}
```

where:
- ``A_1, \ldots, A_p`` are the ``n \times n`` VAR coefficient matrices
- ``I_n`` is the ``n \times n`` identity matrix
- ``\xi_t = (y_t', y_{t-1}', \ldots, y_{t-p+1}')'`` is the stacked state vector

The stability condition is ``|\lambda_i(F)| < 1`` for every eigenvalue ``\lambda_i``.

```@example test_diag
stat = is_stationary(m)
(is_stationary = stat.is_stationary,
 max_modulus = round(stat.max_modulus, digits=4),
 n_eigenvalues = length(stat.eigenvalues))
```

The maximum modulus of 0.497 sits well inside the unit circle, so the fitted VAR(2) is stationary and every downstream object --- impulse responses, forecast error variance decompositions, Granger tests --- is well defined. The six eigenvalues are the ``np = 3 \times 2`` roots of the companion matrix, and they arrive in three complex-conjugate pairs, so the system's dynamics are oscillatory rather than monotone. A modulus close to but below 1.0 signals persistence, common when levels rather than growth rates enter the system; at or above 1.0 the VAR has a unit or explosive root and must be respecified, usually by differencing or by reducing the lag order.

`is_stationary` returns a `VARStationarityResult`, which has no `report` method; use `show(stdout, result)` for the formatted table, as in Recipe 1.

| Field | Type | Description |
|-------|------|-------------|
| `is_stationary` | `Bool` | `true` when every eigenvalue has modulus strictly below 1 |
| `eigenvalues` | `Vector` | Companion matrix eigenvalues (real or complex) |
| `max_modulus` | `T` | Largest eigenvalue modulus |
| `companion_matrix` | `Matrix{T}` | The ``np \times np`` companion matrix ``F`` |

---

## Granger Causality

The Granger causality test (Granger 1969) asks whether lagged values of one variable improve the prediction of another within a VAR. The **pairwise test** checks whether all lag coefficients from variable ``j`` are zero in the equation for variable ``i``; the **block test** generalizes it to a group of causing variables.

For the pairwise test the null is:

```math
H_0: A_1[i,j] = A_2[i,j] = \cdots = A_p[i,j] = 0
```

where ``A_l[i,j]`` is the coefficient on the ``l``-th lag of variable ``j`` in the equation for variable ``i``. The Wald statistic is:

```math
W = \hat{\theta}' \hat{V}^{-1} \hat{\theta} \sim \chi^2(p)
```

where:
- ``\hat{\theta}`` is the ``p \times 1`` vector of restricted coefficients
- ``\hat{V} = \hat{\sigma}_{ii} (X'X)^{-1}`` restricted to the corresponding rows and columns
- ``p`` is the lag order, which is also the degrees of freedom

For a block test with ``m`` causing variables the degrees of freedom become ``p \times m``.

!!! note "Technical Note"
    Granger causality is a **predictive** concept, not a structural one. Variable ``j`` Granger-causes variable ``i`` when lags of ``j`` help forecast ``i`` beyond the information already in the lags of ``i`` and of every other variable in the VAR. Adding or removing a variable from the system can reverse the conclusion, which is precisely why it carries no structural content.

```@example test_diag
report(granger_test(m, 3, 1))
```

The Wald statistic of 7.8995 on 2 degrees of freedom gives a p-value of 0.0193, so the federal funds rate Granger-causes industrial production growth at the 5% level: past policy rates carry information about future output growth beyond what output's own history supplies. The test says nothing about the sign or the transmission channel --- for that, identify the system and read the impulse responses.

The block form tests several causing variables jointly:

```@example test_diag
report(granger_test(m, [2, 3], 1))
```

Adding inflation to the conditioning set raises the statistic to 19.5165 on 4 degrees of freedom, with a p-value below 0.001. The block test is not the sum of two pairwise tests: it accounts for the covariance between the two blocks of coefficients, so it can reject when neither variable rejects alone, or fail to reject when one does. Use it whenever the hypothesis concerns a group --- "do nominal variables help forecast real activity?" --- rather than a single series.

`granger_test_all` returns an ``n \times n`` matrix in which entry ``[i, j]`` tests whether variable ``j`` Granger-causes variable ``i``, with `nothing` on the diagonal. The matrix has no compact display of its own, so extract the fields for a readable screen:

```@example test_diag
names_var = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
g_all = granger_test_all(m)
[(cause = names_var[j], effect = names_var[i],
  statistic = round(g_all[i, j].statistic, digits=3),
  pvalue = round(g_all[i, j].pvalue, digits=4))
 for i in 1:3, j in 1:3 if i != j]
```

Five of the six directions reject at 5%. The exception is industrial production to inflation, at 1.739 with a p-value of 0.419 --- output growth carries no predictive content for the change in inflation, while the reverse link is significant at 10.502 with a p-value of 0.005. The strongest single link is output to the funds rate, at 16.455 with a p-value of 0.0003, which is what a policy reaction function implies. Screening the full matrix before imposing an identification scheme is the cheapest way to see whether the recursive ordering a Cholesky decomposition would assume is consistent with the predictive structure of the data.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Wald ``\chi^2`` statistic |
| `pvalue` | `T` | P-value from ``\chi^2(df)`` |
| `df` | `Int` | Degrees of freedom (number of restrictions) |
| `cause` | `Vector{Int}` | Indices of the causing variable(s), sorted |
| `effect` | `Int` | Index of the effect variable |
| `n` | `Int` | Number of variables in the VAR |
| `p` | `Int` | Lag order |
| `nobs` | `Int` | Effective number of observations |
| `test_type` | `Symbol` | `:pairwise` or `:block` |

---

## Normality Tests

Multivariate normality of the residuals is what licenses exact finite-sample inference on coefficient t-statistics and confidence intervals. OLS estimates stay consistent under any residual distribution, but likelihood-based tests and bootstrap intervals can behave poorly when normality fails badly. Every test below accepts either a `VARModel`, from which residuals are extracted automatically, or a raw residual matrix.

### Jarque-Bera

The multivariate Jarque-Bera test (Jarque & Bera 1980; Lutkepohl 2005, Section 4.5) combines multivariate skewness and kurtosis into one statistic. The `:multivariate` method computes:

```math
\lambda_{JB} = \lambda_s + \lambda_k = \frac{T \, b_{1,k}}{6} + \frac{T \, (b_{2,k} - k(k+2))^2}{24k}
```

where:
- ``b_{1,k} = \frac{1}{T^2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3`` is multivariate skewness
- ``b_{2,k} = \frac{1}{T} \sum_i (u_i' \Sigma^{-1} u_i)^2`` is multivariate kurtosis
- ``k`` is the number of variables
- under ``H_0``, ``\lambda_{JB} \sim \chi^2(2k)``

```@example test_diag
report(jarque_bera_test(m; method=:multivariate))
```

The statistic of 128189.37 on 6 degrees of freedom rejects normality at any level whatsoever. That is the normal state of affairs for monthly macroeconomic residuals: the funds-rate equation alone spans the 1980-82 Volcker episode and the 2008 collapse, and no two-moment family absorbs those. The practical consequence is documented in the pitfalls below --- it invalidates exact small-sample inference, not the coefficient estimates.

The `:component` method runs univariate Jarque-Bera tests on each standardized residual and sums them, which localizes the failure:

```@example test_diag
jb_comp = jarque_bera_test(m; method=:component)
[(variable = v, statistic = round(s, digits=1), pvalue = round(p, digits=4))
 for (v, s, p) in zip(names_var, jb_comp.components, jb_comp.component_pvalues)]
```

Every equation rejects, but not equally: the funds-rate residual contributes 81025.7 and industrial production 77859.0, while the inflation equation contributes only 720.4 --- two orders of magnitude less. Reading the components rather than the aggregate is what turns a rejection into a modelling decision, and here it says the tail problem lives in the real-activity and policy-rate equations, not in the price equation.

### Mardia

Mardia's tests (Mardia 1970) assess multivariate normality through skewness and kurtosis, separately or jointly:

- `:skewness` tests ``H_0: b_{1,k} = 0``, with ``T \cdot b_{1,k} / 6 \sim \chi^2(k(k+1)(k+2)/6)``
- `:kurtosis` tests ``H_0: b_{2,k} = k(k+2)``, with ``(b_{2,k} - k(k+2)) / \sqrt{8k(k+2)/T} \sim N(0,1)``
- `:both` combines the two into a single ``\chi^2`` statistic

```@example test_diag
report(mardia_test(m; type=:skewness))
```

```@example test_diag
report(mardia_test(m; type=:kurtosis))
```

Skewness gives 2135.38 on 10 degrees of freedom and kurtosis gives 275.01 as a standardized deviate; both reject overwhelmingly. Because the two have different null distributions their levels are not directly comparable, but the kurtosis statistic is a ``z``-score of 275 --- an unambiguous statement about tail thickness. That distinction matters operationally: heavy tails argue for robust or bootstrap standard errors, while asymmetry would argue for a transformation or an asymmetric error distribution.

### Doornik-Hansen

The Doornik-Hansen omnibus test (Doornik & Hansen 2008) applies the Bowman-Shenton transformation to each component's skewness and kurtosis, producing approximately ``N(0,1)`` values ``z_1`` and ``z_2``, then sums their squares:

```math
DH = \sum_{j=1}^{k} (z_{1,j}^2 + z_{2,j}^2) \sim \chi^2(2k)
```

The transformation improves finite-sample size relative to the raw Jarque-Bera statistic, which is why it is the default omnibus test in most modern software.

```@example test_diag
report(doornik_hansen_test(m))
```

The statistic of 120768.24 on 6 degrees of freedom rejects, marginally below the multivariate Jarque-Bera value of 128189.37 on the same residuals and the same degrees of freedom. The two agreeing this closely is the expected outcome in a sample of 797: the Bowman-Shenton transformation buys its size improvement in small samples, and there is nothing small about this one. The `components` and `component_pvalues` fields give the per-equation breakdown.

### Henze-Zirkler

The Henze-Zirkler test (Henze & Zirkler 1990) works from the empirical characteristic function and is consistent against *any* non-normal alternative, not only against moment deviations:

```math
T_{\beta} = \frac{1}{n} \sum_{i,j} e^{-\beta^2 D_{ij}/2} - 2(1+\beta^2)^{-k/2} \sum_i e^{-\beta^2 d_i^2/(2(1+\beta^2))} + n(1+2\beta^2)^{-k/2}
```

where:
- ``D_{ij} = (z_i - z_j)'(z_i - z_j)`` is the squared distance between standardized residuals
- ``d_i = z_i' z_i`` is the squared norm of the ``i``-th standardized residual
- ``\beta`` is a bandwidth chosen as a function of ``k`` and ``n``
- the p-value uses a log-normal approximation under ``H_0``

```@example test_diag
report(henze_zirkler_test(m))
```

The statistic of 8.2093 rejects with a p-value below 0.001. Because it is consistent against every alternative, the interesting case for Henze-Zirkler is a rejection alongside a Mardia non-rejection --- non-normality in a shape the first four moments do not capture. Here all four tests agree, so the extra generality buys nothing beyond confirmation.

### Test suite

`normality_test_suite` runs all seven tests --- multivariate Jarque-Bera, component-wise Jarque-Bera, Mardia skewness, Mardia kurtosis, Mardia combined, Doornik-Hansen, and Henze-Zirkler --- and returns a `NormalityTestSuite` with a consolidated display. It also accepts a raw residual matrix, so it works for any model whose residuals can be extracted.

```@example test_diag
suite = normality_test_suite(m)
[(test = r.test_name, statistic = round(r.statistic, digits=2), pvalue = round(r.pvalue, digits=4))
 for r in suite.results]
```

All seven p-values are zero to four decimal places. Unanimity is the usual outcome on macroeconomic residuals and is not itself informative; what the suite buys is the *pattern*. The two `:jarque_bera` rows are the multivariate and component-wise variants, which differ by more than 30,000 here, and the Mardia rows separate a skewness statistic of 2135.38 from a kurtosis ``z``-score of 275.01. Together with the component breakdown above, that pattern points at heavy tails in the real-activity and policy-rate equations rather than at a misspecified conditional mean.

**`NormalityTestResult` fields**

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | Test identifier, e.g. `:jarque_bera`, `:mardia_skewness` |
| `statistic` | `T` | Test statistic |
| `pvalue` | `T` | P-value |
| `df` | `Int` | Degrees of freedom for ``\chi^2`` tests; 0 for Henze-Zirkler |
| `n_vars` | `Int` | Number of variables |
| `n_obs` | `Int` | Number of observations |
| `components` | `Union{Nothing, Vector{T}}` | Per-component statistics when applicable |
| `component_pvalues` | `Union{Nothing, Vector{T}}` | Per-component p-values when applicable |

`NormalityTestSuite` holds `results` (a length-7 vector), the `residuals` matrix tested, `n_vars`, and `n_obs`.

---

## ARCH Diagnostics

Conditional heteroskedasticity violates the constant-variance assumption, degrading the efficiency of OLS and the coverage of standard confidence intervals. Two complementary tests detect it, and both accept either a raw vector or a fitted volatility model, in which case they use the standardized residuals.

### ARCH-LM

The ARCH-LM test (Engle 1982) regresses squared residuals on their own lags. Under the null of no ARCH effects the auxiliary regression

```math
\varepsilon_t^2 = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2 + v_t
```

has no explanatory power, and the statistic is:

```math
LM = T \cdot R^2 \sim \chi^2(q)
```

where:
- ``T`` is the effective sample size after ``q`` lags
- ``R^2`` is the coefficient of determination of the auxiliary regression
- ``q`` is the number of lags

The function returns a plain named tuple `(statistic, pvalue, q)` rather than a result object, so read the fields directly.

```@example test_diag
Random.seed!(12)
h = ones(600); e = zeros(600)
for t in 2:600                                  # GARCH(1,1) with strong persistence
    h[t] = 0.05 + 0.15 * e[t-1]^2 + 0.80 * h[t-1]
    e[t] = sqrt(h[t]) * randn()
end

arch_raw = arch_lm_test(e, 5)
(statistic = round(arch_raw.statistic, digits=3), pvalue = round(arch_raw.pvalue, digits=6), q = arch_raw.q)
```

The statistic of 38.54 on 5 degrees of freedom gives a p-value of zero to six decimal places, so the ARCH effects built into this series are detected easily. Compare with Recipe 4, where the same test on 500 i.i.d. normal draws returned 1.7303 with a p-value of 0.8851 --- the test has both size and power at these sample sizes. The natural next step is to fit a volatility model and re-run the test on its standardized residuals:

```@example test_diag
g_e = estimate_garch(e)
arch_std = arch_lm_test(g_e, 10)
(statistic = round(arch_std.statistic, digits=3), pvalue = round(arch_std.pvalue, digits=4))
```

After fitting the GARCH(1,1), the ARCH-LM statistic on the standardized residuals falls to 6.796 with a p-value of 0.7445 on 10 lags. The conditional variance model has absorbed the heteroskedasticity: nothing is left for a further ARCH term to explain. That before-and-after pair --- reject on the raw series, fail to reject on the standardized residuals --- is the standard adequacy check for a volatility specification.

### Ljung-Box on squared residuals

The Ljung-Box test applied to squared residuals detects serial correlation in the variance process. Under the null of no autocorrelation in ``z_t^2``:

```math
Q = n(n+2) \sum_{k=1}^{K} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(K)
```

where:
- ``\hat{\rho}_k`` is the sample autocorrelation of the squared series at lag ``k``
- ``n`` is the sample size
- ``K`` is the maximum lag

Like `arch_lm_test`, it returns a named tuple, here `(statistic, pvalue, K)`.

```@example test_diag
lb_raw = ljung_box_squared(e, 10)
lb_std = ljung_box_squared(g_e, 10)
(raw = (statistic = round(lb_raw.statistic, digits=3), pvalue = round(lb_raw.pvalue, digits=6)),
 standardized = (statistic = round(lb_std.statistic, digits=3), pvalue = round(lb_std.pvalue, digits=4)))
```

The squared raw series gives ``Q^2 = 72.928`` on 10 lags with a p-value of zero, and the squared standardized residuals give 7.722 with a p-value of 0.656. Ljung-Box and ARCH-LM reach the same verdict because they are near-equivalent here: ARCH-LM is a Lagrange-multiplier test of exactly the autocorrelation that ``Q^2`` measures directly. Running both is cheap insurance, since they weight the lags differently and can diverge when the variance dynamics are long-memory.

!!! note "Technical Note"
    Apply `arch_lm_test` to a raw series to decide *whether* to fit a volatility model, and apply both `arch_lm_test` and `ljung_box_squared` to the fitted model object --- which routes to the standardized residuals ``\hat\varepsilon_t/\hat\sigma_t`` --- to check that the specification is adequate. Running either test on the raw returns after fitting merely re-detects the clustering the model has already accounted for.

---

## BDS Independence Test

Ljung-Box and ARCH-LM detect only *linear* dependence or conditional heteroskedasticity of a known form. The Brock-Dechert-Scheinkman-LeBaron (BDS) test (Brock, Dechert, Scheinkman & LeBaron 1996) detects *any* departure from independence and identical distribution --- nonlinear serial dependence, neglected heteroskedasticity, or deterministic chaos --- which makes it the canonical post-ARIMA and post-GARCH adequacy check.

The test compares the correlation integral of the ``m``-dimensional embedding of the series with what independence would imply. For a distance threshold ``\varepsilon`` and the indicator ``\Theta_{ij} = \mathbf{1}(|y_i - y_j| < \varepsilon)``, the correlation integral is

```math
C_m(\varepsilon) = \frac{2}{T_m(T_m-1)} \sum_{s < t} \prod_{k=0}^{m-1} \Theta_{s+k,\, t+k}, \qquad T_m = T - m + 1 ,
```

and the standardized statistic is

```math
w_m = \sqrt{T}\, \frac{C_m(\varepsilon) - C_1(\varepsilon)^m}{\sigma_m(\varepsilon)} \xrightarrow{d} N(0, 1) ,
```

where:
- ``C_1(\varepsilon)`` is the first-order correlation integral over the full sample
- ``\sigma_m(\varepsilon)`` is the asymptotic standard deviation (Brock et al. 1996), a function of ``C_1`` and the triple-overlap probability ``K``
- ``m`` is the embedding dimension and ``\varepsilon = \texttt{eps\_frac} \cdot \operatorname{sd}(y)``

Under ``H_0`` the observations are i.i.d., and a large ``|w_m|`` rejects independence in a two-sided test. The result carries one cell per ``(m, \varepsilon)`` pair.

```@example test_diag
Random.seed!(13)
report(bds_test(randn(400); m=2:3, eps_frac=1.0))
```

On 400 i.i.d. normal draws both statistics are small --- ``w_2 = -1.155`` and ``w_3 = -0.318``, with p-values of 0.248 and 0.750 --- and independence survives, as it must. The threshold ``\varepsilon`` is one sample standard deviation here, which is the middle of the range Brock et al. recommend; too small and the correlation integrals rest on too few close pairs, too large and every pair counts as close and the test loses power.

Deterministic chaos is the clearest demonstration of what linear diagnostics miss:

```@example test_diag
z_chaos = Vector{Float64}(undef, 400); z_chaos[1] = 0.3
for t in 2:400
    z_chaos[t] = 4 * z_chaos[t-1] * (1 - z_chaos[t-1])     # logistic map
end
report(bds_test(z_chaos; m=2:3, eps_frac=0.7))
```

The logistic map is perfectly deterministic yet has essentially zero autocorrelation, so Ljung-Box would find nothing. BDS returns ``w_2 = 475.83`` and ``w_3 = 556.81``, both with p-values below machine precision, because the correlation integrals of a chaotic orbit are nothing like the ``C_1(\varepsilon)^m`` that independence implies. This is exactly the structure BDS exists to catch, and the reason to run it on residuals that have already passed the linear tests.

For fitted models, pass the object: `bds_test(model)` tests ARIMA residuals, and for volatility models it tests the **standardized** residuals, since running it on raw returns would merely re-detect the clustering the model has removed.

```@example test_diag
report(bds_test(g_e; m=2, eps_frac=[1.0, 1.5]))
```

At ``\varepsilon = 1.0\,\text{sd}`` the statistic is 0.386 with a p-value of 0.700, and at ``1.5\,\text{sd}`` it is 0.280 with 0.780, so the fitted GARCH(1,1) leaves no detectable nonlinear structure. Passing a vector to `eps_frac` produces one row per threshold, which is how to check that a conclusion is not an artefact of a single bandwidth --- a rejection at one ``\varepsilon`` and not another usually means the sample is too short rather than that the truth changes with the threshold.

!!! note "Small samples and the bootstrap"
    The asymptotic ``N(0,1)`` approximation is unreliable for ``T < 200``, and a warning is emitted below that threshold. For short series pass `bootstrap=500` or more: each replication permutes the series to impose the i.i.d. null and recomputes ``w_m``, yielding a permutation p-value in `boot_pvalue` that does not rely on the asymptotic distribution.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `m` | `AbstractVector{Int}` | `2:6` | Embedding dimensions; values with ``T - m + 1 < 2`` are dropped |
| `eps_frac` | `Real` or `AbstractVector` | `0.7` | Threshold multipliers of the sample standard deviation |
| `bootstrap` | `Int` | `0` | Permutation replications for an i.i.d.-null p-value |
| `seed` | `Int` | `1234` | RNG seed for the permutation bootstrap |

`BDSResult` stores `m`, `eps`, `eps_frac`, `sd`, the `statistic` and `pvalue` matrices (``|m| \times |\varepsilon|``), `boot_pvalue`, the raw correlation integrals `C`, `nobs`, `small_sample`, `bootstrap`, and `seed`.

---

## EDF Goodness-of-Fit Tests

Empirical-distribution-function tests compare the sample distribution with a hypothesized continuous distribution through the probability-integral transform ``z_{(i)} = F(y_{(i)}; \theta)``. Unlike the moment-based normality suite above, `edf_test` tests fit against any of seven parametric families and reports one of five statistics. It is the standard tool for checking residuals, PIT transforms, or loss series in risk and forecast-evaluation work.

The statistics summarize the gap between the empirical and hypothesized CDFs differently:

```math
\begin{aligned}
D   &= \max_i\left[\max\left(\tfrac{i}{n} - z_{(i)},\; z_{(i)} - \tfrac{i-1}{n}\right)\right] & &\text{(Kolmogorov-Smirnov)}\\
W^2 &= \frac{1}{12n} + \sum_{i=1}^n\left(z_{(i)} - \frac{2i-1}{2n}\right)^2 & &\text{(Cramér-von Mises)}\\
A^2 &= -n - \frac{1}{n}\sum_{i=1}^n (2i-1)\left[\ln z_{(i)} + \ln\!\left(1 - z_{(n+1-i)}\right)\right] & &\text{(Anderson-Darling)}\\
U^2 &= W^2 - n\left(\bar{z} - \tfrac{1}{2}\right)^2 & &\text{(Watson)}
\end{aligned}
```

where:
- ``z_{(i)}`` are the sorted PIT values
- ``n`` is the sample size
- ``\bar{z}`` is the mean of the PIT values

Anderson-Darling weights the tails most heavily and is the recommended default.

!!! warning "Estimated parameters change the null distribution"
    With `params=:specified` the statistics are distribution-free, and the p-values use the Marsaglia-Tsang-Wang exact Kolmogorov CDF for ``n \le 100``, the Marsaglia ADinf distribution for ``A^2``, and asymptotic tables for ``W^2`` and ``U^2``. With `params=:estimate` they are not distribution-free. Only the **normal** family has tabulated null distributions --- the Stephens (1974) modified statistics with D'Agostino & Stephens (1986) closed-form p-values, and the Dallal-Wilkinson (1986) approximation for Lilliefors. Every other family under `params=:estimate` returns `pvalue = NaN` with the reason in the `case` label, rather than a wrong number.

```@example test_diag
Random.seed!(19)
z_norm = rand(Normal(0.5, 2.0), 300)

report(edf_test(z_norm; dist=:normal, test=:ad, params=:estimate))
```

The modified Anderson-Darling statistic ``A^{2*} = 0.2671``, with mean and variance estimated from the data at 0.434 and 1.781, gives a p-value of 0.6875 against a 5% critical value of 0.787 --- normality is not rejected, the right answer for 300 Gaussian draws. The `case` label records Stephens Case 3, both parameters estimated, which is what selects those modified critical values; the unmodified `raw_statistic` of 0.2664 is reported alongside for reference. Testing the same series against a *fully specified* null takes the distribution-free route and gives a very different answer:

```@example test_diag
report(edf_test(z_norm; dist=:normal, test=:ks, params=:specified, theta=(0.0, 1.0)))
```

The Kolmogorov-Smirnov statistic of 0.2370 against ``N(0,1)`` rejects with a p-value below 0.001, more than three times the 1% critical value of 0.094. That is correct: the data are ``N(0.5, 2.0)``, so the specified null has both the location and the scale wrong. The contrast between the two blocks is the whole point of the `params` keyword --- the first asks "is this Gaussian?", the second asks "is this *this particular* Gaussian?", and only the second can detect a calibration failure in a forecast distribution or a risk model.

Non-normal families are supported for the specified route, and for the estimated route wherever a published null table exists:

```@example test_diag
Random.seed!(20)
d_exp = rand(Exponential(1.5), 200)
r_exp = edf_test(d_exp; dist=:exponential, test=:ad, params=:specified, theta=(1.5,))
(statistic = round(r_exp.statistic, digits=4), pvalue = round(r_exp.pvalue, digits=4), case = r_exp.case)
```

The exponential draws give an Anderson-Darling statistic of 0.7459 with a p-value of 0.5217 against the correct rate, so the fit is accepted. The `case` label reads "Case 0 (fully specified)", which is the signal that the distribution-free ADinf p-value applies; had `params=:estimate` been used here the p-value would have come back `NaN` with the reason spelled out, since no published null table covers the estimated-parameter exponential case.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dist` | `Symbol` | `:normal` | `:normal`, `:exponential`, `:logistic`, `:gumbel`, `:gamma`, `:weibull`, `:chisq` |
| `test` | `Symbol` | `:ad` | `:ks`, `:lilliefors`, `:cvm`, `:ad`, `:watson` |
| `params` | `Symbol` | `:estimate` | `:estimate` (ML fit) or `:specified` (supply `theta`) |
| `theta` | `Tuple`/`Vector` | `nothing` | Parameters when `params=:specified` |

`EDFTestResult` stores `test`, `dist`, `params`, `statistic` (the value compared with the critical values), `raw_statistic` (the unmodified EDF statistic), `pvalue`, `nobs`, the fitted or specified `theta`, a `critical_values` dictionary, and a human-readable `case` label.

---

## Variance-Ratio Tests

The variance-ratio test evaluates the **random-walk (martingale) hypothesis** for a level series such as a log price or log exchange rate. Under a random walk the variance of the ``q``-period increment grows linearly in ``q``, so the ratio equals one at every aggregation:

```math
VR(q) = \frac{\operatorname{Var}(y_t - y_{t-q})}{q \, \operatorname{Var}(y_t - y_{t-1})}.
```

`variance_ratio_test` treats its argument as the **level** series and works internally with the first differences ``x_t = y_t - y_{t-1}``. It reports the overlapping Lo-MacKinlay (1988) estimator with the unbiased normalizer ``m = q(N-q+1)(1-q/N)``, the homoskedastic statistic ``Z(q)`` and the heteroskedasticity-robust ``Z^*(q)`` (both asymptotically ``N(0,1)``), and the Chow-Denning (1993) joint statistic ``\max_q |Z(q)|`` whose p-value comes from the studentized-maximum-modulus complement ``1 - (2\Phi(\cdot) - 1)^m``.

- ``VR(q) > 1`` --- positive autocorrelation in the increments (trending, momentum)
- ``VR(q) < 1`` --- negative autocorrelation (mean reversion)

```@example test_diag
Random.seed!(16)
rw = cumsum(randn(600))                          # a simulated random walk (level series)
report(variance_ratio_test(rw; q=[2, 4, 8, 16]))
```

The ratios range from 1.045 to 1.231 and the joint test does not reject: the Chow-Denning statistic of 1.921 has a p-value of 0.202, so the random-walk null survives. This block is also the cleanest possible illustration of why the joint test exists. The individual ``Z^*(8) = 1.921`` carries a p-value of 0.0548 and picks up a significance star, and a reader scanning the four rows for the smallest p-value would report a near-rejection. The Chow-Denning statistic is the same 1.921 read against the studentized-maximum-modulus distribution instead of the normal, and once the search over four aggregations is priced in, the evidence disappears.

### Wright rank/sign and wild-bootstrap variants

`method=:wright` adds Wright's (2000) rank (``R1``, ``R2``) and sign (``S1``) statistics, whose exact i.i.d.-null distributions are simulated on demand and cached. These are robust to non-normality and often more powerful in small samples. `bootstrap=B` adds Kim's (2006) wild-bootstrap p-values for ``Z^*(q)`` and for the Chow-Denning statistic.

```@example test_diag
Random.seed!(17)
ar1 = zeros(800)
for t in 2:800
    ar1[t] = 0.5 * ar1[t-1] + randn()            # mean-reverting level: not a random walk
end
vr_ar = variance_ratio_test(ar1; q=[2, 4, 8], method=:wright, bootstrap=299)
(vr = round.(vr_ar.vr, digits=3),
 cd_star = round(vr_ar.cd_star_stat, digits=3),
 cd_boot_p = round(vr_ar.cd_boot_pvalue, digits=4))
```

The ratios fall monotonically --- 0.801, 0.490, 0.261 --- exactly the signature of mean reversion, and the robust Chow-Denning statistic of 7.016 with a wild-bootstrap p-value of 0.0033 rejects the random walk decisively. The monotone decline in ``VR(q)`` is more informative than the rejection itself: it says the deviation is a stationary autoregressive component rather than a one-off level shift, which would leave the ratios flat across ``q``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `q` | `Vector{Int}` | `[2,4,8,16]` | Aggregation values, each with ``2 \le q < N`` |
| `method` | `Symbol` | `:lomackinlay` | `:lomackinlay`, or `:wright` to add rank and sign statistics |
| `bootstrap` | `Int` | `0` | Kim (2006) wild-bootstrap replications |
| `robust` | `Bool` | `true` | Report the heteroskedasticity-robust branch as primary |
| `boot_weights` | `Symbol` | `:rademacher` | `:rademacher` or `:normal` wild-bootstrap weights |
| `seed` | `Int` | `1234` | RNG seed for the wild bootstrap |

`VarianceRatioResult` stores `q`, `vr`, the `z`/`z_star` statistics with their p-values, the joint `cd_stat`/`cd_star_stat` and their p-values, the Wright `R1`/`R2`/`S1` vectors with p-values, the bootstrap p-values `z_star_boot_pvalue` and `cd_boot_pvalue`, and `nobs`.

---

## Equality and Rank-Correlation Tests

This battery compares the location, scale, or distribution of a response across the groups of a classifier, and measures the association between two series --- the EViews *Equality Tests by Classification* and *Correlations* dialogs. `equality_test(y, g; test=...)` groups `y` by the distinct values of `g`; `cor_test(x, y; method=...)` returns a rank or product-moment correlation. Both also dispatch on `CrossSectionData` and `PanelData` via column symbols, and `ttest` provides the one-sample and paired forms that a classifier cannot express.

**Location tests**: two-sample and paired ``t`` (pooled or Welch), one-way ANOVA (classic ``F`` or Welch), Mann-Whitney ``U``, Wilcoxon signed-rank, Kruskal-Wallis ``H``, van der Waerden normal scores, and the Mood median ``\chi^2``. **Scale tests**: two-group variance ``F``, Bartlett, Levene (centred at the group mean), Brown-Forsythe (centred at the group median), and Siegel-Tukey. The rank tests use the exact null for small tie-free samples and otherwise a continuity- and tie-corrected normal approximation, matching R's `wilcox.test` and `kruskal.test`.

!!! note "Grouped data versus regression residuals"
    The Levene and Brown-Forsythe tests here operate on **raw data split by a classifier**. For the regression-residual heteroskedasticity variants, which work from the deviations of fitted residuals, use `white_test` and `breusch_pagan_test` on [Linear Regression](@ref regression_page).

```@example test_diag
g1 = [5.1, 4.9, 6.2, 5.7, 6.0, 5.5]
g2 = [6.1, 5.9, 7.2, 6.8, 6.5]
g3 = [7.0, 7.5, 6.9, 8.1, 7.3, 7.8]
y_grp = vcat(g1, g2, g3)
grp = vcat(fill(1, 6), fill(2, 5), fill(3, 6))

report(anova_test(y_grp, grp))
```

The classic equal-variance ``F`` of 21.2302 on ``(2, 14)`` degrees of freedom gives a p-value below 0.001, so the three group means differ. `anova_test` is a thin wrapper for `equality_test(y, g; test=:anova)`; `equal_var=false` switches to Welch's unequal-variance ``F``, which is the safer default whenever the group variances are not known to match.

```@example test_diag
report(equality_test(y_grp, grp; test=:kruskal_wallis))
```

The tie-corrected Kruskal-Wallis ``H`` of 12.1712 on 2 degrees of freedom gives 0.0023 and reaches the same conclusion without assuming normality. Agreement between the parametric and rank tests, as here, means the ANOVA rejection is not driven by an outlier or by skew --- the check worth running whenever a group has fewer than ten observations.

```@example test_diag
report(equality_test(y_grp, grp; test=:brown_forsythe))
```

Brown-Forsythe returns 0.0338 with a p-value of 0.9668, so the group dispersions are indistinguishable and the equal-variance ``F`` above was the right choice. Bartlett's test (`test=:bartlett`) is more powerful when the data really are normal but is notoriously sensitive to departures from it; Brown-Forsythe, centred at the group median, is the robust alternative and the one to reach for by default.

```@example test_diag
y_12 = vcat(g1, g2); grp_12 = vcat(fill(1, 6), fill(2, 5))
report(equality_test(y_12, grp_12; test=:mann_whitney))
```

Restricted to the first two groups, the Mann-Whitney ``U`` of 3.0 gives a p-value of 0.0303 and rejects equality of distributions at 5%. With six and five observations and no ties, the exact null is used rather than the normal approximation, which matters at these sample sizes: the asymptotic p-value would overstate the significance. The two-sample, paired, variance-ratio, and Siegel-Tukey tests all require exactly two groups and raise an `ArgumentError` otherwise.

Association between two series is measured by `cor_test`, whose Kendall statistic counts concordant minus discordant pairs in ``O(n \log n)`` with a merge-sort inversion counter --- ``\tau_a`` under the exact null for small tie-free samples, and the tie-adjusted ``\tau_b`` otherwise:

```@example test_diag
x_cor = [10.0, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
z_cor = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
report(cor_test(x_cor, z_cor; method=:pearson))
```

```@example test_diag
report(cor_test(x_cor, z_cor; method=:kendall))
```

Pearson gives ``r = 0.8164`` with ``t = 4.24``, a p-value of 0.0022, and a 95% interval of ``[0.424, 0.951]``; Kendall gives ``\tau_a = 0.6364`` with an exact p-value of 0.0057. Spearman (`method=:spearman`) sits between them. The three answer different questions --- Pearson measures linear association and assumes it, Spearman measures monotone association on ranks, Kendall measures the probability that two randomly chosen pairs are concordant --- so a large gap between Pearson and the rank measures signals nonlinearity or a single influential point, not a contradiction.

---

## Model Comparison Tests

The classical trinity --- Wald, likelihood ratio, and Lagrange multiplier --- gives asymptotically equivalent tests of nested hypotheses. This package implements the LR and LM tests with automatic detection of which model is restricted.

The **likelihood ratio test** (Wilks 1938) compares maximized log-likelihoods:

```math
LR = -2(\ell_R - \ell_U) \sim \chi^2(df)
```

where:
- ``\ell_R`` is the maximized log-likelihood of the restricted model
- ``\ell_U`` is the maximized log-likelihood of the unrestricted model
- ``df = k_U - k_R`` is the difference in the number of parameters

The **Lagrange multiplier test** (Rao 1948) evaluates the score of the unrestricted log-likelihood at the restricted estimates:

```math
LM = s' (-H)^{-1} s \sim \chi^2(df)
```

where:
- ``s`` is the score vector at the restricted estimates
- ``H`` is the Hessian of the negative log-likelihood at those estimates

!!! note "Technical Note"
    In the linear regression model with normal errors the three statistics satisfy ``W \geq LR \geq LM`` (Berndt & Savin 1977). The ordering is not guaranteed outside that setting, and it need not hold at all when the score and Hessian are obtained numerically, as they are here by central differences. LR needs both models estimated, LM needs only the restricted model plus the unrestricted parameterization for the score, and Wald needs only the unrestricted model.

`lr_test` works for any pair implementing `loglikelihood`, `dof`, and `nobs` from `StatsAPI`. `lm_test` requires same-family nesting, because it must embed the restricted parameters into the unrestricted parameter space.

| Test | Supported pairs | Notes |
|------|-----------------|-------|
| `lr_test` | Any pair with `loglikelihood`, `dof`, `nobs` | Generic; detects restricted vs. unrestricted automatically |
| `lm_test` | ARIMA × ARIMA | Same differencing order ``d`` required |
| `lm_test` | VAR × VAR | Same data matrix, different lag orders |
| `lm_test` | ARCH × ARCH, GARCH × GARCH | Same family, different orders |
| `lm_test` | ARCH × GARCH | Cross-type nesting (ARCH is GARCH with ``p = 0``) |
| `lm_test` | EGARCH × EGARCH, GJR × GJR | Same family, different orders |

```@example test_diag
Random.seed!(18)
y_mc = cumsum(randn(200))
ar2 = estimate_ar(diff(y_mc), 2; method=:mle)
ar4 = estimate_ar(diff(y_mc), 4; method=:mle)

report(lr_test(ar2, ar4))
```

The LR statistic of 7.2828 on 2 degrees of freedom gives a p-value of 0.0262, so the AR(2) is rejected against the AR(4) at the 5% level. The two maximized log-likelihoods, ``-279.41`` and ``-275.77``, differ by only 3.64 --- a reminder of how little separation a rejection at this level represents. The arguments may be passed in either order, since the function identifies the restricted model as the one with fewer parameters.

```@example test_diag
report(lm_test(ar2, ar4))
```

The LM statistic of 7.7057 gives a p-value of 0.0212 and agrees with LR on the decision. It also exceeds the LR statistic of 7.2828, which the Berndt-Savin inequality would forbid in the linear model with normal errors; here the score and Hessian come from central differences of an ARMA likelihood, so a discrepancy of this size is a numerical-differentiation artefact rather than evidence about the models. When the two diverge materially, prefer LR, which uses both maximized likelihoods, over LM, which extrapolates from one.

`lr_test` handles the VAR case identically:

```@example test_diag
report(lr_test(estimate_var(Y, 2), estimate_var(Y, 3)))
```

The LR statistic of 42.8681 on 9 degrees of freedom rejects VAR(2) against VAR(3) with a p-value below 0.001, so the third lag carries information the first two do not. Sequential LR testing of this kind over-selects in large samples relative to BIC, because its critical value does not grow with ``T``; use it as a complement to the information criteria reported by `estimate_var`, not as a substitute.

**`LRTestResult` fields**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LR statistic ``-2(\ell_R - \ell_U)`` |
| `pvalue` | `T` | P-value from ``\chi^2(df)`` |
| `df` | `Int` | Degrees of freedom ``k_U - k_R`` |
| `loglik_restricted` | `T` | Log-likelihood of the restricted model |
| `loglik_unrestricted` | `T` | Log-likelihood of the unrestricted model |
| `dof_restricted` | `Int` | Parameters in the restricted model |
| `dof_unrestricted` | `Int` | Parameters in the unrestricted model |
| `nobs_restricted` | `Int` | Observations in the restricted model |
| `nobs_unrestricted` | `Int` | Observations in the unrestricted model |

**`LMTestResult` fields**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LM statistic ``s'(-H)^{-1}s`` |
| `pvalue` | `T` | P-value from ``\chi^2(df)`` |
| `df` | `Int` | Degrees of freedom ``k_U - k_R`` |
| `nobs` | `Int` | Number of observations |
| `score_norm` | `T` | Euclidean norm of the score vector |

---

## Complete Example

A full post-estimation diagnostic pass on the FRED-MD VAR(2): stability, causal structure, residual distribution, residual dependence, and lag order.

```@example test_diag
show(stdout, is_stationary(m))
```

```@example test_diag
# Standardize each residual before the variance diagnostics — see Pitfall 4
u_std = [m.U[:, j] ./ std(m.U[:, j]) for j in 1:3]

[(variable = v,
  arch_lm_pvalue = round(arch_lm_test(u_std[j], 5).pvalue, digits=4),
  bds_min_pvalue = round(minimum(bds_test(u_std[j]; m=2:3, eps_frac=1.0).pvalue), digits=4))
 for (j, v) in enumerate(names_var)]
```

```@example test_diag
report(lr_test(m, estimate_var(Y, 3)))
```

The VAR(2) is stable with a maximum companion modulus of 0.497, and the Granger screen above showed the funds rate predicting industrial production while industrial production does not predict inflation --- the asymmetry a monetary VAR is built to exploit. The residual diagnostics are the sobering part: all three equations reject no-ARCH and reject independence at any level, so the residuals are neither homoskedastic nor i.i.d. even though the conditional mean is correctly specified in the linear sense. Bootstrap or heteroskedasticity-robust inference is mandatory here, and the LR test's rejection of VAR(2) against VAR(3) at 42.87 should be read with that in mind, since the likelihood it maximizes assumes exactly the Gaussian homoskedastic errors the diagnostics have ruled out.

---

## Common Pitfalls

1. **Reading Granger causality as causation.** Granger causality is predictive and conditional on the information set in the VAR. It can arise from an omitted common cause, from the timing of measurement, or from temporal aggregation. Adding a fourth variable to a three-variable system routinely reverses a Granger conclusion, which is the clearest evidence that it carries no structural content.

2. **Treating a normality rejection as fatal.** OLS coefficients in a VAR stay consistent and asymptotically normal whatever the residual distribution. A rejection says that *exact finite-sample* inference --- t-tests, F-tests, and normal-quantile confidence intervals --- is unreliable, not that the model is wrong. The remedy is robust standard errors or bootstrap intervals, not a different estimator.

3. **Testing raw and standardized residuals interchangeably.** Apply `arch_lm_test` to a raw series to decide whether a volatility model is needed; apply it to the fitted model object afterwards, which routes to the standardized residuals, to check that the model is adequate. Passing the raw returns after fitting simply re-detects the clustering the model already removed, and passing the standardized residuals before fitting is not defined.

4. **Feeding a very small-scale series to `arch_lm_test`.** The auxiliary regression puts an intercept column of ones next to columns of squared residuals. When the series has a standard deviation of order ``10^{-3}`` --- routine for a twice-differenced price index --- those columns differ by twelve orders of magnitude, ``X'X`` becomes numerically singular, and the reported ``T R^2`` collapses toward zero with a p-value near 1. Divide by the sample standard deviation first, as the Complete Example does: on the CPIAUCSL residuals the raw call returns a p-value of exactly 1.0 while the standardized call rejects at any level.

5. **Feeding returns to `variance_ratio_test`.** The function expects the **level** series --- a log price, not a log return --- and differences it internally. Passing returns tests whether the *returns* follow a random walk, which is a different and usually uninteresting hypothesis, and will produce ratios far below one for any stationary return series.

6. **Trusting an estimated-parameter EDF p-value outside the normal family.** Only the normal family has tabulated null distributions for `params=:estimate`. Every other family returns `pvalue = NaN` and records the reason in `case`; that is deliberate, and an `NaN` there must not be read as a non-rejection. Either specify the parameters or use a parametric bootstrap.

7. **Running BDS on short series without the bootstrap.** The asymptotic ``N(0,1)`` approximation is unreliable below ``T = 200``, where the test is badly over-sized. A warning is emitted, but the statistic is still returned; pass `bootstrap=500` or more to get a permutation p-value that does not depend on the asymptotics.

8. **Using `lm_test` across model families.** The Lagrange multiplier test embeds the restricted parameters into the unrestricted parameter space, which is family-specific: an ARIMA model cannot be compared with a GARCH model this way. Use `lr_test`, which needs only `loglikelihood`, `dof`, and `nobs`, for cross-family comparisons.

---

## References

- Anderson, T. W., & Darling, D. A. (1954). A test of goodness of fit. *Journal of the American Statistical Association*, 49(268), 765-769. [DOI](https://doi.org/10.1080/01621459.1954.10501232)

- Bartlett, M. S. (1937). Properties of sufficiency and statistical tests. *Proceedings of the Royal Society A*, 160(901), 268-282. [DOI](https://doi.org/10.1098/rspa.1937.0109)

- Berndt, E. R., & Savin, N. E. (1977). Conflict among criteria for testing hypotheses in the multivariate linear regression model. *Econometrica*, 45(5), 1263-1277. [DOI](https://doi.org/10.2307/1914072)

- Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B. (1996). A test for independence based on the correlation dimension. *Econometric Reviews*, 15(3), 197-235. [DOI](https://doi.org/10.1080/07474939608800353)

- Brock, W. A., Hsieh, D. A., & LeBaron, B. (1991). *Nonlinear Dynamics, Chaos, and Instability: Statistical Theory and Economic Evidence*. Cambridge, MA: MIT Press. ISBN 978-0-262-02329-0.

- Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the equality of variances. *Journal of the American Statistical Association*, 69(346), 364-367. [DOI](https://doi.org/10.1080/01621459.1974.10482955)

- Chow, K. V., & Denning, K. C. (1993). A simple multiple variance ratio test. *Journal of Econometrics*, 58(3), 385-401. [DOI](https://doi.org/10.1016/0304-4076(93)90051-6)

- D'Agostino, R. B., & Stephens, M. A. (1986). *Goodness-of-Fit Techniques*. New York: Marcel Dekker. ISBN 978-0-8247-7487-5.

- Dallal, G. E., & Wilkinson, L. (1986). An analytic approximation to the distribution of Lilliefors's test statistic for normality. *The American Statistician*, 40(4), 294-296. [DOI](https://doi.org/10.1080/00031305.1986.10475419)

- Doornik, J. A., & Hansen, H. (2008). An omnibus test for univariate and multivariate normality. *Oxford Bulletin of Economics and Statistics*, 70(s1), 927-939. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007. [DOI](https://doi.org/10.2307/1912773)

- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438. [DOI](https://doi.org/10.2307/1912791)

- Henze, N., & Zirkler, B. (1990). A class of invariant consistent tests for multivariate normality. *Communications in Statistics --- Theory and Methods*, 19(10), 3595-3617. [DOI](https://doi.org/10.1080/03610929008830400)

- Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters*, 6(3), 255-259. [DOI](https://doi.org/10.1016/0165-1765(80)90024-5)

- Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1/2), 81-93. [DOI](https://doi.org/10.1093/biomet/30.1-2.81)

- Kim, J. H. (2006). Wild bootstrapping variance ratio tests. *Economics Letters*, 92(1), 38-43. [DOI](https://doi.org/10.1016/j.econlet.2006.01.007)

- Knight, W. R. (1966). A computer method for calculating Kendall's tau with ungrouped data. *Journal of the American Statistical Association*, 61(314), 436-439. [DOI](https://doi.org/10.1080/01621459.1966.10480879)

- Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583-621. [DOI](https://doi.org/10.1080/01621459.1952.10483441)

- Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. *Journal of the American Statistical Association*, 62(318), 399-402. [DOI](https://doi.org/10.1080/01621459.1967.10482916)

- Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303. [DOI](https://doi.org/10.1093/biomet/65.2.297)

- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *Review of Financial Studies*, 1(1), 41-66. [DOI](https://doi.org/10.1093/rfs/1.1.41)

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics*, 18(1), 50-60. [DOI](https://doi.org/10.1214/aoms/1177730491)

- Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with applications. *Biometrika*, 57(3), 519-530. [DOI](https://doi.org/10.1093/biomet/57.3.519)

- Marsaglia, G., Tsang, W. W., & Wang, J. (2003). Evaluating Kolmogorov's distribution. *Journal of Statistical Software*, 8(18), 1-4. [DOI](https://doi.org/10.18637/jss.v008.i18)

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)

- Rao, C. R. (1948). Large sample tests of statistical hypotheses concerning several parameters with applications to problems of estimation. *Mathematical Proceedings of the Cambridge Philosophical Society*, 44(1), 50-57. [DOI](https://doi.org/10.1017/S0305004100023987)

- Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72-101. [DOI](https://doi.org/10.2307/1412159)

- Stephens, M. A. (1974). EDF statistics for goodness of fit and some comparisons. *Journal of the American Statistical Association*, 69(347), 730-737. [DOI](https://doi.org/10.1080/01621459.1974.10480196)

- van der Waerden, B. L. (1952). Order tests for the two-sample problem and their power. *Indagationes Mathematicae*, 14, 453-458.

- Watson, G. S. (1961). Goodness-of-fit tests on a circle. *Biometrika*, 48(1/2), 109-114. [DOI](https://doi.org/10.1093/biomet/48.1-2.109)

- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83. [DOI](https://doi.org/10.2307/3001968)

- Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *Annals of Mathematical Statistics*, 9(1), 60-62. [DOI](https://doi.org/10.1214/aoms/1177732360)

- Wright, J. H. (2000). Alternative variance-ratio tests using ranks and signs. *Journal of Business & Economic Statistics*, 18(1), 1-9. [DOI](https://doi.org/10.1080/07350015.2000.10524842)
