# [VAR](@id var_page)

**MacroEconometricModels.jl** provides a complete implementation of Vector Autoregression (VAR) models, from reduced-form estimation through structural identification and robust inference. The VAR framework, introduced by Sims (1980), remains the workhorse of empirical macroeconomics for analyzing the dynamic interactions among multiple time series.

- **Estimation**: OLS estimation of reduced-form VAR(p) with automatic information criteria (AIC, BIC, HQIC) and stability checking
- **Lag Selection**: Data-driven lag order selection via AIC, BIC, or HQIC minimization
- **Structural Identification**: Six methods --- Cholesky (recursive), sign restrictions, narrative restrictions, long-run (Blanchard-Quah), Arias et al. (2018) zero + sign, and Mountford-Uhlig (2009) penalty function
- **Robust Inference**: Newey-West HAC, White heteroscedasticity-robust (HC0), and Driscoll-Kraay panel-robust covariance estimators
- **Innovation Accounting**: IRF, FEVD, and historical decomposition with bootstrap or asymptotic confidence intervals; see [Innovation Accounting](@ref innovation_accounting_page)
- **Forecasting**: Multi-step ahead point forecasts, bootstrap prediction intervals, and Waggoner-Zha conditional forecasts

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization. Bayesian estimation of the same model is documented on the [Bayesian VAR](@ref bvar_page) page; cointegrated systems belong in a [VECM](@ref vecm_page).

The examples throughout use three FRED-MD series under their official transformation codes: `INDPRO` (`tcode=5`, log difference) and `CPIAUCSL` (`tcode=6`, second log difference), both rescaled to percent, together with `FEDFUNDS` (`tcode=2`, first difference), already in percentage points. All three are therefore measured on a common percent scale.

```@setup var
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y[:, 1:2] .*= 100          # log differences to percent; FEDFUNDS is already in percentage points
Y = Y[end-59:end, :]
```

## Quick Start

**Recipe 1: Estimate VAR(p)**

```@example var
model = estimate_var(Y, 4)
report(model)
```

**Recipe 2: Lag selection**

```@example var
# Select lag order minimizing BIC (default)
p_bic = select_lag_order(Y, 4)

# Select via AIC
p_aic = select_lag_order(Y, 4; criterion=:aic)

model = estimate_var(Y, p_bic)
report(model)
```

**Recipe 3: Cholesky IRF**

```@example var
model = estimate_var(Y, 4)

# Cholesky IRF with bootstrap confidence intervals
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
report(result)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/quickstart_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 4: Sign restrictions**

```@example var
model = estimate_var(Y, 4)

# Contractionary monetary shock: FFR rises on impact
check = ir -> ir[1, 3, 3] > 0
result = irf(model, 20; method=:sign, check_func=check)
report(result)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_sign.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 5: Arias identification**

```@example var
model_short = estimate_var(Y, 2)

# Sign restrictions on the monetary policy shock (shock 3)
restrictions = SVARRestrictions(3;
    signs = [sign_restriction(3, 3, :positive),          # FFR rises
             sign_restriction(1, 1, :positive)]          # Output rises to demand shock
)
result = identify_arias(model_short, restrictions, 20; n_draws=500)
report(result)
```

**Recipe 6: Uhlig identification**

```@example var
model_short = estimate_var(Y, 2)

# Mountford-Uhlig penalty function: one optimal rotation
restrictions = SVARRestrictions(3;
    signs = [sign_restriction(1, 1, :positive),     # Fiscal shock raises INDPRO
             sign_restriction(3, 3, :positive)]     # Monetary shock raises FFR
)
result = identify_uhlig(model_short, restrictions, 20)
report(result)
```

---

## Reduced-Form VAR

### The Model

A **VAR(p)** model for an ``n``-dimensional vector of endogenous variables ``y_t`` is:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables at time ``t``
- ``c`` is the ``n \times 1`` vector of intercepts
- ``A_i`` is the ``n \times n`` coefficient matrix for lag ``i = 1, \ldots, p``
- ``u_t`` is the ``n \times 1`` vector of reduced-form innovations with ``E[u_t] = 0`` and ``E[u_t u_t'] = \Sigma``

### OLS Estimation

Stack the observations into matrices. Let ``T`` denote the total sample size and define the effective sample as ``T_{\text{eff}} = T - p`` observations after accounting for lags:

```math
Y = \begin{bmatrix} y_{p+1}' \\ y_{p+2}' \\ \vdots \\ y_T' \end{bmatrix}_{T_{\text{eff}} \times n}, \quad
X = \begin{bmatrix} 1 & y_p' & y_{p-1}' & \cdots & y_1' \\
1 & y_{p+1}' & y_p' & \cdots & y_2' \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & y_{T-1}' & y_{T-2}' & \cdots & y_{T-p}' \end{bmatrix}_{T_{\text{eff}} \times (1+np)}
```

where:
- ``Y`` is the ``T_{\text{eff}} \times n`` matrix of dependent variables
- ``X`` is the ``T_{\text{eff}} \times k`` matrix of regressors with ``k = 1 + np``

The compact form ``Y = XB + U`` yields the OLS estimator:

```math
\hat{B} = (X'X)^{-1} X'Y
```

where:
- ``\hat{B}`` is the ``k \times n`` coefficient matrix ``[c, A_1, \ldots, A_p]'``

The `Sigma` field holds the maximum-likelihood residual covariance:

```math
\hat{\Sigma} = \frac{1}{T_{\text{eff}}} \hat{U}'\hat{U}
```

where:
- ``\hat{U} = Y - X\hat{B}`` is the ``T_{\text{eff}} \times n`` residual matrix

!!! note "Two residual covariances"
    `model.Sigma` is the ML estimator with denominator ``T_{\text{eff}}``, which is what the
    information criteria, the Gaussian log-likelihood, and the impulse responses consume.
    `vcov(model)` instead builds ``\hat{\Sigma}_{\text{dof}} \otimes (X'X)^{-1}`` from the
    small-sample estimator ``\hat{U}'\hat{U} / (T_{\text{eff}} - k)``, so coefficient standard
    errors carry the degrees-of-freedom correction while ``\Sigma`` does not.

```@example var
model = estimate_var(Y, 4; varnames=["INDPRO", "CPI", "FFR"])
report(model)
```

The `report` output displays the VAR specification (number of variables, lags, observations) alongside the AIC, BIC, and HQIC values. The coefficient matrix `model.B` stores the intercept in row 1, followed by ``A_1, A_2, \ldots, A_p`` stacked vertically. To extract lag-``i`` coefficients for an ``n``-variable system: `A_i = model.B[(i-1)*n+2 : i*n+1, :]'`. On this 60-observation sample the residual standard deviations are 0.67 percent for industrial production, 0.22 percent for inflation, and 0.086 percentage points for the funds rate --- the policy rate is by far the least volatile of the three at a monthly frequency.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `check_stability` | `Bool` | `true` | Warn if estimated VAR is non-stationary |
| `varnames` | `Vector{String}` | `nothing` | Variable display names (default: `y1`, `y2`, ...) |

### Return Value

`estimate_var` returns a `VARModel{T}` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original ``T \times n`` data matrix |
| `p` | `Int` | Number of lags |
| `B` | `Matrix{T}` | ``(1+np) \times n`` coefficient matrix ``[c, A_1, \ldots, A_p]'`` |
| `U` | `Matrix{T}` | ``T_{\text{eff}} \times n`` residual matrix |
| `Sigma` | `Matrix{T}` | ``n \times n`` residual covariance matrix |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `hqic` | `T` | Hannan-Quinn Information Criterion |
| `varnames` | `Vector{String}` | Variable display names |

---

## Stability and Lag Selection

### Companion Form and Stability

A VAR(p) is **stable** (stationary) if all eigenvalues of the companion matrix ``F`` lie inside the unit circle. The companion form rewrites the VAR(p) as a VAR(1) in the ``np``-dimensional state vector:

```math
F = \begin{bmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_n & 0 & \cdots & 0 & 0 \\
0 & I_n & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_n & 0
\end{bmatrix}_{np \times np}
```

where:
- ``A_i`` is the ``n \times n`` VAR coefficient matrix for lag ``i``
- ``I_n`` is the ``n \times n`` identity matrix
- ``F`` is the ``np \times np`` companion matrix

The stability condition requires ``|\lambda_i| < 1`` for all eigenvalues ``\lambda_i`` of ``F``. The function `is_stationary` checks this condition and returns the companion matrix eigenvalues:

```@example var
model = estimate_var(Y, 4)
stab = is_stationary(model)
stab
```

`is_stationary` returns a `VARStationarityResult`, not a `Bool`: `stab.is_stationary` is the verdict, `stab.max_modulus` the largest eigenvalue modulus, `stab.eigenvalues` the full companion spectrum, and `stab.companion_matrix` the ``np \times np`` matrix itself. Here the largest modulus is 0.83, comfortably inside the unit circle. Values near 1.0 indicate near-unit-root behavior and suggest the system requires differencing or a [VECM](@ref vecm_page) specification.

### Information Criteria

The lag length minimizes an information criterion that balances fit against model complexity. Each criterion adds a penalty in the number of regressors per equation ``k = 1 + np`` to the log determinant of the ML residual covariance:

```math
\text{AIC}(p) = \log|\hat{\Sigma}_p| + \frac{2k}{T_{\text{eff}}}, \qquad
\text{BIC}(p) = \log|\hat{\Sigma}_p| + \frac{k \log T_{\text{eff}}}{T_{\text{eff}}}, \qquad
\text{HQ}(p) = \log|\hat{\Sigma}_p| + \frac{2k \log(\log T_{\text{eff}})}{T_{\text{eff}}}
```

where:
- ``\hat{\Sigma}_p`` is the ML residual covariance at lag order ``p``
- ``T_{\text{eff}} = T - p`` is the effective sample size
- ``k = n(1 + np)`` is the **system** parameter count: ``n`` equations of ``1 + np`` regressors each
- ``n`` is the number of endogenous variables

AIC tends to overfit in finite samples; BIC penalizes complexity more heavily as ``T`` grows; HQIC sits between them. All three are stored on the fitted model as `aic`, `bic`, and `hqic`.

!!! note "System penalty and a common estimation sample"
    Two conventions make the criteria comparable across ``p``, both following Lütkepohl (2005 §4.3)
    and Stata's `varsoc`. The penalty counts the ``n(1 + np)`` system-wide parameters rather than
    the ``1 + np`` regressors of a single equation, which is a factor of ``n`` heavier. And
    `select_lag_order` estimates every candidate on the common sample ``t = \text{max\_p}+1, \ldots, T``,
    so each ``\log|\hat\Sigma_p|`` is computed on identical data. Per-candidate ragged samples
    combined with the lighter penalty pushed the selection to `max_p` on short samples.

```@example var
# BIC-optimal lag order (default)
p_bic = select_lag_order(Y, 4)

# AIC-optimal lag order
p_aic = select_lag_order(Y, 4; criterion=:aic)

model = estimate_var(Y, p_bic)
report(model)
```

`select_lag_order` evaluates every lag order from 1 to `max_p` and returns the integer minimizing the chosen criterion. The two criteria disagree on this sample: BIC selects ``\hat{p} = 1`` while AIC selects ``\hat{p} = 4``. The log determinant falls steadily with ``p`` --- from ``-7.57`` at ``p = 1`` to ``-8.86`` at ``p = 4`` --- but the system penalty adds ``n(1+np)`` parameters at every step, which is enough for BIC's ``\log T_{\text{eff}}`` factor to overwhelm the fit gain and not enough for AIC's factor of 2. That gap is the textbook one: AIC overfits in finite samples, and 60 observations on three variables is a finite sample. Prefer the BIC order here, and confirm either choice with `is_stationary`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `criterion` | `Symbol` | `:bic` | Information criterion: `:aic`, `:bic`, or `:hqic` |

---

## Structural VAR and Identification

The reduced-form residuals ``u_t`` are linear combinations of orthogonal structural shocks ``\varepsilon_t``:

```math
u_t = B_0 \varepsilon_t
```

where:
- ``B_0`` is the ``n \times n`` contemporaneous impact matrix
- ``\varepsilon_t`` are structural shocks with ``E[\varepsilon_t \varepsilon_t'] = I_n``

The identifying restriction ``\Sigma = B_0 B_0'`` provides ``n(n+1)/2`` equations for ``n^2`` unknowns, leaving ``n(n-1)/2`` free parameters. Additional restrictions are required to achieve **exact identification**. The package provides six identification strategies.

### Cholesky (Recursive)

The Cholesky decomposition imposes a lower triangular structure on ``B_0``:

```math
B_0 = \text{chol}(\Sigma)
```

where:
- ``B_0`` is lower triangular, implying variable ``i`` responds contemporaneously only to shocks ``1, \ldots, i``

The ordering reflects economic assumptions about the speed of adjustment. Variables ordered first respond only to their own shocks on impact. In the standard monetary VAR ordering [INDPRO, CPI, FFR], the federal funds rate shock (shock 3) has no contemporaneous effect on output or prices, consistent with the information and implementation lags in monetary policy transmission (Christiano, Eichenbaum & Evans 1999).

```@example var
model = estimate_var(Y, 4)

# Cholesky IRF with bootstrap 90% CI
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50, conf_level=0.90)
report(result)
```

A one-standard-deviation policy shock raises the funds rate by 0.082 percentage points on impact. Industrial production and prices are zero on impact by construction, then turn negative: output reaches a trough of ``-0.068`` percent at ``h = 5`` and inflation falls 0.046 percent at ``h = 2``. Both responses are economically small and statistically indistinguishable from zero --- the 90% band for the output response at ``h = 3`` runs from ``-0.22`` to ``0.07`` percent, which is what 56 usable observations buy.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification scheme; see [Structural Identification](@ref structural_identification_page) for the full list |
| `ci_type` | `Symbol` | `:none` | `:none`, `:bootstrap` (residual bootstrap), or `:theoretical` (asymptotic delta method) |
| `reps` | `Int` | `200` | Replications used for the bands |
| `conf_level` | `Real` | `0.95` | Band coverage |
| `stationary_only` | `Bool` | `false` | Reject and redraw replications whose re-estimated companion matrix is explosive |
| `check_func` | `Function` | `nothing` | Sign-restriction predicate (`method=:sign`) |
| `narrative_check` | `Function` | `nothing` | Narrative-restriction predicate (`method=:narrative`) |
| `seed` | `Integer` | `nothing` | Seed owning the bootstrap RNG; makes bands reproducible bit-for-bit |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator, when no `seed` is given |

### Bootstrap Schemes and Bias Correction

The residual bootstrap resamples rows i.i.d. by default, which is valid only under conditionally homoskedastic, serially independent errors. Two alternatives relax that, and Kilian's (1998) bias correction addresses the small-sample bias of the OLS coefficients:

```@example var
iid   = irf(model, 20; ci_type=:bootstrap, reps=50, seed=1)
wild  = irf(model, 20; ci_type=:bootstrap, reps=50, bootstrap=:wild, seed=1)
block = irf(model, 20; ci_type=:bootstrap, reps=50, bootstrap=:block, block_length=8, seed=1)
bc    = irf(model, 20; ci_type=:bootstrap, reps=50, bias_correct=true, seed=1)

# 90% band width for the output response to the policy shock at h = 3
[w.ci_upper[3, 1, 3] - w.ci_lower[3, 1, 3] for w in (iid, wild, block, bc)]
```

At a common seed the four schemes disagree by about a quarter of the band width: the wild bootstrap widens the i.i.d. band from 0.292 to 0.313 percentage points, the bias correction gives a similar 0.310, and the moving-block scheme narrows it to 0.250 because concatenating contiguous residual blocks reduces the effective number of independent resampled observations. The spread is the size of the specification choice --- reporting a single scheme without saying which one hides it.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `bootstrap` | `Symbol` | `:iid` | Resampling scheme: `:iid`, `:wild`, `:block` |
| `block_length` | `Int` | `0` | Moving-block length; `0` selects ``\lceil T^{1/3} \rceil`` |
| `wild_dist` | `Symbol` | `:rademacher` | Wild multiplier: `:rademacher` or `:mammen` |
| `bias_correct` | `Bool` | `false` | Kilian (1998) bootstrap-after-bootstrap |
| `bias_reps` | `Int` | `0` | Inner-bootstrap replications; `0` reuses `reps` |

- **Wild** multiplies each residual *row* by a single scalar draw. Because the whole row shares the multiplier, the contemporaneous cross-equation covariance is preserved exactly while the conditional variance is randomised --- the property that makes it robust to conditional heteroskedasticity of unknown form (Gonçalves & Kilian 2004). `:mammen` additionally matches the third moment; `:rademacher` cannot, since its odd moments vanish by symmetry.
- **Block** concatenates contiguous residual blocks, retaining the serial dependence that i.i.d. resampling destroys (Brüggemann, Jentsch & Trenkler 2016).
- **`bias_correct`** runs Kilian's bootstrap-after-bootstrap: an inner bootstrap estimates ``\Psi = E[B^*] - \hat{B}``, the DGP is re-centred at ``\hat{B} - \delta\Psi``, and each outer draw is corrected by the same ``\Psi``.

!!! note "The stationarity shrinkage is not optional"
    Subtracting the raw bias can push the companion matrix outside the unit circle, making the bias-corrected DGP explosive. Kilian's rule is applied: if the uncorrected estimate is already non-stationary no correction is made at all; otherwise ``\delta`` starts at 1 and shrinks by 0.01 until the corrected companion is stable.

    On a persistent AR(1) (``\rho = 0.95``, ``T = 60``) where OLS is badly downward-biased, the correction cuts the mean bias by roughly three quarters and lowers RMSE.

`bootstrap=:iid` with `bias_correct=false` reproduces the previous bands bit-for-bit, so existing results are unchanged. All bands are reproducible at a fixed `seed`.

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Sign Restrictions

Sign restrictions identify structural shocks by constraining the signs of impulse responses at selected horizons, following Rubio-Ramírez, Waggoner & Zha (2010). The algorithm draws random orthogonal matrices ``Q`` from the Haar measure and retains only those producing IRFs consistent with the sign constraints:

1. Compute the Cholesky factor: ``P = \text{chol}(\Sigma)``
2. Draw ``Q`` uniformly from ``O(n)`` via QR decomposition of a random matrix
3. Compute the candidate impact matrix: ``B_0 = PQ``
4. Compute IRFs ``\Theta_0 = B_0, \Theta_1, \ldots`` from the candidate ``B_0``
5. Accept if all sign conditions hold; otherwise discard and repeat

By default `identify_sign` stops at the first rotation that clears the check and returns the pair `(Q, irf)`. With `store_all=true` it instead exhausts `max_draws` and returns a `SignIdentifiedSet` holding every accepted rotation and its IRF, which is what characterizing the full identified set requires (Baumeister & Hamilton 2015). `irf_median` and `irf_bounds` then summarize that set pointwise.

```@example var
model = estimate_var(Y, 4)

# Contractionary monetary shock: FFR rises, INDPRO and CPI fall
check = ir -> ir[1, 3, 3] > 0 && ir[1, 1, 3] < 0 && ir[1, 2, 3] < 0

# Full identified set
id_set = identify_sign(model, 20, check; max_draws=5000, store_all=true)
id_set
```

```@example var
# Pointwise median and 68% bands over the identified set
med = irf_median(id_set)
lower, upper = irf_bounds(id_set; quantiles=[0.16, 0.84])
round.([med[1, 1, 3] lower[1, 1, 3] upper[1, 1, 3]], digits=4)
```

```julia
plot_result(id_set)
```

```@raw html
<iframe src="../assets/plots/svar_setid_band.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

476 of 5000 rotations satisfy all three impact conditions, an acceptance rate of 9.5%. Across that set the median impact response of industrial production to the policy shock is ``-0.27`` percent, with a 68% interval of ``[-0.50, -0.08]`` that excludes zero --- unsurprising, since the sign restriction imposes the negative sign directly. The response decays quickly: by ``h = 6`` the median is ``-0.03`` percent. Rates below 1% suggest the restrictions are overly stringent or nearly contradictory; `irf_bounds` and `irf_median` summarize the set without collapsing it to a single rotation.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_draws` | `Int` | `1000` | Rotation draws attempted |
| `store_all` | `Bool` | `false` | Return the full `SignIdentifiedSet` instead of the first accepted `(Q, irf)` pair |
| `shock_names` | `Vector{String}` | `nothing` | Shock display names |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

**`SignIdentifiedSet{T}` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Q_draws` | `Vector{Matrix{T}}` | Accepted rotation matrices |
| `irf_draws` | `Array{T,4}` | ``n_{\text{accepted}} \times H \times n \times n`` stacked IRFs |
| `n_accepted` | `Int` | Number of accepted draws |
| `n_total` | `Int` | Draws attempted (equal to `max_draws`) |
| `acceptance_rate` | `T` | ``n_{\text{accepted}} / n_{\text{total}}`` |
| `variables` / `shocks` | `Vector{String}` | Variable and shock names |

### Narrative Restrictions

Narrative restrictions augment sign restrictions with historical information about specific shocks at particular dates, following Antolín-Díaz & Rubio-Ramírez (2018). Two types of narrative constraints are supported:

1. **Shock sign narrative**: at date ``t^*``, structural shock ``j`` was positive (or negative)
2. **Shock contribution narrative**: at date ``t^*``, shock ``j`` was the dominant driver of variable ``i``

```@example var
model = estimate_var(Y, 4)

# Sign restrictions on impact
sign_check = ir -> ir[1, 3, 3] > 0 && ir[1, 1, 3] < 0

# Narrative: monetary shock was positive at observation 20
narrative_check = shocks -> shocks[20, 3] > 0

Q, irfs, shocks = identify_narrative(model, 20, sign_check, narrative_check; max_draws=5000)
round.([shocks[20, 3] irfs[1, 3, 3] irfs[1, 1, 3]], digits=4)
```

The algorithm first filters for sign-satisfying rotations, then checks whether the recovered structural shocks ``\varepsilon = B_0^{-1} u`` satisfy the narrative conditions. It returns the first rotation clearing both filters, as a `(Q, irf, shocks)` tuple, and throws an `IdentificationError` if none is found in `max_draws` attempts. The `shocks` matrix is ``T_{\text{eff}} \times n``, here ``56 \times 3``. The accepted rotation puts the date-20 monetary shock at ``+0.030`` and delivers an impact output response of ``-0.087`` percent. Because the narrative condition is imposed on one date of one shock, it discards rotations that the sign restrictions alone would keep, sharply reducing the identified set.

### Long-Run (Blanchard-Quah)

Long-run restrictions constrain the cumulative effect of structural shocks on selected variables. The long-run impact matrix is:

```math
C(1) = (I_n - A_1 - A_2 - \cdots - A_p)^{-1} B_0
```

where:
- ``C(1)`` is the ``n \times n`` long-run cumulative response matrix
- ``A(1) = A_1 + A_2 + \cdots + A_p`` is the sum of VAR coefficient matrices

Blanchard & Quah (1989) impose that ``C(1)`` is lower triangular, so that shocks ordered later have zero long-run effect on variables ordered earlier. The typical application restricts demand shocks to have no long-run effect on output, identifying supply-driven long-run fluctuations.

```@example var
model = estimate_var(Y, 4)
result = irf(model, 40; method=:long_run)

# Cumulative 40-period response of INDPRO to each structural shock
round.([sum(result.values[:, 1, j]) for j in 1:3]', digits=4)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_longrun.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Because industrial production enters as a growth rate, the cumulative sum of its impulse response is the response of the *level*. The restriction shows up exactly there: the first shock moves the level of output permanently by 0.28 percent, while the second and third accumulate to zero to four decimal places. Shock 1 is the only one with a permanent effect on output, which is what the Blanchard-Quah triangularity of ``C(1)`` imposes.

### Arias et al. (2018) Zero + Sign Restrictions

When sign restrictions alone are insufficient, zero restrictions on specific impulse responses can be imposed alongside sign constraints. Arias, Rubio-Ramírez & Waggoner (2018) develop an algorithm that draws rotation matrices ``Q`` uniformly over the set satisfying zero restrictions, then filters for sign satisfaction. Importance weights correct for non-uniform sampling induced by the zero-restriction constraint manifold.

The algorithm constructs ``Q`` column-by-column via QR decomposition in the null space of the zero restriction matrix, then checks sign restrictions on the candidate IRF ``\Theta_h = \Phi_h L Q``.

| Type | Function | Description |
|------|----------|-------------|
| Zero | `zero_restriction(var, shock; horizon=0)` | Variable `var` does not respond to `shock` at `horizon` |
| Long-run zero | `zero_restriction(var, shock; horizon=:long_run)` | ``e_v' C(1) L q_s = 0`` |
| Sign | `sign_restriction(var, shock, :positive; horizon=0)` | Response has required sign at `horizon` |
| Sign (range) | `sign_restriction(var, shock, :positive; horizons=0:K)` | Expands to ``K+1`` sign restrictions |

```@example var
model_short = estimate_var(Y, 2)

# Monetary policy shock (shock 3):
# Sign: FFR rises on impact, output rises to demand shock
restrictions = SVARRestrictions(3;
    signs = [sign_restriction(3, 3, :positive),
             sign_restriction(1, 1, :positive)]
)

result = identify_arias(model_short, restrictions, 20; n_draws=1000)
report(result)
```

```@example var
# Weighted IRF percentiles (importance-weight-corrected)
pct = irf_percentiles(result; quantiles=[0.16, 0.5, 0.84])
round.(pct[1, 3, 3, :]', digits=4)
```

The acceptance rate is 26.2%: roughly one draw in four satisfies both sign conditions. The weighted median impact response of the funds rate to its own shock is 0.052 percentage points, with a 68% interval of ``[0.018, 0.090]``. Because these restrictions are sign-only, the importance weights are uniform and `ess_fraction` is exactly 1 --- all 1000 accepted draws contribute equally. Adding a zero restriction would make the weights uneven and pull `ess_fraction` below 1; low rates (under 1%) instead signal overly stringent or contradictory restrictions.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Target number of accepted draws |
| `n_rotations` | `Int` | `1000` | Maximum attempts per target draw |
| `compute_weights` | `Bool` | `true` | Compute importance weights (set `false` for faster exploratory analysis) |
| `normalize_weights` | `Bool` | `true` | Scale stored weights to sum to 1 (`false` keeps the raw volume-element scale) |

**AriasSVARResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Q_draws` | `Vector{Matrix{T}}` | Accepted rotation matrices |
| `irf_draws` | `Array{T,4}` | ``n_{\text{draws}} \times H \times n \times n`` IRF draws |
| `weights` | `Vector{T}` | Importance weights (normalized to sum to 1) |
| `acceptance_rate` | `T` | Fraction of draws satisfying all restrictions |
| `restrictions` | `SVARRestrictions` | The imposed restrictions |
| `ess` | `T` | Kish effective sample size of the importance weights |
| `ess_fraction` | `T` | ``\mathrm{ESS} / n_{\text{draws}}`` |

`ess_fraction` is exactly 1 under pure sign restrictions (uniform weights) and falls below 1 once zero restrictions make the weights uneven. A value near zero means a handful of draws carry the posterior, so the credible bands rest on far fewer effective draws than `n_draws` implies; see [Structural Identification](@ref structural_identification_page).

### Mountford-Uhlig (2009) Penalty Function

When a single best rotation is preferred over a distribution of draws, Mountford & Uhlig (2009) provide a penalty function approach. Zero restrictions are enforced exactly via null-space projection; sign restrictions are encouraged through a penalty function minimized with two-phase Nelder-Mead optimization.

The penalty for each sign restriction ``s`` is:

```math
\text{penalty} = -\sum_{s} w_s \cdot \text{sign}_s \cdot \frac{\text{IRF}_s}{\sigma_s}
```

where:
- ``w_s = 100`` if the sign restriction is satisfied, ``w_s = 1`` if violated
- ``\text{sign}_s \in \{+1, -1\}`` is the required sign direction
- ``\text{IRF}_s`` is the impulse response value at the restricted horizon
- ``\sigma_s`` is the standard deviation of the response variable (normalization)

!!! note "When to use Uhlig vs Arias"
    Use `identify_uhlig` when a single point-identified rotation is needed --- for example, as a starting point for policy analysis. Use `identify_arias` when the full identified set is required for inference with credible intervals.

```@example var
model_short = estimate_var(Y, 2)

# Fiscal vs monetary separation
restrictions = SVARRestrictions(3;
    signs = [sign_restriction(1, 1, :positive),     # Fiscal shock raises INDPRO
             sign_restriction(3, 3, :positive)]     # Monetary shock raises FFR
)

result = identify_uhlig(model_short, restrictions, 20)
report(result)
```

The optimizer converges with a total penalty of ``-199.24``, split as ``[-99.62, 0, -99.62]`` across the three shocks: shocks 1 and 3 each carry a restriction and each attains the satisfied-restriction weight ``w_s = 100``, while shock 2 is unrestricted and contributes nothing. At that rotation output rises 0.71 percent to its own shock and the funds rate rises 0.108 percentage points to the policy shock, both with the required sign. The `converged` field is `true` only when every sign restriction holds at the optimum; a `false` value means the optimizer settled in a local minimum that violates some condition, and increasing `n_starts` or relaxing restrictions is the remedy.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_starts` | `Int` | `50` | Random starting points for coarse search |
| `n_refine` | `Int` | `10` | Top candidates refined in second phase |
| `max_iter_coarse` | `Int` | `500` | Maximum Nelder-Mead iterations (coarse phase) |
| `max_iter_fine` | `Int` | `2000` | Maximum iterations (refinement phase) |

**UhligSVARResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Optimal rotation matrix |
| `irf` | `Array{T,3}` | ``H \times n \times n`` impulse responses |
| `penalty` | `T` | Total penalty at optimum (more negative = better) |
| `shock_penalties` | `Vector{T}` | Per-shock penalty values |
| `restrictions` | `SVARRestrictions` | The imposed restrictions |
| `converged` | `Bool` | Whether all sign restrictions are satisfied |

---

## Covariance Estimation

### Newey-West HAC Estimator

For robust inference in the presence of heteroscedasticity and autocorrelation, the Newey-West (1987, 1994) estimator computes a heteroscedasticity and autocorrelation consistent (HAC) covariance matrix:

```math
\hat{V}_{\text{NW}} = (X'X)^{-1} \hat{S} (X'X)^{-1}
```

where:
- ``\hat{V}_{\text{NW}}`` is the HAC covariance matrix of the coefficient estimator
- ``\hat{S}`` is the long-run covariance estimator

The long-run covariance ``\hat{S}`` is:

```math
\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j')
```

where:
- ``\hat{\Gamma}_j = \sum_{t=j+1}^{T} \hat{u}_t \hat{u}_{t-j}' x_t x_{t-j}'`` is the ``j``-th order autocovariance
- ``w_j`` is the kernel weight at lag ``j``
- ``m`` is the bandwidth (truncation parameter)

### Kernel Functions

The weight function ``w_j`` depends on the kernel choice:

**Bartlett** (default):

```math
w_j = 1 - \frac{j}{m+1}
```

**Parzen**:

```math
w_j = \begin{cases}
1 - 6x^2 + 6|x|^3 & |x| \leq 0.5 \\
2(1-|x|)^3 & 0.5 < |x| \leq 1
\end{cases}
```

where:
- ``x = j/(m+1)``

**Quadratic spectral** (Andrews 1991):

```math
w_j = \frac{25}{12\pi^2 x^2} \left( \frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5) \right)
```

where:
- ``x = j/(m+1)``

### Automatic Bandwidth Selection

Newey & West (1994) provide a data-driven bandwidth:

```math
m^* = 1.1447 \left( \hat{\alpha} \cdot T \right)^{1/3}
```

where:
- ``\hat{\alpha} = 4\hat{\rho}^2 / (1-\hat{\rho})^4`` is estimated from AR(1) fits to the residuals
- ``T`` is the sample size

### White Heteroscedasticity-Robust Estimator

When errors are heteroscedastic but serially uncorrelated, the White (1980) HC0 estimator provides consistent standard errors without bandwidth selection:

```math
\hat{V}_{W} = (X'X)^{-1} \left( \sum_{t=1}^{T} \hat{u}_t^2 x_t x_t' \right) (X'X)^{-1}
```

where:
- ``\hat{u}_t`` is the OLS residual at time ``t``
- ``x_t`` is the ``k \times 1`` regressor vector at time ``t``

### Driscoll-Kraay Panel-Robust Estimator

For panel data with both cross-sectional and temporal dependence, the Driscoll & Kraay (1998) estimator applies HAC estimation to the cross-sectional averages of moment conditions. This produces standard errors robust to heteroscedasticity, serial correlation, and cross-sectional dependence.

```@example var
using LinearAlgebra

# Construct VAR design matrices
Y_eff, X = construct_var_matrices(Y, 2)
residuals = Y_eff - X * ((X'X) \ (X'Y_eff))

# Newey-West HAC (Bartlett kernel, automatic bandwidth)
V_nw = newey_west(X, residuals; bandwidth=0, kernel=:bartlett)

# White heteroscedasticity-robust (HC0)
V_w = white_vcov(X, residuals)

# Automatic bandwidth selection
bw = optimal_bandwidth_nw(residuals)

# Ratio of HAC to White standard errors, first equation
round.((sqrt.(diag(V_nw)) ./ sqrt.(diag(V_w)))[1:7]', digits=3)
```

Both estimators return the full ``k n_{eq} \times k n_{eq}`` sandwich for the stacked system --- ``21 \times 21`` here, for ``k = 7`` regressors in each of three equations. The Newey-West (1994) rule selects a bandwidth of 1, so only the first autocovariance enters, and the HAC standard errors sit within roughly 4% below to 16% above their White counterparts. That gap is the cost of ignoring residual autocorrelation. Newey-West is the default for VAR and local-projection applications; White is simpler but inconsistent when errors are serially correlated, and Driscoll-Kraay extends HAC to panels whose cross-sectional units are correlated.

---

## Forecasting

The VAR generates multi-step ahead forecasts by iterating the estimated recursion forward from the last ``p`` observations:

```math
\hat{y}_{T+h} = \hat{c} + \hat{A}_1 \hat{y}_{T+h-1} + \cdots + \hat{A}_p \hat{y}_{T+h-p}
```

where:
- ``\hat{y}_{T+h}`` is the ``h``-step ahead point forecast
- ``\hat{y}_{T+j}`` for ``j \leq 0`` uses the observed data

The default `:bootstrap` bands implement Kilian's (1998) bootstrap-B. Each replication builds a pseudo-sample by resampling the residuals, **re-estimates** the VAR on it, and then simulates forward from the true last ``p`` observations using those re-estimated coefficients and fresh resampled shocks. The bands therefore carry both future-innovation and coefficient-estimation uncertainty, at a cost of `reps` VAR re-estimations. The `:analytic` alternative uses the Lütkepohl (2005, §3.5) known-coefficient forecast MSE ``\Sigma_y(h) = \sum_{i=0}^{h-1} \Phi_i \Sigma_u \Phi_i'`` with symmetric Gaussian bands, which is cheaper but ignores parameter uncertainty.

```@example var
model = estimate_var(Y, 4; varnames=["INDPRO", "CPI", "FFR"])
fc = forecast(model, 12; ci_method=:bootstrap, reps=50, conf_level=0.95)
report(fc)
```

```julia
plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_var.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The point forecast reverts to the unconditional mean within a few months: industrial production is projected at 0.12 percent one month ahead and 0.05 percent at twelve, and the funds rate moves from ``-0.06`` to ``+0.05`` percentage points. The bands widen with the horizon as the accumulating shock variance implies --- the 95% interval for the funds rate goes from ``[-0.22, 0.09]`` at ``h = 1`` to ``[-0.31, 0.40]`` at ``h = 12``, and for industrial production from ``[-0.61, 1.11]`` to ``[-1.91, 2.27]`` percent. Bootstrap-B bands widen faster than an analytic MSE band would, because re-estimating the VAR on every replication feeds coefficient uncertainty into the long horizons as well.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:bootstrap` | `:bootstrap` (Kilian bootstrap-B), `:analytic` (known-coefficient MSE), or `:none` |
| `reps` | `Int` | `500` | Number of bootstrap replications |
| `conf_level` | `Real` | `0.95` | Confidence level for intervals |
| `stationary_only` | `Bool` | `false` | Discard bootstrap replications whose re-estimated companion matrix is explosive |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

`VARForecast{T}` return value:

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Matrix{T}` | ``h \times n`` point forecast |
| `ci_lower` / `ci_upper` | `Matrix{T}` | ``h \times n`` interval bounds (zero-width when `ci_method=:none`) |
| `horizon` | `Int` | Forecast horizon |
| `ci_method` | `Symbol` | Interval method used |
| `conf_level` | `T` | Confidence level |
| `varnames` | `Vector{String}` | Variable display names |

### Conditional Forecasts and Scenario Analysis

`conditional_forecast` answers the scenario question — *what happens to everything else if inflation is held at 2% for the next four quarters?* — using Waggoner & Zha (1999). Write the forecast as the unconditional path plus the moving average of the future structural shocks,

```math
y_{T+s} = \hat{y}_{T+s} + \sum_{j=1}^{s} \Psi_{s-j} \, \varepsilon_{T+j},
\qquad \Psi_s = \Phi_s P
```

where:
- ``\hat{y}_{T+s}`` is the unconditional forecast
- ``\Phi_s`` is the reduced-form moving-average coefficient at lag ``s`` (``\Phi_0 = I``)
- ``P`` is the structural impact matrix and ``\varepsilon_{T+j} \sim N(0, I)`` the future shocks

Each restriction is linear in ``\varepsilon``, so stacking them over the conditioning window gives ``R\varepsilon = r``, where ``r`` is the gap between the desired path and the unconditional forecast. The shocks are then drawn from the implied conditional distribution

```math
\varepsilon \sim N\big(R'(RR')^{-1}r, \; I - R'(RR')^{-1}R\big)
```

— the minimum-norm mean plus randomness in the null space of the restrictions. The point forecast uses the conditional mean; the bands come from `reps` draws, so unconstrained variables and horizons keep genuine uncertainty while constrained cells collapse onto their targets.

```@example var
model = estimate_var(Y, 4; varnames=["INDPRO", "CPI", "FFR"])
# Hold the policy rate at 2% for four quarters
scenario = Dict(("FFR", h) => 2.0 for h in 1:4)
cfc = conditional_forecast(model, scenario, 12; reps=200)
report(cfc)
```

Each variable's table places the conditional path next to the unconditional one, so the scenario's effect is read off directly. Over the first four months the funds rate is pinned at exactly 2.0 with a degenerate ``[2.0, 2.0]`` band, against an unconditional path of roughly ``-0.06``; holding the rate two full percentage points above where the VAR would have put it is a very large shock, and the spillovers are correspondingly violent --- industrial production jumps to ``+4.5`` percent in the first month before overshooting to ``-2.7`` percent by the third. Unconstrained variables keep genuine uncertainty: at ``h = 1`` the 95% band for output is ``[3.43, 5.66]`` percent. Conditions are built either as a `Dict` keyed by `(variable, horizon)` or with [`forecast_condition`](@ref), which also builds **soft** conditions — a target with a tolerance:

```@example var
soft = [forecast_condition("FFR", h, 2.0; sd=0.25) for h in 1:4]
cfc_soft = conditional_forecast(model, soft, 12; reps=200)
round.(cfc_soft.forecast[1:4, 3], digits=3)   # shrunk toward 2.0, not pinned to it
```

A soft condition treats the target as a noisy observation with standard deviation `sd`, so the path is shrunk toward it rather than hit exactly, and `sd → 0` recovers the hard condition. With `sd=0.25` the four conditioned months come out at 0.45, 0.73, 0.78 and 1.03 rather than 2.0 --- the tolerance is large relative to what the shocks can deliver, so the restriction is only partly enforced.

```julia
plot_result(cfc)
```

!!! note "Identification and conditional forecasts"
    For conditions on observable paths the conditional forecast is **invariant** to the rotation ``Q``: writing ``P = LQ``, the rotation cancels between ``R`` and the impact matrix, so identification does not change the forecast. It changes only the interpretation of the implied shocks in `result.shocks`, which rotate as ``\varepsilon_L = Q\varepsilon_{LQ}``. Pass `Q` when those structural shocks are themselves of interest.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `Q` | `AbstractMatrix` | `nothing` | Rotation matrix; `nothing` is Cholesky. Affects only the reported `shocks` |
| `reps` | `Int` | `1000` | Draws used for the bands |
| `conf_level` | `Real` | `0.95` | Band coverage |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

`ConditionalForecast{T}` return value:

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Matrix{T}` | ``h \times n`` conditional mean path |
| `ci_lower` / `ci_upper` | `Matrix{T}` | ``h \times n`` band bounds |
| `horizon` | `Int` | Forecast horizon |
| `conf_level` | `T` | Band coverage |
| `varnames` | `Vector{String}` | Variable display names |
| `conditions` | `Vector{ForecastCondition{T}}` | The restrictions imposed, with variable indices resolved |
| `unconditional` | `Matrix{T}` | ``h \times n`` unconditional forecast, for comparison |
| `shocks` | `Matrix{T}` | ``h \times n_{\text{shocks}}`` mean structural shocks implied by the conditions |
| `identification` | `Symbol` | `:cholesky` or `:custom` |
| `n_draws` | `Int` | Draws used for the bands |

The same function dispatches on `BVARPosterior` to integrate the scenario over the posterior — see [Bayesian VAR](@ref bvar_page).

---

## Innovation Accounting and Bayesian VAR

For detailed coverage of impulse response functions, forecast error variance decomposition, and historical decomposition, see the dedicated [Innovation Accounting](@ref innovation_accounting_page) page. For Bayesian VAR estimation with Minnesota priors, conjugate NIW sampling, and hyperparameter optimization, see [Bayesian VAR](@ref bvar_page).

---

## Complete Example

This example demonstrates an end-to-end VAR workflow from data loading through structural analysis using FRED-MD monetary policy variables.

```@example var
# Step 1: Select lag order
p_opt = select_lag_order(Y, 4)

# Step 2: Estimate VAR
model = estimate_var(Y, p_opt; varnames=["INDPRO", "CPI", "FFR"])
report(model)
```

```@example var
# Step 3: Check stability
stab = is_stationary(model)
stab
```

```@example var
# Step 4: Cholesky IRF with bootstrap CI
# Ordering: [INDPRO, CPI, FFR] — monetary policy shock is shock 3
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
report(result)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example var
# Step 5: FEVD
decomp = fevd(model, 20)
report(decomp)
```

```@example var
# Step 6: Historical decomposition
hd = historical_decomposition(model, size(model.U, 1))
verify_decomposition(hd)
```

```@example var
# Step 7: Forecast
fc = forecast(model, 12)
report(fc)
```

BIC selects ``p = 1``, well inside the search range. The Cholesky ordering [INDPRO, CPI, FFR] implements the recursive identification of Christiano, Eichenbaum & Evans (1999): a monetary policy shock (shock 3) raises the federal funds rate on impact while output and prices respond only with a lag. The variance decomposition says the policy shock is a minor driver of industrial production --- 2.0% of its forecast error variance at twenty months, against 92.0% for output's own shock and 6.0% for the price shock. The funds rate is the variable most exposed to the rest of the system: 94.1% own, but 4.8% attributable to output shocks, the signature of a policy rule that reacts to real activity. A parsimonious VAR(1) leaves less room for cross-variable transmission than the VAR(4) that AIC would select, so read these shares as the conservative end of the range. The historical decomposition identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)`` holds to numerical precision over all 59 effective observations, which `verify_decomposition` confirms by returning `true`.

---

## Common Pitfalls

1. **Variable ordering matters for Cholesky identification.** The Cholesky decomposition imposes a recursive causal structure where variable ``i`` responds contemporaneously only to shocks ``1, \ldots, i``. Reordering the columns of ``Y`` changes the economic interpretation of the structural shocks. The standard monetary VAR ordering places slow-moving variables first (output, prices) and the policy instrument last.

2. **Non-stationary VAR produces unreliable inference.** `is_stationary` returns a `VARStationarityResult`, not a `Bool` --- test `is_stationary(model).is_stationary`. When it is `false` the companion matrix has eigenvalues on or outside the unit circle and the asymptotic theory underlying OLS standard errors and bootstrap confidence intervals no longer applies. Difference the data, apply the appropriate transformation codes via `apply_tcode`, or estimate a [VECM](@ref vecm_page) for cointegrated systems.

3. **Too many lags exhaust degrees of freedom.** Each additional lag adds ``n^2`` parameters to the system and ``n`` regressors to every equation. For an ``n = 7`` variable system each lag costs 49 parameters. With moderate sample sizes (``T < 200``) overfitting degrades forecast accuracy and inflates IRF confidence intervals. Cap `max_p` at a value the sample can support, and prefer BIC over AIC when ``T`` is small --- as the Information Criteria section shows, AIC selects four lags on this 60-observation sample where BIC selects one.

4. **Low acceptance rate for sign restrictions.** When `identify_sign` or `identify_arias` reports an acceptance rate below 1%, the imposed sign conditions are difficult to satisfy jointly. This may indicate contradictory economic restrictions or an overidentified specification. Relaxing some conditions (e.g., restricting only impact responses rather than multiple horizons) typically improves acceptance.

5. **Uhlig penalty function finds local minima.** The `identify_uhlig` optimizer uses multi-start Nelder-Mead, but non-convexity of the penalty landscape means the solution depends on initial conditions. If `converged` is `false`, increase `n_starts` or verify that the sign restrictions are economically coherent.

---

## References

- Andrews, D. W. K. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation.
  *Econometrica*, 59(3), 817-858. [DOI](https://doi.org/10.2307/2938229)

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs.
  *American Economic Review*, 108(10), 2802-2829. [DOI](https://doi.org/10.1257/aer.20161852)

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications.
  *Econometrica*, 86(2), 685-720. [DOI](https://doi.org/10.3982/ECTA14468)

- Antolín-Díaz, J., Petrella, I., & Rubio-Ramírez, J. F. (2021). Structural Scenario Analysis with SVARs.
  *Journal of Monetary Economics*, 117, 798-815. [DOI](https://doi.org/10.1016/j.jmoneco.2020.06.001)

- Baumeister, C., & Hamilton, J. D. (2015). Sign Restrictions, Structural Vector Autoregressions, and Useful Prior Information.
  *Econometrica*, 83(5), 1963-1999. [DOI](https://doi.org/10.3982/ECTA12356)

- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances.
  *American Economic Review*, 79(4), 655-673. [JSTOR](https://www.jstor.org/stable/1827924)

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary Policy Shocks: What Have We Learned and to What End?
  In *Handbook of Macroeconomics*, Vol. 1, edited by J. B. Taylor & M. Woodford, 65-148. Amsterdam: Elsevier. [DOI](https://doi.org/10.1016/S1574-0048(99)01005-8)

- Driscoll, J. C., & Kraay, A. C. (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data.
  *Review of Economics and Statistics*, 80(4), 549-560. [DOI](https://doi.org/10.1162/003465398557825)

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. [DOI](https://doi.org/10.1007/978-3-540-27752-1)

- Mountford, A., & Uhlig, H. (2009). What Are the Effects of Fiscal Policy Shocks?
  *Journal of Applied Econometrics*, 24(6), 960-992. [DOI](https://doi.org/10.1002/jae.1079)

- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica*, 55(3), 703-708. [DOI](https://doi.org/10.2307/1913610)

- Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
  *Review of Economic Studies*, 61(4), 631-653. [DOI](https://doi.org/10.2307/2297912)

- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference.
  *Review of Economic Studies*, 77(2), 665-696. [DOI](https://doi.org/10.1111/j.1467-937X.2009.00578.x)

- Sims, C. A. (1980). Macroeconomics and Reality.
  *Econometrica*, 48(1), 1-48. [DOI](https://doi.org/10.2307/1912017)

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 52(2), 381-419. [DOI](https://doi.org/10.1016/j.jmoneco.2004.05.007)

- Waggoner, D. F., & Zha, T. (1999). Conditional Forecasts in Dynamic Multivariate Models.
  *Review of Economics and Statistics*, 81(4), 639-651. [DOI](https://doi.org/10.1162/003465399558508)

- White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity.
  *Econometrica*, 48(4), 817-838. [DOI](https://doi.org/10.2307/1912934)
