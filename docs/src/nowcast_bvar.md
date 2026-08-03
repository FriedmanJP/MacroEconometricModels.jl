# [BVAR Nowcasting](@id nowcast_bvar_page)

The large Bayesian VAR estimates the whole panel jointly rather than compressing it into factors, and relies on a Normal-Inverse-Wishart prior to keep a system with more coefficients than observations estimable. Prior tightness is not fixed by hand: four hyperparameters are chosen by numerical search over the Giannone, Lenza & Primiceri (2015) dummy-observation family, and the ragged edge is filled by a Kalman smoother run on the companion form of the estimated VAR (Cimadomo et al. 2022). The result is a nowcast built from directly interpretable coefficients instead of latent factors.

For the shared data layout, the `nowcast()` interface, and the result visualizations, see [Nowcasting](@ref nowcast_page). Sibling estimators: [DFM Nowcasting](@ref nowcast_dfm_page) and [Bridge Equations](@ref nowcast_bridge_page).

```@setup nc_bvar
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
nM, nQ = 4, 1
N = nM + nQ
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN
T_obs = size(Y, 1)
```

The examples run on a FRED-MD panel in the standard mixed-frequency layout: four monthly indicators (`INDPRO`, `UNRATE`, `CPIAUCSL`, `M2SL`) in the leading columns and one quarterly target (`FEDFUNDS`) in the last, observed only at quarter-end months and missing in the most recent month.

## Quick Start

**Recipe 1: BVAR nowcast with data-driven hyperparameters**

```@example nc_bvar
bvar = nowcast_bvar(Y, nM, nQ; lags=5)
report(bvar)
```

**Recipe 2: Different starting values for the hyperparameter search**

```@example nc_bvar
bvar_alt = nowcast_bvar(Y, nM, nQ; lags=5, lambda0=0.5, theta0=1.0, miu0=1.0, alpha0=1.0)
(lambda=round(bvar_alt.lambda, digits=2), theta=round(bvar_alt.theta, digits=2),
 miu=round(bvar_alt.miu, digits=2), alpha=round(bvar_alt.alpha, digits=2),
 loglik=round(bvar_alt.loglik, digits=2), converged=bvar_alt.converged)
```

**Recipe 3: Read off the nowcast and the next-period forecast**

```@example nc_bvar
result = nowcast(bvar)
report(result)
```

**Recipe 4: Multi-step forecast path**

```@example nc_bvar
forecast(bvar, 6; target_var=N)
```

**Recipe 5: `TimeSeriesData` dispatch**

```@example nc_bvar
ts = TimeSeriesData(Y; varnames=["INDPRO", "UNRATE", "CPI", "M2", "FEDFUNDS"],
                    frequency=Monthly)
report(nowcast_bvar(ts, nM, nQ; lags=5))
```

---

## Model Specification

All ``N`` variables enter one vector autoregression with ``p`` lags and an intercept:

```math
y_t = c + B_1 y_{t-1} + \cdots + B_p y_{t-p} + u_t, \quad u_t \sim N(0, \Sigma)
```

where:
- ``y_t`` is the ``N \times 1`` vector of variables at time ``t``
- ``c`` is the ``N \times 1`` intercept
- ``B_1, \ldots, B_p`` are the ``N \times N`` coefficient matrices
- ``\Sigma`` is the ``N \times N`` innovation covariance

The parameter count is the reason a prior is needed at all: with ``N = 5`` and ``p = 5`` each equation carries ``1 + Np = 26`` coefficients, so the full system has 130 free parameters against 55 usable observations.

### Prior Structure

The Normal-Inverse-Wishart prior is imposed through **dummy observations** — artificial rows appended to the regression before OLS — following the four-block scheme of Giannone, Lenza & Primiceri (2015). Writing ``\sigma_i`` for the residual standard deviation of an AR(1) fitted to variable ``i`` and ``\bar{y}_i`` for its pre-sample mean, the blocks are:

| Block | Hyperparameter | Dummy entry | Shrinks toward |
|-------|----------------|-------------|----------------|
| Minnesota, variable ``i`` at lag ``l`` | ``\lambda``, ``\theta`` | ``\sigma_i \, l^{\theta} / \lambda`` | Random walk at lag 1, zero beyond |
| Sum-of-coefficients | ``\mu`` | ``\bar{y}_i / \mu`` | Own lags summing to one (unit root) |
| Co-persistence | ``\alpha`` | ``\bar{y}_i / \alpha`` | A single common stochastic trend |
| Inverse-Wishart scale | --- | ``\text{diag}(\sigma)`` against ``X = 0`` | ``\mathbb{E}[\Sigma] \propto \text{diag}(\hat\sigma_i^2)`` |

Every Minnesota row carries exactly one nonzero regressor entry — the coefficient on ``y_{i,t-l}`` in equation ``i`` — and targets 1 at lag 1 and 0 at every longer lag, so the block restricts each coefficient on its own rather than restricting a sum of them. Every entry is a scale: the larger the dummy, the more prior information it injects. Dividing by ``\lambda`` therefore makes a smaller ``\lambda`` a tighter overall prior, and the same holds for ``\mu`` and ``\alpha`` within their blocks. The exponent ``\theta`` sets how fast tightness grows with the lag: at ``\theta = 0`` every lag is shrunk alike, and the larger ``\theta`` the harder the long lags are pushed to zero.

!!! note "Why the conjugate prior has no cross-variable tightness parameter"
    Dummy observations imply a prior variance ``\Sigma_{mm} (X_d' X_d)^{-1}_{cc}`` for coefficient ``c`` of equation ``m``, so for any given regressor the cross-to-own prior standard-deviation ratio is ``\sqrt{\Sigma_{mm} / \Sigma_{jj}}`` — the classical Minnesota asymmetry — whatever row scale is chosen. Relative own-versus-cross tightness is structurally fixed in a conjugate Normal-Inverse-Wishart prior and cannot be a free hyperparameter, which is why ``\theta`` is the lag-decay exponent (Litterman's ``d``, GLP's ``\alpha``) rather than a cross-variable weight. Set `prior=:litterman` to get that knob; see [The Litterman prior](@ref nowcast_bvar_litterman) below.

!!! note "Technical Note"
    Stacking the dummies on the data and running OLS returns the posterior mode of the conjugate Normal-Inverse-Wishart model in closed form, so no MCMC is involved and a fit costs one least-squares solve. The ``\sigma_i`` used to scale the prior are estimated from the data beforehand, which puts each equation's prior on the scale of its own residual variance. The closing inverse-Wishart block sets ``X = 0`` and therefore constrains ``\Sigma`` alone: the random-walk prior mean satisfies every other dummy row exactly, so without this block the prior sum of squares is singular and the marginal likelihood degenerates.

---

## Estimation

Estimation runs on the **complete block** of the panel. Scanning back from the end, `nowcast_bvar` finds `t_complete`, the last row with no missing values, and estimates on rows ``1`` through `t_complete`; if that block is shorter than `lags + 2` rows the routine falls back to the full sample. Missing values inside the block are replaced by their column means before the search begins.

```@example nc_bvar
t_complete = findlast(t -> !any(isnan, Y[t, :]), 1:T_obs)
(last_complete_row=t_complete, mean_filled_entries=count(isnan, Y[1:t_complete, :]))
```

With the standard quarterly mask the last complete row is month 57 — month 60 is the ragged edge and months 58 and 59 are not quarter ends — and 38 entries of the quarterly column inside that window are set to the column mean. This is the estimator's mixed-frequency approximation, and it is coarse: the quarterly series contributes its own variation only one month in three. Panels whose series mostly share a frequency suit this estimator best; for a genuinely mixed-frequency panel the [DFM](@ref nowcast_dfm_page) treats the quarterly block properly through temporal aggregation instead of imputing it.

The hyperparameters are then searched by Nelder-Mead over ``(\log \lambda, \log \theta, \log \mu, \log \alpha)``, maximizing the closed-form Normal-Inverse-Wishart **log marginal likelihood** of Giannone, Lenza & Primiceri (2015) — the criterion that integrates the coefficients and ``\Sigma`` out, carrying the ``|X_d'X_d| / |X_*'X_*|`` determinant ratio and the multivariate gamma terms, rather than a fit evaluated at the mode. The search is confined to the box ``|\log(\cdot)| \le 5``, so each hyperparameter is restricted to ``[e^{-5}, e^{5}] \approx [0.0067, 148.41]``. The optimum is reported in the `lambda`, `theta`, `miu` and `alpha` fields and the attained log marginal likelihood in `loglik`.

---

## Hyperparameter Diagnostics

Because the objective integrates the coefficients and ``\Sigma`` out instead of evaluating a fit at the posterior mode, it carries a genuine complexity penalty and can be read as a model-selection criterion over the lag order:

```@example nc_bvar
map(2:5) do L
    b = nowcast_bvar(Y, nM, nQ; lags=L)
    (lags=L, loglik=round(b.loglik, digits=1), lambda=round(b.lambda, digits=1),
     optimum=b.converged ? "interior" : "boundary")
end
```

Two lags maximize the marginal likelihood at 657.1, four lags minimize it at 642.8, and five recovers to 645.8 — a non-monotone path, which is the signature of the penalty an in-sample fit measure lacks. The spread is 14 log points across the whole range on 55 usable observations, so read the ranking as weak evidence rather than a verdict: five lags for a monthly panel remains the conventional choice, and this criterion does not contradict it strongly enough to overturn it.

The search can still terminate on the wall of the box. The `converged` field records whether the optimum is interior: it is `false` when any log-hyperparameter reaches ``\pm 5`` to within ``10^{-3}``, and `report` then prints an explicit warning instead of presenting the boundary values as an estimate.

!!! warning "Check `converged` before quoting a hyperparameter"
    A value reported at 148.41 has hit ``e^{5}`` and one at 0.0067 has hit ``e^{-5}``: the objective was still improving when the search ran out of box, so the number is a truncation point, not an optimum. Every fit on this panel lands in the interior, but a flat objective on a short or near-collinear panel does reach the wall.

```@example nc_bvar
bvar_short = nowcast_bvar(Y, nM, nQ; lags=3)
report(bvar_short)
```

At three lags the search settles at ``\lambda = 0.7350``, ``\theta = 1.4378``, ``\mu = 1.0759`` and ``\alpha = 0.6399``, all interior. The five-lag fit is slightly looser overall (``\lambda = 0.8391``) but far tighter on the two persistence blocks (``\mu = 0.1557``, ``\alpha = 0.1595``), which is how the longer system holds itself together: the extra lags are free to move only because the unit-root and common-trend dummies pin down their sum. Recipe 2 starts the same five-lag search from ``\lambda_0 = 0.5`` and ``\alpha_0 = 1.0`` and reaches that optimum to four decimals. Agreement across starts is not something a Nelder-Mead search over four correlated hyperparameters can be assumed to give, so repeat it from a second start whenever the shrinkage values themselves are part of the answer.

---

## [The Litterman Prior](@id nowcast_bvar_litterman)

`prior=:litterman` replaces the conjugate Normal-Inverse-Wishart prior with Litterman's (1986) original formulation: ``\Sigma`` is **fixed** at ``\text{diag}(\hat\sigma_1^2, \ldots, \hat\sigma_N^2)`` rather than integrated out. Fixing ``\Sigma`` separates the equations, and separated equations are what make a cross-variable tightness parameter possible. The prior standard deviation of the coefficient on lag ``l`` of variable ``j`` in equation ``m`` is

```math
\text{sd}(B^{(l)}_{mj}) =
\begin{cases}
\lambda \, / \, l^{\theta} & j = m \quad \text{(own lag)} \\[4pt]
\lambda \, \theta_{\text{cross}} \, \sigma_m \, / \, (l^{\theta} \sigma_j) & j \neq m \quad \text{(cross lag)}
\end{cases}
```

where:
- ``\lambda`` is overall tightness, ``\theta`` the lag-decay exponent — both as in the conjugate prior
- ``\theta_{\text{cross}}`` is the cross-variable relative tightness: ``\theta_{\text{cross}} = 1`` reproduces the conjugate prior's ``\sigma_m/\sigma_j`` asymmetry exactly, and ``\theta_{\text{cross}} < 1`` shrinks other variables' lags harder than own lags
- ``\sigma_m / \sigma_j`` puts the restriction on the ratio of residual standard deviations, so it is invariant to the units each series is measured in

The prior mean is a random walk (1 on the own lag-1 coefficient, 0 elsewhere), the intercept prior is diffuse, and the sum-of-coefficients and co-persistence blocks carry over unchanged so that the two priors impose comparable long-run structure.

```@example nc_bvar
bvar_lit = nowcast_bvar(Y, nM, nQ; lags=5, prior=:litterman)
report(bvar_lit)
```

On this panel the criterion drives both ``\theta_{\text{cross}}`` and ``\alpha`` to the floor of the box, ``e^{-5} \approx 0.0067``: with 26 coefficients per equation and 55 usable observations it prefers to shrink cross-variable dynamics essentially to zero while tightening the common-trend block as hard as it can, leaving something close to five separate autoregressions tied together only by a common trend. That is a substantive finding rather than a numerical failure, and `report` names which hyperparameters hit which edge instead of presenting the boundary values as estimates. A panel long enough to identify cross-variable transmission would leave ``\theta_{\text{cross}}`` interior.

The nowcast moves from 0.0739 under the conjugate prior to 0.0207 under Litterman — the gap is the cross-variable channel, which the conjugate prior cannot switch off and this one shuts down almost entirely.

!!! warning "`loglik` is not comparable across priors"
    The conjugate value integrates out the coefficients **and** ``\Sigma``; the Litterman value integrates out the coefficients with ``\Sigma`` held fixed. They are different objectives over different parameter spaces, so the Litterman fit's higher number is not evidence that it fits better. Compare lag orders and hyperparameters *within* a prior; compare the two priors by out-of-sample nowcast accuracy.

Because the search is five-dimensional rather than four, `max_iter` defaults to `2000` under `:litterman` (against `200` under `:conjugate`) — roughly a thousand Nelder-Mead iterations are needed to reach a stationary point on a panel this size. An explicit `max_iter` is always honoured, and setting it too low stops the search early at a point that is not an optimum.

Passing `theta_cross0` to the conjugate prior throws rather than being silently ignored, because no dummy-observation prior can express it.

---

## Filling the Ragged Edge

Missing entries are filled by a genuine Kalman smoother, not by projecting the VAR forward from the last complete row. The estimated BVAR is cast in companion state-space form with state ``[y_t; \ldots; y_{t-p+1}]``, transition built from the lag blocks of `beta`, state noise `sigma`, observation matrix ``C = [I \;\; 0]`` and a negligible measurement ridge; the panel is centred on the implied steady-state mean ``(I - \sum_i B_i)^{-1} c`` before smoothing, falling back to column means when that inverse is ill-conditioned near a unit root.

Because the missing-data smoother drops only the unobserved rows in each period, a series released this month updates the states of a series that has not been released yet, through the state covariance. Interior gaps — including the two-months-in-three holes in the quarterly column — are filled on the same pass as the ragged edge.

```@example nc_bvar
(missing_input=count(isnan, bvar.data), missing_output=count(isnan, bvar.X_sm),
 nowcast=round(bvar.X_sm[end, N], digits=4))
```

All 41 missing entries are filled, and the last row of the target column is the current-quarter nowcast that `report` displays.

---

## Forecasting

`forecast` iterates the estimated VAR forward from the smoothed panel, feeding each step's prediction back in as the next step's lag:

```math
\hat{y}_{T+h} = \hat{c} + \hat{B}_1 \hat{y}_{T+h-1} + \cdots + \hat{B}_p \hat{y}_{T+h-p}
```

where ``\hat{y}_{T+h-i}`` is the smoothed observation when ``h - i \le 0`` and a previously generated forecast otherwise.

```@example nc_bvar
forecast(bvar, 6; target_var=N)
```

The path oscillates — 0.0184, 0.0336, 0.0617 over the first three months — rather than decaying smoothly, because with 26 coefficients per equation the estimated companion matrix carries complex roots that a factor model would never produce. Omitting `target_var` returns the whole ``h \times N`` panel. `nowcast(bvar)` reports the single-step version of the same recursion as its `forecast` field.

---

## Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `5` | Number of VAR lags |
| `thresh` | `Real` | ``10^{-6}`` | Relative-tolerance stopping rule for the Nelder-Mead search |
| `max_iter` | `Int` or `nothing` | `nothing` | Maximum Nelder-Mead iterations; resolves to `200` under `:conjugate` and `2000` under `:litterman` |
| `lambda0` | `Real` | `0.2` | Starting value for overall shrinkage |
| `theta0` | `Real` | `1.0` | Starting value for the lag-decay exponent |
| `miu0` | `Real` | `1.0` | Starting value for the sum-of-coefficients weight |
| `alpha0` | `Real` | `2.0` | Starting value for the co-persistence weight |
| `prior` | `Symbol` | `:conjugate` | `:conjugate` for the GLP Normal-Inverse-Wishart prior, `:litterman` for the non-conjugate prior with fixed ``\Sigma`` |
| `theta_cross0` | `Real` or `nothing` | `nothing` | Starting value for cross-variable relative tightness; `:litterman` only, and throws under `:conjugate` |

---

## NowcastBVAR Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed panel with every missing entry filled |
| `beta` | `Matrix{T}` | Posterior mode coefficients, ``(1 + Np) \times N``, first row the intercept |
| `sigma` | `Matrix{T}` | Innovation covariance, ``N \times N`` — the posterior mode under `:conjugate`, the fixed ``\text{diag}(\hat\sigma^2)`` under `:litterman` |
| `lambda` | `T` | Overall shrinkage at the optimum |
| `theta` | `T` | Lag-decay exponent at the optimum |
| `miu` | `T` | Sum-of-coefficients weight at the optimum |
| `alpha` | `T` | Co-persistence weight at the optimum |
| `theta_cross` | `T` | Cross-variable relative tightness at the optimum; `NaN` under `:conjugate`, where it is not a free parameter |
| `prior` | `Symbol` | `:conjugate` or `:litterman` |
| `lags` | `Int` | Number of VAR lags |
| `loglik` | `T` | Log marginal likelihood attained at the optimum, for the prior actually used — not comparable across priors |
| `nM` | `Int` | Number of monthly variables |
| `nQ` | `Int` | Number of quarterly variables |
| `data` | `Matrix{T}` | Original input panel, NaN included |
| `converged` | `Bool` | `false` when any log-hyperparameter ended on the ``\pm 5`` box edge |

`StatsAPI` methods are defined: `loglikelihood(bvar)` returns `loglik`, `predict(bvar)` returns `X_sm`, and `nobs(bvar)` returns the number of periods.

---

## Complete Example

```@example nc_bvar
# === Step 1: Estimate the BVAR ===
bvar = nowcast_bvar(Y, nM, nQ; lags=5, max_iter=200)
report(bvar)
```

```@example nc_bvar
# === Step 2: Current quarter and next period ===
result = nowcast(bvar)
report(result)
```

```@example nc_bvar
# === Step 3: Cross-check against the DFM on the same panel ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1)
(bvar=round(nowcast(bvar).nowcast, digits=4),
 dfm=round(nowcast(dfm).nowcast, digits=4))
```

**Interpretation.** The BVAR estimates all 130 coefficients of the five-variable, five-lag system from 55 usable observations, which only the dummy-observation prior makes possible. The marginal likelihood picks a moderate overall tightness (``\lambda = 0.8391``) with a lag decay of ``\theta = 1.2717``, so the dummy that shrinks a lag-5 coefficient is ``5^{1.2717} \approx 7.7`` times the size of its lag-1 counterpart, and it pulls the two persistence blocks tight. The Kalman smoother fills the quarterly mask and the ragged edge in one pass, giving a current-quarter estimate of 0.0739 against the DFM's 0.0375 on the same data. A gap that wide is itself the finding: the DFM reads the quarter off two factors extracted from all five series, whereas the BVAR imputes two of every three quarterly observations with a column mean before it ever sees the data. Where the two disagree by more than their historical revision spread, prefer the DFM on mixed-frequency panels and treat the BVAR as the robustness check.

---

## Common Pitfalls

1. **Check `converged` before quoting hyperparameters.** A value of 148.41 is ``e^{5}``, the edge of the search box, and means the objective never turned around. `report` prints a warning in that case; the estimates remain usable, but the hyperparameters are not.

2. **Do not read small `loglik` differences as a lag-order decision.** The objective is a marginal likelihood, so it does penalize complexity and is comparable across ``p`` — but the spread across two to five lags on a 60-month panel is 14 log points, well inside what a different sample would reshuffle. Corroborate with the data frequency or an out-of-sample criterion.

3. **Column-mean imputation degrades a sparse quarterly block.** Everything missing inside the estimation window is replaced by a column mean before the prior is calibrated, so a quarterly series contributes real variation in only one month of three. Use the DFM when the quarterly block is large relative to the panel.

4. **The estimation window ends at the last complete row.** Rows after `t_complete` are filled by the smoother but never enter the likelihood. A panel with a deep ragged edge therefore estimates on materially fewer observations than `nobs` reports, and if the complete block is shorter than `lags + 2` rows the routine silently falls back to the full sample.

5. **Lags cost parameters quadratically in ``N``.** Each additional lag adds ``N`` coefficients to every one of the ``N`` equations. With ``N = 10`` and `lags=5` the system carries 510 coefficients; the prior keeps this estimable, but the flatter the resulting objective, the more likely the hyperparameter search ends on the box edge.

6. **Do not compare `loglik` across priors.** The conjugate and Litterman criteria integrate out different things, so the two numbers are not on a common scale. Choosing between the priors is an out-of-sample question.

---

## References

- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71--92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436--451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions --- Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25--38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)
- Kadiyala, K. Rao, and Sune Karlsson. 1997. "Numerical Methods for Estimation and Inference in Bayesian VAR-Models." *Journal of Applied Econometrics* 12 (2): 99--132. [https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A](https://doi.org/10.1002/(SICI)1099-1255(199703)12:2%3C99::AID-JAE429%3E3.0.CO;2-A)
