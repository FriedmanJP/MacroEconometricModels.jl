# [Structural Identification](@id structural_identification_page)

Structural identification recovers the mapping from reduced-form VAR residuals to economically interpretable structural shocks. The reduced-form covariance ``\Sigma = B_0 B_0'`` provides ``n(n+1)/2`` equations for ``n^2`` unknowns in the impact matrix ``B_0``, leaving ``n(n-1)/2`` free parameters. Additional restrictions --- economic, statistical, or a combination --- pin down the remaining degrees of freedom. This page documents the six schemes that impose *economic* restrictions on a rotation of the Cholesky factor. Non-recursive contemporaneous zeros estimated by ML live on [AB-Model SVAR](@ref id_ab_page).

- **Cholesky (recursive)** --- lower-triangular ``B_0`` via Cholesky decomposition (Christiano, Eichenbaum & Evans 1999)
- **Sign restrictions** --- set identification via random rotations satisfying inequality constraints (Rubio-Ramírez, Waggoner & Zha 2010)
- **Narrative restrictions** --- sign restrictions augmented with historical event constraints (Antolín-Díaz & Rubio-Ramírez 2018)
- **Long-run (Blanchard-Quah)** --- lower-triangular long-run cumulative impact matrix (Blanchard & Quah 1989)
- **Zero + sign restrictions** --- exact zero restrictions with sign constraints and importance-weighted inference (Arias, Rubio-Ramírez & Waggoner 2018)
- **Penalty function (Mountford-Uhlig)** --- point-identified rotation via constrained optimization (Mountford & Uhlig 2009)

For identification from an external instrument (high-frequency surprises, narrative shocks) see [Proxy SVAR](@ref id_proxy_page). For AB-model maximum likelihood with patterns on ``A`` and ``B`` see [AB-Model SVAR](@ref id_ab_page). For the shock that maximises a target variable's forecast-error variance or spectral share see [Max-Share Identification](@ref id_maxshare_page). Cointegrated systems use the structural VECM route on [Vector Error Correction Models](@ref vecm_page) (`identify_svec`, `method=:svec`). For statistical identification via non-Gaussianity or heteroskedasticity --- 15 further schemes reachable through the same `method=` keyword --- see [Statistical Identification](@ref nongaussian_page). Once ``B_0`` is identified, the impact matrix feeds the impulse responses, variance decompositions, and historical decompositions of [Innovation Accounting](@ref innovation_accounting_page).

```@setup sid
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
post_rb = estimate_bvar(Y, 2; n_draws=100, seed=11)
```

## Quick Start

**Recipe 1: Recursive (Cholesky) IRFs with bootstrap bands**

```@example sid
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50, seed=101)
report(result)
```

**Recipe 2: Sign restrictions --- the full identified set**

```@example sid
# Contractionary monetary shock: FFR rises, output and prices fall on impact
check = resp -> resp[1, 3, 3] > 0 && resp[1, 1, 3] < 0 && resp[1, 2, 3] < 0
id_set = identify_sign(model, 20, check; max_draws=5000, store_all=true,
                       rng=MersenneTwister(42))
id_set
```

**Recipe 3: Long-run (Blanchard-Quah) identification**

```@example sid
result_lr = irf(model, 40; method=:long_run)
report(result_lr)
```

**Recipe 4: Zero + sign restrictions (Arias et al. 2018)**

```@example sid
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(1, 1)],                  # output fixed on impact
    signs = [sign_restriction(3, 1, :positive),        # FFR up
             sign_restriction(2, 1, :negative)])       # prices down
arias = identify_arias(model, restrictions, 20; n_draws=200, rng=MersenneTwister(21))
(acceptance = round(arias.acceptance_rate, digits=4),
 ess = round(arias.ess, digits=1),
 ess_fraction = round(arias.ess_fraction, digits=4))
```

**Recipe 5: Penalty function (Mountford-Uhlig 2009)**

```@example sid
signs_only = SVARRestrictions(3;
    signs = [sign_restriction(3, 3, :positive),
             sign_restriction(1, 3, :negative),
             sign_restriction(2, 3, :negative)])
uhlig = identify_uhlig(model, signs_only, 20; rng=MersenneTwister(9))
(penalty = round(uhlig.penalty, digits=2), converged = uhlig.converged)
```

**Recipe 6: Reuse the identification downstream**

```@example sid
decomp = fevd(model, 20; method=:long_run)
report(decomp)
```

---

## The Identification Problem

A reduced-form VAR(p) produces residuals ``u_t`` with covariance ``\Sigma``. The structural decomposition posits:

```math
u_t = B_0 \, \varepsilon_t, \qquad E[\varepsilon_t \varepsilon_t'] = I_n
```

where:
- ``B_0`` is the ``n \times n`` contemporaneous impact matrix (maps structural shocks to reduced-form innovations)
- ``\varepsilon_t`` are orthogonal structural shocks with unit variance

The restriction ``\Sigma = B_0 B_0'`` is satisfied by any ``B_0 = P Q`` where ``P = \text{chol}(\Sigma)`` and ``Q`` is an orthogonal rotation (``Q'Q = I``). The choice of ``Q`` determines the economic interpretation of the shocks. Every identification scheme in this package reduces to selecting ``Q`` under a different constraint set.

The six economic schemes, the instrument / pattern / share schemes, and the fifteen statistical schemes share the `method=` keyword of `irf`, `fevd`, and `historical_decomposition`. Set-identified schemes that need a restriction object also have dedicated entry points.

| Scheme | Public interface | Extra arguments |
|--------|------------------|-----------------|
| Cholesky | `irf(m, H; method=:cholesky)` | none |
| Sign | `irf(m, H; method=:sign, check_func=f)` or `identify_sign` | `check_func`, `max_draws` |
| Narrative (ADRR) | `identify_arias(m, r, H)` with `narrative_shock_restriction` / `narrative_contribution_restriction` | `n_narrative_sims`, `n_draws` |
| Long-run | `irf(m, H; method=:long_run)` | none |
| Structural VECM | `irf(vecm, H; method=:svec)` | `VECMModel`; `method=:long_run` is an alias |
| Zero + sign | `identify_arias(m, restrictions, H)` or `method=:arias` | `SVARRestrictions` |
| Penalty function | `identify_uhlig(m, restrictions, H)` or `method=:uhlig` | `SVARRestrictions` |
| Proxy | `irf(m, H; method=:proxy, instruments=Z)` | `instruments` |
| AB-model | `irf(m, H; method=:ab, pattern=…)` | `pattern` |
| Max-share | `irf(m, H; method=:max_share, target=…)` | `target` |

The `method=` keyword accepts **twenty-five** symbols: the ten above plus fifteen statistical schemes (five ICA, four ML, the `:nongaussian_ml` dispatcher, GMM, and four heteroskedasticity estimators). Every one of them returns the same `ImpulseResponse` object, so switching identification never changes the downstream code.

!!! note "Technical Note"
    `irf(m, H; method=:sign)` and `irf(m, H; method=:narrative)` run `identify_sign` / `identify_narrative` with `store_all=true` and return the **pointwise median** of the identified set. Bands are identified-set quantiles (`ci_type = :identified_set`); `max_draws` defaults to 1000. That median path is a set summary, not an IRF (Fry & Pagan 2011) --- use `median_target` for the admissible rotation closest to it.

---

## Cholesky (Recursive)

The Cholesky decomposition sets ``Q = I``, imposing a lower-triangular structure on ``B_0``:

```math
B_0 = \text{chol}(\Sigma)
```

where:
- ``B_0`` is lower triangular: variable ``i`` responds contemporaneously only to shocks ``1, \ldots, i``

The ordering reflects economic assumptions about the speed of adjustment. Variables ordered first are the most exogenous --- they respond only to their own shocks on impact. In the standard monetary VAR ordering [output, prices, interest rate], the interest rate shock has no contemporaneous effect on output or prices, consistent with the information and implementation lags in monetary policy transmission (Christiano, Eichenbaum & Evans 1999).

```@example sid
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50,
             conf_level=0.90, seed=101)
B0 = identify_cholesky(model)
round.(B0, digits=5)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The impact matrix is exactly lower triangular: the monetary shock moves the federal funds rate by ``0.105`` percentage points on impact while industrial production and prices are pinned at zero by construction. Reading down the first column, a one-standard-deviation output shock raises industrial production by ``0.0072`` log points (0.72 percent) and simultaneously raises the funds rate by ``0.019`` percentage points --- the systematic policy response, not a policy shock. Because the FRED-MD transformation codes put CPI in second log differences, the price entries are responses of monthly *inflation*, not of the price level. [`identify_cholesky`](@ref) returns this matrix directly, without running the IRF pipeline.

Cholesky identification is exact (point identification). Different variable orderings produce different ``B_0`` and hence different IRFs --- there is no statistical test for the "correct" ordering. Economic theory must justify the assumed causal ordering.

---

## Sign Restrictions

**Sign restrictions** identify structural shocks by constraining the signs of impulse responses at selected horizons (Rubio-Ramírez, Waggoner & Zha 2010). The algorithm draws random orthogonal matrices ``Q`` from the Haar measure and retains only those producing IRFs consistent with the sign constraints:

```math
Q \in O(n): \quad \text{sign}(\Theta_h(Q))_{i,j} = s_{i,j,h} \quad \forall (i, j, h) \in \mathcal{S}
```

where:
- ``O(n)`` is the orthogonal group (``Q'Q = I``)
- ``\Theta_h(Q) = \Phi_h \, P \, Q`` is the IRF at horizon ``h`` for rotation ``Q``
- ``\mathcal{S}`` is the set of sign restrictions ``(variable, shock, horizon, sign)``
- ``\Phi_h`` is the ``h``-th MA coefficient from the companion form

The algorithm:
1. Draw ``Q`` uniformly from ``O(n)`` via QR decomposition of a random Gaussian matrix
2. Compute the candidate impact matrix ``B_0 = PQ``
3. Compute IRFs from the candidate ``B_0``
4. Accept if all sign conditions hold; otherwise discard and repeat

The check function receives the full ``H \times n \times n`` IRF array, indexed `[horizon, variable, shock]`, so `resp[1, 3, 3]` is the impact response of variable 3 to shock 3.

```@example sid
# Contractionary monetary shock: FFR rises, INDPRO and CPI fall on impact
check = resp -> resp[1, 3, 3] > 0 && resp[1, 1, 3] < 0 && resp[1, 2, 3] < 0

result_sign = irf(model, 20; method=:sign, check_func=check, rng=MersenneTwister(3))
(impact = round.(result_sign.values[1, :, 3], digits=4),
 ci_type = result_sign.ci_type,
 impact_95 = round.((result_sign.ci_lower[1, 1, 3], result_sign.ci_upper[1, 1, 3]), digits=4))
```

```julia
plot_result(result_sign)
```

```@raw html
<iframe src="../assets/plots/irf_sign.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The reported path is the pointwise median of the identified set (`ci_type = :identified_set`): industrial production falls ``0.0037`` log points on impact against a ``0.0346`` percentage-point rise in the funds rate, with a 95% identified-set band of ``[-0.0066, -0.0003]``. `max_draws` defaults to 1000. The pointwise median is a set summary, not an IRF --- no single admissible ``Q`` is required to match it at every horizon (Fry & Pagan 2011).

With `store_all=true`, `identify_sign` returns a `SignIdentifiedSet` holding every accepted rotation and its IRFs:

```@example sid
id_set = identify_sign(model, 20, check; max_draws=5000, store_all=true,
                       rng=MersenneTwister(42))
id_set
```

```@example sid
med = irf_median(id_set)
lower, upper = irf_bounds(id_set; quantiles=[0.16, 0.84])
(impact_median = round(med[1, 1, 3], digits=4),
 impact_68 = round.((lower[1, 1, 3], upper[1, 1, 3]), digits=4),
 h6_median = round(med[6, 1, 3], digits=4))
```

```julia
plot_result(id_set)
```

```@raw html
<iframe src="../assets/plots/svar_setid_band.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

498 of 5000 rotations satisfy all three impact conditions, an acceptance rate of ``10.0\%``. Across that set the median impact response of industrial production is ``-0.0030`` log points with a 68% interval of ``[-0.0058, -0.0010]``, which excludes zero --- unsurprisingly, since the restriction imposes the negative sign directly. The response dies out quickly: by ``h = 6`` the median is ``-0.0001``. The width of that band is identification uncertainty alone; it carries no estimation uncertainty, because every rotation is applied to the same point estimate of ``(B, \Sigma)``. Pick a single admissible path with `median_target` (below) rather than treating this median as a structural IRF.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_draws` | `Int` | `1000` | Maximum rotation draws |
| `store_all` | `Bool` | `false` | Return a `SignIdentifiedSet` with all accepted draws |
| `shock_names` | `Union{Nothing,Vector{String}}` | `nothing` | Shock labels (defaults to the variable names) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

!!! note "The median is not an estimator"
    The acceptance rate is the fraction of random draws satisfying all sign conditions simultaneously. Rates below 1% signal restrictions that are overly stringent or nearly contradictory. The median response across admissible rotations is a summary of the set, not a point estimate --- report the full identified set (Baumeister & Hamilton 2015).

---

## Summarising the Identified Set

The pointwise median of a sign- or Arias-identified set need not equal the IRF of any single admissible rotation (Fry & Pagan 2011). Four summaries of a `SignIdentifiedSet` or `AriasSVARResult` pick either one stored draw or a joint band.

- **`median_target`** --- Fry–Pagan (2011): the stored ``Q`` whose IRF is closest to the pointwise median in standardised Euclidean distance
- **`modal_model`** --- Inoue & Kilian (2013): the stored draw at which a Gaussian KDE over vectorised, cell-standardised IRFs attains its maximum
- **`joint_band`** --- Inoue & Kilian (2022): the componentwise envelope of the lowest-loss draws whose weights sum to `level`, always including the median-target IRF
- **`sup_t_band`** --- Montiel Olea & Plagborg-Møller (2019): ``\mathrm{median} \pm c\,\sigma``, with ``c`` the `level` quantile of ``\max_{h,i,j} |\mathrm{IRF}-\mathrm{median}|/\sigma``

```@example sid
mt = median_target(id_set)
mm = modal_model(id_set)
jb_lo, jb_hi = joint_band(id_set; level=0.68)
st_lo, st_hi = sup_t_band(id_set; level=0.68)
(median_target_index = mt.index,
 modal_index = mm.index,
 impact_mt = round.(mt.irf[1, :, 3], digits=4),
 joint_68 = round.((jb_lo[1, 1, 3], jb_hi[1, 1, 3]), digits=4))
```

```julia
plot_result(id_set; view=:median_target)
```

```@raw html
<iframe src="../assets/plots/svar_median_target.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The median-target rotation is draw 312 of 498. Its impact on industrial production is ``-0.0030`` log points --- close to the pointwise median --- against a ``0.0541`` percentage-point rise in the funds rate. The modal draw is a different rotation (index 150). The 68% joint band for the output response on impact is ``[-0.0067, -0.0000]``, wider than the pointwise 68% interval because a path that stays inside the joint band at every ``(h,i,j)`` is a stricter requirement than a pointwise quantile. `plot_result(id_set; view=:median_target)` draws that single admissible IRF; `view=:joint` draws the joint band.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `level` | `Real` | `0.68` | Coverage of `joint_band` / `sup_t_band` |
| `loss` | `Symbol` | `:absolute` | Loss for `joint_band` (`:absolute` is L1 to the pointwise median) |
| `bandwidth` | `Union{Nothing,Real}` | `nothing` | KDE bandwidth for `modal_model` (mean pairwise distance on 32 draws) |

---

## Narrative Restrictions

**Narrative restrictions** constrain structural shocks and historical decompositions at documented dates (Antolín-Díaz & Rubio-Ramírez 2018). They sit on the same `SVARRestrictions` object as traditional signs and zeros; `identify_arias` accepts a rotation only if it also matches the narrative, then reweights the draw by ``1/\hat\omega``.

Two typed constraints are supported:

1. **Shock sign**: at residual-sample dates ``t^*``, structural shock ``j`` was positive (or negative)
2. **Contribution**: over a window, shock ``j``'s historical-decomposition contribution to variable ``i`` satisfies one of
   - Type A (`kind=:most_important`, default): ``|H_{i,j}| > \max_{k \neq j} |H_{i,k}|``
   - Type B (`kind=:overwhelming`): ``|H_{i,j}| > \sum_{k \neq j} |H_{i,k}|``
   - least important (`kind=:least_important`): ``|H_{i,j}| < \min_{k \neq j} |H_{i,k}|``

The contribution of shock ``j`` to the unexpected change in variable ``i`` between dates ``t`` and ``t+h`` is

```math
H_{i,j,t,t+h} = \sum_{\ell=0}^{h} \Theta_\ell[i,j] \, \varepsilon_{j,t+h-\ell}
```

where:
- ``\Theta_\ell`` is the structural IRF at horizon ``\ell``
- ``\varepsilon_{j,s}`` is structural shock ``j`` at date ``s`` (residual-sample index)

Traditional sign restrictions truncate the prior on ``Q``. Narrative restrictions truncate the likelihood, so discarding the rotations that fail the narrative is not enough: draws that are more likely to satisfy it would otherwise be over-weighted. After a draw clears the sign/zero and narrative filters, ``\hat\omega`` is the fraction of `n_narrative_sims` independent paths ``\varepsilon^* \sim N(0,I)`` over the restricted dates that also satisfy the narrative. The Arias importance weight is multiplied by ``1/\hat\omega``. Consequently `ess_fraction < 1` whenever narrative restrictions are present, even with no zeros.

```@example sid
r_nar = SVARRestrictions(3;
    signs = [sign_restriction(3, 3, :positive),
             sign_restriction(1, 3, :negative),
             narrative_shock_restriction(3, [20], :positive)])
arias_nar = identify_arias(model, r_nar, 20; n_draws=200, n_narrative_sims=400,
                           rng=MersenneTwister(4))
pct_nar = irf_percentiles(arias_nar; quantiles=[0.16, 0.5, 0.84])
(impact_median = round.(pct_nar[1, :, 3, 2], digits=4),
 ess_fraction = round(arias_nar.ess_fraction, digits=3),
 n_narrative_sims = arias_nar.n_narrative_sims)
```

The weighted median impact of the monetary shock is a ``0.0036`` log-point drop in industrial production against a ``0.0573`` percentage-point rise in the funds rate, with a 68% band of ``[-0.0057, -0.0012]`` for output. `ess_fraction = 0.998` of 200 draws: a single shock-sign date has ``\hat\omega \approx 1/2`` for every rotation, so the ``1/\hat\omega`` correction is almost uniform and Kish's ESS barely moves. Contribution restrictions and many dates make ``\hat\omega`` depend on ``Q`` and pull `ess_fraction` well below 1. `historical_decomposition(model, r_nar, H)` uses the same weighted draws.

`identify_narrative(model, r_nar, H)` is a thin wrapper around `identify_arias`. The closure form `identify_narrative(model, H, sign_check, narrative_check)` remains the set-aware path for `irf(; method=:narrative)` and `compute_Q(:narrative)`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Target number of accepted rotations |
| `n_rotations` | `Int` | `1000` | Maximum attempts per accepted draw |
| `n_narrative_sims` | `Int` | `1000` | Monte Carlo draws of ``ε*`` used to estimate ``\hat\omega`` |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

| Field | Type | Description |
|-------|------|-------------|
| `ess_fraction` | `T` | Kish ESS over accepted draws; ``< 1`` with narrative restrictions |
| `n_narrative_sims` | `Int` | ``M`` used for ``\hat\omega``; `0` when no narrative restriction is present |
| `weights` | `Vector{T}` | Normalized importance weights (Arias volume element times ``1/\hat\omega``) |

---

## Long-Run (Blanchard-Quah)

**Long-run restrictions** constrain the cumulative effect of structural shocks on selected variables (Blanchard & Quah 1989). The long-run impact matrix is:

```math
C(1) = (I_n - A_1 - A_2 - \cdots - A_p)^{-1} \, B_0
```

where:
- ``C(1)`` is the ``n \times n`` long-run cumulative response matrix
- ``A(1) = A_1 + A_2 + \cdots + A_p`` is the sum of VAR coefficient matrices

Blanchard & Quah (1989) impose that ``C(1)`` is lower triangular, so shocks ordered later have zero long-run effect on variables ordered earlier. The typical application restricts demand shocks to have no long-run effect on output, identifying supply-driven long-run fluctuations.

!!! note "Sign normalization"
    [`identify_long_run`](@ref) returns the orthogonal rotation ``Q``; the structural impact matrix is ``\text{chol}(\Sigma) \cdot Q``. Shocks are sign-normalized so that every shock has a non-negative long-run effect on its own variable, which makes the shock signs deterministic rather than an artifact of the Cholesky factor.

```@example sid
result_lr = irf(model, 40; method=:long_run)
cum_own = [round(sum(result_lr.values[:, j, j]), digits=4) for j in 1:3]
```

```julia
plot_result(result_lr)
```

```@raw html
<iframe src="../assets/plots/irf_longrun.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Every shock has a non-negative cumulative own effect over 40 months --- ``0.0051`` for the output shock, ``0.0013`` for the price shock, ``0.4971`` for the funds-rate shock --- confirming the normalization. On impact, industrial production moves ``0.0052`` log points to the first shock and ``-0.0033`` to the second, so the recursive long-run structure loads the durable output movement on the shock ordered first while leaving the remaining shocks free to move output transitorily. Compare this with the Cholesky column ordering above: the two schemes agree on the impact scale of the output shock but disagree entirely on which shock the interest rate belongs to.

---

## Arias et al. (2018) Zero + Sign Restrictions

When sign restrictions alone are insufficient, **zero restrictions** on specific impulse responses can be imposed alongside sign constraints. Arias, Rubio-Ramírez & Waggoner (2018) develop an importance-sampling algorithm that draws ``Q`` uniformly over the set satisfying the zero restrictions, then filters for sign satisfaction. Importance weights correct for the non-uniform sampling induced by the zero-restriction constraint manifold (Proposition 4).

The algorithm constructs ``Q`` column by column via QR decomposition in the null space of the zero-restriction matrix, then checks the sign restrictions on the candidate IRF ``\Theta_h = \Phi_h \, L \, Q``.

| Type | Constructor | Role |
|------|-------------|------|
| Zero | `zero_restriction(var, shock; horizon=0)` | Linear: variable `var` does not respond to `shock` at `horizon` |
| Long-run zero | `zero_restriction(var, shock; horizon=:long_run)` | Linear: ``e_v' C(1) L q_s = 0`` with ``C(1)=(I-\sum A_i)^{-1}`` |
| Sign | `sign_restriction(var, shock, :positive; horizon=0)` | Rejection: response has the required sign at `horizon` |
| Sign (range) | `sign_restriction(var, shock, :positive; horizons=0:K)` | Expands to ``K+1`` sign restrictions |
| ``A_0`` / ``A_+`` zero | `a0_zero_restriction(eq, shock)`, `aplus_zero_restriction(eq, shock; lag=1)` | Linear in ``Q``: zeros ``A_0[\mathrm{eq},\mathrm{shock}]`` in the RWZ form ``A_0 = L^{-T} Q`` (``y'A_0``), not the column-convention impact ``(LQ)[\mathrm{eq},\mathrm{shock}]`` (Arias, Caldara & Rubio-Ramírez 2019) |
| Elasticity | `elasticity_bound(num, den, shock; lower, upper)` | Rejection: ``\Theta_{\mathrm{num}}/\Theta_{\mathrm{den}}`` in ``[\mathrm{lower},\mathrm{upper}]`` (Kilian & Murphy 2012) |
| Magnitude | `magnitude_bound(var, shock; lower, upper)` | Rejection: IRF entry in a closed interval |
| FEVD share | `fevd_share_restriction(var, shock; horizon, lower, upper)` | Rejection on the shock's forecast-error variance share |
| Cumulative | `cumulative_restriction(var, shock, :positive; horizons=0:H)` | Rejection on the cumulated IRF |
| Narrative shock | `narrative_shock_restriction(shock, dates, :positive)` | Rejection: ``ε_{j,t}`` has the given sign at residual-sample dates |
| Narrative contribution | `narrative_contribution_restriction(var, shock, window; kind=:most_important)` | Rejection: Type A most important (``|H_j| > \max_{k\neq j}|H_k|``); `kind=:overwhelming` is Type B, `:least_important` is ``|H_j| < \min_{k\neq j}|H_k|`` |

`SVARRestrictions` stores linear zeros and rejection restrictions in two lists. `sign_check(r)` returns an `irf -> Bool` closure for `identify_sign`, structural DFM, and counterfactuals; a handwritten predicate remains an escape hatch. Imposing all ``n(n-1)/2`` long-run zeros (on early-ordered shocks) recovers Blanchard–Quah as a just-identified special case, unique up to column sign. Cointegrated systems throw `IdentificationError` --- difference the data or use [`identify_svec`](@ref) on the VECM ([Vector Error Correction Models](@ref vecm_page)).

```@example sid
r_h = SVARRestrictions(3;
    signs = [sign_restriction(3, 1, :positive; horizons=0:2),
             sign_restriction(2, 1, :negative; horizons=0:2)])
id_typed = identify_sign(model, 20, sign_check(r_h); max_draws=2000, store_all=true,
                         rng=MersenneTwister(21))
(n_accepted = id_typed.n_accepted,
 rate = round(id_typed.acceptance_rate; digits=3))
```

`horizons=0:2` expands each sign to three restrictions (impact through two months). `sign_check(r_h)` is the predicate `identify_sign` consumes; the same container feeds `identify_arias`.

!!! warning "Zero restrictions must go on early-ordered shocks"
    Column ``j`` of ``Q`` is drawn from the null space of ``j-1`` orthogonality constraints plus its own zero restrictions, so shock ``j`` admits at most ``n - j`` zeros. Loading a zero restriction onto the last shock of an ``n``-variable system over-constrains the draw and raises `IdentificationError`. Order the restricted shock first.

```@example sid
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(1, 1)],
    signs = [sign_restriction(3, 1, :positive),
             sign_restriction(2, 1, :negative)])

result = identify_arias(model, restrictions, 20; n_draws=200, rng=MersenneTwister(21))
report(result)
```

```@example sid
pct = irf_percentiles(result; quantiles=[0.16, 0.5, 0.84])
(shape = size(pct),
 output_impact = maximum(abs, result.irf_draws[:, 1, 1, 1]),
 ffr_impact_median = round(pct[1, 3, 1, 2], digits=4),
 cpi_h4_median = round(pct[4, 2, 1, 2], digits=5))
```

The sampler accepts 200 draws from 975 attempts (``20.5\%``) and the impact response of industrial production is exactly zero in every one of them, as the zero restriction requires. Under the median weighted draw the funds rate rises ``0.063`` percentage points on impact and monthly inflation is ``-0.00047`` four months out. `irf_percentiles` returns an ``H \times n \times n \times |q|`` array, so the quantile index comes last: `pct[1, 3, 1, 2]` is the median (second quantile) impact response of variable 3 to shock 1.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Target number of accepted draws |
| `n_rotations` | `Int` | `1000` | Maximum attempts per target draw |
| `compute_weights` | `Bool` | `true` | Compute importance weights |
| `normalize_weights` | `Bool` | `true` | Scale stored weights to sum to 1 (`false` keeps the raw volume-element scale) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

| Field | Type | Description |
|-------|------|-------------|
| `Q_draws` | `Vector{Matrix{T}}` | Accepted rotation matrices |
| `irf_draws` | `Array{T,4}` | ``n_{\text{draws}} \times H \times n \times n`` IRF draws |
| `weights` | `Vector{T}` | Importance weights (normalized to sum to 1 by default) |
| `acceptance_rate` | `T` | Fraction of attempts satisfying all restrictions |
| `restrictions` | `SVARRestrictions` | Imposed restrictions |
| `ess` | `T` | Kish effective sample size of the importance weights |
| `ess_fraction` | `T` | ``\mathrm{ESS} / n_{\text{draws}}`` |

### Rank and order conditions

Linear zeros identify the rotation almost everywhere when they satisfy the
Rubio-Ramírez, Waggoner & Zha (2010) rank condition. Order the zeros so shock
``j`` carries ``q_j`` of them. Global identification holds at a generic
``(A_0, A_+)`` if and only if

```math
\mathrm{rank}\bigl(M_j(f(A_0, A_+))\bigr) = n - j, \qquad j = 1,\ldots,n
```

where:
- ``M_j`` stacks the linear zero-restriction rows for shock ``j`` (finite-horizon IRF, long-run, ``A_0``, and ``A_+``)
- ``n - j`` is the number of free directions remaining after ``j-1`` orthogonality constraints
- the **order condition** ``q_j \ge n - j`` is necessary but not sufficient: linearly dependent rows (a repeated impact zero, or an impact zero stacked with a long-run zero when ``C(1) = I``) drop the rank below the count

`check_identification` returns an `IdentificationStatus` with `status ∈ (:exact, :over, :under, :set)`. `:under` and `:over` throw `IdentificationError` from `identify_arias` and `identify_uhlig`. Sign restrictions do not enter the rank; a drawable shortfall of zeros with at least one sign is `:set`. Uhlig still returns a point from that set, stores the RWZ status on the result, and `report` says so.

```@example sid
rec = SVARRestrictions(3; zeros=[zero_restriction(2, 1), zero_restriction(3, 1),
                                 zero_restriction(3, 2)])
st_rec = check_identification(rec, model)
st_sign = check_identification(SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)]), 3)
(recursive = (st_rec.status, st_rec.ranks),
 sign_only = st_sign.status)
```

The three recursive impact zeros recover Cholesky: ``\mathrm{rank}(M_j) = (2, 1, 0) = n - j`` and the status is `:exact`. The sign-only container has no linear zeros, so the rank condition fails and the status is `:set` --- Arias samples the identified set; Uhlig reports one penalty-optimal rotation from it.

| Field | Type | Description |
|-------|------|-------------|
| `status` | `Symbol` | `:exact`, `:over`, `:under`, or `:set` |
| `ranks` | `Vector{Int}` | ``\mathrm{rank}(M_j)`` for shock ``j`` (generic point) |
| `orders` | `Vector{Int}` | Number of linear zeros on shock ``j`` |
| `n_overidentifying` | `Int` | ``\sum_j \max(\mathrm{rank}_j - (n-j), 0)`` |

### Effective Sample Size

Uneven importance weights mean the weighted IRF summaries rest on fewer draws than the nominal count. The result reports Kish's (1965) **effective sample size**

```math
\mathrm{ESS} = \frac{\left(\sum_s w_s\right)^2}{\sum_s w_s^2}
```

where:
- ``w_s`` is draw ``s``'s importance weight (the ratio is scale-invariant, so normalization does not matter)
- ``\mathrm{ESS} = n`` exactly when the weights are uniform, and approaches 1 when one draw carries all the mass

Under pure sign restrictions the weights are uniform and `ess_fraction` is exactly 1 --- nothing is lost, because no importance sampling takes place. Zero restrictions make the weights uneven, so `ess_fraction` falls below 1; a value near zero means the posterior rests on a handful of draws and the credible bands are far less precise than `n_draws` suggests. Below 10% of the draw count the sampler is reported as degenerate and `identify_arias` warns.

```@example sid
sign_only = SVARRestrictions(3;
    signs = [sign_restriction(3, 1, :positive),
             sign_restriction(2, 1, :negative)])
arias_sign = identify_arias(model, sign_only, 20; n_draws=200, rng=MersenneTwister(21))

(with_zero = (round(result.ess, digits=1), round(result.ess_fraction, digits=3)),
 sign_only = (round(arias_sign.ess, digits=1), round(arias_sign.ess_fraction, digits=3)),
 weight_ratio = round(maximum(result.weights) / minimum(result.weights), digits=0))
```

Dropping the single zero restriction moves the effective sample from ``47.3`` of 200 draws (``23.6\%``) to the full ``200``. The 200 accepted zero-restricted draws therefore support inference no sharper than about 47 equally weighted ones, because the heaviest weight in that sample is some 71000 times the lightest. This is the same degeneracy signal that triggers resampling in sequential Monte Carlo. Treat a low value as evidence that the zero and sign restrictions are close to contradictory, not as a reason to draw more rotations --- the weight distribution is a property of the restriction set.

### Bayesian Integration

For Bayesian VARs, `identify_arias_bayesian` applies the Arias algorithm to each posterior draw, producing posterior-weighted IRF bands that account for both parameter and identification uncertainty:

```julia
post = estimate_bvar(Y, 2; n_draws=500)
res = identify_arias_bayesian(post, restrictions, 20;
    n_rotations=100, quantiles=[0.16, 0.5, 0.84])
res.irf_quantiles, res.irf_mean, res.ess, res.ess_fraction
```

Weights are pooled across posterior draws on the raw volume-element scale and normalized once at the end, so `ess` measures degeneracy over the whole pooled sample. Each per-draw call accepts a single rotation, so normalizing within it would force every weight to 1 and reduce the weighted summaries to unweighted ones.

The same restriction object drives historical decompositions: `historical_decomposition(model, restrictions, horizon)` runs `identify_arias` internally and returns weighted posterior quantiles of the shock contributions.

Haar / uniform-on-``O(n)`` Bayesian inference is a **single-prior** procedure: the rotation prior is unrevisable given ``(B, \Sigma)``. Giacomini & Kitagawa (2021) drop that prior. `identify_robust_bayes` reports, for every IRF entry,

- the **set of posterior means** of the identified-set bounds ``[\ell(B,\Sigma), u(B,\Sigma)]``
- a **robust credible region** --- the smallest interval containing `level` of those sets
- the Haar single-prior equal-tailed interval from `identify_arias_bayesian`, for comparison
- the informativeness diagnostic ``\kappa = 1 - |C_{\mathrm{single}}| / |C_{\mathrm{robust}}|`` (clipped to ``[0,1]``) and the empty-set probability ``\Pi(\emptyset)``

``\kappa`` near 0 means the Haar interval is essentially prior-free; ``\kappa`` near 1 means it is prior-driven. The robust region is **not** expanded post-hoc to nest the Haar interval.

```@example sid
r_gk = identify_robust_bayes(post_rb, signs_only, 12; level=0.68,
                             solver=:draws, n_draws=50, n_rotations=50,
                             rng=MersenneTwister(11))
report(r_gk)
```

```@example sid
(informativeness = round(r_gk.informativeness, digits=3),
 empty_set_prob = round(r_gk.empty_set_prob, digits=3),
 output_robust = round.((r_gk.robust_lower[1, 1, 3], r_gk.robust_upper[1, 1, 3]), digits=4))
```

```julia
plot_result(r_gk)
```

```@raw html
<iframe src="../assets/plots/svar_robust_bayes.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

On this 100-draw posterior ``\kappa = 0.31`` and the identified set is never empty. The Haar interval is about 30% narrower than the robust region because the uniform-on-``O(n)`` prior concentrates mass inside the identified set. Report both. `solver=:optimize` is the closed form for ``n = 2`` with linear-in-``Q`` restrictions; ``n = 3`` uses Haar draws of the identified-set bounds (`solver=:draws`).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `level` | `Real` | `0.68` | Credibility level of the robust region |
| `solver` | `Symbol` | `:optimize` | `:optimize` (``n=2`` closed form) or `:draws` |
| `n_draws` | `Int` | `200` | Haar draws per posterior draw when `solver=:draws` |
| `n_rotations` | `Int` | `100` | Arias attempts per posterior draw for the single-prior interval |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

| Field | Type | Description |
|-------|------|-------------|
| `lower`, `upper` | `Array{T,3}` | Posterior means of the identified-set bounds |
| `robust_lower`, `robust_upper` | `Array{T,3}` | Robust credible region |
| `single_prior_lower`, `single_prior_upper` | `Array{T,3}` | Haar equal-tailed interval |
| `informativeness` | `T` | ``\kappa \in [0,1]`` |
| `empty_set_prob` | `T` | Posterior probability that the identified set is empty |
| `level` | `T` | Credibility level |

---

## Mountford-Uhlig (2009) Penalty Function

When a single best rotation is preferred over a distribution of draws, Mountford & Uhlig (2009) provide a **penalty function** approach. Zero restrictions of every linear type (finite IRF, long-run, ``A_0``, ``A_+``) are enforced exactly via null-space projection; `SignRestriction`s are encouraged through a penalty minimized with two-phase Nelder-Mead. Elasticity, magnitude, FEVD, cumulative, and ``A_0``/``A_+`` *sign* restrictions are not in the penalty --- `identify_uhlig` throws `ArgumentError` on those mixed containers; use `identify_arias`.

```math
x_s = -\mathrm{sign}_s \cdot \frac{\mathrm{IRF}_s}{\sigma_s}, \qquad
f(x) = \begin{cases} x & x \le 0 \\ 100\, x & x > 0 \end{cases}, \qquad
\mathrm{penalty} = \sum_{s \in \mathcal{S}} f(x_s)
```

where:
- ``x_s > 0`` if and only if sign restriction ``s`` is violated
- ``\mathrm{sign}_s \in \{+1,-1\}`` is the required sign direction
- ``\sigma_s = \sqrt{\Sigma_{ii}}`` is the reduced-form residual standard deviation of the response variable

Weight 1 applies when the restriction is satisfied and weight 100 when it is violated (Uhlig 2005, §3.3; Mountford & Uhlig 2009, §3). Minimization therefore makes violations prohibitively expensive rather than rewarding large already-admissible responses.

```@example sid
result_uhlig = identify_uhlig(model, signs_only, 20; rng=MersenneTwister(9))
report(result_uhlig)
```

```@example sid
(penalty = round(result_uhlig.penalty, digits=2),
 per_shock = round.(result_uhlig.shock_penalties, digits=2),
 impact = round.(result_uhlig.irf[1, :, 3], digits=4))
```

All three sign conditions are satisfied (`converged = true`) at a total penalty of ``-1.55``, carried entirely by shock 3 --- the only shock carrying restrictions. Because satisfied restrictions enter at weight 1, that figure is the sum of the three normalised IRFs, not a hundred-fold reward. The selected rotation puts the impact response of industrial production at ``-0.0039`` log points and the funds rate at ``0.0474`` percentage points, close to the median of the sign-restricted set; the penalty's job is to rule out violations, not to inflate admissible magnitudes.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_starts` | `Int` | `50` | Random starting points for the coarse search |
| `n_refine` | `Int` | `10` | Local refinements in the second phase |
| `max_iter_coarse` | `Int` | `500` | Maximum Nelder-Mead iterations (coarse) |
| `max_iter_fine` | `Int` | `2000` | Maximum iterations (refinement) |
| `tol_coarse` | `Real` | ``10^{-4}`` | Convergence tolerance (coarse) |
| `tol_fine` | `Real` | ``10^{-8}`` | Convergence tolerance (refinement) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Optimal rotation matrix |
| `irf` | `Array{T,3}` | ``H \times n \times n`` impulse responses |
| `penalty` | `T` | Total penalty at the optimum (more negative is better) |
| `shock_penalties` | `Vector{T}` | Per-shock penalty contributions |
| `restrictions` | `SVARRestrictions` | Imposed restrictions |
| `converged` | `Bool` | Whether all sign restrictions are satisfied |

!!! note "When to use Uhlig vs Arias"
    Use `identify_uhlig` when a single point-identified rotation is needed --- for example, as a starting point for policy analysis. Use `identify_arias` when the full identified set is required for inference with credible intervals.

---

## Choosing an Identification Scheme

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Recursive contemporaneous ordering | Cholesky | Simple, exactly identified |
| Long-run neutrality (stationary VAR) | Blanchard-Quah | Supply versus demand |
| Permanent vs transitory (VECM) | [SVEC](@ref vecm_page) | Common trends; ``C(1)`` is singular |
| External instrument for one (or ``k``) shock | [Proxy](@ref id_proxy_page) | High-frequency or narrative proxy |
| Non-recursive short-run zeros | [AB-model](@ref id_ab_page) | Amisano–Giannini ML with LR over-ID |
| Agnostic sign constraints | Sign restrictions | Avoids specifying exact zeros |
| Exact zeros plus signs | Arias et al. | Importance-weighted identified set |
| Single optimal rotation from signs | Uhlig penalty | Fast, deterministic |
| Shock that dominates a target | [Max-share](@ref id_maxshare_page) | News and main-cycle shocks |
| Documented historical episodes | Narrative (ADRR) | Truncates the likelihood, reweights by ``1/\hat\omega`` |
| Independent non-Gaussian shocks | [ICA / ML](@ref id_nongaussian_page) | No economic restrictions |
| Independence via higher moments | [GMM](@ref id_nongaussian_page) | Sandwich SEs and Hansen ``J`` |
| Changing shock volatility | [Heteroskedastic](@ref id_heteroskedastic_page) | Regime covariances identify ``B_0`` |
| Prior-robust set inference | Robust Bayes | Giacomini–Kitagawa multiple priors |

---

## Complete Example

This example identifies a contractionary monetary policy shock in the FRED-MD system [output, prices, interest rate] two ways --- recursively via Cholesky and via sign restrictions --- and compares the impact response of industrial production. Both schemes share the same reduced-form VAR(2).

```@example sid
# Reduced-form VAR on the monetary system (FFR ordered last)
svar = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])

# Recursive (Cholesky) identification of the monetary shock
chol = irf(svar, 20; method=:cholesky, ci_type=:bootstrap, reps=50, seed=7)
report(chol)
```

```@example sid
# Sign-restricted identification: FFR up, INDPRO and CPI down on impact
sign_set = identify_sign(svar, 20, check; max_draws=5000, store_all=true,
                         rng=MersenneTwister(42))
sign_med = irf_median(sign_set)

(cholesky_impact = round(chol.values[1, 1, 3], digits=4),
 sign_median_impact = round(sign_med[1, 1, 3], digits=4),
 acceptance_rate = round(sign_set.acceptance_rate, digits=3))
```

With the interest rate ordered last, the Cholesky scheme forces the contemporaneous response of industrial production to the monetary shock to be exactly ``0.0`` --- output cannot react within the month. Sign restrictions require only that the impact response be negative, so the median across the 498 admissible rotations is a nonzero contraction of ``-0.0030`` log points. The two schemes encode different assumptions about within-period monetary transmission, and the gap between their impact responses is the price of the recursive zero restriction. Neither number is more "correct" than the other; the choice is an economic argument, and the sign-restricted answer is a set summary that must be reported with `irf_bounds`.

---

## Common Pitfalls

1. **Variable ordering matters for Cholesky.** Different orderings produce different IRFs. There is no statistical test for the correct ordering --- economic theory must justify the assumed causal structure.

2. **Sign restrictions are set-identified.** The median response across admissible rotations is a summary statistic, not a point estimate. Report the full credible set to avoid overstating precision (Uhlig 2005).

3. **The pointwise median is not an IRF.** `irf(; method=:sign)` and `method=:narrative` report the pointwise median of the identified set; that path need not correspond to any single admissible rotation (Fry & Pagan 2011). Use `median_target` for the admissible rotation closest to that median, or `modal_model` for the Inoue–Kilian modal draw. With tight restrictions that exhaust `max_draws` (default 1000), raise `max_draws` on `irf` itself.

4. **Low acceptance rates.** If `identify_sign` or `identify_arias` produces acceptance rates below 1%, the restrictions may be nearly contradictory. Relax the restrictions or increase `max_draws`.

5. **Reading `identify_arias` draw counts at face value.** The acceptance rate says how many draws survived the restrictions; `ess_fraction` says how many of them actually count. With zero or narrative restrictions the two diverge --- check `ess_fraction` before trusting the width of a credible band. Narrative restrictions reweight by ``1/\hat\omega``; simply dropping the rotations that fail the historical constraint overstates the posterior of those more likely to match the episode (Antolín-Díaz & Rubio-Ramírez 2018).

6. **Zero restrictions on late-ordered shocks over-constrain the draw.** Shock ``j`` admits at most ``n - j`` zero restrictions in both `identify_arias` and `identify_uhlig`. Reorder the system so the heavily restricted shock comes first. `check_identification` reports `:under` when the RWZ rank or order condition fails and `:over` when extra independent zeros empty the null space; both routines throw `IdentificationError` before sampling.

7. **Uhlig may not converge.** If `result.converged == false`, increase `n_starts` or relax the sign restrictions. The optimizer found a local minimum where some sign conditions are violated.

8. **Uhlig only penalizes `SignRestriction`.** Elasticity, magnitude, FEVD, cumulative, and ``A_0``/``A_+`` signs throw `ArgumentError` from `identify_uhlig`; use `identify_arias`. ``A_0``/``A_+`` and long-run *zeros* remain null-space rows. `A0[eq, shock]` is the RWZ ``y'A_0`` entry ``(L^{-T} Q)[\mathrm{eq},\mathrm{shock}]``, not the impact ``(LQ)[\mathrm{eq},\mathrm{shock}]``.

9. **Long-run identification requires stationarity.** If the VAR has a near-unit root, ``(I - A(1))`` is nearly singular and the long-run matrix ``C(1)`` explodes; `identify_long_run` warns when the condition number crosses ``1/\sqrt{\varepsilon}``. Use a VECM specification for cointegrated systems.

10. **The order condition is not the rank condition.** Loading ``n(n-1)/2`` zeros on shock 1 satisfies the *count* for that shock and starves shocks ``2,\ldots,n``. An impact zero stacked with a long-run zero on the same entry can also pass the count while ``\mathrm{rank}(M_j) < n - j``. Read `IdentificationStatus.ranks`, not just `orders`.

---

## References

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs.
  *American Economic Review*, 108(10), 2802--2829. [DOI](https://doi.org/10.1257/aer.20161852)

- Arias, J. E., Caldara, D., & Rubio-Ramírez, J. F. (2019). The Systematic Component of Monetary Policy in SVARs: An Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 101, 1--13. [DOI](https://doi.org/10.1016/j.jmoneco.2018.07.010)

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications.
  *Econometrica*, 86(2), 685--720. [DOI](https://doi.org/10.3982/ECTA14468)

- Baumeister, C., & Hamilton, J. D. (2015). Sign Restrictions, Structural Vector Autoregressions, and Useful Prior Information.
  *Econometrica*, 83(5), 1963--1999. [DOI](https://doi.org/10.3982/ECTA12356)

- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances.
  *American Economic Review*, 79(4), 655--673. [JSTOR](https://www.jstor.org/stable/1827924)

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary Policy Shocks: What Have We Learned and to What End?
  In *Handbook of Macroeconomics*, Vol. 1A, 65--148. [DOI](https://doi.org/10.1016/S1574-0048(99)01005-8)

- Fry, R., & Pagan, A. (2011). Sign Restrictions in Structural Vector Autoregressions: A Critical Review.
  *Journal of Economic Literature*, 49(4), 938--960. [DOI](https://doi.org/10.1257/jel.49.4.938)

- Giacomini, R., & Kitagawa, T. (2021). Robust Bayesian Inference for Set-Identified Models.
  *Econometrica*, 89(4), 1519--1556. [DOI](https://doi.org/10.3982/ECTA16773)

- Inoue, A., & Kilian, L. (2013). Inference on Impulse Response Functions in Structural VAR Models.
  *Journal of Econometrics*, 177(1), 1--13. [DOI](https://doi.org/10.1016/j.jeconom.2013.04.013)

- Inoue, A., & Kilian, L. (2022). Joint Bayesian Inference about Impulse Responses in VAR Models.
  *Journal of Econometrics*, 231(2), 457--476. [DOI](https://doi.org/10.1016/j.jeconom.2021.05.010)

- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Kilian, L., & Murphy, D. P. (2012). Why Agnostic Sign Restrictions Are Not Enough: Understanding the Dynamics of Oil Market VAR Models.
  *Journal of the European Economic Association*, 10(5), 1166--1188. [DOI](https://doi.org/10.1111/j.1542-4774.2012.01080.x)

- Kish, L. (1965). *Survey Sampling*. Wiley. ISBN 978-0-471-48900-9.

- Montiel Olea, J. L., & Plagborg-Møller, M. (2019). Simultaneous Confidence Bands: Theory, Implementation, and an Application to SVARs.
  *Journal of Applied Econometrics*, 34(1), 1--17. [DOI](https://doi.org/10.1002/jae.2656)

- Mountford, A., & Uhlig, H. (2009). What Are the Effects of Fiscal Policy Shocks?
  *Journal of Applied Econometrics*, 24(6), 960--992. [DOI](https://doi.org/10.1002/jae.1079)

- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference.
  *Review of Economic Studies*, 77(2), 665--696. [DOI](https://doi.org/10.1111/j.1467-937X.2009.00578.x)

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 52(2), 381--419. [DOI](https://doi.org/10.1016/j.jmoneco.2004.05.007)
