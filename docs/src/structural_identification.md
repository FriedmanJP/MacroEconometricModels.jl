# [Structural Identification](@id structural_identification_page)

Structural identification recovers the mapping from reduced-form VAR residuals to economically interpretable structural shocks. The reduced-form covariance ``\Sigma = B_0 B_0'`` provides ``n(n+1)/2`` equations for ``n^2`` unknowns in the impact matrix ``B_0``, leaving ``n(n-1)/2`` free parameters. Additional restrictions --- economic, statistical, or a combination --- pin down the remaining degrees of freedom. MacroEconometricModels.jl implements six structural identification schemes:

- **Cholesky (recursive)** --- lower-triangular ``B_0`` via Cholesky decomposition (Christiano, Eichenbaum & Evans 1999)
- **Sign restrictions** --- set identification via random rotations satisfying inequality constraints (Rubio-Ramírez, Waggoner & Zha 2010)
- **Narrative restrictions** --- sign restrictions augmented with historical event constraints (Antolín-Díaz & Rubio-Ramírez 2018)
- **Long-run (Blanchard-Quah)** --- lower-triangular long-run cumulative impact matrix (Blanchard & Quah 1989)
- **Zero + sign restrictions** --- exact zero restrictions with sign constraints and importance-weighted inference (Arias, Rubio-Ramírez & Waggoner 2018)
- **Penalty function (Mountford-Uhlig)** --- point-identified rotation via constrained optimization (Mountford & Uhlig 2009)

For statistical identification via non-Gaussianity or heteroskedasticity (18 additional methods), see [Statistical Identification](@ref nongaussian_page).

```@setup sid
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

**Recipe 1: Cholesky IRF**

```@example sid
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
report(result)
```

**Recipe 2: Sign restrictions with identified set**

```@example sid
check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0 && irf[1, 2, 3] < 0
id_set = identify_sign(model, 20, check; max_draws=5000, store_all=true)
id_set
```

**Recipe 3: Long-run (Blanchard-Quah) identification**

```@example sid
result_lr = irf(model, 40; method=:long_run)
report(result_lr)
```

**Recipe 4: Arias et al. zero + sign restrictions**

```@example sid
restrictions = SVARRestrictions(3;
    signs = [sign_restriction(3, 3, :positive),
             sign_restriction(1, 1, :positive)]
)
result_arias = identify_arias(model, restrictions, 20; n_draws=500)
result_arias
```

**Recipe 5: Uhlig penalty function**

```@example sid
result_uhlig = identify_uhlig(model, restrictions, 20)
result_uhlig
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

The restriction ``\Sigma = B_0 B_0'`` is satisfied by any ``B_0 = P Q`` where ``P = \text{chol}(\Sigma)`` and ``Q`` is an orthogonal rotation (``Q'Q = I``). The choice of ``Q`` determines the economic interpretation of the shocks. All identification schemes in this package reduce to selecting ``Q`` under different constraint sets.

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
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50, conf_level=0.90)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The Cholesky identification is exact (point identification). Different variable orderings produce different ``B_0`` and hence different IRFs --- there is no statistical test for the "correct" ordering. Economic theory must justify the assumed causal ordering.

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
2. Compute the candidate impact matrix: ``B_0 = PQ``
3. Compute IRFs from the candidate ``B_0``
4. Accept if all sign conditions hold; otherwise discard and repeat

```@example sid
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])

# Contractionary monetary shock: FFR rises, INDPRO and CPI fall
check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0 && irf[1, 2, 3] < 0

# Full identified set
id_set = identify_sign(model, 20, check; max_draws=5000, store_all=true)
id_set
```

With `store_all=true`, `identify_sign` returns a `SignIdentifiedSet` containing all accepted rotations and their IRFs. Use `irf_bounds` and `irf_median` to summarize:

```@example sid
med = irf_median(id_set)
lower, upper = irf_bounds(id_set; quantiles=[0.16, 0.84])
nothing # hide
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_draws` | `Int` | `1000` | Maximum rotation draws |
| `store_all` | `Bool` | `false` | Return `SignIdentifiedSet` with all accepted draws |

!!! note "Technical Note"
    The acceptance rate indicates what fraction of random draws satisfy all sign conditions simultaneously. Rates below 1% suggest the restrictions may be overly stringent or nearly contradictory. The median response across admissible rotations is a summary statistic, not a point estimate --- report the full identified set (Baumeister & Hamilton 2015).

---

## Narrative Restrictions

**Narrative restrictions** augment sign restrictions with historical information about specific shocks at particular dates (Antolín-Díaz & Rubio-Ramírez 2018). Two types of narrative constraints:

1. **Shock sign narrative**: at date ``t^*``, structural shock ``j`` was positive (or negative)
2. **Shock contribution narrative**: at date ``t^*``, shock ``j`` was the dominant driver of variable ``i``

The algorithm first filters for sign-satisfying rotations, then checks whether the recovered structural shocks ``\varepsilon = B_0^{-1} u`` satisfy the narrative conditions. This sequential filtering sharply reduces the identified set.

```@example sid
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])

sign_check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0
narrative_check = shocks -> shocks[20, 3] > 0

Q, irfs, shocks = identify_narrative(model, 20, sign_check, narrative_check; max_draws=5000)
nothing # hide
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_draws` | `Int` | `1000` | Maximum rotation draws |

---

## Long-Run (Blanchard-Quah)

**Long-run restrictions** constrain the cumulative effect of structural shocks on selected variables (Blanchard & Quah 1989). The long-run impact matrix is:

```math
C(1) = (I_n - A_1 - A_2 - \cdots - A_p)^{-1} \, B_0
```

where:
- ``C(1)`` is the ``n \times n`` long-run cumulative response matrix
- ``A(1) = A_1 + A_2 + \cdots + A_p`` is the sum of VAR coefficient matrices

Blanchard & Quah (1989) impose that ``C(1)`` is lower triangular, so that shocks ordered later have zero long-run effect on variables ordered earlier. The typical application restricts demand shocks to have no long-run effect on output, identifying supply-driven long-run fluctuations.

```@example sid
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
result = irf(model, 40; method=:long_run)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_longrun.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Arias et al. (2018) Zero + Sign Restrictions

When sign restrictions alone are insufficient, **zero restrictions** on specific impulse responses can be imposed alongside sign constraints. Arias, Rubio-Ramírez & Waggoner (2018) develop an importance-sampling algorithm that draws ``Q`` uniformly over the set satisfying zero restrictions, then filters for sign satisfaction. Importance weights correct for non-uniform sampling induced by the zero-restriction constraint manifold (Proposition 4).

The algorithm constructs ``Q`` column-by-column via QR decomposition in the null space of the zero restriction matrix, then checks sign restrictions on the candidate IRF ``\Theta_h = \Phi_h \, L \, Q``.

| Type | Function | Description |
|------|----------|-------------|
| Zero | `zero_restriction(var, shock; horizon=0)` | Variable `var` does not respond to `shock` at `horizon` |
| Sign | `sign_restriction(var, shock, :positive; horizon=0)` | Response has required sign at `horizon` |

```@example sid
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])

restrictions = SVARRestrictions(3;
    signs = [sign_restriction(3, 3, :positive),
             sign_restriction(1, 1, :positive)]
)

result = identify_arias(model, restrictions, 20; n_draws=500)
result
```

```@example sid
pct = irf_percentiles(result; quantiles=[0.16, 0.5, 0.84])
nothing # hide
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Target number of accepted draws |
| `n_rotations` | `Int` | `1000` | Maximum attempts per target draw |
| `compute_weights` | `Bool` | `true` | Compute importance weights |

| Field | Type | Description |
|-------|------|-------------|
| `Q_draws` | `Vector{Matrix{T}}` | Accepted rotation matrices |
| `irf_draws` | `Array{T,4}` | ``n_{\text{draws}} \times H \times n \times n`` IRF draws |
| `weights` | `Vector{T}` | Importance weights (normalized to sum to 1) |
| `acceptance_rate` | `T` | Fraction of draws satisfying all restrictions |
| `restrictions` | `SVARRestrictions` | Imposed restrictions |

### Bayesian Integration

For Bayesian VARs, `identify_arias_bayesian` applies the Arias algorithm to each posterior draw, producing posterior-weighted IRF bands that account for both parameter and identification uncertainty:

```julia
post = estimate_bvar(Y, 2; n_draws=500)
irf_q, irf_m, acc, total, w = identify_arias_bayesian(post, restrictions, 20;
    n_rotations=100, quantiles=[0.16, 0.5, 0.84])
```

---

## Mountford-Uhlig (2009) Penalty Function

When a single best rotation is preferred over a distribution of draws, Mountford & Uhlig (2009) provide a **penalty function** approach. Zero restrictions are enforced exactly via null-space projection; sign restrictions are encouraged through a penalty function minimized with two-phase Nelder-Mead optimization.

```math
\text{penalty} = -\sum_{s \in \mathcal{S}} w_s \cdot \text{sign}_s \cdot \frac{\text{IRF}_s}{\sigma_s}
```

where:
- ``w_s = 100`` if the sign restriction is satisfied, ``w_s = 1`` if violated
- ``\text{sign}_s \in \{+1, -1\}`` is the required sign direction
- ``\sigma_s`` is the standard deviation of the response variable (normalization)

```@example sid
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])

restrictions = SVARRestrictions(3;
    signs = [sign_restriction(1, 1, :positive),
             sign_restriction(3, 3, :positive)]
)

result = identify_uhlig(model, restrictions, 20)
result
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_starts` | `Int` | `50` | Random starting points for coarse search |
| `n_refine` | `Int` | `10` | Top candidates refined in second phase |
| `max_iter_coarse` | `Int` | `500` | Maximum Nelder-Mead iterations (coarse) |
| `max_iter_fine` | `Int` | `2000` | Maximum iterations (refinement) |

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Optimal rotation matrix |
| `irf` | `Array{T,3}` | ``H \times n \times n`` impulse responses |
| `penalty` | `T` | Total penalty at optimum |
| `converged` | `Bool` | Whether all sign restrictions satisfied |

!!! note "When to use Uhlig vs Arias"
    Use `identify_uhlig` when a single point-identified rotation is needed --- for example, as a starting point for policy analysis. Use `identify_arias` when the full identified set is required for inference with credible intervals.

---

## Choosing an Identification Scheme

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Baseline recursive IRFs | Cholesky | Simple, transparent, widely used |
| Agnostic about magnitudes | Sign restrictions | Avoids specifying exact zeros |
| Historical event information | Narrative | Sharply reduces identified set |
| Long-run neutrality | Blanchard-Quah | Natural for supply vs demand |
| Exact zero + sign constraints | Arias et al. | Importance-weighted inference |
| Single optimal rotation | Uhlig penalty | Fast, deterministic |
| Non-Gaussian shocks | [Statistical ID](@ref nongaussian_page) | 18 methods via `compute_Q` |

---

## Complete Example

This example identifies a contractionary monetary policy shock in the FRED-MD system [output, prices, interest rate] two ways --- recursively via Cholesky and via sign restrictions --- and compares the impact response of industrial production. Both schemes share the same reduced-form VAR(2).

```@example sid
# Reduced-form VAR on the monetary system (FFR ordered last)
svar = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])

# Recursive (Cholesky) identification of the monetary shock
chol = irf(svar, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
report(chol)
```

```@example sid
# Sign-restricted identification: FFR up, INDPRO and CPI down on impact
check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0 && irf[1, 2, 3] < 0
sign_set = identify_sign(svar, 20, check; max_draws=5000, store_all=true)
sign_med = irf_median(sign_set)

# Impact response of INDPRO to the monetary shock under each scheme
(cholesky_indpro_impact = round(chol.values[1, 1, 3], digits=4),
 sign_median_indpro_impact = round(sign_med[1, 1, 3], digits=4))
```

With the interest rate ordered last, the Cholesky scheme forces the contemporaneous response of industrial production to the monetary shock to be exactly zero --- output cannot react within the period. Sign restrictions instead require only that the impact response be negative, so the median across admissible rotations is a nonzero contraction (summarized further by `irf_median` and `irf_bounds`). The two schemes encode different identifying assumptions about the within-period transmission of monetary policy, and the gap between their impact responses is the cost of the recursive zero restriction.

---

## Common Pitfalls

1. **Variable ordering matters for Cholesky.** Different orderings produce different IRFs. There is no statistical test for the correct ordering --- economic theory must justify the assumed causal structure.

2. **Sign restrictions are set-identified.** The median response across admissible rotations is a summary statistic, not a point estimate. Report the full credible set to avoid overstating precision (Uhlig 2005).

3. **Low acceptance rates.** If `identify_sign` or `identify_arias` produces acceptance rates below 1%, the restrictions may be nearly contradictory. Relax restrictions or increase `max_draws`.

4. **Uhlig may not converge.** If `result.converged == false`, increase `n_starts` or relax sign restrictions. The optimizer found a local minimum where some sign conditions are violated.

5. **Long-run identification requires stationarity.** If the VAR has a near-unit root, ``(I - A(1))`` is nearly singular and the long-run matrix ``C(1)`` explodes. Use a VECM specification for cointegrated systems.

---

## References

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs.
  *American Economic Review*, 108(10), 2802--2829. [DOI](https://doi.org/10.1257/aer.20161852)

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications.
  *Econometrica*, 86(2), 685--720. [DOI](https://doi.org/10.3982/ECTA14468)

- Baumeister, C., & Hamilton, J. D. (2015). Sign Restrictions, Structural Vector Autoregressions, and Useful Prior Information.
  *Econometrica*, 83(5), 1963--1999. [DOI](https://doi.org/10.3982/ECTA12356)

- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances.
  *American Economic Review*, 79(4), 655--673. [JSTOR](https://www.jstor.org/stable/1827924)

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary Policy Shocks: What Have We Learned and to What End?
  In *Handbook of Macroeconomics*, Vol. 1A, 65--148. [DOI](https://doi.org/10.1016/S1574-0048(99)01005-8)

- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Mountford, A., & Uhlig, H. (2009). What Are the Effects of Fiscal Policy Shocks?
  *Journal of Applied Econometrics*, 24(6), 960--992. [DOI](https://doi.org/10.1002/jae.1079)

- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference.
  *Review of Economic Studies*, 77(2), 665--696. [DOI](https://doi.org/10.1111/j.1467-937X.2009.00578.x)

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 52(2), 381--419. [DOI](https://doi.org/10.1016/j.jmoneco.2004.05.007)
