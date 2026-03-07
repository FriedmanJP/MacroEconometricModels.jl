# [Historical Decomposition](@id ia_hd_page)

Historical Decomposition (HD) decomposes observed variable movements into contributions from individual structural shocks over time. While FEVD answers "which shocks matter in general?", HD answers "which shocks drove this specific episode?" -- making it an indispensable tool for narrative analysis of macroeconomic events.

- **Frequentist HD**: Exact additive decomposition with verification of the identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)``
- **Bayesian HD**: Posterior distributions over shock contributions with credible intervals
- **Accessor functions**: `contribution()`, `total_shock_contribution()`, and `verify_decomposition()` for programmatic analysis

```@setup ia_hd
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
model = estimate_var(Y, 4; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
post = estimate_bvar(Y, 4; n_draws=100, varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

**Recipe 1: Basic historical decomposition**

```@example ia_hd
# HD with Cholesky identification
hd = historical_decomposition(model)
report(hd)
```

**Recipe 2: Verify the decomposition identity**

```@example ia_hd
hd = historical_decomposition(model)

# The identity y_t = sum_j HD_j(t) + initial(t) holds to machine precision
verified = verify_decomposition(hd)
```

**Recipe 3: Extract individual shock contributions**

```@example ia_hd
hd = historical_decomposition(model)

# Contribution of monetary shock (shock 3) to output (variable 1)
monetary_to_output = contribution(hd, 1, 3)

# Total shock-driven component for variable 1 (excludes initial conditions)
total = total_shock_contribution(hd, 1)
```

**Recipe 4: Bayesian historical decomposition**

```@example ia_hd
bhd = historical_decomposition(post; method=:cholesky)
report(bhd)
```

**Recipe 5: HD visualization**

```@example ia_hd
hd = historical_decomposition(model)
nothing # hide
```

```julia
plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Frequentist HD

Historical decomposition derives from the structural VMA representation of the VAR. The observed data at time ``t`` is decomposed into the cumulative effect of all past structural shocks plus a contribution from initial (pre-sample) conditions (Kilian & Lutkepohl 2017, Chapter 4).

```math
y_t = \sum_{s=0}^{t-1} \Theta_s \varepsilon_{t-s} + \text{initial conditions}
```

where:
- ``y_t`` is the ``n \times 1`` vector of observed variables at time ``t``
- ``\Theta_s = \Phi_s P`` are the ``n \times n`` structural MA coefficients at lag ``s``
- ``\Phi_s`` are the reduced-form MA coefficients from the VMA representation
- ``P = L Q`` is the ``n \times n`` structural impact matrix (Cholesky factor ``L`` times rotation ``Q``)
- ``\varepsilon_t = Q' L^{-1} u_t`` are the ``n \times 1`` structural shocks
- The initial conditions capture the contribution of pre-sample values through the VAR dynamics

### Contribution Formula

The contribution of shock ``j`` to variable ``i`` at time ``t`` is:

```math
\text{HD}_{ij}(t) = \sum_{s=0}^{t-1} (\Theta_s)_{ij} \, \varepsilon_j(t-s)
```

where:
- ``\text{HD}_{ij}(t)`` is the contribution of shock ``j`` to variable ``i`` at time ``t``
- ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the structural MA coefficient at lag ``s``
- ``\varepsilon_j(t-s)`` is the realized structural shock ``j`` at time ``t-s``

### Additive Identity

The decomposition satisfies an exact identity — the sum of all shock contributions plus initial conditions recovers the observed data:

```math
y_{i,t} = \sum_{j=1}^{n} \text{HD}_{ij}(t) + \text{initial}_i(t)
```

The `verify_decomposition()` function checks this identity to machine precision (default tolerance ``10^{-10}``). A failure indicates a numerical problem in the MA coefficient computation.

### Code Example

```@example ia_hd
# Historical decomposition with Cholesky identification
hd = historical_decomposition(model)
report(hd)

# Verify the additive identity holds
verified = verify_decomposition(hd)

# Extract contribution of monetary policy shock (shock 3) to output (variable 1)
monetary_to_output = contribution(hd, 1, 3)

# Total shock-driven component of INDPRO (excludes initial conditions)
total = total_shock_contribution(hd, 1)

# Tabular output for the last 10 periods
print_table(stdout, hd, 1; periods=(hd.T_eff-9):hd.T_eff)
```

The HD reveals which structural shocks drove specific historical episodes. Large positive contributions from the monetary shock to industrial production indicate periods of accommodative policy, while large negative contributions signal contractionary episodes. The initial conditions component is typically large at the beginning of the sample and decays toward zero as the sample grows, reflecting the diminishing influence of pre-sample values.

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Helper Functions

| Function | Description |
|----------|-------------|
| `contribution(hd, var, shock)` | Time series of shock ``j``'s contribution to variable ``i``; accepts `Int` indices or `String` names |
| `total_shock_contribution(hd, var)` | Sum of all shock contributions for variable ``i`` (excludes initial conditions) |
| `verify_decomposition(hd; tol)` | Check that the additive identity holds to tolerance `tol` |

### HistoricalDecomposition Return Values

| Field | Type | Description |
|-------|------|-------------|
| `contributions` | `Array{T,3}` | ``T_{eff} \times n \times n`` shock contributions: `contributions[t, i, j]` = contribution of shock ``j`` to variable ``i`` at time ``t`` |
| `initial_conditions` | `Matrix{T}` | ``T_{eff} \times n`` initial condition component |
| `actual` | `Matrix{T}` | ``T_{eff} \times n`` actual data values |
| `shocks` | `Matrix{T}` | ``T_{eff} \times n`` structural shocks |
| `T_eff` | `Int` | Effective number of time periods (sample size minus lag order) |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `method` | `Symbol` | Identification method (`:cholesky`, `:sign`, `:long_run`, etc.) |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `max_draws` | `Int` | `1000` | Maximum draws for sign/narrative identification |

---

## Bayesian HD

Bayesian HD computes the historical decomposition for each posterior draw from a Bayesian VAR, producing posterior distributions over shock contributions (Kilian & Lutkepohl 2017, Chapter 12). Non-stationary posterior draws are discarded to ensure economically meaningful decompositions.

```@example ia_hd
# Bayesian HD with 68% credible intervals
bhd = historical_decomposition(post; method=:cholesky,
                               quantiles=[0.16, 0.5, 0.84])
report(bhd)

# Access posterior mean contribution of shock 3 to variable 1
monetary_contrib = contribution(bhd, 1, 3; stat=:mean)

# Verify the mean decomposition identity (approximate, due to averaging)
verified = verify_decomposition(bhd)
```

The Bayesian HD iterates over all stationary posterior draws, computing the structural MA coefficients and shock contributions for each. The `point_estimate` array contains the posterior mean (or median) contributions, while the `quantiles` array stores the full posterior distribution. Wide credible bands around a shock's contribution indicate that the data do not clearly attribute a specific episode to that shock.

!!! note "Technical Note"
    The `verify_decomposition` tolerance for Bayesian HD (default ``10^{-6}``) is looser than for frequentist HD (``10^{-10}``) because the point estimate is an average across draws, and the additive identity holds exactly only for each individual draw. The per-draw identity is always exact.

```@raw html
<iframe src="../assets/plots/hd_bayesian.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### BayesianHistoricalDecomposition Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``T_{eff} \times n \times n \times n_q`` contribution quantiles |
| `point_estimate` | `Array{T,3}` | ``T_{eff} \times n \times n`` posterior point estimate contributions |
| `initial_quantiles` | `Array{T,3}` | ``T_{eff} \times n \times n_q`` initial condition quantiles |
| `initial_point_estimate` | `Matrix{T}` | ``T_{eff} \times n`` posterior point estimate initial conditions |
| `shocks_point_estimate` | `Matrix{T}` | ``T_{eff} \times n`` posterior point estimate structural shocks |
| `actual` | `Matrix{T}` | ``T_{eff} \times n`` actual data values |
| `T_eff` | `Int` | Effective number of time periods |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |
| `method` | `Symbol` | Identification method |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |

---

## Complete Example

This example combines IRF, FEVD, and HD for a complete structural analysis of a three-variable monetary policy VAR. The workflow moves from shock identification to dynamic responses, variance contributions, and finally episode-level attribution.

```@example ia_hd
# Step 1: Impulse responses — how do variables respond to shocks?
irfs = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
report(irfs)
```

```@example ia_hd
# Step 2: Variance decomposition — which shocks matter most?
decomp = fevd(model, 20)
report(decomp)
print_table(stdout, decomp, 1; horizons=[1, 4, 8, 20])
```

```@example ia_hd
# Step 3: Historical decomposition — which shocks drove specific episodes?
hd = historical_decomposition(model)
report(hd)

# Verify the additive identity
verified = verify_decomposition(hd)

# Monetary policy contribution to output over time
monetary_to_output = contribution(hd, 1, 3)
nothing # hide
```

```julia
plot_result(irfs)
plot_result(decomp)
plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The IRFs reveal the dynamic transmission mechanism: a contractionary monetary shock (positive FFR innovation) reduces industrial production with a lag of several quarters. The FEVD shows that monetary shocks explain a modest but non-trivial share of output forecast error variance at business-cycle horizons. The HD identifies specific historical episodes where monetary shocks made large positive or negative contributions to output movements, providing a narrative interpretation of the estimated structural model.

---

## Common Pitfalls

1. **HD depends on the full identification scheme.** The historical decomposition inherits all identification assumptions from the structural VAR. With Cholesky identification, reordering the variables changes every shock contribution. With sign restrictions, the HD reports a representative draw from the identified set — re-run with multiple seeds to assess robustness.

2. **Initial conditions dominate early in the sample.** The first ``p`` to ``2p`` periods of the HD are driven primarily by initial conditions rather than structural shocks, because the VMA representation has not yet accumulated enough lags. Focus interpretation on the middle and end of the sample.

3. **Bayesian HD discards non-stationary draws.** If many posterior draws are non-stationary (common with diffuse priors or short samples), the effective number of draws for the Bayesian HD is reduced. A warning is issued when more than half of the draws are discarded. Tighten the Minnesota prior or increase `n_draws` to compensate.

4. **The horizon argument caps MA coefficient computation.** For frequentist HD, the `horizon` argument controls how many structural MA coefficients ``\Theta_s`` are computed. Setting `horizon` less than `T_eff` truncates the decomposition and the additive identity holds only approximately. The default uses the full effective sample size.

5. **String indexing requires matching variable names.** The `contribution(hd, "INDPRO", "FEDFUNDS")` syntax requires that the variable names stored in the model match the strings exactly. Use `hd.variables` and `hd.shock_names` to inspect the available names.

---

## References

- Kilian, Lutz, and Helmut Lutkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [DOI: 10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
