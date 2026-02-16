# Innovation Accounting

Innovation accounting refers to the collection of tools for analyzing the dynamic effects of structural shocks in VAR models. This includes Impulse Response Functions (IRF), Forecast Error Variance Decomposition (FEVD), and Historical Decomposition (HD).

## Quick Start

```julia
irfs = irf(model, 20; method=:cholesky)                          # Frequentist IRF
irfs_ci = irf(model, 20; ci_type=:bootstrap, reps=1000)          # With bootstrap CI
birfs = irf(post, 20; method=:cholesky)                          # Bayesian IRF
decomp = fevd(model, 20)                                         # FEVD
hd = historical_decomposition(model, 198)                        # Historical decomposition
report(irfs)                                                      # Publication-quality summary
```

---

## Impulse Response Functions (IRF)

### Definition

The impulse response function ``\Theta_h`` measures the effect of a one-unit structural shock at time ``t`` on the endogenous variables at time ``t+h``:

```math
\Theta_h = \frac{\partial y_{t+h}}{\partial \varepsilon_t'}
```

where
- ``\Theta_h`` is the ``n \times n`` impulse response matrix at horizon ``h``
- ``y_{t+h}`` is the ``n \times 1`` vector of endogenous variables at time ``t+h``
- ``\varepsilon_t`` is the ``n \times 1`` vector of structural shocks at time ``t``

For a VAR, the IRF at horizon ``h`` is computed recursively:

```math
\Theta_h = \sum_{i=1}^{\min(h,p)} A_i \Theta_{h-i}
```

where
- ``A_i`` are the ``n \times n`` VAR coefficient matrices for lag ``i``
- ``\Theta_0 = B_0`` is the ``n \times n`` structural impact matrix
- ``p`` is the VAR lag order

### Companion Form Representation

Using the companion form, IRFs can be computed as:

```math
\Theta_h = J F^h J' B_0
```

where
- ``J = [I_n, 0, \ldots, 0]`` is the ``n \times np`` selection matrix
- ``F`` is the ``np \times np`` companion matrix
- ``B_0`` is the ``n \times n`` structural impact matrix

### Cumulative IRF

The cumulative impulse response up to horizon ``H`` is:

```math
\Theta^{cum}_H = \sum_{h=0}^{H} \Theta_h
```

where ``\Theta^{cum}_H`` accumulates the impulse responses from impact through horizon ``H``, measuring the total cumulative effect of a structural shock. This is particularly relevant for variables in growth rates, where the cumulative IRF represents the effect on the level.

### Confidence Intervals

**Bootstrap (Frequentist)**: Residual bootstrap of Kilian (1998):
1. Estimate the VAR and save residuals ``\hat{u}_t``
2. Generate bootstrap sample by resampling residuals with replacement
3. Re-estimate the VAR and compute IRFs
4. Repeat ``B`` times to build the distribution

**Credible Intervals (Bayesian)**: For each posterior draw, compute IRFs and report posterior quantiles (e.g., 16th and 84th percentiles for 68% intervals).

### Usage

```julia
using MacroEconometricModels

Y = randn(200, 3)
model = estimate_var(Y, 2)

# Basic IRF (Cholesky identification)
irf_result = irf(model, 20)

# With bootstrap confidence intervals
irf_ci = irf(model, 20; ci_type=:bootstrap, reps=1000)

# Sign restrictions
sign_constraints = [1 1 0; -1 0 0; 0 0 1]
irf_sign = irf(model, 20; method=:sign, sign_restrictions=sign_constraints)
```

The basic `irf(model, 20)` call uses Cholesky identification by default. Adding `ci_type=:bootstrap` generates pointwise confidence bands via Kilian's (1998) residual bootstrap — `reps=1000` draws are recommended for publication-quality bands. Sign restrictions produce a set of admissible IRFs satisfying the constraints; the returned values are the median (or a representative draw), with the set-identified nature reflected in wider credible bands.

!!! note "Technical Note"
    The `ci_lower` and `ci_upper` arrays are only populated when `ci_type=:bootstrap` (frequentist) or when using the Bayesian `irf(post, ...)` method. With `ci_type=:none` (the default), these arrays contain zeros. Always check `irf_result.ci_type` before interpreting confidence bands.

### ImpulseResponse Return Values

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Array{T,3}` | ``(H+1) \times n \times n`` IRF array: `values[h+1, i, j]` = response of variable ``i`` to shock ``j`` at horizon ``h`` |
| `ci_lower` | `Array{T,3}` | Lower confidence bound (same shape as `values`) |
| `ci_upper` | `Array{T,3}` | Upper confidence bound |
| `horizon` | `Int` | Maximum IRF horizon ``H`` |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `ci_type` | `Symbol` | CI method used (`:bootstrap`, `:none`, etc.) |

### BayesianImpulseResponse Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``(H+1) \times n \times n \times 3``: dimension 4 = [16th pctl, median, 84th pctl] |
| `mean` | `Array{T,3}` | ``(H+1) \times n \times n`` posterior mean IRF |
| `horizon` | `Int` | Maximum IRF horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |

**Reference**: Kilian (1998), Lütkepohl (2005, Chapter 3)

---

## Forecast Error Variance Decomposition (FEVD)

### Definition

The FEVD measures the proportion of the ``h``-step ahead forecast error variance of variable ``i`` attributable to structural shock ``j``:

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{n} (\Theta_s)_{ik}^2}
```

where
- ``\text{FEVD}_{ij}(h)`` is the share of variable ``i``'s ``h``-step forecast error variance due to shock ``j``
- ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the impulse response matrix at horizon ``s``
- The numerator sums the squared contributions of shock ``j`` through horizon ``h-1``
- The denominator sums contributions from all ``n`` shocks, ensuring ``\sum_j \text{FEVD}_{ij}(h) = 1``

### Properties

- ``0 \leq \text{FEVD}_{ij}(h) \leq 1`` for all ``i, j, h``
- ``\sum_{j=1}^{n} \text{FEVD}_{ij}(h) = 1`` for all ``i, h``
- As ``h \to \infty``, FEVD converges to the unconditional variance decomposition

### Usage

```julia
# Basic FEVD
fevd_result = fevd(model, 20)

# With bootstrap CI
fevd_ci = fevd(model, 20; ci_type=:bootstrap, reps=500)

# Access decomposition for variable 1
fevd_var1 = fevd_result.decomposition[:, 1, :]  # horizons × shocks
```

The `proportions` array satisfies ``\sum_j \text{proportions}[h, i, j] = 1`` for all horizons ``h`` and variables ``i``. At short horizons, own shocks typically dominate (large diagonal entries). As ``h \to \infty``, the FEVD converges to the unconditional variance decomposition, revealing which shocks are the dominant long-run drivers of each variable's fluctuations. Adding `ci_type=:bootstrap` produces bootstrap CIs that quantify estimation uncertainty in the FEVD shares.

### FEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `decomposition` | `Array{T,3}` | ``H \times n \times n`` raw variance contributions |
| `proportions` | `Array{T,3}` | ``H \times n \times n`` proportion of FEV: `proportions[h, i, j]` = share of variable ``i``'s FEV due to shock ``j`` at horizon ``h`` |

### BayesianFEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times 3``: dimension 4 = [16th pctl, median, 84th pctl] |
| `mean` | `Array{T,3}` | ``H \times n \times n`` posterior mean FEVD proportions |
| `horizon` | `Int` | Maximum horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels |

**Reference**: Lütkepohl (2005, Section 2.3.3)

---

## Historical Decomposition (HD)

### Definition

Historical decomposition decomposes observed variable movements into contributions from individual structural shocks over time:

```math
y_t = \sum_{s=0}^{t-1} \Theta_s \varepsilon_{t-s} + \text{initial conditions}
```

where
- ``y_t`` is the ``n \times 1`` vector of observed variables at time ``t``
- ``\Theta_s = \Phi_s P`` are the ``n \times n`` structural MA coefficients at lag ``s``
- ``\Phi_s`` are the reduced-form MA coefficients (from the VMA representation)
- ``P = L Q`` is the ``n \times n`` impact matrix (Cholesky factor ``L`` times rotation ``Q``)
- ``\varepsilon_t = Q' L^{-1} u_t`` are the ``n \times 1`` structural shocks
- The initial conditions capture the contribution of pre-sample values

### Contribution of Shock j to Variable i at Time t

```math
\text{HD}_{ij}(t) = \sum_{s=0}^{t-1} (\Theta_s)_{ij} \, \varepsilon_j(t-s)
```

where
- ``\text{HD}_{ij}(t)`` is the contribution of shock ``j`` to variable ``i`` at time ``t``
- ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the structural MA coefficient at lag ``s``
- ``\varepsilon_j(t-s)`` is the realized structural shock ``j`` at time ``t-s``

The decomposition satisfies the identity:

```math
y_t = \sum_{j=1}^{n} \text{HD}_{ij}(t) + \text{initial}_i(t)
```

### Usage

```julia
# Basic historical decomposition
hd = historical_decomposition(model, 198)

# Verify decomposition identity
verify_decomposition(hd)  # returns true if identity holds

# Get contribution of shock 1 to variable 2
contrib = contribution(hd, 2, 1)

# Total shock contribution (excluding initial conditions)
total = total_shock_contribution(hd, 1)

# With different identification
hd_sign = historical_decomposition(model, 198; method=:sign,
    sign_restrictions=sign_constraints)
```

The `contributions[t, i, j]` array gives the contribution of shock ``j`` to variable ``i`` at time ``t``. Summing across shocks plus the initial conditions recovers the actual data: `verify_decomposition(hd)` checks this identity holds to numerical precision. The `total_shock_contribution(hd, i)` function sums all shock contributions for variable ``i``, providing the "shock-driven" component of the series with initial conditions removed.

### HistoricalDecomposition Return Values

| Field | Type | Description |
|-------|------|-------------|
| `contributions` | `Array{T,3}` | ``T_{eff} \times n \times n`` shock contributions: `contributions[t, i, j]` = contribution of shock ``j`` to variable ``i`` at time ``t`` |
| `initial_conditions` | `Matrix{T}` | ``T_{eff} \times n`` initial condition component |
| `actual` | `Matrix{T}` | ``T_{eff} \times n`` actual data values |
| `shocks` | `Matrix{T}` | ``T_{eff} \times n`` structural shocks |
| `T_eff` | `Int` | Effective number of time periods |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `method` | `Symbol` | Identification method (`:cholesky`, `:sign`, etc.) |

### BayesianHistoricalDecomposition Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``T_{eff} \times n \times n \times n_q`` contribution quantiles |
| `mean` | `Array{T,3}` | ``T_{eff} \times n \times n`` mean contributions |
| `initial_quantiles` | `Array{T,3}` | ``T_{eff} \times n \times n_q`` initial condition quantiles |
| `initial_mean` | `Matrix{T}` | ``T_{eff} \times n`` mean initial conditions |
| `shocks_mean` | `Matrix{T}` | ``T_{eff} \times n`` mean structural shocks |
| `actual` | `Matrix{T}` | ``T_{eff} \times n`` actual data values |
| `T_eff` | `Int` | Effective number of time periods |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels |
| `method` | `Symbol` | Identification method |

**Reference**: Kilian & Lütkepohl (2017, Chapter 4)

---

## LP-Based Innovation Accounting

Structural Local Projections provide the same innovation accounting tools (IRF, FEVD, HD) as standard VAR, but via LP estimation. This offers robustness to VAR dynamic misspecification at the cost of some efficiency. For full theoretical background, see [Local Projections](lp.md).

### IRF from Structural LP

The `irf()` function dispatches on `StructuralLP` to return the pre-computed 3D impulse response:

```julia
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
irf_result = irf(slp)   # Returns ImpulseResponse from the StructuralLP

# Access: irf_result.values[h, i, j] = response of var i to shock j at horizon h
println("Impact of shock 1 on var 2: ", irf_result.values[1, 2, 1])
```

The LP-based IRFs are numerically close to VAR-based IRFs under correct specification (Plagborg-Møller & Wolf 2021), but the LP standard errors stored in `slp.se` are wider because each horizon is estimated independently without imposing cross-horizon restrictions.

### FEVD from Structural LP

The `fevd()` method for `StructuralLP` dispatches to the R²-based LP-FEVD of Gorodnichenko & Lee (2019):

```julia
decomp = fevd(slp, 20)  # Returns LPFEVD

# Bias-corrected shares
println("Var 1 explained by Shock 1 at h=8: ",
        round(decomp.bias_corrected[1, 1, 8] * 100, digits=1), "%")
```

Unlike VMA-based FEVD, LP-FEVD estimates variance shares directly via R² regressions, so they do not depend on the invertibility of the VAR lag polynomial. See [LP-Based FEVD](lp.md#LP-Based-FEVD) for the three estimator variants (R², LP-A, LP-B) and bias correction details.

### Historical Decomposition from Structural LP

```julia
hd = historical_decomposition(slp)
verify_decomposition(hd)  # Check additive identity
```

The LP-based historical decomposition uses the structural shocks recovered from the VAR identification step (``\hat{\varepsilon}_t = Q'L^{-1}\hat{u}_t``) combined with LP-estimated IRF coefficients to decompose observed variable movements into shock contributions.

### Cumulative IRF

For variables measured in growth rates (e.g., log-differenced GDP), the cumulative IRF shows the effect on the level:

```julia
lp_model = estimate_lp(Y, 1, 20; lags=4)
lp_irfs = lp_irf(lp_model)
cum_irfs = cumulative_irf(lp_irfs)
```

The `cumulative_irf` function sums the pointwise IRF from horizon 0 through ``h``, propagating standard errors via the delta method. This is especially useful for comparing LP and VAR results in levels versus differences.

---

## Summary Tables

The package provides publication-quality summary tables using a unified interface with multiple dispatch.

### Functions

| Function | Description |
|----------|-------------|
| `report(obj)` | Print comprehensive summary to stdout |
| `table(obj, ...)` | Extract results as a matrix |
| `print_table(io, obj, ...)` | Print formatted table to IO stream |

### Usage Examples

```julia
using MacroEconometricModels

Y = randn(200, 3)
model = estimate_var(Y, 2)
irf_result = irf(model, 20)
fevd_result = fevd(model, 20)
hd_result = historical_decomposition(model, 198)

# Print summaries
report(model)
report(irf_result)
report(fevd_result)
report(hd_result)

# Extract as DataFrames for further analysis
df_irf = table(irf_result, 1, 1)                    # response of var 1 to shock 1
df_irf_sel = table(irf_result, 1, 1; horizons=[1, 4, 8, 12, 20])

df_fevd = table(fevd_result, 1)                     # FEVD for variable 1
df_fevd_sel = table(fevd_result, 1; horizons=[1, 4, 8, 12])

df_hd = table(hd_result, 1)                         # HD for variable 1
df_hd_sel = table(hd_result, 1; periods=180:198)    # specific periods

# Print formatted tables to stdout or file
print_table(stdout, irf_result, 1, 1; horizons=[1, 4, 8, 12])
print_table(stdout, fevd_result, 1; horizons=[1, 4, 8, 12])
print_table(stdout, hd_result, 1; periods=190:198)

# Write to file
open("results.txt", "w") do io
    print_table(io, irf_result, 1, 1)
    print_table(io, fevd_result, 1)
end
```

### String Indexing

Variables and shocks can be indexed by name:

```julia
# If variable names are set
df = table(irf_result, "GDP", "Monetary Shock")
df = table(fevd_result, "Inflation")
df = table(hd_result, "Output")
```

---

## Display Backends

All `show`, `print_table`, and `report` methods route through a unified PrettyTables backend. Switching from terminal text to LaTeX or HTML output requires a single call:

```julia
# Switch output format globally
set_display_backend(:text)    # Terminal-friendly (default)
set_display_backend(:latex)   # LaTeX \begin{tabular} output
set_display_backend(:html)    # HTML <table> output

# Check current backend
get_display_backend()         # :text
```

This applies to all innovation accounting results — IRF, FEVD, and HD tables:

```julia
# IRF table in LaTeX for a paper
set_display_backend(:latex)
open("tables/irf_table.tex", "w") do io
    print_table(io, irfs, 1, 1; horizons=[1, 4, 8, 12, 20])
end

# FEVD table in HTML for slides
set_display_backend(:html)
open("slides/fevd.html", "w") do io
    print_table(io, fevd_result, 1; horizons=[1, 4, 8, 12, 20])
end

# Reset to text for interactive work
set_display_backend(:text)
```

For a comprehensive display backend workflow, see [Example 15: Table Output](examples.md#Example-15:-Table-Output-Text,-LaTeX,-and-HTML).

---

## Bibliographic References

The `refs()` function returns bibliographic references for any model or identification method. This integrates with innovation accounting results:

```julia
# References for a VAR model
refs(model)                           # AEA text format (default)
refs(model; format=:bibtex)           # BibTeX format
refs(model; format=:latex)            # LaTeX \bibitem format

# References by identification method
refs(:cholesky)                       # Cholesky decomposition references
refs(:fastica)                        # FastICA references
refs(:sign)                           # Sign restriction references

# Write BibTeX to file
open("references.bib", "w") do io
    refs(io, model; format=:bibtex)
end
```

For a comprehensive references workflow, see [Example 16: Bibliographic References](examples.md#Example-16:-Bibliographic-References).

!!! note "Technical Note"
    For non-Gaussian identification methods (ICA, ML, heteroskedasticity-based), see [Non-Gaussian Identification](nongaussian.md). All 18 identification methods work seamlessly with `irf()`, `fevd()`, and `historical_decomposition()` via the `method` keyword.

---

## Complete Example

This example combines IRF, FEVD, and HD for a three-variable VAR.

```julia
using MacroEconometricModels
using Random

Random.seed!(42)

# Simulate a 3-variable VAR(2)
T, n, p = 200, 3, 2
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(n)
end

model = estimate_var(Y, p)

# IRF with bootstrap confidence intervals
H = 20
irfs = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=500)
println("Shock 1 → Var 1 at h=0: ", round(irfs.values[1, 1, 1], digits=3))
println("Shock 1 → Var 1 at h=8: ", round(irfs.values[9, 1, 1], digits=3))

# FEVD
decomp = fevd(model, H)
println("\nFEVD for Var 1 at h=1: shock shares = ",
        round.(decomp.proportions[1, 1, :] .* 100, digits=1), "%")
println("FEVD for Var 1 at h=20: shock shares = ",
        round.(decomp.proportions[20, 1, :] .* 100, digits=1), "%")

# Historical decomposition
hd = historical_decomposition(model, size(model.U, 1))
println("\nDecomposition identity holds: ", verify_decomposition(hd))

# Summary tables
df_irf = table(irfs, 1, 1; horizons=[0, 4, 8, 12, 20])
df_fevd = table(decomp, 1; horizons=[1, 4, 8, 20])
```

The IRF values show the dynamic propagation of structural shocks through the system. At impact (``h=0``), the Cholesky identification imposes a lower-triangular structure, so shock 1 affects only the first variable contemporaneously. By ``h=8``, cross-variable transmission is visible. The FEVD reveals whether the first variable's forecast uncertainty is dominated by its own shocks or by spillovers from other variables. At short horizons own shocks typically dominate; as ``h \to \infty``, the FEVD converges to the unconditional variance decomposition. The HD passes the verification check, confirming the additive identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)`` holds to numerical precision.

---

## References

- Kilian, Lutz. 1998. "Small-Sample Confidence Intervals for Impulse Response Functions." *Review of Economics and Statistics* 80 (2): 218–230. [https://doi.org/10.1162/003465398557465](https://doi.org/10.1162/003465398557465)
- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [https://doi.org/10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
