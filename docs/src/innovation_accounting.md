# Innovation Accounting

Innovation accounting refers to the collection of tools for analyzing the dynamic effects of structural shocks in VAR models. This includes Impulse Response Functions (IRF), Forecast Error Variance Decomposition (FEVD), and Historical Decomposition (HD).

## Impulse Response Functions (IRF)

### Definition

The impulse response function ``\Theta_h`` measures the effect of a one-unit structural shock at time ``t`` on the endogenous variables at time ``t+h``:

```math
\Theta_h = \frac{\partial y_{t+h}}{\partial \varepsilon_t'}
```

For a VAR, the IRF at horizon ``h`` is computed recursively:

```math
\Theta_h = \sum_{i=1}^{\min(h,p)} A_i \Theta_{h-i}
```

with ``\Theta_0 = B_0`` (the structural impact matrix).

### Companion Form Representation

Using the companion form, IRFs can be computed as:

```math
\Theta_h = J F^h J' B_0
```

where ``J = [I_n, 0, \ldots, 0]`` is an ``n \times np`` selection matrix and ``F`` is the companion matrix.

### Cumulative IRF

The cumulative impulse response up to horizon ``H`` is:

```math
\Theta^{cum}_H = \sum_{h=0}^{H} \Theta_h
```

### Confidence Intervals

**Bootstrap (Frequentist)**: Residual bootstrap of Kilian (1998):
1. Estimate the VAR and save residuals ``\hat{u}_t``
2. Generate bootstrap sample by resampling residuals with replacement
3. Re-estimate the VAR and compute IRFs
4. Repeat ``B`` times to build the distribution

**Credible Intervals (Bayesian)**: For each MCMC draw, compute IRFs and report posterior quantiles (e.g., 16th and 84th percentiles for 68% intervals).

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

**Reference**: Kilian (1998), Lütkepohl (2005, Chapter 3)

---

## Forecast Error Variance Decomposition (FEVD)

### Definition

The FEVD measures the proportion of the ``h``-step ahead forecast error variance of variable ``i`` attributable to structural shock ``j``:

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{n} (\Theta_s)_{ik}^2}
```

where ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the impulse response matrix at horizon ``s``.

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

**Reference**: Lütkepohl (2005, Section 2.3.3)

---

## Historical Decomposition (HD)

### Definition

Historical decomposition decomposes observed variable movements into contributions from individual structural shocks over time:

```math
y_t = \sum_{s=0}^{t-1} \Theta_s \varepsilon_{t-s} + \text{initial conditions}
```

where:
- ``\Theta_s = \Phi_s P`` are structural moving average (MA) coefficients
- ``P = L Q`` is the impact matrix (Cholesky factor ``L`` times rotation ``Q``)
- ``\varepsilon_t = Q' L^{-1} u_t`` are structural shocks

### Contribution of Shock j to Variable i at Time t

```math
\text{HD}_{ij}(t) = \sum_{s=0}^{t-1} (\Theta_s)_{ij} \, \varepsilon_j(t-s)
```

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

**Reference**: Kilian & Lütkepohl (2017, Chapter 4)

---

## Summary Tables

The package provides publication-quality summary tables using a unified interface with multiple dispatch.

!!! note "Name conflict with Base.summary"
    The `summary` function may conflict with `Base.summary`. Use the fully qualified name `MacroEconometricModels.summary(obj)` or import explicitly with `using MacroEconometricModels: summary`.

### Functions

| Function | Description |
|----------|-------------|
| `summary(obj)` | Print comprehensive summary to stdout |
| `table(obj, ...)` | Extract results as a DataFrame |
| `print_table(io, obj, ...)` | Print formatted table to IO stream |

### Usage Examples

```julia
using MacroEconometricModels

Y = randn(200, 3)
model = estimate_var(Y, 2)
irf_result = irf(model, 20)
fevd_result = fevd(model, 20)
hd_result = historical_decomposition(model, 198)

# Print summaries (use fully qualified name to avoid Base.summary conflict)
MacroEconometricModels.summary(model)
MacroEconometricModels.summary(irf_result)
MacroEconometricModels.summary(fevd_result)
MacroEconometricModels.summary(hd_result)

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

## References

- Kilian, L. (1998). "Small-Sample Confidence Intervals for Impulse Response Functions." *Review of Economics and Statistics*, 80(2), 218-230.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
