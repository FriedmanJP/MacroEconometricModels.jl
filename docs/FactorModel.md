# Factor Model Documentation

## Overview

The `MacroEconometricModels` package now includes comprehensive support for **Static Factor Models** estimated using Principal Component Analysis (PCA). Factor models are essential tools in macroeconometrics for:

- Dimensionality reduction of large datasets
- Extracting common factors from panels of economic variables
- Forecasting with factor-augmented models (FAVAR)
- Structural analysis and identification
- Data summarization and visualization

## Mathematical Background

### The Static Factor Model

The static factor model represents an N-dimensional vector of observables **X**ₜ as:

```
Xₜ = Λ Fₜ + eₜ
```

where:
- **X**ₜ is the (N × 1) vector of observables at time t
- **F**ₜ is the (r × 1) vector of common factors
- **Λ** is the (N × r) matrix of factor loadings
- **e**ₜ is the (N × 1) vector of idiosyncratic errors

### Assumptions

1. The factors **F**ₜ are orthogonal: E[**F**ₜ **F**ₜ'] = **I**ᵣ
2. The idiosyncratic errors are uncorrelated with factors: E[**e**ₜ **F**ₜ'] = 0
3. The idiosyncratic errors can be weakly correlated across series

### Estimation via PCA

The model is estimated using the principal components of the sample covariance matrix:

1. Standardize the data (optional): **X̃** = (**X** - μ) / σ
2. Compute the sample covariance matrix: **Σ̂** = **X̃**' **X̃** / T
3. Perform eigenvalue decomposition: **Σ̂** = **V** **D** **V**'
4. Extract factor loadings: **Λ̂** = **V**ᵣ √**D**ᵣ (first r eigenvectors scaled by sqrt of eigenvalues)
5. Compute factors: **F̂** = **X̃** **V**ᵣ

## Quick Start

### Basic Usage

```julia
using MacroEconometricModels
using Random

# Generate synthetic data
Random.seed!(123)
T, N, r = 100, 20, 3
X = randn(T, N)  # Your data: T observations × N variables

# Estimate factor model with r factors
model = estimate_factors(X, r)

# Access results
factors = model.factors      # T × r matrix of estimated factors
loadings = model.loadings    # N × r matrix of factor loadings
variance_explained = model.explained_variance
```

### Determining the Number of Factors

```julia
# Use information criteria (Bai & Ng, 2002)
max_factors = 10
ic = ic_criteria(X, max_factors)

println("Optimal number of factors:")
println("  IC1: $(ic.r_IC1)")
println("  IC2: $(ic.r_IC2)")
println("  IC3: $(ic.r_IC3)")

# Estimate with optimal number
r_optimal = ic.r_IC2
model = estimate_factors(X, r_optimal)
```

### Model Diagnostics

```julia
# Get fitted values
X_fitted = predict(model)

# Compute residuals
residuals = residuals(model)

# R² for each variable
r2_values = r2(model)
println("Mean R²: ", mean(r2_values))

# Scree plot data
scree_data = scree_plot_data(model)
# Can be used with Plots.jl, Makie.jl, etc.
```

## API Reference

### Types

#### `FactorModel <: StatsAPI.StatisticalModel`

Stores the results of factor model estimation.

**Fields:**
- `X::Matrix{Float64}`: Original data matrix (T × N)
- `factors::Matrix{Float64}`: Estimated common factors (T × r)
- `loadings::Matrix{Float64}`: Factor loadings (N × r)
- `eigenvalues::Vector{Float64}`: All eigenvalues from PCA
- `explained_variance::Vector{Float64}`: Proportion of variance explained by each factor
- `cumulative_variance::Vector{Float64}`: Cumulative proportion of variance
- `r::Int`: Number of factors
- `standardized::Bool`: Whether data was standardized

### Functions

#### `estimate_factors(X, r; standardize=true)`

Estimate a static factor model using PCA.

**Arguments:**
- `X::Matrix{Float64}`: Data matrix (T × N)
- `r::Int`: Number of factors to extract

**Keyword Arguments:**
- `standardize::Bool=true`: Standardize data before estimation

**Returns:** `FactorModel`

**Example:**
```julia
model = estimate_factors(X, 3; standardize=true)
```

---

#### `predict(model::FactorModel)`

Reconstruct the original data using estimated factors and loadings.

**Returns:** Matrix{Float64} of fitted values (T × N)

**Formula:** **X̂** = **F** **Λ**'

---

#### `residuals(model::FactorModel)`

Compute the idiosyncratic component (residuals).

**Returns:** Matrix{Float64} of residuals (T × N)

**Formula:** **e** = **X** - **X̂**

---

#### `r2(model::FactorModel)`

Compute R² (proportion of variance explained) for each variable.

**Returns:** Vector{Float64} of R² values (length N)

**Formula:** R²ᵢ = 1 - Var(eᵢ) / Var(Xᵢ)

---

#### `ic_criteria(X, max_factors; standardize=true)`

Compute information criteria for determining optimal number of factors.

**Arguments:**
- `X::Matrix{Float64}`: Data matrix (T × N)
- `max_factors::Int`: Maximum number of factors to consider

**Returns:** NamedTuple with fields:
- `IC1::Vector{Float64}`: IC₁ criterion values
- `IC2::Vector{Float64}`: IC₂ criterion values
- `IC3::Vector{Float64}`: IC₃ criterion values
- `r_IC1::Int`: Optimal r according to IC₁
- `r_IC2::Int`: Optimal r according to IC₂
- `r_IC3::Int`: Optimal r according to IC₃

**Information Criteria (Bai & Ng, 2002):**

```
IC₁(r) = log(V(r)) + r * ((N+T)/(NT)) * log(NT/(N+T))
IC₂(r) = log(V(r)) + r * ((N+T)/(NT)) * log(min(N,T))
IC₃(r) = log(V(r)) + r * (log(min(N,T))/min(N,T))
```

where V(r) is the average sum of squared residuals.

**Example:**
```julia
ic = ic_criteria(X, 10)
r_optimal = ic.r_IC2  # Often performs well in practice
```

---

#### `scree_plot_data(model::FactorModel)`

Extract data for creating a scree plot.

**Returns:** NamedTuple with fields:
- `factors::Vector{Int}`: Factor numbers
- `explained_variance::Vector{Float64}`: Individual variance explained
- `cumulative_variance::Vector{Float64}`: Cumulative variance

**Example:**
```julia
scree_data = scree_plot_data(model)

using Plots
plot(scree_data.factors, scree_data.explained_variance,
     marker=:circle, label="Individual",
     xlabel="Factor Number", ylabel="Variance Explained")
plot!(scree_data.factors, scree_data.cumulative_variance,
      marker=:square, label="Cumulative")
```

---

#### `nobs(model::FactorModel)`

Get number of observations.

**Returns:** Int (number of time periods T)

---

#### `dof(model::FactorModel)`

Compute degrees of freedom.

**Returns:** Int

**Formula:** N×r + T×r - r² (accounting for normalization constraints)

## StatsAPI Interface

`FactorModel` implements the `StatsAPI.StatisticalModel` interface, providing compatibility with the Julia statistics ecosystem.

**Supported methods:**
- `predict(model)`: Fitted values
- `residuals(model)`: Model residuals
- `r2(model)`: R² statistics
- `nobs(model)`: Number of observations
- `dof(model)`: Degrees of freedom

## Practical Examples

### Example 1: Macroeconomic Forecasting

```julia
# Load macroeconomic dataset
# X contains GDP, inflation, unemployment, etc.
T, N = size(X)  # e.g., 200 periods × 100 variables

# Determine optimal factors
ic = ic_criteria(X, 10)
r = ic.r_IC2  # e.g., 5 factors

# Estimate factor model
fm = estimate_factors(X, r)

# Use factors in a VAR for forecasting
# (Combine with existing VAR functionality)
factors = fm.factors

# Forecast one period ahead using VAR on factors
# Then map back to original variables using loadings
```

### Example 2: Data Summarization

```julia
# Large panel dataset
X = load_large_panel()  # 500 × 200 matrix

# Extract small number of factors
model = estimate_factors(X, 5)

println("5 factors explain $(round(model.cumulative_variance[5] * 100, digits=1))% of variance")

# Analyze factor loadings to interpret factors
for i in 1:5
    top_variables = sortperm(abs.(model.loadings[:, i]), rev=true)[1:10]
    println("Factor $i most related to variables: $top_variables")
end
```

### Example 3: Structural Analysis

```julia
# Estimate factors from large info set
model = estimate_factors(X, 3)

# Extract specific factor of interest (e.g., monetary policy)
monetary_factor = model.factors[:, 2]

# Use in structural VAR or local projections
# Analyze impulse responses to monetary shocks
```

## Advanced Topics

### Factor Identification

Factors are identified only up to rotation. The PCA solution corresponds to one specific rotation (principal components). For structural interpretation:

1. Use domain knowledge to interpret factors based on loadings
2. Consider rotation methods (varimax, etc.) if needed
3. Sign normalization is arbitrary

### Standardization

**When to standardize (default):**
- Variables measured in different units
- Variables with vastly different variances
- You want equal weight for all variables

**When not to standardize:**
- Variables already in comparable units
- Relative variances contain information
- Specific application requires original scale

### Large N, Large T

For very large datasets:
- Consider randomized SVD methods (not yet implemented)
- Use sparse matrix techniques if applicable
- Parallel computation for cross-validation

### Missing Data

Current implementation requires complete data. For missing values:
1. Use interpolation/imputation before estimation
2. Consider EM algorithm (future extension)
3. Use subset of complete observations

## References

1. **Stock, J. H., & Watson, M. W. (2002).** "Forecasting using principal components from a large number of predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.

2. **Bai, J., & Ng, S. (2002).** "Determining the number of factors in approximate factor models." *Econometrica*, 70(1), 191-221.

3. **Bai, J., & Ng, S. (2008).** "Large dimensional factor analysis." *Foundations and Trends in Econometrics*, 3(2), 89-163.

4. **Bernanke, B. S., Boivin, J., & Eliasz, P. (2005).** "Measuring the effects of monetary policy: a factor-augmented vector autoregressive (FAVAR) approach." *Quarterly Journal of Economics*, 120(1), 387-422.

## Future Enhancements

Potential extensions to the factor model functionality:

- Dynamic factor models with Kalman filter
- Factor models with missing data (EM algorithm)
- Rotation methods for interpretation
- Bootstrap inference for loadings
- Mixed-frequency factor models
- Bayesian factor models
- Factor-augmented VAR (FAVAR) integration

## Getting Help

For issues or questions:
1. Check the examples in `/examples/factor_model_example.jl`
2. Review the test suite in `/test/test_factormodel.jl`
3. Consult the references listed above

## See Also

- `VARModel`: Vector autoregression models
- `estimate_bvar`: Bayesian VAR estimation
- `irf`: Impulse response functions
