# [AB-Model SVAR](@id id_ab_page)

The AB-model writes the structural relation among VAR residuals as ``A u_t = B \varepsilon_t`` with ``\varepsilon_t \sim (0, I)`` (Amisano and Giannini 1997; Lütkepohl 2005, ch. 9). Zero and fixed-value patterns on ``A`` and ``B`` encode non-recursive contemporaneous restrictions (Sims 1986). The just-identified Blanchard–Quah long-run pattern reproduces [`identify_long_run`](@ref); mixed short- and long-run zeros throw `ArgumentError`. This is the default SVAR tool in EViews, Stata, and JMulTi. Restriction-based alternatives live on [Structural Identification](@ref structural_identification_page); external-instrument identification is [Proxy SVAR](@ref id_proxy_page).

- **A-model / B-model / AB-model** patterns with `NaN` free and numbers fixed
- **Recursive pattern** that reproduces Cholesky (``Q \approx I``, LR df = 0)
- **Blanchard–Quah** long-run pattern that reproduces [`identify_long_run`](@ref)
- **Rank/order** diagnosis via [`check_identification`](@ref) (Amisano–Giannini)
- **Overidentification LR** ``T(\log|\hat\Sigma_r| - \log|\hat\Sigma|) \sim \chi^2``

```@setup ab
using MacroEconometricModels, Random, LinearAlgebra
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
n = nvars(model)
```

## Quick Start

**Recipe 1: Recursive AB-model (Cholesky as ML)**

```@example ab
pat = recursive_pattern(n)
svar = estimate_svar(model, pat; rng=MersenneTwister(15))
report(svar)
```

**Recipe 2: Impulse responses through `method=:ab`**

```@example ab
ir_ab = irf(model, 12; method=:ab, pattern=pat, rng=MersenneTwister(15))
report(ir_ab)
```

```julia
plot_result(ir_ab)
```

**Recipe 3: Rank and order conditions**

```@example ab
st = check_identification(pat, model; rng=MersenneTwister(15))
report(st)
```

**Recipe 4: Overidentified A-model and the LR test**

```@example ab
A_over = recursive_pattern(n).A
A_over[3, 1] = 0          # extra contemporaneous zero
pat_over = SVARPattern(A_over, recursive_pattern(n).B)
st_over = check_identification(pat_over, n)
svar_over = estimate_svar(model, pat_over; n_starts=2, rng=MersenneTwister(15))
(status = st_over.status, lr_df = svar_over.lr_df,
 lr_stat = round(svar_over.lr_stat, digits=2),
 lr_pvalue = round(svar_over.lr_pvalue, digits=3))
```

**Recipe 5: Blanchard–Quah long-run pattern**

```@example ab
pat_bq = blanchard_quah_pattern(n)
svar_bq = estimate_svar(model, pat_bq; rng=MersenneTwister(15))
Q_lr = identify_long_run(model)
(Q_ab = round.(svar_bq.Q, digits=3), Q_long_run = round.(Q_lr, digits=3))
```

**Recipe 6: FEVD of the recursive AB rotation**

```@example ab
decomp = fevd(model, 12; method=:ab, pattern=pat, rng=MersenneTwister(15))
report(decomp)
```

---

## The AB-Model

The reduced-form residual satisfies ``u_t = A^{-1} B \varepsilon_t``. With unit-variance structural shocks the implied covariance is

```math
\Sigma = A^{-1} B B' A^{-\top}
```

where:
- ``A`` is the ``n \times n`` contemporaneous left-hand matrix
- ``B`` is the ``n \times n`` shock-loading matrix
- ``\Sigma`` is the reduced-form residual covariance

A **pattern** marks each entry as free (`NaN`) or fixed (a number, typically 0 or 1). The concentrated Gaussian log-likelihood is

```math
\ell(A,B) = -\frac{T}{2}\bigl[\log|B|^2 - \log|A|^2 + \mathrm{tr}(B^{-1} A \hat\Sigma A' B^{-\top})\bigr]
```

[`estimate_svar`](@ref) maximises ``\ell`` in the free entries with `Optim.LBFGS` and a `ForwardDiff` gradient, using several random starts. Column signs are normalised so the impact (or long-run impact) diagonal is positive. The rotation used by `irf`/`fevd`/`hd` is

```math
Q = L^{-1} A^{-1} B, \qquad L = \mathrm{chol}(\hat\Sigma)
```

Just-identified recursive and Blanchard–Quah patterns use the Cholesky and long-run closed forms, so they reproduce [`identify_cholesky`](@ref) and [`identify_long_run`](@ref) to machine precision.

!!! note "Technical Note"
    The order condition is ``n_{\mathrm{free}} \le n(n+1)/2 + n_{\mathrm{LR}}``. The local rank condition is that the Jacobian of ``(\mathrm{vech}(A^{-1}BB'A^{-\top}), g_{\mathrm{LR}})`` has full column rank at a generic point (Amisano and Giannini 1997, Prop. 3). [`check_identification`](@ref) on an `SVARPattern` reports `:exact`, `:over`, or `:under`. `:under` throws [`IdentificationError`](@ref) from [`estimate_svar`](@ref); `:over` is estimable and is tested by the LR statistic.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_starts` | `Int` | `5` | Random starts around the Cholesky fill |
| `max_iter` | `Int` | `400` | `Optim.LBFGS` iteration cap |
| `rng` | `AbstractRNG` | default RNG | Starts and rank-condition probes |

**Return value** (`SVARModel`):

| Field | Type | Description |
|-------|------|-------------|
| `A`, `B` | `Matrix{T}` | Estimated contemporaneous matrices |
| `Q` | `Matrix{T}` | Rotation ``L^{-1} A^{-1} B`` |
| `loglik` | `T` | Concentrated log-likelihood |
| `lr_stat`, `lr_df`, `lr_pvalue` | `T`, `Int`, `T` | Overidentification LR test |
| `identification` | `IdentificationStatus` | `:exact`, `:over`, or `:under` |
| `pattern` | `SVARPattern{T}` | The restriction pattern |

Convenience constructors: [`recursive_pattern`](@ref), [`blanchard_quah_pattern`](@ref), [`a_model_pattern`](@ref), [`b_model_pattern`](@ref), [`ab_model_pattern`](@ref).

---

## Recursive Identification as an A-Model

A unit-lower-triangular ``A`` and a diagonal ``B`` are the textbook recursive form (Lütkepohl 2005, Example 9.1). The product ``A^{-1} B`` is the Cholesky factor of ``\hat\Sigma``, so ``Q = I``.

```@example ab
L = cholesky_factor(model)
B0 = svar.A \ svar.B
(Q_norm = round(norm(svar.Q - I(n)), digits=8),
 lr_df = svar.lr_df,
 B0_matches_chol = B0 ≈ L)
```

On this sample the rotation is the identity to numerical noise and the LR degrees of freedom are zero: the pattern is just-identified. The impact of the last shock on industrial production is the ``(1,3)`` entry of ``B_0``, which Cholesky forces to zero by the upper triangle of ``A``.

---

## Overidentification

Each extra independent zero beyond ``n(n-1)/2`` (plus normalisations) is an overidentifying restriction. The LR statistic compares the restricted covariance ``\hat\Sigma_r = A^{-1} B B' A^{-\top}`` to the unrestricted VAR covariance:

```math
\mathrm{LR} = T\bigl(\log|\hat\Sigma_r| - \log|\hat\Sigma|\bigr) \;\sim\; \chi^2(n_{\mathrm{over}})
```

Recipe 4 zeros ``A_{31}`` on top of the recursive pattern. That cell is free in the just-identified model; fixing it raises ``n_{\mathrm{over}}`` by one. A small p-value rejects the extra contemporaneous restriction.

---

## Blanchard–Quah Long-Run Restrictions

[`blanchard_quah_pattern`](@ref) is the supported long-run case: ``A = I``, ``B`` free, and ``C(1)B`` lower triangular. [`estimate_svar`](@ref) uses the closed form and reproduces [`identify_long_run`](@ref). Mixed short- and long-run zeros (Galí 1992) are a quadratic penalty, not a likelihood constraint, and throw `ArgumentError`; structural VECM long-run restrictions are SID-16.

```@example ab
(lr_df_bq = svar_bq.lr_df,
 status_bq = svar_bq.identification.status,
 Q_agree = svar_bq.Q ≈ Q_lr)
```

On a stationary VAR the AB long-run MLE and [`identify_long_run`](@ref) return the same rotation. Near a unit root ``C(1)`` is ill-conditioned; difference the data or move to a structural VECM.

---

## Complete Example

A three-variable system of industrial production, inflation, and the funds rate is identified two ways: recursively (Cholesky via the AB-model) and by Blanchard–Quah long-run zeros. The two schemes agree only if the extra short-run zeros happen to be true.

```@example ab
pat_rec = recursive_pattern(n)
svar_rec = estimate_svar(model, pat_rec; rng=MersenneTwister(21))
ir_rec = irf(model, 12; method=:ab, pattern=pat_rec, rng=MersenneTwister(21))
ir_bq = irf(model, 12; method=:ab, pattern=blanchard_quah_pattern(n),
            rng=MersenneTwister(21))
(rec_ip_mp = round(ir_rec.values[1, 1, 3], digits=4),
 bq_ip_mp = round(ir_bq.values[1, 1, 3], digits=4),
 rec_lr_df = svar_rec.lr_df)
```

The recursive scheme forces the contemporaneous production response to the funds-rate shock to zero (the ``(1,3)`` impact). Blanchard–Quah does not: it zeros the long-run production response of the last shock instead. The two impact numbers differ because they encode different identifying assumptions, not because the estimators disagree on a common parameter.

```@example ab
refs(svar_rec)
```

---

## Common Pitfalls

1. **Underidentification is an error, not a warning.** A pattern with too many free entries (`ab_model_pattern` of two fully free matrices) throws [`IdentificationError`](@ref). Run [`check_identification`](@ref) on the pattern before estimating.

2. **The order condition is not the rank condition.** Two linearly dependent zeros count twice in the order statistic and once in the Jacobian rank. Read `IdentificationStatus` rather than counting `NaN`s by hand.

3. **Overidentified ``Q`` is not orthogonal.** When ``\hat\Sigma_r \neq \hat\Sigma``, ``Q = L^{-1} A^{-1} B`` satisfies ``B_0 = L Q`` but not ``Q'Q = I``. Impulse responses remain ``\Phi_h B_0``; FEVD proportions need not sum to one.

4. **Sign normalisation flips columns.** The estimator enforces a positive impact (or long-run) diagonal. Comparing ``A`` entries across samples requires the same sign convention.

5. **Only Blanchard–Quah long-run is supported.** A `long_run` pattern that is not `blanchard_quah_pattern` throws `ArgumentError`. Mixed short- and long-run zeros are not a concentrated-ML constraint.

6. **Long-run patterns need stationarity.** ``C(1) = (I - \sum A_i)^{-1}`` explodes at a unit root, the same caveat as [`identify_long_run`](@ref).

7. **Multiple starts.** The concentrated likelihood is not globally concave. Raise `n_starts` if `loglik` jumps across seeds on an overidentified pattern.

---

## References

- Amisano, Gianni, and Carlo Giannini. 1997. *Topics in Structural VAR Econometrics*. 2nd ed. Springer. [DOI](https://doi.org/10.1007/978-3-642-60623-6)

- Blanchard, Olivier Jean, and Danny Quah. 1989. "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review* 79 (4): 655--673. [JSTOR](https://www.jstor.org/stable/1827924)

- Galí, Jordi. 1992. "How Well Does the IS-LM Model Fit Postwar U.S. Data?" *Quarterly Journal of Economics* 107 (2): 709--738. [DOI](https://doi.org/10.2307/2118487)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Springer. ISBN 978-3-540-40172-8.

- Sims, Christopher A. 1986. "Are Forecasting Models Usable for Policy Analysis?" *Federal Reserve Bank of Minneapolis Quarterly Review* 10 (1): 2--16.
