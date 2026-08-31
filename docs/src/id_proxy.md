# [Proxy SVAR](@id id_proxy_page)

Proxy (external-instrument) identification recovers one or more structural shocks from a series ``z_t`` that is correlated with the target shock and orthogonal to the others (Stock & Watson 2018; Mertens & Ravn 2013). High-frequency monetary surprises and narrative tax changes are the leading applications. This page documents the VAR implementation; the local-projection counterpart is [Local Projections](@ref lp_page). Restriction-based schemes live on [Structural Identification](@ref structural_identification_page); news and main-business-cycle shocks are [Max-Share Identification](@ref id_maxshare_page).

- **One instrument**: closed-form impact column ``b_1 \propto \mathrm{Cov}(\hat u_t, z_t)``
- **``k`` instruments**: Mertens–Ravn (2013, Appendix A) column-space identification
- **Relevance**: HAC first-stage ``F`` (warns when ``F < 10``) and Mertens–Ravn reliability
- **Inference**: Jentsch–Lunsford (2019) moving-block bootstrap; Anderson–Rubin bands for ``k = 1``

```@setup proxy
using MacroEconometricModels, Random
Random.seed!(42)
mp = load_example(:mp_shocks)
row(yr, q) = (yr - 1960) * 4 + q
rows = row(1988, 4):row(2012, 2)
Y = mp.data[rows, 1:3]
z = mp.data[rows, 6]
Z = reshape(z, :, 1)
model = estimate_var(Y, 4; varnames=["ygap", "infl", "ffr"])
```

## Quick Start

**Recipe 1: Identify a monetary shock with a high-frequency proxy**

```@example proxy
proxy = identify_proxy(model, Z; normalize=:unit_effect, normalize_var=3)
report(proxy)
```

**Recipe 2: Impulse responses of the instrumented shock**

```@example proxy
ir_proxy = irf(model, 12; method=:proxy, instruments=Z,
               normalize=:unit_effect, normalize_var=3,
               shock_names=["MP", "u2", "u3"])
report(ir_proxy)
```

```julia
plot_result(ir_proxy)
```

**Recipe 3: First-stage strength and reliability**

```@example proxy
(F = round(proxy.first_stage_F, digits=1),
 reliability = round(proxy.reliability, digits=3),
 partial = proxy.is_partial)
```

**Recipe 4: Unit-variance rotation for FEVD**

```@example proxy
decomp = fevd(model, 12; method=:proxy, instruments=Z,
              normalize=:unit_variance, normalize_var=3,
              shock_names=["MP", "u2", "u3"])
report(decomp)
```

**Recipe 5: Weak-instrument-robust Anderson–Rubin bands**

```@example proxy
ar = proxy_ar_band(model, z; horizon=4, normalize_var=3, n_grid=81, span=8)
report(ar)
```

---

## External-Instrument Identification

The reduced-form residual satisfies ``u_t = B_0 \varepsilon_t``. An instrument ``z_t`` (or a ``k``-vector ``Z_t``) is **relevant** for the first ``k`` shocks and **exogenous** to the rest:

```math
\mathbb{E}[z_t \varepsilon_{1t}'] = \Phi, \qquad \mathbb{E}[z_t \varepsilon_{2t}'] = 0
```

where:
- ``\varepsilon_{1t}`` is the ``k \times 1`` vector of instrumented shocks
- ``\Phi`` is a ``k \times k`` invertible relevance matrix
- ``\varepsilon_{2t}`` holds the remaining ``n-k`` shocks

Then ``\Sigma_{uz} = \mathbb{E}[u_t z_t'] = B_{01} \Phi'``, so the column space of ``B_{01}`` is identified. For ``k = 1`` the impact column is unique up to scale:

```math
b_1 \propto \Sigma_{uz}
```

Stock & Watson (2018) **unit-variance** normalisation takes the QR of ``L^{-1} \Sigma_{uz}`` with ``L = \mathrm{chol}(\Sigma)``, yielding an orthogonal ``Q`` and ``B_0 = L Q``. **Unit-effect** normalisation rescales the identified column so a chosen variable (the federal funds rate in Recipe 1) moves by one unit on impact.

Missing instrument observations are dropped pairwise against the VAR residuals (`align=true`), the Gertler & Karadi (2015) convention.

!!! note "Technical Note"
    For ``k > 1``, Mertens & Ravn (2013, Appendix A) recover relative impulse responses ``s = \Sigma_{2z} \Sigma_{1z}^{-1}`` from a partition of the residual vector and complete ``Q`` by QR. Individual shocks inside the ``k``-dimensional subspace are identified only up to a ``k \times k`` rotation. FEVD and historical-decomposition shares of the unidentified complement are not identified (`is_partial=true` when ``k < n``).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shocks` | `AbstractVector{Int}` | `1:k` | Columns of ``Q`` assigned to the instrumented shocks |
| `normalize` | `Symbol` | `:unit_effect` | `:unit_effect` or `:unit_variance` |
| `normalize_var` | `Int` | `1` | Variable with unit impact under `:unit_effect` when ``k = 1`` |
| `align` | `Bool` | `true` | Drop `NaN` rows pairwise; accept length ``T`` or ``T_{\mathrm{eff}}`` |
| `instrument_names` | `Vector{String}` | `["z1", …]` | Labels for the instruments |
| `shock_names` | `Vector{String}` | `"Proxy"` / `"Unidentified j"` | Shock labels |

**Return value** (`ProxySVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Rotation; first ``k`` columns (after `shocks`) are identified |
| `B0` | `Matrix{T}` | Impact matrix ``B_0 = \mathrm{chol}(\Sigma)\, Q`` (then scaled if unit-effect) |
| `k` | `Int` | Number of instruments / instrumented shocks |
| `first_stage_F` | `T` | HAC first-stage ``F`` of the normalising residual on ``Z`` |
| `reliability` | `T` | Mertens–Ravn reliability in ``[0, 1]`` |
| `is_partial` | `Bool` | `true` when ``k < n`` |

The one-argument method `identify_proxy(model, z::AbstractVector)` keeps the original NamedTuple `(Q, b1, first_stage_F, z_eff)` used by structural DFM.

---

## Relevance and Reliability

The first-stage regression is the residual of the normalising variable on the instruments, with HAC standard errors. Stock & Watson (2018) and Montiel Olea, Stock & Watson (2021) treat ``F < 10`` as a weak instrument; the estimator still returns a result and emits a warning.

Mertens–Ravn **reliability** is the share of instrumented-shock variance captured by the proxy:

```math
\mathcal{R} = \frac{1}{k}\,\mathrm{tr}\!\big(\Sigma_{uz}' \Sigma^{-1} \Sigma_{uz}\, \Sigma_z^{-1}\big)
```

where:
- ``\Sigma_{uz} = \mathrm{Cov}(\hat u_t, z_t)`` is the ``n \times k`` residual–instrument covariance
- ``\Sigma_z`` is the instrument covariance
- ``k = 1`` reduces to ``\phi^2 / \mathrm{Var}(z_t)`` under unit-variance shocks

```@example proxy
(F = round(proxy.first_stage_F, digits=1),
 reliability = round(proxy.reliability, digits=3))
```

Gertler–Karadi `mp1` on this 1988Q4–2012Q2 window produces a first-stage ``F`` well above 10, so the Wald IRF bands are not automatically invalid. Reliability below one is expected: the surprise series is a noisy reading of the structural monetary shock, not the shock itself.

---

## Inference

The i.i.d. residual bootstrap is invalid for proxy SVARs (Jentsch & Lunsford 2019). `irf` with `method=:proxy` and `ci_type=:bootstrap` jointly resamples ``(u_t, z_t)`` by the moving-block bootstrap, defaulting `bootstrap=:iid` to `:block`.

```@example proxy
ir_mbb = irf(model, 8; method=:proxy, instruments=Z,
             normalize=:unit_effect, normalize_var=3,
             shock_names=["MP", "u2", "u3"],
             ci_type=:bootstrap, bootstrap=:block, reps=50, seed=740)
(impact_ffr = round(ir_mbb.values[1, 3, 1], digits=3),
 band = (round(ir_mbb.ci_lower[1, 1, 1], digits=3),
         round(ir_mbb.ci_upper[1, 1, 1], digits=3)))
```

For ``k = 1``, [`proxy_ar_band`](@ref) builds the observationally equivalent LP-IV (Plagborg-Møller & Wolf 2021, Proposition 1) with the VAR lags as controls and reuses [`lp_iv_ar_band`](@ref). The Anderson–Rubin set has correct coverage whether or not the instrument is strong.

---

## Complete Example

The shipped `:mp_shocks` panel aligns Gertler–Karadi `mp1` with the quarterly macro block. A funds-rate unit-effect proxy SVAR and LP-IV estimate the same impact (Plagborg-Møller & Wolf 2021).

```@example proxy
proxy_ffr = identify_proxy(model, Z; normalize=:unit_effect, normalize_var=3)
lpiv = estimate_lp_iv(Y, 3, Z, 0; lags=4, varnames=["ygap", "infl", "ffr"])
lpir = lp_iv_irf(lpiv)
(proxy_impact = round.(proxy_ffr.B0[:, 1], digits=3),
 lp_impact = round.(lpir.values[1, :], digits=3),
 F = round(proxy_ffr.first_stage_F, digits=1))
```

```@example proxy
ir_full = irf(model, 12; method=:proxy, instruments=Z,
              normalize=:unit_effect, normalize_var=3,
              shock_names=["MP", "u2", "u3"])
report(ir_full)
```

On impact the instrumented shock raises the funds rate by construction (unit-effect on `ffr`). The output-gap and inflation impacts from the proxy SVAR and from LP-IV at ``h = 0`` agree to sampling error, as the two estimands coincide when the controls are the same VAR lags. First-stage ``F`` on this window is large enough that the point IRFs are not a weak-instrument artefact; the Anderson–Rubin bands in Recipe 5 remain the conservative report when ``F`` is closer to 10.

---

## Common Pitfalls

1. **Weak instruments.** ``F < 10`` does not stop estimation; it invalidates Wald bands. Read `first_stage_F`, and use `proxy_ar_band` when the statistic is small.

2. **i.i.d. residual bootstrap.** Resampling ``u_t`` alone breaks the ``(u_t, z_t)`` dependence that identifies the shock. Pass `ci_type=:bootstrap` (the proxy path uses the moving-block scheme) rather than a residual i.i.d. bootstrap of the VAR only.

3. **Partial identification.** With ``k < n`` the complement columns of ``Q`` are an arbitrary orthogonal completion. FEVD and historical-decomposition shares of those unidentified shocks are not identified; interpret only the instrumented columns.

4. **Missing proxy dates.** Narrative and high-frequency series are `NaN` outside their published samples. `align=true` drops those rows pairwise; passing a window that is all `NaN` throws.

5. **``k > 1`` rotations.** Two instruments identify a two-dimensional subspace, not two labelled shocks, unless further restrictions pin down the ``k \times k`` rotation.

6. **Unit-effect versus unit-variance.** Unit-effect ``Q`` is not orthogonal, so ``B_0 B_0' \neq \Sigma``. Use `normalize=:unit_variance` before `fevd` if the identified shock's variance share is the object of interest.

---

## References

- Gertler, Mark, and Peter Karadi. 2015. "Monetary Policy Surprises, Credit Costs, and Economic Activity." *American Economic Journal: Macroeconomics* 7 (1): 44--76. [DOI](https://doi.org/10.1257/mac.20130329)

- Jentsch, Carsten, and Kurt G. Lunsford. 2019. "The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States: Comment." *American Economic Review* 109 (7): 2655--2678. [DOI](https://doi.org/10.1257/aer.20162011)

- Mertens, Karel, and Morten O. Ravn. 2013. "The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States." *American Economic Review* 103 (4): 1212--1247. [DOI](https://doi.org/10.1257/aer.103.4.1212)

- Montiel Olea, José L., James H. Stock, and Mark W. Watson. 2021. "Inference in Structural Vector Autoregressions Identified with an External Instrument." *Journal of Econometrics* 225 (1): 74--87. [DOI](https://doi.org/10.1016/j.jeconom.2020.05.014)

- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica* 89 (2): 955--980. [DOI](https://doi.org/10.3982/ECTA17813)

- Stock, James H., and Mark W. Watson. 2018. "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *Economic Journal* 128 (610): 917--948. [DOI](https://doi.org/10.1111/ecoj.12593)
