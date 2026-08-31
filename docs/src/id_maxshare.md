# [Max-Share Identification](@id id_maxshare_page)

Max-share identification selects the structural shock that explains the largest fraction of a target variable's forecast-error variance over a finite window (Uhlig 2004; Francis, Owyang, Roush and DiCecio 2014) or of its spectral mass on a frequency band (Angeletos, Collard and Dellas 2020). The leading application is a **news shock**: the disturbance that best accounts for future productivity, orthogonal to a surprise-TFP column (Barsky and Sims 2011). Restriction-based schemes live on [Structural Identification](@ref structural_identification_page); external instruments are [Proxy SVAR](@ref id_proxy_page).

- **Time domain**: leading eigenvector of ``S = \sum_{h\in\mathcal{H}} (e_i'\Phi_h L)'(e_i'\Phi_h L)``
- **Frequency domain**: Gauss–Legendre quadrature of the same quadratic form on ``[\omega_1,\omega_2]``
- **Sequential shocks**: `previous` restricts the maximisation to an orthogonal complement
- **Partial identification**: remaining columns of ``Q`` are unidentified

```@setup maxshare
using MacroEconometricModels, Random, LinearAlgebra
Random.seed!(42)
qd = load_example(:fred_qd)
Y = to_matrix(apply_tcode(qd[:, ["OPHNFB", "GDPC1", "UNRATE"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
model = estimate_var(Y, 2; varnames=["prod", "gdp", "unrate"])
```

## Quick Start

**Recipe 1: Identify the shock that moves productivity**

```@example maxshare
ms = identify_max_share(model; target="prod", horizons=0:16)
report(ms)
```

**Recipe 2: Impulse responses of the max-share shock**

```@example maxshare
ir_ms = irf(model, 16; method=:max_share, target="prod", horizons=0:16,
            shock_names=["News", "u2", "u3"])
report(ir_ms)
```

```julia
plot_result(ir_ms)
```

**Recipe 3: FEVD share of the identified shock**

```@example maxshare
decomp = fevd(model, 16; method=:max_share, target="prod", horizons=0:16,
              shock_names=["News", "u2", "u3"])
report(decomp)
```

**Recipe 4: Frequency-domain (business-cycle) band**

```@example maxshare
ω_bc = (2π / 32, 2π / 6)
ms_bc = identify_max_share(model; target="gdp", band=ω_bc)
(share = round(ms_bc.share, digits=3),
 q_prod = round.(ms_bc.q, digits=3))
```

**Recipe 5: Barsky–Sims news shock (orthogonal to a surprise)**

```@example maxshare
surprise = identify_max_share(model; target="prod", horizons=0:0)
news = identify_max_share(model; target="prod", horizons=0:16, previous=surprise.q)
(orthogonality = round(dot(news.q, surprise.q), digits=8),
 news_share = round(news.share, digits=3))
```

**Recipe 6: Historical decomposition of the news shock**

```@example maxshare
hd_ms = historical_decomposition(model; method=:max_share, target="prod",
                                 horizons=0:16, shock_names=["News", "u2", "u3"])
report(hd_ms)
```

---

## Time-Domain Max-Share

A reduced-form VAR has moving-average coefficients ``\Phi_h`` and Cholesky factor ``L`` of ``\hat\Sigma``. The contribution of a unit-length rotation ``q`` to variable ``i`` at horizon ``h`` is ``e_i'\Phi_h L q``. Summing squared contributions over a window ``\mathcal{H}`` yields the quadratic form

```math
S = \sum_{h\in\mathcal{H}} (e_i'\Phi_h L)'(e_i'\Phi_h L)
```

where:
- ``e_i`` is the selection vector for the target variable
- ``\Phi_h`` is the reduced-form MA coefficient at lag ``h`` (``\Phi_0 = I``)
- ``L = \mathrm{chol}(\hat\Sigma)`` so ``B_0 = L Q``

The max-share shock is the leading eigenvector of ``S``. Because the denominator of the FEV share (total MSE of variable ``i`` over ``\mathcal{H}``) does not depend on ``q``, maximising the contribution is equivalent to maximising the share. That share equals ``\lambda_{\max}(S)/\mathrm{tr}(S)`` and is stored as `share`. With `cumulative=true` the same construction uses the partial sums ``\Psi_h = \sum_{k=0}^{h}\Phi_k`` in place of ``\Phi_h``.

!!! note "Technical Note"
    Horizons are 0-indexed: `horizons=0:16` sums impact through lag 16 (17 MA matrices). `ma_coefficients` and [`cholesky_factor`](@ref) supply ``\Phi_h`` and ``L``. The first `n_shocks` columns of ``Q`` are the leading eigenvectors; the orthogonal complement is unidentified (`is_partial=true` when ``n_{\mathrm{shocks}} < n``). Sign normalisation makes the target impact ``(Lq)_i`` positive.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `target` | `Int` or `String` | required | Target variable (index or name) |
| `horizons` | `UnitRange{Int}` | `0:20` | Time-domain window (ignored if `band` is set) |
| `band` | `Tuple` | `nothing` | Frequency band ``(\omega_1,\omega_2)`` in radians |
| `cumulative` | `Bool` | `false` | Use cumulative MA coefficients ``\Psi_h`` |
| `n_shocks` | `Int` | `1` | Number of sequential max-share shocks |
| `previous` | `AbstractVecOrMat` | `nothing` | Already-identified columns to orthogonalise against |
| `n_quad` | `Int` | `32` | Gauss–Legendre nodes for `band` |

**Return value** (`MaxShareResult`):

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Orthogonal rotation; leading columns identified |
| `q` | `Vector{T}` | First identified column of ``Q`` |
| `share` | `T` | ``\lambda_{\max}/\mathrm{tr}(S)`` |
| `eigvals` | `Vector{T}` | Eigenvalues of ``S``, descending |
| `is_partial` | `Bool` | `true` when unidentified columns remain |

```@example maxshare
fv_match = fevd(model, 17; method=:max_share, target="prod", horizons=0:16)
(share = round(ms.share, digits=3),
 fevd_news = round(fv_match.proportions[1, 1, 17], digits=3))
```

On this sample the max-share shock accounts for most of productivity's 16-quarter forecast-error variance — that is the identifying criterion, so a large own-share is the definition of success rather than an empirical finding about the data.

---

## Frequency Domain

Parseval's identity equates the infinite-horizon FEV with the integral of the spectral density. Restricting the integral to a band ``[\omega_1,\omega_2]`` identifies the shock that dominates those frequencies (Angeletos, Collard and Dellas 2020). The criterion matrix is

```math
S = \int_{\omega_1}^{\omega_2} \mathrm{Re}\bigl[(e_i' C(e^{-i\omega}) L)^{*} (e_i' C(e^{-i\omega}) L)\bigr]\,d\omega
```

where:
- ``C(z) = (I - A_1 z - \cdots - A_p z^p)^{-1}`` is the VAR transfer function
- the integral is Gauss–Legendre quadrature (`n_quad=32` nodes by default)

A business-cycle band of 6–32 quarters is ``(2\pi/32,\, 2\pi/6)``. Passing both `horizons` and `band` throws `ArgumentError`.

```@example maxshare
ms_low = identify_max_share(model; target="gdp", band=(0.0, 2π / 32))
(low_share = round(ms_low.share, digits=3),
 alignment = round(abs(dot(ms_bc.q, ms_low.q)), digits=3))
```

The business-cycle shock and the low-frequency shock need not coincide: GDP's trend-cycle split can load on different rotations. Alignment near one would say that the same disturbance dominates both bands on this sample.

---

## News Shocks

Barsky and Sims (2011) identify a **surprise** productivity shock as the disturbance that explains TFP on impact, then a **news** shock as the max-share of TFP over a longer window in the orthogonal complement of that surprise. Recipe 5 is that construction with labour productivity (`OPHNFB`) standing in for utilisation-adjusted TFP.

```math
q_{\mathrm{news}} = \arg\max_{q'q=1,\; q\perp q_{\mathrm{surprise}}} q'S_{\mathcal{H}} q
```

`previous` accepts a vector or an ``n\times k`` matrix of earlier columns; they are orthonormalised by QR before the projection ``S \leftarrow P S P`` with ``P = I - Q_{\mathrm{prev}}Q_{\mathrm{prev}}'``.

```@example maxshare
ir_news = irf(model, 16; method=:max_share, target="prod", horizons=0:16,
              previous=surprise.q, shock_names=["News", "u2", "u3"])
(impact_prod = round(ir_news.values[1, 1, 1], digits=4),
 h8_gdp = round(ir_news.values[9, 2, 1], digits=4))
```

A news shock can move productivity little on impact (the surprise column already owns that variation) and still move output at business-cycle horizons if agents react to the signal. The GDP response at ``h=8`` is that delayed transmission on this window.

---

## Complete Example

A three-variable quarterly system — labour productivity, real GDP, and the unemployment rate from `:fred_qd` — is identified twice: a raw max-share of productivity over 16 quarters, and a Barsky–Sims news shock orthogonal to the impact surprise.

```@example maxshare
ms_raw = identify_max_share(model; target="prod", horizons=0:16)
surp = identify_max_share(model; target="prod", horizons=0:0)
news2 = identify_max_share(model; target="prod", horizons=0:16, previous=surp.q)
fv_raw = fevd(model, 17; method=:max_share, target="prod", horizons=0:16)
(raw_share = round(ms_raw.share, digits=3),
 news_share = round(news2.share, digits=3),
 fevd_window = round(fv_raw.proportions[1, 1, 17], digits=3),
 q_dot = round(abs(dot(ms_raw.q, news2.q)), digits=3))
```

The raw max-share of productivity and the news column after removing the surprise are not the same rotation: the surprise is part of what the unrestricted maximisation would claim. FEVD over ``0:16`` matches `share` because that is ``\lambda_{\max}/\mathrm{tr}(S)`` for the same window.

```@example maxshare
refs(ms_raw)
```

---

## Common Pitfalls

1. **Partial identification.** Only the leading `n_shocks` columns of ``Q`` are identified. FEVD and historical-decomposition shares of the complement are not identified; the functions warn once and still fill those columns numerically.

2. **Horizon 0 versus 1.** `horizons=0:H` includes impact. Passing `1:H` drops the contemporaneous contribution and can relabel a news shock as a surprise.

3. **`horizons` and `band` together.** The two windows are alternative criteria, not a combination. Specify one.

4. **News without a surprise.** Max-sharing TFP over ``0:H`` without `previous` folds the surprise into the "news" column. Follow Recipe 5 when the object is a news shock.

5. **Sign of the target impact.** Columns are flipped so ``(Lq)_i > 0``. Comparing ``q`` across samples requires the same convention.

6. **Column matching is off.** Identified max-share columns are already labelled. Bootstrap draws are not permuted against a reference impact (`_should_match_columns(:max_share) == false`), the same treatment as `:proxy`.

7. **Unit-root targets.** The transfer ``C(e^{-i\omega})`` is ill-conditioned at ``\omega=0`` when the VAR has a unit root. Difference the target or drop frequencies at the origin.

---

## References

- Angeletos, George-Marios, Fabrice Collard, and Harris Dellas. 2020. "Business-Cycle Anatomy." *American Economic Review* 110 (10): 3030--3070. [DOI](https://doi.org/10.1257/aer.20181080)

- Barsky, Robert B., and Eric R. Sims. 2011. "News Shocks and Business Cycles." *Journal of Monetary Economics* 58 (3): 273--289. [DOI](https://doi.org/10.1016/j.jmoneco.2011.03.001)

- Francis, Neville, Michael T. Owyang, Jennifer E. Roush, and Riccardo DiCecio. 2014. "A Flexible Finite-Horizon Alternative to Long-Run Restrictions with an Application to Technology Shocks." *Journal of Money, Credit and Banking* 46 (2-3): 343--370. [DOI](https://doi.org/10.1111/jmcb.12105)

- Uhlig, Harald. 2004. "Do Technology Shocks Lead to a Fall in Total Hours Worked?" *Journal of the European Economic Association* 2 (2-3): 361--371. [DOI](https://doi.org/10.1162/154247604323068041)
