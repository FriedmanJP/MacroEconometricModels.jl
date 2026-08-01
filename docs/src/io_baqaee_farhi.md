# [Baqaee & Farhi (2019) Nonlinear Input-Output](@id io_baqaee_farhi_page)

Hulten's (1978) theorem says that, to first order, the effect of a sector's productivity on aggregate output is its sales share and nothing else — the shape of the production network is irrelevant. Baqaee & Farhi (2019) show that this is an artefact of the first order: as soon as shocks are large enough for second-order terms to matter, the network reasserts itself through the elasticities of substitution that govern how inputs are reallocated. This page computes both orders from an [`IOData`](@ref) table. See [Input-Output Analysis](@ref io_page) for the container and [Classical Analysis](@ref io_classical_page) for the linear multipliers this decomposition generalizes.

- **First order**: Domar weights and Hulten's theorem, exact under Cobb-Douglas technology
- **Second order**: the "beyond Hulten" Hessian of log output in log productivities, parameterized by production and consumption substitution elasticities
- **Centralities**: the influence vector, upstreamness, and downstreamness of each sector in the production network

```@setup io_baqaee_farhi
using MacroEconometricModels
```

## Quick Start

**Recipe 1: Domar weights**

```@example io_baqaee_farhi
io = load_example(:wiot)
domar_weights(io)
```

**Recipe 2: The full decomposition**

```@example io_baqaee_farhi
bf = baqaee_farhi(io)
report(bf)
```

**Recipe 3: Gross substitutes in production**

```@example io_baqaee_farhi
baqaee_farhi(io; theta=2.0).second_order
```

**Recipe 4: Complements in production**

```@example io_baqaee_farhi
baqaee_farhi(io; theta=0.5).second_order
```

**Recipe 5: Network centralities**

```@example io_baqaee_farhi
bf.upstreamness, bf.downstreamness
```

---

## Domar Weights and Hulten's Theorem

The **Domar weight** of a sector is its sales divided by GDP. Because intermediate sales are counted in both the numerator of each buyer and the numerator of each seller, Domar weights sum to more than one — the ratio of gross output to value added (Domar 1961).

```math
\lambda_i = \frac{p_i y_i}{\text{GDP}} = \frac{x_i}{\mathbf{1}'V\mathbf{1}},
\qquad \frac{d \log Y}{d \log A_i} = \lambda_i
```

where:
- ``\lambda_i`` is the Domar weight of sector ``i``
- ``x_i`` is gross output, that is total sales including intermediate sales
- ``\mathbf{1}'V\mathbf{1}`` is GDP, the sum of every entry of the value-added matrix
- ``A_i`` is the Hicks-neutral productivity of sector ``i``
- ``Y`` is aggregate real output

The right-hand identity is **Hulten's theorem**: a sufficient statistic for the first-order impact of a microeconomic productivity shock is the sector's sales share. No elasticity, no network position, and no measure of centrality adds anything at this order.

```@example io_baqaee_farhi
domar_weights(io)
```

```@example io_baqaee_farhi
sum(domar_weights(io))
```

Agriculture's weight of 0.488 and manufacturing's 0.976 say that a one percent productivity gain in manufacturing raises output twice as much as the same gain in agriculture, which is exactly the ratio of their sales. Their sum of 1.463 is the economy's gross output of 3000 over its GDP of 2050: the "intermediate-input multiplier" by which the network amplifies value added into sales. Hulten's theorem holds exactly by construction in this package's implementation, so `first_order` is a copy of `domar`:

```@example io_baqaee_farhi
bf.first_order == domar_weights(io)
```

---

## The Second-Order "Beyond Hulten" Term

The second-order term is the Hessian of log output in log productivities. It aggregates the input-output covariances of the network, weighted by how easily producers and consumers substitute away from a damaged sector.

```math
H_{jk} = \frac{d^2 \log Y}{d \log A_j \, d \log A_k}
= (\theta - 1)\sum_{i=1}^{n} \lambda_i \operatorname{Cov}_{\Omega^{(i)}}\!\left(\Psi_{(j)}, \Psi_{(k)}\right)
+ (\sigma - 1)\operatorname{Cov}_{\beta}\!\left(\Psi_{(j)}, \Psi_{(k)}\right)
```

where:
- ``\Psi = L = (I-A)^{-1}`` is the Leontief inverse and ``\Psi_{(j)}`` its ``j``-th column
- ``\Omega^{(i)} = A_{\cdot i} / \sum_{l} A_{li}`` is sector ``i``'s vector of input-cost shares, normalized to sum to one
- ``\beta = y / \mathbf{1}'y`` is the vector of final-demand shares
- ``\operatorname{Cov}_{\omega}(u, v) = \sum_{l} \omega_l (u_l - \bar{u})(v_l - \bar{v})`` with ``\bar{u} = \sum_l \omega_l u_l`` is the covariance under the weights ``\omega``
- ``\theta`` is the elasticity of substitution across intermediate inputs in production
- ``\sigma`` is the elasticity of substitution across goods in consumption
- ``\lambda_i`` are the Domar weights, which weight the producer-side term by each sector's size

The two terms carry the whole economics. The producer term is positive when ``\theta > 1``, because firms facing **gross substitutes** reallocate spending away from the sector whose price has risen. It is negative when ``\theta < 1``, because firms facing **complements** cannot substitute and are dragged along by the bottleneck. The consumer term does the same for households through ``\sigma``.

```math
\Delta \log Y \approx \sum_{i} \lambda_i \, \Delta \log A_i
+ \frac{1}{2} \sum_{j} \sum_{k} H_{jk} \, \Delta \log A_j \, \Delta \log A_k
```

!!! note "Cobb-Douglas is the exact-Hulten case"
    With ``\theta = \sigma = 1`` — the default when `theta` and `sigma` are not supplied —
    both coefficients vanish and the Hessian is exactly zero. This is not a numerical
    accident: under Cobb-Douglas technology cost shares are invariant to prices, there is
    nothing to reallocate, and Hulten's theorem holds globally rather than locally.

```@example io_baqaee_farhi
baqaee_farhi(io).second_order
```

```@example io_baqaee_farhi
baqaee_farhi(io; theta=2.0).second_order
```

With gross substitutes the diagonal is positive: ``H_{11} = 0.250`` makes log output **convex** in agricultural productivity. A ten percent productivity gain in agriculture is worth ``0.488 \times 0.10 = 4.88`` percent at first order plus ``\tfrac{1}{2} \times 0.250 \times 0.01 = 0.13`` percent of reallocation gain, for 5.00 percent in total; a ten percent *loss* costs only 4.75 percent, because the economy substitutes away from the damaged sector. Complements reverse both signs:

```@example io_baqaee_farhi
baqaee_farhi(io; theta=0.5).second_order
```

Here ``H_{11} = -0.125`` makes output **concave**: the same ten percent loss now costs 4.94 percent rather than 4.88, and the same gain is worth only 4.82. This asymmetry — losses amplified, gains dampened — is the central result of Baqaee & Farhi (2019), and it is invisible to any first-order or purely linear input-output calculation.

The off-diagonal entry of ``-0.200`` at ``\theta = 2`` is what makes uniform shocks different from idiosyncratic ones. Applying a ten percent gain to *both* sectors gives a quadratic form of ``\tfrac{1}{2}(0.250 - 2 \times 0.200 + 0.160) \times 0.01 \approx 0.005`` percent, essentially nothing: a common productivity shock changes no relative price, so there is no reallocation to capture. Second-order effects require dispersion in productivity across sectors, not aggregate TFP movements.

Consumption substitution enters through ``\sigma`` with the same structure:

```@example io_baqaee_farhi
baqaee_farhi(io; sigma=2.0).second_order
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `theta` | `Union{Real,Nothing}` | `nothing` | Elasticity of substitution across intermediate inputs; `nothing` is Cobb-Douglas, ``\theta = 1`` |
| `sigma` | `Union{Real,Nothing}` | `nothing` | Elasticity of substitution in consumption; `nothing` is Cobb-Douglas, ``\sigma = 1`` |

---

## Network Centralities

Alongside the two orders, the decomposition reports three summaries of where each sector sits in the production network. They are the input-output objects that Acemoglu et al. (2012) and Carvalho & Tahbaz-Salehi (2019) use to explain why idiosyncratic shocks fail to wash out in aggregate.

```math
v = \beta' \Psi, \qquad
\text{up}_i = \sum_{j} \Psi_{ij}, \qquad
\text{down}_j = \sum_{i} \Psi_{ij}
```

where:
- ``v_j`` is the **influence** of sector ``j``, the final-demand-weighted column aggregation of the Leontief inverse
- ``\text{up}_i`` is the **upstreamness** of sector ``i``, the row sum of ``\Psi`` — the output of ``i`` required by one unit of final demand for every product
- ``\text{down}_j`` is the **downstreamness** of sector ``j``, the column sum of ``\Psi`` — the total production triggered by one unit of final demand for ``j``
- ``\beta`` is the vector of final-demand shares, as above

```@example io_baqaee_farhi
bf.influence
```

```@example io_baqaee_farhi
bf.upstreamness, bf.downstreamness
```

Agriculture is the more **upstream** sector (1.584 against 1.386): a unit of final demand for either product draws more heavily on farm output than on manufacturing output, relative to the two sectors' sizes. Manufacturing is the less **downstream** one (1.452 against 1.518), because it buys fewer intermediates per unit of output. These two vectors are not new quantities — `downstreamness` is exactly the backward linkage and the Type I output multiplier, and `upstreamness` is exactly the Chenery-Watanabe forward linkage of the [Classical Analysis](@ref io_classical_page) page. Antràs et al. (2012) build a closely related upstreamness index from the allocation coefficients rather than the Leontief inverse.

!!! note "`influence` is a column aggregation, not the Domar weight"
    Under this package's orientation of ``\Psi`` — where ``x = \Psi y`` — the Domar weights are
    recovered by the *row* aggregation ``\Psi\beta``, which reproduces ``[0.488, 0.976]``
    exactly. The `influence` field reports the *column* aggregation ``\beta'\Psi``, here
    ``[0.433, 0.987]``. The two agree only when ``\Psi`` is symmetric, so read `influence` as a
    demand-weighted average requirement rather than as an alternative Domar weight.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `domar` | `Vector{Float64}` | Domar weights ``\lambda_i = x_i / \text{GDP}`` |
| `first_order` | `Vector{Float64}` | Hulten first-order elasticities, a copy of `domar` |
| `second_order` | `Matrix{Float64}` | ``n \times n`` symmetrized "beyond Hulten" Hessian |
| `influence` | `Vector{Float64}` | Influence vector ``\beta'\Psi`` |
| `upstreamness` | `Vector{Float64}` | Row sums of ``\Psi`` |
| `downstreamness` | `Vector{Float64}` | Column sums of ``\Psi`` |
| `sectors` | `Vector{String}` | Sector labels |

---

## Complete Example

This example prices a large negative productivity shock two ways: with Hulten's first-order rule, and with the second-order correction that a calibrated network implies. The gap between them is what the Baqaee & Farhi decomposition buys.

```@example io_baqaee_farhi
tbl = load_example(:wiot)

# Calibration: intermediate inputs are gross substitutes, consumption goods
# are mild complements
bf_cal = baqaee_farhi(tbl; theta=2.0, sigma=0.9)
report(bf_cal)
```

```@example io_baqaee_farhi
bf_cal.second_order
```

```@example io_baqaee_farhi
# A ten percent productivity loss confined to agriculture
dlogA = [-0.10, 0.0]

hulten = bf_cal.first_order' * dlogA
correction = 0.5 * dlogA' * bf_cal.second_order * dlogA
(hulten, correction, hulten + correction)
```

```@example io_baqaee_farhi
# The same shock hitting both sectors
dlogA_common = [-0.10, -0.10]
(bf_cal.first_order' * dlogA_common,
 0.5 * dlogA_common' * bf_cal.second_order * dlogA_common)
```

Hulten prices the agricultural shock at a 4.88 percent fall in aggregate output. The second-order term adds back 0.12 percentage points, because substitution toward manufacturing partly replaces the lost farm output, so the calibrated network puts the true cost at 4.76 percent. The common shock tells the opposite story: its first-order cost of 14.63 percent — the sum of both Domar weights — comes with a correction of essentially zero, since a shock that hits every sector equally leaves no relative price to substitute against. The correction is therefore not a uniform haircut on Hulten; it is a function of how concentrated the shock is.

---

## Common Pitfalls

1. **A zero `second_order` matrix is the expected default.** With no `theta` or `sigma` the model is Cobb-Douglas, both coefficients ``(\theta-1)`` and ``(\sigma-1)`` vanish, and Hulten is exact. Supply at least one elasticity before concluding that the network does not matter.

2. **`theta` and `sigma` are scalars, not vectors.** One production elasticity applies to every sector and one consumption elasticity to every good. Sector-specific elasticities, which Baqaee & Farhi (2019) allow, are not supported — calibrate a representative value or run the decomposition separately under bracketing values.

3. **Domar weights sum to more than one, and should.** Their sum is gross output over GDP. A sum near one indicates a table with almost no intermediate trade, not a normalization error.

4. **GDP is the sum of the entire `va` matrix.** Every value-added category counts, including net taxes on production and any residual row. A table that carries taxes as a separate value-added category yields Domar weights on a different denominator than one that nets them out, so weights are comparable across tables only when the value-added blocks are defined the same way.

5. **`second_order` is the Hessian, not the contribution.** The factor of ``\tfrac{1}{2}`` belongs to the Taylor expansion, not to the returned matrix. Forgetting it doubles the correction.

6. **The expansion is local.** Two orders approximate well for shocks of a few percent. For a shock large enough to shut a sector down, use [hypothetical extraction](@ref io_classical_page), which solves the counterfactual exactly instead of approximating it.

7. **Aggregation changes the answer.** Cost shares are read straight off the table, so a two-sector table and a four-hundred-sector table of the same economy give different Hessians. Substitution possibilities that are within-sector in a coarse classification become across-sector — and therefore visible to the decomposition — in a fine one.

---

## API Reference

```@docs
domar_weights
baqaee_farhi
```

---

## References

- Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The Network Origins of Aggregate Fluctuations.
  *Econometrica*, 80(5), 1977--2016. [DOI](https://doi.org/10.3982/ECTA9623)

- Antras, P., Chor, D., Fally, T., & Hillberry, R. (2012). Measuring the Upstreamness of Production and Trade Flows.
  *American Economic Review*, 102(3), 412--416. [DOI](https://doi.org/10.1257/aer.102.3.412)

- Baqaee, D. R., & Farhi, E. (2019). The Macroeconomic Impact of Microeconomic Shocks: Beyond Hulten's Theorem.
  *Econometrica*, 87(4), 1155--1203. [DOI](https://doi.org/10.3982/ECTA15202)

- Baqaee, D. R., & Farhi, E. (2020). Productivity and Misallocation in General Equilibrium.
  *The Quarterly Journal of Economics*, 135(1), 105--163. [DOI](https://doi.org/10.1093/qje/qjz030)

- Carvalho, V. M., & Tahbaz-Salehi, A. (2019). Production Networks: A Primer.
  *Annual Review of Economics*, 11, 635--663. [DOI](https://doi.org/10.1146/annurev-economics-080218-030212)

- Domar, E. D. (1961). On the Measurement of Technological Change.
  *The Economic Journal*, 71(284), 709--729. [DOI](https://doi.org/10.2307/2228246)

- Hulten, C. R. (1978). Growth Accounting with Intermediate Inputs.
  *The Review of Economic Studies*, 45(3), 511--518. [DOI](https://doi.org/10.2307/2297252)

- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.).
  Cambridge University Press. ISBN 978-0-521-51713-3. [DOI](https://doi.org/10.1017/CBO9780511626982)
