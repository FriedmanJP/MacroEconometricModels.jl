# [Baqaee & Farhi (2019) Nonlinear Input-Output](@id io_baqaee_farhi_page)

Hulten's (1978) theorem says that, to first order, the effect of a sector's productivity on aggregate output is its sales share and nothing else — the shape of the production network is irrelevant. Baqaee & Farhi (2019) show that this is an artefact of the first order: as soon as shocks are large enough for second-order terms to matter, the network reasserts itself through the elasticities of substitution that govern how inputs are reallocated. Baqaee & Farhi (2020) extend the framework to *inefficient* economies with markups and wedges, separating pure technology effects from changes in allocative efficiency. This page takes an [`IOData`](@ref) table from first-order Domar weights through the full standard-form nested-CES economy — exact nonlinear counterfactuals with endogenous prices, a generalized multi-factor Hessian, factor-price incidence, and wedge decompositions. See [Input-Output Analysis](@ref io_page) for the container and [Classical Analysis](@ref io_classical_page) for the linear multipliers this decomposition generalizes.

- **First order**: Domar weights and Hulten's theorem, exact under Cobb-Douglas technology
- **Standard form**: nested CES production networks with heterogeneous elasticities and multiple primary factors
- **Exact counterfactuals**: nonlinear equilibrium prices and quantities for arbitrary (large) productivity and factor-supply shocks
- **Second order**: the full multi-factor "beyond Hulten" Hessian, consistent with the exact solver by construction
- **Incidence**: factor-price, goods-price, and Domar-share responses to productivity shocks
- **Wedges**: cost-based vs revenue-based Domar weights, markups ``\mu \ge 1``, and the B&F (2020) Theorem 1 technology / allocative-efficiency decomposition

```@setup io_baqaee_farhi
using MacroEconometricModels
io = load_example(:wiot)
```

## Quick Start

**Recipe 1: Domar weights (legacy scalar API)**

```@example io_baqaee_farhi
domar_weights(io)
```

**Recipe 2: Local Hessian on an `IOData` table**

```@example io_baqaee_farhi
bf = baqaee_farhi(io; theta=2.0, sigma=0.9)
report(bf)
```

**Recipe 3: Standard-form network + generalized local approximation**

```@example io_baqaee_farhi
net = production_network(io; theta=0.5, sigma=0.9)
local_bf = baqaee_farhi(net)
report(local_bf)
```

**Recipe 4: Exact counterfactual for a large productivity shock**

```@example io_baqaee_farhi
eq = bf_equilibrium(net; dlogA=[-0.10, 0.0])
(eq.dlogY, eq.hulten, eq.converged)
```

**Recipe 5: Shock curve — exact vs Hulten vs second-order**

```@example io_baqaee_farhi
sc = bf_shock_curve(net, 1; range=(-0.3, 0.3), points=7)
report(sc)
```

**Recipe 6: Markups and Theorem 1 decomposition**

```@example io_baqaee_farhi
net_μ = production_network(io; theta=0.5, sigma=0.9, mu=[1.2, 1.1])
(cost_based_domar(net_μ), revenue_based_domar(net_μ))
```

```@example io_baqaee_farhi
w = bf_wedge_decomp(net_μ; dlogA=[0.05, 0.0], dlogmu=[0.0, 0.02])
report(w)
```

```julia
plot_result(sc)
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
bf0 = baqaee_farhi(io)
bf0.first_order == domar_weights(io)
```

---

## The Second-Order "Beyond Hulten" Term (Legacy Scalar API)

The legacy method `baqaee_farhi(io; theta, sigma)` returns a [`BaqaeeFarhiResult`](@ref) whose second-order term is the Hessian of log output in log productivities under one scalar production elasticity ``\theta`` and one consumption elasticity ``\sigma``. With the Cobb-Douglas default (`theta=sigma=1`) the Hessian is exactly zero and Hulten is exact.

```math
H_{jk} = (\theta - 1)\sum_{i=1}^{n} \lambda_i \operatorname{Cov}_{\Omega^{(i)}}\!\left(\Psi_{(j)}, \Psi_{(k)}\right)
+ (\sigma - 1)\operatorname{Cov}_{\beta}\!\left(\Psi_{(j)}, \Psi_{(k)}\right)
```

where:
- ``\Psi = L = (I-A)^{-1}`` is the Leontief inverse (column orientation of the classical IO page)
- ``\Omega^{(i)}`` is sector ``i``'s intermediate cost-share vector, renormalized to sum to one
- ``\beta = y / \mathbf{1}'y`` is the vector of final-demand shares
- ``\theta`` is the elasticity of substitution across intermediate inputs
- ``\sigma`` is the elasticity of substitution across goods in consumption

```@example io_baqaee_farhi
baqaee_farhi(io).second_order
```

```@example io_baqaee_farhi
baqaee_farhi(io; theta=2.0).second_order
```

With gross substitutes the diagonal is positive: log output is **convex** in sectoral productivity, and the economy substitutes away from a damaged sector. Complements reverse the sign:

```@example io_baqaee_farhi
baqaee_farhi(io; theta=0.5).second_order
```

!!! note "Cobb-Douglas is the exact-Hulten case"
    With ``\theta = \sigma = 1`` — the default when `theta` and `sigma` are not supplied —
    both coefficients vanish and the Hessian is exactly zero. Under Cobb-Douglas, cost shares
    are price-invariant, there is nothing to reallocate, and Hulten's theorem holds *globally*.

### Keyword Arguments (legacy)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `theta` | `Union{Real,Nothing}` | `nothing` | Elasticity of substitution across intermediate inputs; `nothing` is Cobb-Douglas, ``\theta = 1`` |
| `sigma` | `Union{Real,Nothing}` | `nothing` | Elasticity of substitution in consumption; `nothing` is Cobb-Douglas, ``\sigma = 1`` |

### Return Values ([`BaqaeeFarhiResult`](@ref))

| Field | Type | Description |
|-------|------|-------------|
| `domar` | `Vector{Float64}` | Domar weights ``\lambda_i = x_i / \text{GDP}`` |
| `first_order` | `Vector{Float64}` | Hulten first-order elasticities, a copy of `domar` |
| `second_order` | `Matrix{Float64}` | ``n \times n`` symmetrized Hessian |
| `influence` | `Vector{Float64}` | Influence vector ``\Psi\beta``, equal to `domar` under Hulten |
| `upstreamness` | `Vector{Float64}` | Row sums of ``\Psi`` |
| `downstreamness` | `Vector{Float64}` | Column sums of ``\Psi`` |
| `sectors` | `Vector{String}` | Sector labels |

The legacy path is **frozen** for backward compatibility. For heterogeneous elasticities, multiple factors, nests, exact large shocks, or the full B&F (2019) §4 Hessian, use the standard-form API below.

---

## Standard Form and Nests

Baqaee & Farhi write the economy in **standard form**: every node is a single-elasticity CES aggregator, and arbitrary nesting is encoded by adding *fictitious* producer nodes. Calibration from a column-oriented [`IOData`](@ref) happens **once** in [`production_network`](@ref), which converts shares into **row orientation** (``\Omega[i,j]`` = expenditure share of buyer ``i`` on input ``j``).

```math
\begin{aligned}
\theta_i \neq 1: &\quad
  p_i = A_i^{-1}\Bigl[\sum_j \tilde\Omega_{ij}\, p_j^{1-\theta_i}\Bigr]^{1/(1-\theta_i)} \\
\theta_i = 1: &\quad
  \log p_i = -\log A_i + \sum_j \tilde\Omega_{ij}\,\log p_j
\end{aligned}
```

Node layout (1-based): index 1 is the household (elasticity ``\sigma``); indices ``2,\ldots,M+1`` are producers (real sectors plus fictitious nests); indices ``M+2,\ldots,M+F+1`` are primary factors.

### Nesting schemes

| `nests` | Structure | ``M`` | Typical use |
|---------|-----------|-------|-------------|
| `:single` | One CES node per real sector over all inputs (intermediates + factors) | ``n`` | Transparent multi-factor Hessian; default |
| `:two` | Outer node (``\varepsilon``) buys an intermediate bundle (``\theta``) and a VA bundle (``\eta``) | ``3n`` | Atalay (2017) / B&F quantitative calibrations |

```@example io_baqaee_farhi
net_single = production_network(io; theta=0.5, sigma=0.9, factors=:single)
(net_single.n, net_single.M, net_single.F, net_single.nests)
```

```@example io_baqaee_farhi
net_two = production_network(io; nests=:two, theta=0.1, epsilon=0.5, eta=1.0,
                             sigma=0.9, factors=:va_cats)
(net_two.n, net_two.M, net_two.F)
```

```@example io_baqaee_farhi
report(net_two)
```

Literature calibration guidance (defaults stay Cobb-Douglas = 1.0 for backward compatibility): Atalay (2017) finds intermediate inputs are strong complements (``\theta \approx 0.1``), with outer ``\varepsilon \approx 0.5``–``1``, across-factor ``\eta \approx 1``, and consumption ``\sigma \approx 0.9``–``1``. Baqaee & Farhi (2019) use similar values in their quantitative section.

### Keyword Arguments ([`production_network`](@ref))

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `theta` | scalar or length-`n` | `1.0` | Elasticity across intermediate inputs (`:single`: all inputs; `:two`: within intermediate bundle) |
| `sigma` | scalar | `1.0` | Household elasticity across goods |
| `epsilon` | scalar or length-`n` | `1.0` | Outer VA-vs-intermediate elasticity (`:two` only) |
| `eta` | scalar or length-`n` | `1.0` | Across-factor elasticity inside the VA bundle (`:two` only) |
| `nests` | `:single` or `:two` | `:single` | Nesting scheme |
| `factors` | `:single`, `:va_cats`, or `F×n` matrix | `:single` | Factor mapping |
| `check` | `Bool` | `true` | Error if clipped negative share mass exceeds 1% of a row |

Negative table entries (net taxes, inventory drawdowns) are clipped to zero and the row is renormalized, with a single `@warn`. CES cost shares must be non-negative.

---

## Exact Counterfactuals with Endogenous Prices

The headline deliverable of Baqaee & Farhi is **not** the local Hessian — it is the exact nonlinear equilibrium under large shocks. [`bf_equilibrium`](@ref) solves the nested-CES general equilibrium for arbitrary Hicks-neutral productivity shocks `dlogA` (length `n`) and factor-supply shocks `dlogL` (length `F`).

**Numéraire**: nominal GDP ``E = 1``. Base prices are one; factor supplies satisfy ``L_f = \tilde\Lambda_f`` at the base so base wages equal one.

```@example io_baqaee_farhi
net = production_network(io; theta=0.5, sigma=0.9)
eq = bf_equilibrium(net; dlogA=[-0.20, 0.0])
report(eq)
```

```@example io_baqaee_farhi
(eq.dlogY, eq.hulten, eq.dlogY - eq.hulten)
```

Under complements (``\theta = 0.5``) a 20% agricultural productivity loss costs more than Hulten's first-order prediction — the network amplifies bottlenecks. Under Cobb-Douglas the two numbers coincide for any shock size:

```@example io_baqaee_farhi
net_cd = production_network(io)   # all elasticities = 1
eq_cd = bf_equilibrium(net_cd; dlogA=[-0.50, 0.30])
(eq_cd.dlogY, eq_cd.hulten, abs(eq_cd.dlogY - eq_cd.hulten) < 1e-10)
```

### Shock curve: the signature concavity figure

[`bf_shock_curve`](@ref) sweeps one sector's shock over a grid and returns exact ``\Delta\log Y``, the Hulten line, and the second-order Taylor curve. Under complements the exact path lies below the Hulten line for negative shocks (losses amplified) and below it for positive shocks (gains dampened) — the Baqaee–Farhi asymmetry.

```@example io_baqaee_farhi
sc = bf_shock_curve(net, "Agriculture"; range=(-0.4, 0.4), points=9)
(sc.shocks[1], sc.exact[1], sc.hulten[1], sc.second_order[1])
```

```julia
plot_result(sc)
```

```@raw html
<iframe src="../assets/plots/bf_shock_curve.html" style="width:100%;height:420px;border:none;"></iframe>
```

### Keyword Arguments ([`bf_equilibrium`](@ref))

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dlogA` | length-`n` | zeros | Hicks-neutral productivity shocks on real-sector outer nodes |
| `dlogL` | length-`F` | zeros | Factor-supply shocks |
| `method` | `:newton` or `:fixedpoint` | `:newton` | Inner price fixed-point algorithm |
| `tol` | `Real` | `1e-10` | Convergence tolerance |
| `maxiter` | `Int` | `500` | Max iterations |
| `damping` | `Real` | `0.5` | Damping for Picard / outer loop |

Unconverged solves `@warn` and set `converged=false` rather than returning silently bad numbers. Strong complements (``\theta \to 0``) with large negative shocks are the hard case — that is bottleneck economics, not a bug.

---

## Generalized Local Approximation (Multi-Factor Hessian)

On a [`ProductionNetwork`](@ref), `baqaee_farhi(net)` returns a [`BFLocal`](@ref) with the full B&F (2019) §4 Hessian: heterogeneous elasticities per node, multiple primary factors, and endogenous factor prices. First order is still Hulten on real-sector outer nodes. Second order assembles Cov-blocks without 4-nested loops:

```math
\begin{aligned}
K &= \sum_{i=1}^{M+1}(\theta_i-1)\,\tilde\lambda_i\,
     \bigl(\operatorname{diag}(\omega^{(i)}) - \omega^{(i)}(\omega^{(i)})'\bigr) \\
\Gamma &= \Psi_F' K \Psi_F,\qquad
X = \Psi_F' K \Psi_P \\
\bigl[\operatorname{diag}(\tilde\Lambda) + \Gamma\bigr]\, d\log w &= X\, d\log A \\
H &= \Psi_P' K \Psi_P - X'(d\log w / d\log A)
\end{aligned}
```

where ``\Psi_P`` / ``\Psi_F`` are the producer (outer-node) / factor column blocks of ``\tilde\Psi = (I-\tilde\Omega)^{-1}``, obtained by sparse solves. Single factor implies ``\Psi_f = \mathbf{1}`` and the factor-price correction vanishes. The Hessian is reported at the ``n`` real sectors; for ``n > 500`` the default is `hessian=:none` and [`bf_quadratic`](@ref) evaluates ``v'Hv`` without forming ``H``.

```@example io_baqaee_farhi
local_bf = baqaee_farhi(net)
local_bf.second_order
```

```@example io_baqaee_farhi
# Quadratic form without the dense matrix
bf_quadratic(net, [0.1, -0.05])
```

```@example io_baqaee_farhi
# Matches the dense path
v = [0.1, -0.05]
v' * local_bf.second_order * v
```

!!! note "Legacy scalar vs standard-form Hessian"
    The legacy `baqaee_farhi(io; theta, sigma)` Hessian uses intermediate-only cost
    shares (column-oriented Leontief). The standard-form Hessian includes primary
    factors in each CES nest (B&F 2019 §4). First-order Domar weights match; second-order
    numbers generally differ. Prefer `baqaee_farhi(net)` whenever you also use
    `bf_equilibrium` — the two are cross-validated by finite differences.

### Keyword Arguments ([`baqaee_farhi(net)`](@ref baqaee_farhi))

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `hessian` | `:auto`, `:full`, or `:none` | `:auto` | Form `n×n` H when `n ≤ 500` (`:auto`); always / never with `:full` / `:none` |
| `elasticities` | `Bool` | `true` | Attach a [`BFElasticities`](@ref) block |

---

## Factor Prices and Incidence

The same local system that builds the Hessian yields first-order distributional objects via [`bf_elasticities`](@ref) (also attached to [`BFLocal`](@ref) as `.elasticities`):

| Field | Shape | Content |
|-------|-------|---------|
| `dlogw_dlogA` | ``F \times n`` | Factor-price incidence ``∂\log w_f / ∂\log A_j`` |
| `dlogp_dlogA` | ``n \times n`` | Real-sector price incidence |
| `dlambda_dlogA` | ``n \times n`` | Domar-share reallocation (equals ``H``) |

```@example io_baqaee_farhi
e = bf_elasticities(net)
e.dlogp_dlogA
```

```@example io_baqaee_farhi
net_mf = production_network(io; theta=0.5, sigma=0.9, factors=:va_cats)
e_mf = bf_elasticities(net_mf)
e_mf.dlogw_dlogA
```

With a single factor, wages are pinned by the GDP numéraire at fixed supply, so `dlogw_dlogA` is zero. With multiple factors, productivity shocks reallocate income across factors and the incidence matrix is generally nonzero.

```julia
plot_result(e)          # price-incidence heatmap
plot_result(local_bf)   # Hulten bar chart
```

---

## Wedges and Allocative Efficiency

Hulten's theorem is a *macro-envelope* result: it requires the initial equilibrium to be efficient. With markups or other wedges the envelope fails, and a productivity shock can change output both by shifting the production frontier and by reallocating resources across distorted producers (Baqaee & Farhi 2020).

### Cost-based vs revenue-based Domar weights

Pass sectoral markups `mu ≥ 1` to [`production_network`](@ref). The **cost-share** matrix ``\tilde\Omega`` is unchanged (expenditure as a share of costs); **revenue** shares are ``\Omega_{ij} = \tilde\Omega_{ij}/\mu_i`` on producer rows. That yields two Domar vectors:

| Object | Definition | Role |
|--------|------------|------|
| Cost-based ``\tilde\lambda`` | ``e_1'(I - \tilde\Omega)^{-1}`` | Weights the pure technology effect |
| Revenue-based ``\lambda`` | ``e_1'(I - \Omega)^{-1}`` | Sales / GDP; factor entries are income shares |

When ``\mu \equiv 1`` the two coincide. When ``\mu > 1``, revenue-based factor shares sum to less than one (the residual is the profit share of GDP), and profits are rebated lump-sum to the household so that nominal GDP ``E = 1`` still.

```@example io_baqaee_farhi
net_w = production_network(io; mu=[1.25, 1.10])
report(net_w)
```

```@example io_baqaee_farhi
(sum(net_w.lambda[net_w.M+2:end]),          # cost-based Λ̃ sums to 1
 sum(net_w.lambda_rev[net_w.M+2:end]))      # revenue-based Λ < 1
```

### Exact equilibrium with wedges

[`bf_equilibrium`](@ref) prices producers at markup over marginal cost, ``p_i = \mu_i c_i``, and clears factor markets with GDP equal to factor income plus profits. Markup shocks enter as `dlogmu` (length ``n``, mapped to outer real-sector nodes; fictitious nest nodes stay competitive).

```@example io_baqaee_farhi
eq_w = bf_equilibrium(net_w; dlogA=[0.05, -0.02], dlogmu=[0.0, 0.03])
(eq_w.dlogY, eq_w.technology, eq_w.allocative, eq_w.profit_share, eq_w.converged)
```

### Theorem 1 decomposition

Baqaee & Farhi (2020, Theorem 1, eq. 4) decompose the first-order change in aggregate output as

```math
d\log Y = \underbrace{\tilde\lambda'\, d\log A}_{\Delta\text{Technology}}
\;-\;
\underbrace{\tilde\lambda'\, d\log\mu \;+\; \tilde\Lambda'\, d\log\Lambda}_{\Delta\text{Allocative efficiency}}.
```

- **Technology**: holding the allocation of resources fixed, a productivity shock raises output in proportion to the producer's *cost-based* Domar weight.
- **Allocative efficiency**: equilibrium reallocation of resources. For pure productivity shocks the sufficient statistic is ``-\tilde\Lambda' d\log\Lambda`` (a weighted change in revenue factor shares). Markup shocks add the direct term ``-\tilde\lambda' d\log\mu``.

[`bf_wedge_decomp`](@ref) solves the exact equilibrium and returns a [`BFWedgeDecomp`](@ref) with both pieces. For infinitesimal shocks, `dlogY ≈ technology + allocative`; for large shocks the split remains the first-order formula while `dlogY` is exact.

```@example io_baqaee_farhi
decomp = bf_wedge_decomp(net_w; dlogA=[0.05, 0.0])
report(decomp)
```

```@example io_baqaee_farhi
# Pure markup shock: technology term is zero; all of the FO effect is allocative
decomp_μ = bf_wedge_decomp(net_w; dlogmu=[0.02, -0.01])
(decomp_μ.technology, decomp_μ.allocative, decomp_μ.dlogY)
```

```julia
plot_result(net_w)    # cost vs revenue Domar bars
plot_result(decomp)   # same Domar comparison from the decomp object
```

!!! note "μ ≡ 1 recovers Hulten"
    With `mu=1` (the default) the efficient solver is recovered bit-for-bit: cost and revenue Domar coincide, `profit_share = 0`, and the allocative term is zero to first order (Corollary 1 of B&F 2020).

---

## Network Centralities

Alongside the two orders, the legacy decomposition reports three summaries of where each sector sits in the production network (Acemoglu et al. 2012; Carvalho & Tahbaz-Salehi 2019).

```math
v = \Psi \beta, \qquad
\text{up}_i = \sum_{j} \Psi_{ij}, \qquad
\text{down}_j = \sum_{i} \Psi_{ij}
```

```@example io_baqaee_farhi
bf.upstreamness, bf.downstreamness
```

!!! note "`influence` reproduces the Domar weights"
    Under this package's column orientation of ``\Psi`` — where ``x = \Psi y`` — the
    aggregation ``\Psi\beta`` *is* the Domar weight vector. That identity is Hulten's
    theorem restated on the network. Baqaee & Farhi write the same object as
    ``\beta'\Psi`` under row orientation; the standard-form layer uses row orientation
    internally and converts once from `IOData`.

---

## Complete Example

Price a large negative agricultural productivity shock three ways: Hulten, local second-order, and exact equilibrium. Under calibrated complements the three diverge, and the exact answer is the one to trust for large shocks.

```@example io_baqaee_farhi
tbl = load_example(:wiot)

# Atalay-style complements on intermediates, mild consumption complements
net_cal = production_network(tbl; theta=0.5, sigma=0.9)
loc = baqaee_farhi(net_cal)
report(loc)
```

```@example io_baqaee_farhi
dlogA = [-0.20, 0.0]
hulten = loc.first_order' * dlogA
correction = 0.5 * dlogA' * loc.second_order * dlogA
eq_big = bf_equilibrium(net_cal; dlogA=dlogA)
(hulten, hulten + correction, eq_big.dlogY)
```

```@example io_baqaee_farhi
sc_ag = bf_shock_curve(net_cal, 1; range=(-0.3, 0.3), points=7)
report(sc_ag)
```

Hulten prices a 20% agricultural loss at about 9.8% of aggregate output. The second-order term (complements ⇒ concave) makes the local prediction more severe, and the exact solver confirms the direction: bottlenecks amplify losses. For shocks of a few percent the three nearly coincide; for tens of percent, use `bf_equilibrium`.

---

## Common Pitfalls

1. **A zero `second_order` matrix is the expected default.** With no `theta` or `sigma` (legacy) or all elasticities equal to 1 (standard form) the model is Cobb-Douglas and Hulten is exact. Supply non-unitary elasticities before concluding that the network does not matter.

2. **Orientation: row vs column.** Classical IO in this package is *column* oriented (`A[i,j]` = input of `i` per unit output of `j`). The B&F standard-form layer is *row* oriented (`Ω[i,j]` = expenditure share of buyer `i` on input `j`). Conversion happens once inside `production_network`. Do not mix conventions inside a single calculation.

3. **Negative-share clipping.** Real SUT/MRIO tables carry net taxes and inventory drawdowns as negatives. CES shares must be ≥ 0: `production_network` clips and renormalizes (with `@warn`); `check=true` errors if clipped mass exceeds 1% of any row. The calibrated economy can differ from the raw table — inspect the warning.

4. **The Hessian is local — use `bf_equilibrium` for large shocks.** Two orders approximate well for shocks of a few percent. For a 20–50% sectoral collapse, solve the exact nonlinear equilibrium. The shock-curve plot is the diagnostic that shows when the Taylor expansion leaves the exact path.

5. **`second_order` is the Hessian, not the contribution.** The factor of ``\tfrac{1}{2}`` belongs to the Taylor expansion, not to the returned matrix. Forgetting it doubles the correction.

6. **Legacy vs standard-form second order.** `baqaee_farhi(io; theta, sigma)` freezes the intermediate-only scalar formula. `baqaee_farhi(net)` implements B&F §4 with factors inside each CES nest. First-order Domar weights match; Hessian numbers generally do not. Prefer the network path when combining with `bf_equilibrium`.

7. **Domar weights sum to more than one, and should.** Their sum is gross output over GDP. A sum near one indicates almost no intermediate trade, not a normalization error.

8. **GDP is the sum of the entire `va` matrix.** Every value-added category counts. Weights are comparable across tables only when the value-added blocks are defined the same way.

9. **Aggregation changes the answer.** Cost shares are read straight off the table, so a two-sector table and a four-hundred-sector table of the same economy give different Hessians. Substitution possibilities that are within-sector in a coarse classification become across-sector — and therefore visible to the decomposition — in a fine one.

10. **Numéraire interpretation of prices.** Equilibrium prices are relative to nominal GDP ``E = 1``. Real consumption is ``Y = E / P_c``, so ``\Delta\log Y = -\Delta\log P_c``. Factor wages and goods prices are not in currency units of the raw table.

11. **Large MRIO Hessians.** Full ``H`` is ``n^2``. Default `hessian=:auto` forms it only for ``n \le 500``; above that use `hessian=:none` and `bf_quadratic(net, v)`.

12. **Cost-based vs revenue-based Domar under wedges.** With `mu > 1`, use `cost_based_domar` (or `net.lambda` on outer nodes) for the pure technology weight and `revenue_based_domar` for sales/GDP. Confusing the two mis-states Hulten-style counterfactuals in inefficient economies.

13. **Theorem 1 is first-order.** `technology` and `allocative` on [`BFEquilibrium`](@ref) / [`BFWedgeDecomp`](@ref) are the B&F (2020) first-order split; `dlogY` is the exact nonlinear change. They add up only for small shocks.

---

## API Reference

```@docs
domar_weights
baqaee_farhi
production_network
ProductionNetwork
bf_equilibrium
BFEquilibrium
bf_shock_curve
BFShockCurve
bf_elasticities
BFElasticities
BFLocal
bf_quadratic
bf_wedge_decomp
BFWedgeDecomp
cost_based_domar
revenue_based_domar
```

---

## References

- Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The Network Origins of Aggregate Fluctuations.
  *Econometrica*, 80(5), 1977--2016. [DOI](https://doi.org/10.3982/ECTA9623)

- Antras, P., Chor, D., Fally, T., & Hillberry, R. (2012). Measuring the Upstreamness of Production and Trade Flows.
  *American Economic Review*, 102(3), 412--416. [DOI](https://doi.org/10.1257/aer.102.3.412)

- Atalay, E. (2017). How Important Are Sectoral Shocks?
  *American Economic Journal: Macroeconomics*, 9(4), 254--280. [DOI](https://doi.org/10.1257/mac.20160353)

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
