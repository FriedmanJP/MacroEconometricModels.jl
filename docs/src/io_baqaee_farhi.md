# Baqaee & Farhi (2019) Nonlinear IO

`baqaee_farhi` implements the production-network decomposition of Baqaee &
Farhi (2019, *Econometrica*), which characterizes how microeconomic
productivity shocks propagate to aggregate output beyond Hulten's theorem.

```@example iobf
using MacroEconometricModels
io = load_example(:wiot)
bf = baqaee_farhi(io)
report(bf)
```

## Domar weights and Hulten's theorem

The **Domar weight** of a sector is its sales over GDP, ``\lambda_i = p_i y_i /
\text{GDP}``. Hulten's theorem states that, to first order, the elasticity of
aggregate output to a sector's productivity equals its Domar weight:

```math
\frac{d\log Y}{d\log A_i} = \lambda_i.
```

```@example iobf
domar_weights(io)
```

```@example iobf
bf.first_order ≈ domar_weights(io)     # Hulten holds exactly at first order
```

## The second-order "beyond Hulten" term

The second-order term is the Hessian of log output in log productivities,
capturing reallocation. It is parameterized by production-substitution
elasticities `theta` and consumption-substitution elasticities `sigma`. Under
the Cobb-Douglas default (`theta = sigma = 1`) the term vanishes and Hulten is
exact; with gross substitutes (`theta > 1`) the diagonal rises.

```@example iobf
baqaee_farhi(io).second_order                  # zero (Cobb-Douglas)
```

```@example iobf
baqaee_farhi(io; theta=2.0).second_order       # nonzero reallocation
```

## Network centralities

The result also reports the influence vector ``\beta' L``, upstreamness (row
sums of `L`), and downstreamness (column sums of `L`).

```@example iobf
bf.influence
```

## API

```@docs
domar_weights
baqaee_farhi
```
