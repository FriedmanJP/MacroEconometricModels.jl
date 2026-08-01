# [Heterogeneity & Continuous Time](@id dsge_heterogeneity)

The representative-agent models of the [DSGE Models](@ref dsge_page) suite collapse the cross-section of households into a single decision-maker. That assumption is what makes the Sims (2002) canonical form possible, and it is exactly what fails when the question involves who holds the wealth, who bears the risk, or how long anyone lives. The three families on this page each break the representative agent in a different way, and each therefore carries its own solver rather than routing through `linearize` and `solve`.

**Heterogeneous-agent models** keep the wealth distribution itself as a state variable: households face uninsurable idiosyncratic income risk and a borrowing constraint, so their marginal propensities to consume differ and aggregate dynamics depend on who receives a shock. **Overlapping-generations models** replace the infinitely-lived household with cohorts that die and are replaced, so finite horizons drive a wedge between the interest rate and the discount rate and government debt becomes net wealth. **Continuous-time models** solve the same incomplete-markets problem as a pair of partial differential equations on a sparse grid, which buys accuracy on fine wealth grids that discrete-time simulation cannot match.

```@setup dsge_het
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

Solve the Krusell-Smith (1998) economy to its stationary equilibrium. The endogenous grid method computes the household savings policy, a Young (2010) histogram tracks the wealth distribution, and bisection on the interest rate continues until the capital market clears:

```@example dsge_het
spec = load_ha_example(:krusell_smith)
ss = compute_steady_state(spec; K_init=10.0, r_bounds=(-0.02, 0.04),
                          max_iter=80, tol=1e-4)
report(ss)
```

Bisection settles on an aggregate capital stock of 42.44 supporting output of 3.85, with excess capital demand driven down to ``2.3 \times 10^{-5}``. The stationary distribution is the object a representative agent assumes away: its Gini coefficient is 0.56, median wealth of 25.96 sits far below mean wealth of 42.44, and 6.31% of households are pinned at the borrowing floor. Those constrained households consume their entire income, which is why aggregate consumption in this class of model responds to the distribution of a transfer and not only to its size.

---

## Choosing a Method

The economic mechanism that matters selects the family, and the family selects the solver:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Wealth distribution as a state variable | [Heterogeneous Agents](@ref dsge_ha) | Young histogram with three aggregate solvers |
| MPC heterogeneity and HANK policy analysis | [Heterogeneous Agents](@ref dsge_ha) | One- and two-asset HANK built in |
| Aggregate dynamics from a steady state alone | [Heterogeneous Agents](@ref dsge_ha) | Sequence-space Jacobian needs no simulation |
| Finite horizons and Ricardian failure | [Overlapping Generations](@ref dsge_olg) | Generational turnover raises the interest rate |
| Age-earnings profiles and retirement | [Overlapping Generations](@ref dsge_olg) | Backward induction over age |
| Accuracy on fine wealth grids | [Continuous Time](@ref dsge_continuous) | Implicit upwind scheme, no simulation |
| Stationary distribution from a sparse operator | [Continuous Time](@ref dsge_continuous) | One generator drives HJB and KFE |
| Liquid/illiquid portfolio choice | [Continuous Time](@ref dsge_continuous) | Kaplan-Moll-Violante two-asset machinery |

Discrete and continuous time are alternative formulations of the same incomplete-markets economy, not competing models. Choose continuous time when the wealth grid must be fine or the stationary distribution is the object of interest, and discrete time when the model must be estimated or embedded in a sequence-space general-equilibrium block.

---

## Child Pages

- [Heterogeneous Agents](@ref dsge_ha) --- EGM and VFI individual problems, Young (2010) distribution tracking, sequence-space Jacobian, Reiter, and Krusell-Smith aggregate solvers, Winberry parametric distributions, two-asset HANK, and Bayesian estimation
- [Overlapping Generations](@ref dsge_olg) --- Blanchard (1985) perpetual-youth steady state and saddle-path dynamics, annuity markets, non-Ricardian government debt, and age-dependent life-cycle EGM
- [Continuous Time](@ref dsge_continuous) --- Achdou et al. (2022) HJB and Kolmogorov-Forward finite differences, MIT-shock transitions, and two-asset Kaplan-Moll-Violante HANK

---

## References

- Achdou, Yves, Jiequn Han, Jean-Michel Lasry, Pierre-Louis Lions, and Benjamin Moll. 2022. "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89 (1): 45--86. [DOI](https://doi.org/10.1093/restud/rdab002)

- Aiyagari, S. Rao. 1994. "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109 (3): 659--684. [DOI](https://doi.org/10.2307/2118417)

- Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub. 2021. "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica* 89 (5): 2375--2408. [DOI](https://doi.org/10.3982/ECTA17434)

- Blanchard, Olivier J. 1985. "Debt, Deficits, and Finite Horizons." *Journal of Political Economy* 93 (2): 223--247. [DOI](https://doi.org/10.1086/261297)

- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697--743. [DOI](https://doi.org/10.1257/aer.20160042)

- Krusell, Per, and Anthony A. Smith Jr. 1998. "Income and Wealth Heterogeneity in the Macroeconomy." *Journal of Political Economy* 106 (5): 867--896. [DOI](https://doi.org/10.1086/250034)

- Young, Eric R. 2010. "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell--Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34 (1): 36--41. [DOI](https://doi.org/10.1016/j.jedc.2008.11.010)
