# [Policy Counterfactuals API](@id api_counterfactual)

Complete API reference for the policy-counterfactual module (`src/counterfactual/`): sufficient-statistic containers, rule and loss templates, the McKay–Wolf engine, the Barnichon–Mesters OPP, the Caravello–McKay–Wolf model bank, and the spanning/forecast-sufficiency diagnostics. See [Policy Counterfactuals](@ref counterfactual_page) for the methodology.

## Containers and Templates

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["counterfactual/types.jl", "counterfactual/rules.jl", "counterfactual/loss.jl"]
Order   = [:type, :function]
```

---

## Empirical and Model Inputs

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["counterfactual/empirical.jl", "counterfactual/forecast.jl",
           "counterfactual/irf_target.jl", "counterfactual/model_dsge.jl",
           "counterfactual/model_ha.jl", "counterfactual/behavioral.jl"]
Order   = [:type, :function]
```

---

## Counterfactual Engines

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["counterfactual/counterfactual.jl", "counterfactual/optimal_policy.jl",
           "counterfactual/moments.jl", "counterfactual/opp.jl",
           "counterfactual/constrained.jl", "counterfactual/opp_sequence.jl"]
Order   = [:type, :function]
```

---

## Model Bank, History, and Diagnostics

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["counterfactual/model_bank.jl", "counterfactual/historical.jl",
           "counterfactual/diagnostics.jl", "counterfactual/show.jl"]
Order   = [:type, :function]
```
