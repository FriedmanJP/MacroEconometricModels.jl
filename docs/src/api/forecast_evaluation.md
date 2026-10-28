# [Forecast Evaluation API Reference](@id api_fceval_page)

Complete API documentation for forecast evaluation and combination tools in **MacroEconometricModels.jl**. This catalog covers point forecast accuracy metrics, equal-predictive-ability hypothesis tests (Diebold-Mariano, Clark-West), Mincer-Zarnowitz regression-based unbiasedness tests, forecast encompassing tests, and optimal forecast combination schemes.

See [Forecast Evaluation & Combination](@ref forecast_evaluation_page) for theory, methodology, and runnable examples.

---

## Evaluation and Combination Result Types

Result containers storing summary metrics, test statistics, p-values, regression estimates, and combination weights.

```@docs
ForecastEvaluation
DMTestResult
ClarkWestResult
MincerZarnowitzResult
ForecastEncompassingResult
ForecastCombination
```

---

## Accuracy Evaluation

Point forecast accuracy evaluation across multiple error metrics (RMSE, MAE, MAPE, MASE, Theil's U).

```@docs
forecast_evaluate
```

---

## Predictive Accuracy and Encompassing Tests

Hypothesis tests for comparing competing forecasts: equal predictive accuracy for non-nested models (Diebold-Mariano) and nested models (Clark-West), regression-based unbiasedness (Mincer-Zarnowitz), and forecast encompassing.

```@docs
diebold_mariano
clark_west
mincer_zarnowitz
forecast_encompassing
```

---

## Forecast Combination

Algorithms for combining competing forecasts using equal weighting, inverse mean squared error, and constrained or unconstrained regression weights.

```@docs
combine_forecasts
```
