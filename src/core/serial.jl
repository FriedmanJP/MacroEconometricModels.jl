# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Versioned result serialization (T248 / #347, extended to full coverage #505)
# =============================================================================
# `save_model(result, path)` / `load_model(path)` persist a fitted model — or a
# data container — to disk in a self-describing, version-tagged container so
# files survive a package upgrade. Bare `Base.serialize` is deliberately NOT the
# on-disk format — it breaks across Julia versions and struct-layout changes.
#
# Design:
#   1. `_to_serializable(m)` reduces a result to a `Dict{String,Any}` of PLAIN
#      values only — numbers, strings, symbols, bools, arrays, `nothing`, and
#      nested `Dict`s. No custom struct survives into the payload: the
#      reproducibility manifest, the LP covariance estimator, and every *nested*
#      MacroEconometricModels struct (a `VECMModel`'s `JohansenResult`, an
#      `IOData`'s `IOMetaData`, a `StructuralDFM`'s wrapped `VARModel`, …) are
#      flattened recursively to tagged dicts. Leaf codecs also flatten `Expr`,
#      `NamedTuple`, `Pair`, named `Function`, `SparseMatrixCSC`, and named
#      `Distributions.jl` objects (`__distribution__`); anonymous functions and
#      `Factorization`s drop to `nothing`. User-defined `<: Distribution`
#      subtypes are rejected.
#   2. `_build_container(m)` wraps that payload with a metadata header: the
#      `format_version`, the package + Julia versions, a timestamp, the result
#      type name, and (when present) the reproducibility manifest.
#   3. Disk read/write is the JLD2 package-dependency backend
#      (`_write_model_container` / `_read_model_container` in api.jl).
#   4. `load_model` validates the `format_version` and type tag and raises a
#      typed `SerializationError` naming the expected-vs-found version on a
#      mismatch, rather than returning a corrupted object.
#
# Reduction is generic — `_capture_fields` walks the public fields and
# `_ser_field` flattens each — so adding a type is largely a matter of listing it
# in `_SERIALIZABLE_TYPES`. Reconstruction is a generic positional-constructor
# call (`_generic_construct`), overridden explicitly only where a type takes a
# keyword-defaulted field (e.g. `BVARPosterior`'s `manifest`).
#
# `ModelSpec` is registered: `residual_fns` are rebuilt from stored
# `NamedEquation.expr` (AST allowlist + `Core.eval` + `invokelatest`). `ss_fn`
# is rebuilt from a stored `:steady_state` IR declaration, or a `linear` zeros
# closure; a programmatic closure warns once and drops. Representative-agent
# solutions (`LinearDSGE`, `DSGESolution`, perturbation / projection / OccBin,
# …) round-trip through the registry; cached `Factorization`s drop to `nothing`.
# Bayesian DSGE results (`BayesianDSGE`, `DSGEPrior`, the three state-space
# types, predictive / identification companions) round-trip; `H_inv` /
# `log_det_H` are recomputed by the state-space constructors. Household
# problems (`IndividualProblem`, `HouseholdSystem`) round-trip via
# `CRRAUtility` / named budget functions. HA results (`HASteadyState`,
# `HADSGESolution`, `KrusellSmithSolution`, Winberry / Den Haan / grid
# diagnostics) round-trip; OLG / CT results are still follow-up.

include("serial/registry.jl")
include("serial/codecs.jl")
include("serial/reconstruct.jl")
include("serial/api.jl")
