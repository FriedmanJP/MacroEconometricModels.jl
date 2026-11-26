# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

# =============================================================================
# Domain Dispatch for default_ci_level (CI-01 / #936)
# =============================================================================
# Loaded after all model and result types are defined, before `summary.jl`,
# `core/tables.jl`, and `plotting/*.jl`.

# ── Microeconometric, cross-sectional, discrete-choice, count, micro-panel,
#    RDD, GMM/SMM, systems, and non-LP DiD types → DEFAULT_CI_MICRO (0.90) ───
const _MICRO_CI_TYPES = (
    RegModel,
    RobustRegModel,
    LogitModel,
    ProbitModel,
    OrderedLogitModel,
    OrderedProbitModel,
    MultinomialLogitModel,
    PoissonModel,
    NegBinModel,
    TobitModel,
    TruncRegModel,
    HeckmanModel,
    QuantileRegModel,
    RDDResult,
    MarginalEffects,
    MultinomialMarginalEffects,
    OddsRatio,
    RILogitModel,
    WildClusterBootstrap,
    AndersonRubinCI,
    AndersonRubinTest,
    KernelRegression,
    PanelRegModel,
    PanelIVModel,
    PanelLogitModel,
    PanelProbitModel,
    QuantileFactorPanelModel,
    SURModel,
    ThreeSLSModel,
    GMMModel,
    SMMModel,
    DIDResult,
    EventStudyLP,
    HonestDiDResult,
    PretrendTestResult,
)

for MT in _MICRO_CI_TYPES
    @eval default_ci_level(::Type{<:$MT}) = DEFAULT_CI_MICRO
end

# ── Stored-level result instances honor their estimation-time CI level ────────
default_ci_level(r::RDDResult) = Float64(r.level)
default_ci_level(r::DIDResult) = Float64(r.conf_level)
default_ci_level(r::EventStudyLP) = Float64(r.conf_level)
default_ci_level(r::LPDiDResult) = Float64(r.conf_level)
default_ci_level(r::HonestDiDResult) = Float64(r.conf_level)
default_ci_level(me::MarginalEffects) = Float64(me.conf_level)
default_ci_level(r::OddsRatio) = Float64(r.conf_level)
default_ci_level(m::QuantileRegModel) = Float64(1 - m.alpha)
default_ci_level(m::ThresholdModel) = Float64(m.gamma_ci_level)
default_ci_level(r::BaiPerronResult) = Float64(r.conf_level)
default_ci_level(r::CorTestResult) = Float64(r.conf_level)
