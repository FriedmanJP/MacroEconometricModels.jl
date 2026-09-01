# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    SERIALIZATION_FORMAT_VERSION

On-disk schema version written into every model file by [`save_model`](@ref) and
checked by [`load_model`](@ref). Bumped only on a breaking change to the payload
layout; a file whose version this build does not recognize is rejected with a
[`SerializationError`](@ref) rather than mis-read.
"""
const SERIALIZATION_FORMAT_VERSION = 1

# Result / data types with round-trip support. Maps the stored type tag → the
# concrete type, so `load_model` dispatches `_from_serializable` on the recorded
# name. Only TOP-LEVEL `save_model` targets are listed here; nested structs are
# resolved by name at load time (see `_resolve_ser_type`) and so need no entry.
const _SERIALIZABLE_TYPES = Dict{String,Type}(
    # ── core VAR family (original T248 set) ──────────────────────────────────
    "VARModel"                      => VARModel,
    "BVARPosterior"                 => BVARPosterior,
    "RegModel"                      => RegModel,
    "LogitModel"                    => LogitModel,
    "ProbitModel"                   => ProbitModel,
    "LPModel"                       => LPModel,
    # ── data containers ──────────────────────────────────────────────────────
    "TimeSeriesData"                => TimeSeriesData,
    "PanelData"                     => PanelData,
    "CrossSectionData"              => CrossSectionData,
    "IOData"                        => IOData,
    "IOMetaData"                    => IOMetaData,
    "ExtractionResult"              => ExtractionResult,
    "PriceModelResult"              => PriceModelResult,
    "ImpactResult"                  => ImpactResult,
    "NetworkStatsResult"            => NetworkStatsResult,
    "VerticalSpecialization"        => VerticalSpecialization,
    "ExportDecomposition"           => ExportDecomposition,
    "RegionalFootprintResult"       => RegionalFootprintResult,
    "BaqaeeFarhiResult"             => BaqaeeFarhiResult,
    "ProductionNetwork"             => ProductionNetwork,
    "SDAResult"                     => SDAResult,
    "RASResult"                     => RASResult,
    "BFLocal"                       => BFLocal,
    "BFElasticities"                => BFElasticities,
    "BFShockCurve"                  => BFShockCurve,
    "BFEquilibrium"                 => BFEquilibrium,
    "BFWedgeDecomp"                 => BFWedgeDecomp,
    "BFMisallocation"               => BFMisallocation,
    # ── cointegration / VECM ─────────────────────────────────────────────────
    "VECMModel"                     => VECMModel,
    "CointRegModel"                 => CointRegModel,
    "PanelCointRegModel"            => PanelCointRegModel,
    # ── panel / PVAR ─────────────────────────────────────────────────────────
    "PVARModel"                     => PVARModel,
    "PanelRegModel"                 => PanelRegModel,
    "PanelIVModel"                  => PanelIVModel,
    "PanelLogitModel"               => PanelLogitModel,
    "PanelProbitModel"              => PanelProbitModel,
    # ── ordered / multinomial choice ─────────────────────────────────────────
    "OrderedLogitModel"             => OrderedLogitModel,
    "OrderedProbitModel"            => OrderedProbitModel,
    "MultinomialLogitModel"         => MultinomialLogitModel,
    # ── local-projection variants ────────────────────────────────────────────
    "LPIVModel"                     => LPIVModel,
    "SmoothLPModel"                 => SmoothLPModel,
    "StateLPModel"                  => StateLPModel,
    "PropensityLPModel"             => PropensityLPModel,
    # ── systems / GMM ────────────────────────────────────────────────────────
    "SURModel"                      => SURModel,
    "GMMModel"                      => GMMModel,
    "SMMModel"                      => SMMModel,
    # ── volatility ───────────────────────────────────────────────────────────
    "ARCHModel"                     => ARCHModel,
    "GARCHModel"                    => GARCHModel,
    "EGARCHModel"                   => EGARCHModel,
    "GJRGARCHModel"                 => GJRGARCHModel,
    "APARCHModel"                   => APARCHModel,
    "CGARCHModel"                   => CGARCHModel,
    "FIGARCHModel"                  => FIGARCHModel,
    "FIEGARCHModel"                 => FIEGARCHModel,
    "GarchMidasModel"               => GarchMidasModel,
    "SVModel"                       => SVModel,
    "MGARCHModel"                   => MGARCHModel,
    # ── factor / FAVAR ───────────────────────────────────────────────────────
    "FactorModel"                   => FactorModel,
    "DynamicFactorModel"            => DynamicFactorModel,
    "GeneralizedDynamicFactorModel" => GeneralizedDynamicFactorModel,
    "FAVARModel"                    => FAVARModel,
    "StructuralDFM"                 => StructuralDFM,
    # ── ARIMA / ARDL / nonlinear / MIDAS / state space ───────────────────────
    "ARModel"                       => ARModel,
    "MAModel"                       => MAModel,
    "ARMAModel"                     => ARMAModel,
    "ARIMAModel"                    => ARIMAModel,
    "ARFIMAModel"                   => ARFIMAModel,
    "ARDLModel"                     => ARDLModel,
    "NARDLModel"                    => NARDLModel,
    "PMGModel"                      => PMGModel,
    "ThresholdModel"                => ThresholdModel,
    "MidasModel"                    => MidasModel,
    "StateSpaceModel"               => StateSpaceModel,
    # ── SVAR identification results (SID-24 / #753) ──────────────────────────
    "ProxySVARResult"               => ProxySVARResult,
    "SVARModel"                     => SVARModel,
    "MaxShareResult"                => MaxShareResult,
    "SVECResult"                    => SVECResult,
    "ICASVARResult"                 => ICASVARResult,
    "NonGaussianMLResult"           => NonGaussianMLResult,
    "NonGaussianGMMResult"          => NonGaussianGMMResult,
    "MarkovSwitchingSVARResult"     => MarkovSwitchingSVARResult,
    "GARCHSVARResult"               => GARCHSVARResult,
    "SmoothTransitionSVARResult"    => SmoothTransitionSVARResult,
    "ExternalVolatilitySVARResult"  => ExternalVolatilitySVARResult,
    "AriasSVARResult"               => AriasSVARResult,
    "UhligSVARResult"               => UhligSVARResult,
    "BayesianSetIdentifiedSVAR"     => BayesianSetIdentifiedSVAR,
    "SignIdentifiedSet"             => SignIdentifiedSet,
    "RobustBayesResult"             => RobustBayesResult,
    # ── DSGE / HA / OLG / CT ─────────────────────────────────────────────────
    "ModelSpec"                     => ModelSpec,
    "NamedEquation"                 => NamedEquation,
    "ModelIR"                       => ModelIR,
    "IREquation"                    => IREquation,
    "IRDecl"                        => IRDecl,
    "TimingInfo"                    => TimingInfo,
    # ── DSER-04 representative-agent solutions ───────────────────────────────
    "LinearDSGE"                    => LinearDSGE,
    "DSGESolution"                  => DSGESolution,
    "PerturbationSolution"          => PerturbationSolution,
    "ProjectionSolution"            => ProjectionSolution,
    "PerfectForesightPath"          => PerfectForesightPath,
    "PrunedStateSpace"              => PrunedStateSpace,
    "DeterminacyMap"                => DeterminacyMap,
    "KalmanSmootherResult"          => KalmanSmootherResult,
    "DSGEEstimation"                => DSGEEstimation,
    "OccBinConstraint"              => OccBinConstraint,
    "OccBinRegime"                  => OccBinRegime,
    "OccBinSolution"                => OccBinSolution,
    "OccBinIRF"                     => OccBinIRF,
    # ── DSER-05 Bayesian DSGE ────────────────────────────────────────────────
    "DSGEPrior"                     => DSGEPrior,
    "DSGEStateSpace"                => DSGEStateSpace,
    "NonlinearStateSpace"           => NonlinearStateSpace,
    "ProjectionStateSpace"          => ProjectionStateSpace,
    "BayesianDSGE"                  => BayesianDSGE,
    "PosteriorMode"                 => PosteriorMode,
    "BayesianDSGESimulation"        => BayesianDSGESimulation,
    "MCMCDiagnostics"               => MCMCDiagnostics,
    "IdentificationDiagnostics"     => IdentificationDiagnostics,
    "LearningRateCheck"             => LearningRateCheck,
    "PriorPosteriorOverlap"         => PriorPosteriorOverlap,
    "PriorPredictiveResult"         => PriorPredictiveResult,
    "PosteriorPredictiveCheck"      => PosteriorPredictiveCheck,
    "PrefilterSpec"                 => PrefilterSpec,
    "ObservationTrends"             => ObservationTrends,
    # ── DSER-06 HA household problem ─────────────────────────────────────────
    "IndividualProblem"             => IndividualProblem,
    "HouseholdSystem"               => HouseholdSystem,
    # ── DSER-07 HA results ───────────────────────────────────────────────────
    "HAGrid"                        => HAGrid,
    "IncomeProcess"                 => IncomeProcess,
    "HASteadyState"                 => HASteadyState,
    "HADSGESolution"                => HADSGESolution,
    "KrusellSmithSolution"          => KrusellSmithSolution,
    "WinberryFamily"                => WinberryFamily,
    "DenHaanAccuracy"               => DenHaanAccuracy,
    "HAGridDiagnostics"             => HAGridDiagnostics,
    # ── DSER-08 SSJ blocks ───────────────────────────────────────────────────
    "SimpleBlock"                   => SimpleBlock,
    "HetBlock"                      => HetBlock,
    "MitBlock"                      => MitBlock,
    "SSJModel"                      => SSJModel,
    "SSJGEJacobian"                 => SSJGEJacobian,
    "SSJImpulseResponse"            => SSJImpulseResponse,
    # ── DSER-10 DCEGM / firms / intermediary ─────────────────────────────────
    "DCEGMProblem"                  => DCEGMProblem,
    "DCEGMSystem"                   => DCEGMSystem,
    "DCEGMSolution"                 => DCEGMSolution,
    "DCEGMDistribution"             => DCEGMDistribution,
    "DCEGMFirm"                     => DCEGMFirm,
    "DCEGMEquilibrium"              => DCEGMEquilibrium,
    "DCEGMTransition"               => DCEGMTransition,
    "FirmSystem"                    => FirmSystem,
    "KhanThomasSteadyState"         => KhanThomasSteadyState,
    "KhanThomasTransition"          => KhanThomasTransition,
    "IntermediarySystem"            => IntermediarySystem,
    "IntermediaryPE"                => IntermediaryPE,
    "IntermediarySteadyState"       => IntermediarySteadyState,
    "IntermediaryTransition"        => IntermediaryTransition,
)
