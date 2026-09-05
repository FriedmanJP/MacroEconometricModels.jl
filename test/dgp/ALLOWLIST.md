# DGP white-noise lint allowlist (DGP-01 #790, consumed by `test_dgp_lint.jl`)

Format: one `test/...` path per bullet. The lint parses every bullet under
both headings; a file on either list is exempt.

- **Permanent**: render / dispatch / throws-only / oracle-regeneration files.
  An allowlisted file must not carry a statistical claim (`@test` on a
  statistical quantity). Finalised by DGP-18 (#807).
- **Grandfathered**: files that still violate today and are tracked by a
  per-module issue (DGP-02…18). Each module issue deletes its files from
  this section as it migrates them to `dgp.*` simulators. Nothing is added
  here except by landing the owning module issue.

## Permanent (render/dispatch/throws-only — statistical claims forbidden)

- test/plotting/plot_test_helpers.jl
- test/plotting/test_plot_forecast_filters.jl
- test/plotting/test_plot_irf_fevd_hd.jl
- test/plotting/test_plot_models.jl
- test/plotting/test_plot_nowcast.jl
- test/plotting/test_plot_reg_micro.jl
- test/plotting/test_plot_render.jl
- test/plotting/test_plot_wave2_laneA.jl
- test/plotting/test_plot_wave2_laneB.jl
- test/plotting/test_plot_wave2_laneC.jl
- test/plotting/test_plot_wave2_laneD.jl
- test/plotting/test_plot_wave2_laneE.jl
- test/plotting/test_plot_wave2_laneF.jl
- test/display/display_helpers.jl
- test/display/test_display_goldens.jl
- test/display/test_display_invariants.jl
- test/data/test_data.jl
- test/ext/test_constrained_ext.jl
- test/serialization_helpers.jl
- test/gen_serialization_v1_fixtures.jl
- test/dsge/gen_serialization_v1_fixtures.jl
- test/core/test_serialization.jl
- test/bvar/test_bvar_serialization.jl
- test/vecm/test_vecm_serialization.jl
- test/var/test_var_serialization.jl
- test/lp/test_lp_serialization.jl
- test/did/test_did_serialization.jl
- test/gmm/test_gmm_serialization.jl
- test/midas/test_midas_serialization.jl
- test/fceval/test_fceval_serialization.jl
- test/reg/test_reg_serialization.jl
- test/teststat/test_teststat_serialization.jl
- test/arima/test_arima_serialization.jl
- test/system/test_system_serialization.jl
- test/volatility/test_volatility_serialization.jl
- test/filters/test_filters_serialization.jl
- test/nonlinear/test_nonlinear_serialization.jl
- test/io/test_io_serialization.jl
- test/counterfactual/test_counterfactual_serialization.jl
- test/dsge/test_dsge_serialization.jl

## Grandfathered (pending per-module DGP migration — remove as DGP-02…18 land)

<!-- DGP-12 (#801) -->
- test/teststat/test_bds.jl
- test/teststat/test_bubble.jl
- test/teststat/test_cointegration_resid.jl
- test/teststat/test_dumitrescu_hurlin.jl
- test/teststat/test_granger.jl
- test/teststat/test_hegy.jl
- test/teststat/test_model_comparison.jl
- test/teststat/test_normality.jl
- test/teststat/test_structural_break.jl
- test/teststat/test_unitroot.jl
- test/teststat/test_variance_ratio.jl
<!-- DGP-13 (#802) -->
- test/dsge/test_bayesian_dsge.jl
- test/dsge/test_dsge.jl
- test/dsge/test_ha_dsge.jl
- test/dsge/test_ha_dsge_advanced.jl
<!-- DGP-14 (#803) -->
- test/reg/test_heckman.jl
- test/reg/test_multinomial.jl
- test/reg/test_ordered.jl
- test/reg/test_reg.jl
- test/reg/test_robust.jl
<!-- DGP-15 (#804) -->
- test/preg/test_panel_nonlinear.jl
- test/preg/test_panel_reg.jl
<!-- DGP-16 (#805) -->
- test/did/test_did.jl
<!-- DGP-17 (#806) -->
- test/empirical/test_irf_ci_bands.jl
<!-- DGP-18 (#807) -->
- test/core/test_display_backends.jl
- test/coverage/test_codecov_gaps.jl
- test/coverage/test_data_types_coverage.jl
- test/coverage/test_dsge_bayes_coverage.jl
- test/coverage/test_dsge_coverage.jl
- test/coverage/test_dsge_statid_coverage.jl
- test/coverage/test_gmm_ext_coverage.jl
- test/coverage/test_misc_coverage.jl
- test/coverage/test_pvar_nongaussian_coverage.jl
- test/coverage/test_teststat_break_panel_coverage.jl
- test/gen_ev_fixtures.jl
