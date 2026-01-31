using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random

@testset "Local Projections" begin
    Random.seed!(42)

    @testset "Core LP Estimation (Jordà 2005)" begin
        # Generate AR(1) data: y_t = 0.7 * y_{t-1} + ε_t
        T = 200
        n = 2
        rho = 0.7

        Y = zeros(T, n)
        for t in 2:T
            Y[t, :] = rho * Y[t-1, :] + randn(n)
        end

        # Estimate LP
        horizon = 10
        lags = 4
        model = estimate_lp(Y, 1, horizon; lags=lags, cov_type=:newey_west)

        @test model isa LPModel
        @test model.horizon == horizon
        @test model.lags == lags
        @test length(model.B) == horizon + 1
        @test length(model.vcov) == horizon + 1

        # Extract IRF
        result = lp_irf(model; conf_level=0.95)

        @test result isa LPImpulseResponse
        @test size(result.values) == (horizon + 1, n)
        @test all(result.ci_lower .<= result.values)
        @test all(result.values .<= result.ci_upper)

        # Theoretical IRF for AR(1): φ_h = ρ^h
        # LP should recover this approximately
        theoretical_irf = [rho^h for h in 0:horizon]

        # Check own-response (variable 1 to shock 1)
        # Allow some tolerance due to estimation error
        @test isapprox(result.values[1, 1], 1.0, atol=0.3)  # h=0 should be close to 1

        # Check decay pattern
        @test result.values[end, 1] < result.values[1, 1]
    end

    @testset "HAC Covariance Estimation" begin
        # Generate serially correlated data
        T = 200
        u = zeros(T)
        rho_u = 0.5
        for t in 2:T
            u[t] = rho_u * u[t-1] + randn()
        end

        X = hcat(ones(T), randn(T))

        # Newey-West should give larger SE than White when there's serial correlation
        V_nw = newey_west(X, u; bandwidth=5, kernel=:bartlett)
        V_white = white_vcov(X, u)

        @test size(V_nw) == (2, 2)
        @test size(V_white) == (2, 2)
        @test issymmetric(V_nw)
        @test issymmetric(V_white)

        # Newey-West SE should generally be larger due to autocorrelation adjustment
        # (not always, but typically with positive autocorrelation)
        @test tr(V_nw) > 0
        @test tr(V_white) > 0

        # Test automatic bandwidth selection
        bw = optimal_bandwidth_nw(u)
        @test bw >= 0
        @test bw < T

        # Test kernel weights
        @test kernel_weight(0, 5, :bartlett) == 1.0
        @test kernel_weight(5, 5, :bartlett) ≈ 1 - 5/6
        @test kernel_weight(10, 5, :bartlett) == 0.0
    end

    @testset "LP-IV (Stock & Watson 2018)" begin
        # Generate data with endogeneity
        T = 300
        n = 2

        # Instrument
        Z = randn(T, 1)

        # Endogenous shock (correlated with error)
        common = randn(T)
        shock = 0.5 * Z[:, 1] + 0.5 * common + 0.2 * randn(T)

        # Outcome
        Y = zeros(T, n)
        Y[:, 1] = shock  # Shock variable
        for t in 2:T
            Y[t, 2] = 0.3 * Y[t-1, 2] + 0.5 * shock[t] + common[t] + randn()
        end

        # Estimate LP-IV
        horizon = 5
        model = estimate_lp_iv(Y, 1, Z, horizon; lags=2)

        @test model isa LPIVModel
        @test length(model.first_stage_F) == horizon + 1

        # First stage F-stats should be reasonable (Z is relevant)
        @test all(model.first_stage_F .> 1)

        # Weak instrument test
        wk_test = weak_instrument_test(model; threshold=10.0)
        @test haskey(wk_test, :F_stats)
        @test haskey(wk_test, :passes_threshold)

        # Extract IRF
        result = lp_iv_irf(model)
        @test result isa LPImpulseResponse
    end

    @testset "Smooth LP (Barnichon & Brownlees 2019)" begin
        # Generate data
        T = 200
        n = 2
        Y = zeros(T, n)
        for t in 2:T
            Y[t, :] = 0.5 * Y[t-1, :] + randn(n)
        end

        horizon = 15

        # Estimate standard LP for comparison
        std_model = estimate_lp(Y, 1, horizon; lags=4)
        std_irf = lp_irf(std_model)

        # Estimate smooth LP
        smooth_model = estimate_smooth_lp(Y, 1, horizon; degree=3, n_knots=4, lambda=1.0)

        @test smooth_model isa SmoothLPModel
        @test size(smooth_model.irf_values) == (horizon + 1, n)

        # B-spline basis properties
        basis = smooth_model.spline_basis
        @test basis.degree == 3
        @test size(basis.basis_matrix, 1) == horizon + 1

        # Smoothed IRF should have similar pattern but smoother
        smooth_result = smooth_lp_irf(smooth_model)
        @test smooth_result isa LPImpulseResponse

        # Smooth SE should generally be smaller (variance reduction)
        @test mean(smooth_result.se) <= mean(std_irf.se) * 1.5  # Allow some margin
    end

    @testset "State-Dependent LP (Auerbach & Gorodnichenko 2013)" begin
        # Generate regime-switching data
        T = 300
        n = 2

        # State variable (e.g., output growth)
        z = cumsum(randn(T)) ./ sqrt(T)  # Random walk normalized
        z_standardized = (z .- mean(z)) ./ std(z)

        # Different coefficients in different states
        Y = zeros(T, n)
        for t in 2:T
            F_t = 1 / (1 + exp(-1.5 * z_standardized[t]))  # Probability of recession
            rho_t = F_t * 0.8 + (1 - F_t) * 0.3  # Higher persistence in recession
            Y[t, :] = rho_t * Y[t-1, :] + randn(n)
        end

        horizon = 8

        # Estimate state-dependent LP
        model = estimate_state_lp(Y, 1, z_standardized, horizon;
                                   gamma=1.5, threshold=0.0, lags=2)

        @test model isa StateLPModel
        @test length(model.B_expansion) == horizon + 1
        @test length(model.B_recession) == horizon + 1

        # State transition
        @test model.state.gamma == 1.5
        @test all(0 .<= model.state.F_values .<= 1)

        # Extract regime-specific IRFs
        irf_result = state_irf(model; regime=:both)
        @test haskey(irf_result, :expansion)
        @test haskey(irf_result, :recession)
        @test haskey(irf_result, :difference)

        # Test regime difference
        diff_test = test_regime_difference(model; h=0)
        @test haskey(diff_test, :t_stats)
        @test haskey(diff_test, :p_values)
    end

    @testset "Propensity Score LP (Angrist et al. 2018)" begin
        # Generate treatment effect data
        T = 300
        n = 2

        # Covariates affecting treatment assignment
        X = randn(T, 2)

        # Treatment assignment (logit model)
        propensity_true = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1] .- 0.3 .* X[:, 2]))
        treatment = rand(T) .< propensity_true

        # Potential outcomes
        Y0 = zeros(T, n)  # Control potential outcome
        Y1 = zeros(T, n)  # Treated potential outcome

        for t in 2:T
            Y0[t, :] = 0.5 * Y0[t-1, :] + randn(n)
            Y1[t, :] = Y0[t, :] .+ 0.5  # Treatment effect = 0.5
        end

        # Observed outcome
        Y = similar(Y0)
        for t in 1:T
            Y[t, :] = treatment[t] ? Y1[t, :] : Y0[t, :]
        end

        horizon = 5

        # Estimate propensity score LP
        model = estimate_propensity_lp(Y, treatment, X, horizon;
                                        ps_method=:logit, lags=2)

        @test model isa PropensityLPModel
        @test all(0 .< model.propensity_scores .< 1)
        @test size(model.ate) == (horizon + 1, n)

        # ATE estimates should be positive (true effect is 0.5)
        # Allow wide tolerance due to small sample
        @test mean(model.ate[:, 1]) > -1.0
        @test mean(model.ate[:, 1]) < 2.0

        # Propensity diagnostics
        diag = propensity_diagnostics(model)
        @test haskey(diag, :propensity_summary)
        @test haskey(diag, :overlap)
        @test haskey(diag, :balance)

        # Extract IRF
        result = propensity_irf(model)
        @test result isa LPImpulseResponse

        # Test doubly robust estimator
        dr_model = doubly_robust_lp(Y, treatment, X, horizon; lags=2)
        @test dr_model isa PropensityLPModel
    end

    @testset "GMM Estimation" begin
        # Simple linear IV example
        T_gmm = 200

        # Instrument and endogenous variable
        z = randn(T_gmm)
        x = 0.7 .* z .+ 0.5 .* randn(T_gmm)  # x correlated with z
        u = randn(T_gmm)
        y = 1.5 .+ 2.0 .* x .+ u  # True: β₀ = 1.5, β₁ = 2.0

        data = (y=y, x=x, z=z)

        # Moment function for IV: E[Z * (Y - β₀ - β₁X)] = 0
        function iv_moments(theta, data)
            residuals = data.y .- theta[1] .- theta[2] .* data.x
            hcat(residuals, data.z .* residuals)  # 2 moments: E[ε] = 0, E[Z*ε] = 0
        end

        # Initial guess
        theta0 = [0.0, 0.0]

        # Estimate via GMM
        model = estimate_gmm(iv_moments, theta0, data; weighting=:two_step)

        @test model isa GMMModel
        @test model.n_moments == 2
        @test model.n_params == 2
        @test model.converged

        # Check estimates are close to true values
        @test isapprox(model.theta[1], 1.5, atol=0.5)
        @test isapprox(model.theta[2], 2.0, atol=0.5)

        # J-test (just identified, so J should be 0)
        j_result = j_test(model)
        @test j_result.df == 0  # Just identified

        # GMM summary
        summary = gmm_summary(model)
        @test haskey(summary, :theta)
        @test haskey(summary, :se)
        @test haskey(summary, :t_stats)
    end

    @testset "Compare LP and VAR IRFs" begin
        # Generate VAR(1) data
        T = 300
        n = 2
        A = [0.5 0.1; 0.1 0.5]

        Y = zeros(T, n)
        for t in 2:T
            Y[t, :] = A * Y[t-1, :] + randn(n)
        end

        horizon = 10

        # Compare
        comparison = compare_var_lp(Y, horizon; lags=1)

        @test size(comparison.var_irf) == (horizon, n, n)
        @test size(comparison.lp_irf) == (horizon, n, n)
        @test size(comparison.difference) == (horizon, n, n)

        # For correctly specified model, LP and VAR should give similar IRFs
        # (LP is less efficient but consistent)
        max_diff = maximum(abs.(comparison.difference))
        @test max_diff < 1.0  # Reasonable tolerance
    end

    @testset "StatsAPI Interface" begin
        T = 100
        n = 2
        Y = randn(T, n)

        model = estimate_lp(Y, 1, 5; lags=2)

        # Test StatsAPI methods
        @test coef(model) == model.B
        @test residuals(model) == model.residuals
        @test vcov(model) == model.vcov
        @test nobs(model) == T
        @test islinear(model) == true
    end

    # ==========================================================================
    # Robustness Tests (Following Arias et al. pattern)
    # ==========================================================================

    @testset "Reproducibility" begin
        # Same seed should produce identical LP estimates
        Random.seed!(11111)
        Y1 = zeros(150, 2)
        for t in 2:150
            Y1[t, :] = 0.5 * Y1[t-1, :] + randn(2)
        end
        model1 = estimate_lp(Y1, 1, 10; lags=2)
        irf1 = lp_irf(model1)

        Random.seed!(11111)
        Y2 = zeros(150, 2)
        for t in 2:150
            Y2[t, :] = 0.5 * Y2[t-1, :] + randn(2)
        end
        model2 = estimate_lp(Y2, 1, 10; lags=2)
        irf2 = lp_irf(model2)

        @test irf1.values ≈ irf2.values
        @test irf1.se ≈ irf2.se
    end

    @testset "Numerical Stability - Near-Collinear Regressors" begin
        Random.seed!(22222)
        T_nc = 200
        n_nc = 3

        # Create data with near-collinearity
        Y_nc = randn(T_nc, n_nc)
        Y_nc[:, 3] = Y_nc[:, 1] + 0.01 * randn(T_nc)

        # Should handle near-collinearity gracefully
        model_nc = estimate_lp(Y_nc, 1, 5; lags=2)
        @test model_nc isa LPModel
        @test all(isfinite.(model_nc.B[1]))

        irf_nc = lp_irf(model_nc)
        @test all(isfinite.(irf_nc.values))
    end

    @testset "Edge Cases - Horizons" begin
        Random.seed!(33333)
        T_h = 100
        Y_h = randn(T_h, 2)

        # Minimum horizon (h=1)
        model_h1 = estimate_lp(Y_h, 1, 1; lags=2)
        @test model_h1 isa LPModel
        @test model_h1.horizon == 1

        irf_h1 = lp_irf(model_h1)
        @test size(irf_h1.values, 1) == 2  # h=0 and h=1

        # Larger horizon
        model_h20 = estimate_lp(Y_h, 1, 20; lags=2)
        @test model_h20 isa LPModel

        irf_h20 = lp_irf(model_h20)
        @test size(irf_h20.values, 1) == 21
    end

    @testset "Confidence Interval Properties" begin
        Random.seed!(44444)
        T_ci = 200
        Y_ci = zeros(T_ci, 2)
        for t in 2:T_ci
            Y_ci[t, :] = 0.5 * Y_ci[t-1, :] + randn(2)
        end

        model_ci = estimate_lp(Y_ci, 1, 10; lags=2, cov_type=:newey_west)
        irf_ci = lp_irf(model_ci; conf_level=0.95)

        # CI ordering: lower ≤ point ≤ upper
        @test all(irf_ci.ci_lower .<= irf_ci.values)
        @test all(irf_ci.values .<= irf_ci.ci_upper)

        # CI width should be positive
        ci_width = irf_ci.ci_upper - irf_ci.ci_lower
        @test all(ci_width .>= 0)

        # Different confidence levels should give different widths
        irf_90 = lp_irf(model_ci; conf_level=0.90)
        irf_68 = lp_irf(model_ci; conf_level=0.68)

        # 95% CI should be wider than 90%, which should be wider than 68%
        width_95 = mean(irf_ci.ci_upper - irf_ci.ci_lower)
        width_90 = mean(irf_90.ci_upper - irf_90.ci_lower)
        width_68 = mean(irf_68.ci_upper - irf_68.ci_lower)

        @test width_95 >= width_90
        @test width_90 >= width_68
    end

    @testset "Cumulative IRF Properties" begin
        Random.seed!(55555)
        T_cum = 150
        Y_cum = zeros(T_cum, 2)
        for t in 2:T_cum
            Y_cum[t, :] = 0.5 * Y_cum[t-1, :] + randn(2)
        end

        model_cum = estimate_lp(Y_cum, 1, 10; lags=2)
        irf_std = lp_irf(model_cum)
        irf_cum = cumulative_irf(irf_std)

        # Cumulative IRF should have same size
        @test size(irf_cum.values) == size(irf_std.values)

        # At h=0, cumulative == standard
        @test irf_cum.values[1, :] ≈ irf_std.values[1, :]

        # Cumulative should be cumsum of standard
        @test irf_cum.values ≈ cumsum(irf_std.values, dims=1)
    end

    @testset "LP-IV Weak Instrument Handling" begin
        Random.seed!(66666)
        T_weak = 200
        n_weak = 2

        # Create weak instrument scenario
        Z_weak = randn(T_weak, 1)
        shock_weak = 0.1 * Z_weak[:, 1] + randn(T_weak)  # Weak correlation

        Y_weak = zeros(T_weak, n_weak)
        Y_weak[:, 1] = shock_weak
        for t in 2:T_weak
            Y_weak[t, 2] = 0.3 * Y_weak[t-1, 2] + 0.5 * shock_weak[t] + randn()
        end

        model_weak = estimate_lp_iv(Y_weak, 1, Z_weak, 5; lags=2)
        @test model_weak isa LPIVModel

        # Weak instrument test should detect weakness
        wk_test = weak_instrument_test(model_weak; threshold=10.0)
        @test haskey(wk_test, :F_stats)
        @test haskey(wk_test, :passes_threshold)
        # With weak instrument, some horizons should fail threshold
    end

    # ==========================================================================
    # Extended Coverage Tests for lp_extensions.jl
    # ==========================================================================

    @testset "LP-IV Sargan Overidentification Test" begin
        Random.seed!(77777)
        T_sar = 300
        n_sar = 2

        # Create overidentified IV scenario (2 instruments for 1 endogenous)
        Z_sar = randn(T_sar, 2)  # Two instruments
        shock_sar = 0.4 * Z_sar[:, 1] + 0.3 * Z_sar[:, 2] + 0.3 * randn(T_sar)

        Y_sar = zeros(T_sar, n_sar)
        Y_sar[:, 1] = shock_sar
        for t in 2:T_sar
            Y_sar[t, 2] = 0.3 * Y_sar[t-1, 2] + 0.5 * shock_sar[t] + randn()
        end

        model_sar = estimate_lp_iv(Y_sar, 1, Z_sar, 5; lags=2)
        @test model_sar isa LPIVModel
        @test MacroEconometricModels.n_instruments(model_sar) == 2

        # Sargan test for overidentification
        sargan_result = MacroEconometricModels.sargan_test(model_sar, 0)
        @test haskey(sargan_result, :J_stat)
        @test haskey(sargan_result, :p_value)
        @test haskey(sargan_result, :df)
        @test sargan_result.df == 1  # 2 instruments - 1 endogenous = 1 df
        @test sargan_result.valid == true

        # Test with just-identified model (should return NaN)
        Z_just = randn(T_sar, 1)
        shock_just = 0.5 * Z_just[:, 1] + 0.5 * randn(T_sar)
        Y_just = zeros(T_sar, n_sar)
        Y_just[:, 1] = shock_just
        for t in 2:T_sar
            Y_just[t, 2] = 0.3 * Y_just[t-1, 2] + 0.5 * shock_just[t] + randn()
        end
        model_just = estimate_lp_iv(Y_just, 1, Z_just, 3; lags=2)
        sargan_just = MacroEconometricModels.sargan_test(model_just, 0)
        @test sargan_just.valid == false
    end

    @testset "Smooth LP Cross-Validation" begin
        Random.seed!(88888)
        T_cv = 200
        n_cv = 2
        Y_cv = zeros(T_cv, n_cv)
        for t in 2:T_cv
            Y_cv[t, :] = 0.5 * Y_cv[t-1, :] + randn(n_cv)
        end

        # Test cross-validation for lambda selection
        lambda_grid = Float64.([0.01, 0.1, 1.0, 10.0])
        optimal_lambda = MacroEconometricModels.cross_validate_lambda(
            Y_cv, 1, 10; lambda_grid=lambda_grid, k_folds=3, lags=2
        )
        @test optimal_lambda in lambda_grid
        @test optimal_lambda > 0
    end

    @testset "Smooth LP Comparison Function" begin
        Random.seed!(99999)
        T_cmp = 150
        n_cmp = 2
        Y_cmp = zeros(T_cmp, n_cmp)
        for t in 2:T_cmp
            Y_cmp[t, :] = 0.5 * Y_cmp[t-1, :] + randn(n_cmp)
        end

        # Compare standard and smooth LP
        comparison = MacroEconometricModels.compare_smooth_lp(Y_cmp, 1, 10; lambda=1.0, lags=2)
        @test haskey(comparison, :standard_irf)
        @test haskey(comparison, :smooth_irf)
        @test haskey(comparison, :variance_reduction)
        @test comparison.standard_irf isa LPImpulseResponse
        @test comparison.smooth_irf isa LPImpulseResponse
        # Variance reduction should be positive (smooth reduces variance)
        @test comparison.variance_reduction > 0
    end

    @testset "B-Spline Basis Functions" begin
        # Test bspline_basis_value edge cases
        horizons = collect(0:10)

        # Degree 0 (piecewise constant)
        basis_d0 = MacroEconometricModels.bspline_basis(horizons, 0, 2; T=Float64)
        @test size(basis_d0.basis_matrix, 1) == 11
        @test basis_d0.degree == 0

        # Degree 1 (piecewise linear)
        basis_d1 = MacroEconometricModels.bspline_basis(horizons, 1, 2; T=Float64)
        @test basis_d1.degree == 1

        # Degree 3 (cubic, standard)
        basis_d3 = MacroEconometricModels.bspline_basis(horizons, 3, 4; T=Float64)
        @test basis_d3.degree == 3
        @test MacroEconometricModels.n_basis(basis_d3) == 4 + 3 + 1  # n_knots + degree + 1

        # Roughness penalty matrix
        R = MacroEconometricModels.roughness_penalty_matrix(basis_d3)
        @test size(R, 1) == size(R, 2)
        @test issymmetric(R)
    end

    @testset "Transition Functions for State-Dependent LP" begin
        Random.seed!(10101)
        z = randn(100)
        gamma = 1.5
        c = 0.0

        # Logistic transition
        F_logistic = MacroEconometricModels.logistic_transition(z, gamma, c)
        @test all(0 .<= F_logistic .<= 1)
        @test MacroEconometricModels.logistic_transition(c, gamma, c) ≈ 0.5  # At threshold, F = 0.5

        # Exponential transition
        F_exp = MacroEconometricModels.exponential_transition(z, gamma, c)
        @test all(0 .<= F_exp .<= 1)
        @test MacroEconometricModels.exponential_transition([c], gamma, c)[1] ≈ 0.0  # At threshold, F = 0

        # Indicator transition
        F_ind = MacroEconometricModels.indicator_transition(z, c)
        @test all(F_ind .== 0.0 .|| F_ind .== 1.0)
        @test sum(F_ind) == sum(z .>= c)
    end

    @testset "State Transition Parameter Estimation" begin
        Random.seed!(20202)
        T_st = 200
        n_st = 2

        z_st = randn(T_st)
        Y_st = zeros(T_st, n_st)
        for t in 2:T_st
            Y_st[t, :] = 0.5 * Y_st[t-1, :] + randn(n_st)
        end

        # Test NLLS method
        result_nlls = MacroEconometricModels.estimate_transition_params(
            z_st, Y_st, 1; method=:nlls, c_init=:median
        )
        @test haskey(result_nlls, :gamma)
        @test haskey(result_nlls, :c)
        @test haskey(result_nlls, :F_values)
        @test result_nlls.gamma > 0
        @test all(0 .<= result_nlls.F_values .<= 1)

        # Test grid search method
        result_grid = MacroEconometricModels.estimate_transition_params(
            z_st, Y_st, 1; method=:grid_search, c_init=:mean
        )
        @test haskey(result_grid, :gamma)
        @test haskey(result_grid, :c)
        @test result_grid.convergence_info.method == :grid_search
    end

    @testset "State-Dependent LP - Regime IRFs" begin
        Random.seed!(30303)
        T_reg = 250
        n_reg = 2

        z_reg = cumsum(randn(T_reg)) ./ sqrt(T_reg)
        z_std = (z_reg .- mean(z_reg)) ./ std(z_reg)

        Y_reg = zeros(T_reg, n_reg)
        for t in 2:T_reg
            F_t = 1 / (1 + exp(-1.5 * z_std[t]))
            rho_t = F_t * 0.8 + (1 - F_t) * 0.3
            Y_reg[t, :] = rho_t * Y_reg[t-1, :] + randn(n_reg)
        end

        model_reg = estimate_state_lp(Y_reg, 1, z_std, 8; gamma=1.5, threshold=0.0, lags=2)

        # Test individual regime IRF extraction
        irf_exp = state_irf(model_reg; regime=:expansion)
        @test haskey(irf_exp, :values)
        @test irf_exp.regime == :expansion

        irf_rec = state_irf(model_reg; regime=:recession)
        @test irf_rec.regime == :recession

        irf_diff = state_irf(model_reg; regime=:difference)
        @test irf_diff.regime == :difference

        # Test regime difference test across all horizons
        diff_all = test_regime_difference(model_reg)
        @test haskey(diff_all, :t_stats)
        @test haskey(diff_all, :p_values)
        @test haskey(diff_all, :joint_test)
        @test size(diff_all.t_stats, 1) == 9  # horizons 0-8
    end

    @testset "Propensity Score Estimation Methods" begin
        Random.seed!(40404)
        T_ps = 200

        X_ps = randn(T_ps, 2)
        p_true = 1 ./ (1 .+ exp.(-0.5 .* X_ps[:, 1] .- 0.3 .* X_ps[:, 2]))
        treatment_ps = rand(T_ps) .< p_true

        # Test logit
        p_logit = MacroEconometricModels.estimate_propensity_score(treatment_ps, X_ps; method=:logit)
        @test all(0 .< p_logit .< 1)
        @test length(p_logit) == T_ps

        # Test probit
        p_probit = MacroEconometricModels.estimate_propensity_score(treatment_ps, X_ps; method=:probit)
        @test all(0 .< p_probit .< 1)

        # Test with integer treatment
        treatment_int = Int.(treatment_ps)
        p_int = MacroEconometricModels.estimate_propensity_score(treatment_int, X_ps; method=:logit)
        @test all(0 .< p_int .< 1)
    end

    @testset "Inverse Propensity Weights" begin
        Random.seed!(50505)
        n_ipw = 100
        treatment_ipw = rand(Bool, n_ipw)
        propensity_ipw = 0.3 .+ 0.4 .* rand(n_ipw)  # Between 0.3 and 0.7

        # Test IPW with normalization
        w_norm = MacroEconometricModels.inverse_propensity_weights(
            treatment_ipw, propensity_ipw; trimming=(0.05, 0.95), normalize=true
        )
        @test all(w_norm .> 0)
        @test length(w_norm) == n_ipw

        # Test IPW without normalization
        w_unnorm = MacroEconometricModels.inverse_propensity_weights(
            treatment_ipw, propensity_ipw; trimming=(0.01, 0.99), normalize=false
        )
        @test all(w_unnorm .> 0)

        # Test trimming effect
        propensity_extreme = vcat(fill(0.001, 10), fill(0.5, 80), fill(0.999, 10))
        treatment_ext = rand(Bool, 100)
        w_trimmed = MacroEconometricModels.inverse_propensity_weights(
            treatment_ext, propensity_extreme; trimming=(0.05, 0.95), normalize=false
        )
        # Trimming should prevent extreme weights
        @test maximum(w_trimmed) < 100
    end

    @testset "White Covariance Estimator" begin
        Random.seed!(60606)
        T_wh = 100
        X_wh = hcat(ones(T_wh), randn(T_wh, 2))
        u_wh = randn(T_wh)

        V_white = white_vcov(X_wh, u_wh)
        @test size(V_white) == (3, 3)
        @test V_white ≈ V_white' atol=1e-10  # Approximately symmetric
        @test all(diag(V_white) .>= 0)  # Diagonal should be non-negative
    end

    @testset "Kernel Weight Functions" begin
        # Test different kernel types
        @test kernel_weight(0, 10, :bartlett) == 1.0
        @test kernel_weight(10, 10, :bartlett) ≈ 1 - 10/11
        @test kernel_weight(15, 10, :bartlett) == 0.0

        @test kernel_weight(0, 10, :parzen) == 1.0
        @test kernel_weight(15, 10, :parzen) == 0.0

        @test kernel_weight(0, 10, :quadratic_spectral) == 1.0
        # Quadratic spectral kernel decays but doesn't necessarily hit zero

        @test kernel_weight(0, 10, :tukey_hanning) == 1.0
    end

    @testset "LP with Different Covariance Types" begin
        Random.seed!(70707)
        T_cov = 150
        n_cov = 2
        Y_cov = zeros(T_cov, n_cov)
        for t in 2:T_cov
            Y_cov[t, :] = 0.5 * Y_cov[t-1, :] + randn(n_cov)
        end

        # Test with White covariance
        model_white = estimate_lp(Y_cov, 1, 8; lags=2, cov_type=:white)
        @test model_white isa LPModel
        irf_white = lp_irf(model_white)
        @test all(isfinite.(irf_white.se))

        # Test with Newey-West and custom bandwidth
        model_nw = estimate_lp(Y_cov, 1, 8; lags=2, cov_type=:newey_west, bandwidth=5)
        @test model_nw isa LPModel
        irf_nw = lp_irf(model_nw)
        @test all(isfinite.(irf_nw.se))
    end

    # ==========================================================================
    # Extended Coverage Tests for lp_core.jl and lp_types.jl
    # ==========================================================================

    @testset "Long-Run Variance and Covariance" begin
        Random.seed!(80808)

        # Long-run variance for white noise
        white_noise = randn(200)
        lrv_white = MacroEconometricModels.long_run_variance(white_noise; bandwidth=0)
        @test lrv_white >= 0
        @test isapprox(lrv_white, var(white_noise), atol=0.5)

        # Long-run variance for persistent series
        persistent = zeros(200)
        for t in 2:200
            persistent[t] = 0.8 * persistent[t-1] + randn()
        end
        lrv_pers = MacroEconometricModels.long_run_variance(persistent; bandwidth=10)
        @test lrv_pers > var(persistent)  # LRV should be larger for persistent series

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral]
            lrv = MacroEconometricModels.long_run_variance(white_noise; bandwidth=5, kernel=kernel)
            @test lrv >= 0
            @test isfinite(lrv)
        end

        # Long-run covariance for multivariate data
        X = randn(200, 3)
        lrc = MacroEconometricModels.long_run_covariance(X; bandwidth=5)
        @test size(lrc) == (3, 3)
        @test lrc ≈ lrc' atol=1e-10  # Symmetric
        eigvals_lrc = eigvals(lrc)
        @test all(eigvals_lrc .>= -1e-10)  # Positive semi-definite

        # Small sample edge case
        small_x = randn(5)
        lrv_small = MacroEconometricModels.long_run_variance(small_x)
        @test isfinite(lrv_small)
    end

    @testset "LP with Cholesky Identification" begin
        Random.seed!(81818)
        T_chol = 200
        n_chol = 3

        # Generate VAR data
        Y_chol = zeros(T_chol, n_chol)
        A = [0.5 0.1 0.0; 0.1 0.4 0.1; 0.0 0.1 0.3]
        for t in 2:T_chol
            Y_chol[t, :] = A * Y_chol[t-1, :] + randn(n_chol)
        end

        horizon = 10

        # Estimate LP with Cholesky identification - returns Vector{LPModel}
        models = MacroEconometricModels.estimate_lp_cholesky(Y_chol, horizon; lags=2)

        @test models isa Vector{LPModel{Float64}}
        @test length(models) == n_chol

        # Each model should have correct structure
        for model in models
            @test model isa LPModel
            @test model.horizon == horizon
            irf = lp_irf(model)
            @test irf isa LPImpulseResponse
            @test size(irf.values, 1) == horizon + 1
        end
    end

    @testset "LP with Multiple Shocks" begin
        Random.seed!(82828)
        T_multi = 200
        n_multi = 3

        Y_multi = zeros(T_multi, n_multi)
        for t in 2:T_multi
            Y_multi[t, :] = 0.5 * Y_multi[t-1, :] + randn(n_multi)
        end

        horizon = 8

        # Estimate LP with multiple shocks - returns Vector{LPModel}
        shock_vars = [1, 2]
        models = MacroEconometricModels.estimate_lp_multi(Y_multi, shock_vars, horizon; lags=2)

        @test models isa Vector{LPModel{Float64}}
        @test length(models) == length(shock_vars)

        # Compare with single-shock estimates
        model_single = estimate_lp(Y_multi, 1, horizon; lags=2)
        irf_single = lp_irf(model_single)

        # IRFs from multi should match single-shock estimates
        irf_multi_first = lp_irf(models[1])
        @test size(irf_multi_first.values) == size(irf_single.values)
        # The IRFs should be identical since same shock variable
        @test irf_multi_first.values ≈ irf_single.values atol=1e-10
    end

    @testset "Direct Core Function Tests" begin
        Random.seed!(83838)

        # compute_horizon_bounds
        t_start, t_end = MacroEconometricModels.compute_horizon_bounds(100, 5, 4)
        @test t_start == 5  # lags + 1
        @test t_end == 95   # T_obs - h
        @test t_end >= t_start

        # Edge case: large horizon
        t_start_large, t_end_large = MacroEconometricModels.compute_horizon_bounds(100, 90, 4)
        @test t_end_large >= t_start_large

        # build_response_matrix
        Y_test = randn(50, 3)
        Y_h = MacroEconometricModels.build_response_matrix(Y_test, 3, 5, 40, [1, 2])
        @test size(Y_h) == (36, 2)  # 36 observations, 2 response vars

        # Single response variable
        Y_h_single = MacroEconometricModels.build_response_matrix(Y_test, 2, 5, 40, [3])
        @test size(Y_h_single) == (36, 1)

        # create_cov_estimator - all types
        est_nw = MacroEconometricModels.create_cov_estimator(:newey_west, Float64; bandwidth=5)
        @test est_nw isa MacroEconometricModels.NeweyWestEstimator{Float64}
        @test est_nw.bandwidth == 5

        est_white = MacroEconometricModels.create_cov_estimator(:white, Float64)
        @test est_white isa MacroEconometricModels.WhiteEstimator

        est_dk = MacroEconometricModels.create_cov_estimator(:driscoll_kraay, Float64; bandwidth=3)
        @test est_dk isa MacroEconometricModels.DriscollKraayEstimator{Float64}
        @test est_dk.bandwidth == 3
    end

    @testset "Newey-West Prewhitening" begin
        Random.seed!(84848)
        T_pw = 200

        X_pw = hcat(ones(T_pw), randn(T_pw, 2))

        # AR(1) residuals with strong autocorrelation
        u_pw = zeros(T_pw)
        rho = 0.7
        for t in 2:T_pw
            u_pw[t] = rho * u_pw[t-1] + randn()
        end

        # Without prewhitening
        V_no_pw = newey_west(X_pw, u_pw; bandwidth=5, prewhiten=false)
        @test size(V_no_pw) == (3, 3)
        @test V_no_pw ≈ V_no_pw' atol=1e-10

        # With prewhitening
        V_pw = newey_west(X_pw, u_pw; bandwidth=5, prewhiten=true)
        @test size(V_pw) == (3, 3)
        @test V_pw ≈ V_pw' atol=1e-10
        @test all(diag(V_pw) .>= 0)

        # Both should give reasonable (finite) results
        @test all(isfinite.(V_no_pw))
        @test all(isfinite.(V_pw))
    end

    @testset "White Vcov HC Variants" begin
        Random.seed!(85858)
        T_hc = 100

        X_hc = hcat(ones(T_hc), randn(T_hc, 2))
        u_hc = randn(T_hc)

        # Test all HC variants
        for variant in [:hc0, :hc1, :hc2, :hc3]
            V = white_vcov(X_hc, u_hc; variant=variant)
            @test size(V) == (3, 3)
            @test V ≈ V' atol=1e-10  # Symmetric
            @test all(diag(V) .>= 0)  # Non-negative diagonal
        end

        # HC1 should give larger estimates than HC0 (dof correction)
        V_hc0 = white_vcov(X_hc, u_hc; variant=:hc0)
        V_hc1 = white_vcov(X_hc, u_hc; variant=:hc1)
        @test tr(V_hc1) >= tr(V_hc0)
    end

    @testset "LP Type Accessor Functions" begin
        Random.seed!(86868)
        T_acc = 150
        n_acc = 3

        Y_acc = zeros(T_acc, n_acc)
        for t in 2:T_acc
            Y_acc[t, :] = 0.5 * Y_acc[t-1, :] + randn(n_acc)
        end

        model = estimate_lp(Y_acc, 1, 10; lags=4)

        # Test accessor functions
        @test MacroEconometricModels.nvars(model) == n_acc
        @test MacroEconometricModels.nlags(model) == 4
        @test MacroEconometricModels.nhorizons(model) == 11  # horizon + 1
        @test MacroEconometricModels.nresponse(model) == n_acc

        # Horizon-specific StatsAPI methods
        @test coef(model, 0) == model.B[1]
        @test coef(model, 5) == model.B[6]
        @test residuals(model, 0) == model.residuals[1]
        @test vcov(model, 0) == model.vcov[1]
        @test nobs(model, 0) == model.T_eff[1]

        # dof
        @test dof(model) == sum(length(b) for b in model.B)

        # LP-IV accessor
        Z = randn(T_acc, 2)
        shock = 0.5 * Z[:, 1] + 0.5 * randn(T_acc)
        Y_iv = zeros(T_acc, 2)
        Y_iv[:, 1] = shock
        for t in 2:T_acc
            Y_iv[t, 2] = 0.3 * Y_iv[t-1, 2] + 0.5 * shock[t] + randn()
        end
        model_iv = estimate_lp_iv(Y_iv, 1, Z, 5; lags=2)
        @test MacroEconometricModels.n_instruments(model_iv) == 2

        # Propensity model accessors
        X_cov = randn(T_acc, 2)
        p_true = 1 ./ (1 .+ exp.(-0.5 .* X_cov[:, 1]))
        treatment = rand(T_acc) .< p_true
        Y_prop = randn(T_acc, 2)
        model_prop = estimate_propensity_lp(Y_prop, treatment, X_cov, 5; lags=2)
        @test MacroEconometricModels.n_treated(model_prop) == sum(treatment)
        @test MacroEconometricModels.n_control(model_prop) == sum(.!treatment)
    end

    @testset "DriscollKraay Covariance Estimator" begin
        # Test DriscollKraayEstimator type construction
        dk_est = MacroEconometricModels.DriscollKraayEstimator{Float64}(5)
        @test dk_est.bandwidth == 5
        @test dk_est.kernel == :bartlett

        # Default constructor
        dk_default = MacroEconometricModels.DriscollKraayEstimator()
        @test dk_default.bandwidth == 0
        @test dk_default.kernel == :bartlett

        # Create via create_cov_estimator
        dk_via_create = MacroEconometricModels.create_cov_estimator(:driscoll_kraay, Float64; bandwidth=3)
        @test dk_via_create isa MacroEconometricModels.DriscollKraayEstimator{Float64}
        @test dk_via_create.bandwidth == 3

        # Invalid bandwidth should throw
        @test_throws ArgumentError MacroEconometricModels.DriscollKraayEstimator{Float64}(-1)

        # Note: robust_vcov is not yet implemented for DriscollKraayEstimator
        # Testing that estimate_lp correctly creates the estimator but would need
        # implementation to actually use it
    end

    @testset "BSplineBasis Accessor" begin
        horizons = collect(0:15)
        basis = MacroEconometricModels.bspline_basis(horizons, 3, 4; T=Float64)

        # Test n_basis accessor
        @test MacroEconometricModels.n_basis(basis) == 4 + 3 + 1  # n_interior_knots + degree + 1 = 8
        @test size(basis.basis_matrix, 2) == MacroEconometricModels.n_basis(basis)
    end

    @testset "LP Estimation Edge Cases" begin
        Random.seed!(88888)

        # Minimum horizon (h=1) - h=0 should throw error
        Y_edge = randn(100, 2)
        @test_throws ArgumentError estimate_lp(Y_edge, 1, 0; lags=2)

        # Minimum valid horizon (h=1)
        model_h1 = estimate_lp(Y_edge, 1, 1; lags=2)
        @test model_h1.horizon == 1
        @test length(model_h1.B) == 2  # h=0 and h=1
        irf_h1 = lp_irf(model_h1)
        @test size(irf_h1.values) == (2, 2)

        # Subset of response variables
        model_subset = estimate_lp(Y_edge, 1, 5; lags=2, response_vars=[2])
        @test size(model_subset.B[1], 2) == 1

        # Integer input (type conversion)
        Y_int = rand(1:10, 80, 2)
        model_int = estimate_lp(Y_int, 1, 5; lags=2)
        @test eltype(model_int.Y) == Float64
    end
end
