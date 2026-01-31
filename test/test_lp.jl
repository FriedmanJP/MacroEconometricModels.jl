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
end
