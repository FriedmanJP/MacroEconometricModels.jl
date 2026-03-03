# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

@testset "FAVAR Tests" begin

    # =========================================================================
    # Shared test data generation
    # =========================================================================
    function make_favar_data(; T_obs=200, N=30, r_true=3, n_key=2, seed=42)
        rng = Random.MersenneTwister(seed)
        F_true = randn(rng, T_obs, r_true)
        Lambda_true = randn(rng, N, r_true)
        noise = 0.3 * randn(rng, T_obs, N)
        X = F_true * Lambda_true' + noise
        return X, n_key
    end

    # =========================================================================
    # Task 3: FAVARModel Type + Display + Accessors
    # =========================================================================

    @testset "FAVARModel construction and field access" begin
        X, n_key = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        @test favar isa FAVARModel{Float64}
        @test favar.n_factors == 2
        @test favar.n_key == 2
        @test favar.p == 2
        @test size(favar.Y, 2) == 4  # r + n_key
        @test size(favar.B, 1) == 1 + 4 * 2  # 1 + n * p
        @test size(favar.B, 2) == 4
        @test size(favar.U, 2) == 4
        @test size(favar.Sigma) == (4, 4)
        @test length(favar.varnames) == 4
        @test favar.varnames[1] == "F1"
        @test favar.varnames[2] == "F2"
        @test size(favar.X_panel) == (200, 30)
        @test length(favar.panel_varnames) == 30
        @test favar.Y_key_indices == [1, 2]
        @test size(favar.factors) == (200, 2)
        @test size(favar.loadings) == (30, 2)
        @test favar.factor_model isa FactorModel{Float64}
        @test isfinite(favar.aic)
        @test isfinite(favar.bic)
        @test isfinite(favar.loglik)
    end

    @testset "FAVARModel accessors" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 3, 1)

        @test nvars(favar) == 5  # 3 factors + 2 key
        @test nlags(favar) == 1
        @test ncoefs(favar) == 1 + 5 * 1
        @test effective_nobs(favar) == 200 - 1
        @test length(varnames(favar)) == 5
        @test varnames(favar)[1] == "F1"
        @test varnames(favar)[4] == "X1"  # key variable name from panel_varnames
    end

    @testset "to_var() conversion" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)
        vm = to_var(favar)

        @test vm isa VARModel{Float64}
        @test nvars(vm) == nvars(favar)
        @test nlags(vm) == nlags(favar)
        @test size(vm.B) == size(favar.B)
        @test vm.varnames == favar.varnames
        @test isfinite(vm.aic)
        @test isfinite(vm.bic)
    end

    @testset "Display output" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        # FAVARModel display
        io = IOBuffer()
        show(io, favar)
        output = String(take!(io))
        @test length(output) > 100
        @test occursin("FAVAR", output)
        @test occursin("Factors", output)
        @test occursin("Key variables", output)
        @test occursin("Variance Explained", output)
    end

    @testset "BayesianFAVAR display" begin
        # Construct a BayesianFAVAR manually to test display
        T_obs = 50; r = 2; n_key = 1; n_var = 3; p = 1; k = 1 + n_var * p; n_draws = 10
        bfavar = BayesianFAVAR{Float64}(
            randn(n_draws, k, n_var),        # B_draws
            randn(n_draws, n_var, n_var),     # Sigma_draws
            randn(n_draws, T_obs, r),         # factor_draws
            randn(n_draws, 20, r),            # loadings_draws
            randn(T_obs, 20),                 # X_panel
            ["X$i" for i in 1:20],            # panel_varnames
            [1],                              # Y_key_indices
            r,                                # n_factors
            n_key,                            # n_key
            n_var,                            # n
            p,                                # p
            randn(T_obs, n_var),              # data
            ["F1", "F2", "Y1"]                # varnames
        )
        io = IOBuffer()
        show(io, bfavar)
        output = String(take!(io))
        @test length(output) > 100
        @test occursin("Bayesian FAVAR", output)
        @test occursin("Equation", output)
    end

    # =========================================================================
    # Task 4: Two-Step FAVAR Estimation
    # =========================================================================

    @testset "Two-step estimation dimensions" begin
        X, _ = make_favar_data(T_obs=200, N=30, r_true=3)
        favar = estimate_favar(X, [1, 5], 3, 2)

        @test favar.n_factors == 3
        @test favar.n_key == 2
        @test nvars(favar) == 5
        @test size(favar.factors) == (200, 3)
        @test size(favar.loadings) == (30, 3)
        @test size(favar.B, 1) == 1 + 5 * 2  # intercept + 5 vars * 2 lags
        @test size(favar.B, 2) == 5
        @test effective_nobs(favar) == 198
    end

    @testset "Matrix dispatch (Y_key as matrix)" begin
        X, _ = make_favar_data()
        Y_key = X[:, [1, 5]]
        favar = estimate_favar(X, Y_key, 2, 2)

        @test favar isa FAVARModel{Float64}
        @test favar.n_factors == 2
        @test favar.n_key == 2
        # Y_key_indices should be found by column matching
        @test favar.Y_key_indices == [1, 5]
    end

    @testset "Column indices dispatch" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [3, 10, 20], 2, 1)

        @test favar.Y_key_indices == [3, 10, 20]
        @test favar.n_key == 3
        @test nvars(favar) == 5  # 2 factors + 3 key
    end

    @testset "Panel varnames keyword" begin
        X, _ = make_favar_data(N=20)
        pvn = ["macro_$i" for i in 1:20]
        favar = estimate_favar(X, [1, 2], 2, 1; panel_varnames=pvn)

        @test favar.panel_varnames == pvn
        @test favar.varnames[3] == "macro_1"  # key var names from panel_varnames
        @test favar.varnames[4] == "macro_2"
    end

    @testset "Double-counting removal" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        # Slow-moving factors should be roughly orthogonal to Y_key
        Y_key = X[:, [1, 2]]
        F_tilde = favar.factors
        for j in 1:favar.n_factors
            for k in 1:size(Y_key, 2)
                corr = abs(cor(F_tilde[:, j], Y_key[:, k]))
                # After removing double-counting, correlation should be small
                @test corr < 0.3
            end
        end
    end

    @testset "Float fallback (Integer matrix)" begin
        X_int = round.(Int, randn(100, 20) * 10)
        Y_key_int = X_int[:, [1, 2]]
        favar = estimate_favar(X_int, Y_key_int, 2, 1)
        @test favar isa FAVARModel{Float64}
    end

    # =========================================================================
    # Validation errors
    # =========================================================================

    @testset "Validation errors" begin
        X, _ = make_favar_data(T_obs=200, N=30)

        # r too large
        @test_throws ArgumentError estimate_favar(X, [1, 2], 31, 1)

        # p too large (not enough observations)
        @test_throws ArgumentError estimate_favar(X, [1, 2], 2, 198)

        # Key index out of range
        @test_throws ArgumentError estimate_favar(X, [1, 31], 2, 1)
        @test_throws ArgumentError estimate_favar(X, [0, 1], 2, 1)

        # Mismatched rows
        @test_throws ArgumentError estimate_favar(randn(100, 30), randn(50, 2), 2, 1)

        # Wrong panel_varnames length
        @test_throws ArgumentError estimate_favar(X, [1, 2], 2, 1;
            panel_varnames=["a", "b"])

        # r < 1
        @test_throws ArgumentError estimate_favar(X, [1, 2], 0, 1)

        # p < 1
        @test_throws ArgumentError estimate_favar(X, [1, 2], 2, 0)

        # Invalid method
        @test_throws ArgumentError estimate_favar(X, [1, 2], 2, 1; method=:invalid_method)
    end

    @testset "NaN/Inf data validation" begin
        X_nan = randn(100, 20)
        X_nan[5, 3] = NaN
        @test_throws ArgumentError estimate_favar(X_nan, [1, 2], 2, 1)

        X_ok = randn(100, 20)
        Y_key_inf = copy(X_ok[:, [1, 2]])
        Y_key_inf[1, 1] = Inf
        @test_throws ArgumentError estimate_favar(X_ok, Y_key_inf, 2, 1)
    end

    # =========================================================================
    # Task 5: Structural Analysis Delegation
    # =========================================================================

    @testset "IRF dispatch" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        irf_result = irf(favar, 10)
        @test irf_result isa ImpulseResponse{Float64}
        @test irf_result.horizon == 10
        @test size(irf_result.values) == (10, 4, 4)
        @test length(irf_result.variables) == 4
        @test length(irf_result.shocks) == 4
    end

    @testset "IRF with bootstrap CI" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        irf_ci = irf(favar, 10; ci_type=:bootstrap, reps=50)
        @test irf_ci.ci_type == :bootstrap
        @test size(irf_ci.ci_lower) == (10, 4, 4)
        @test size(irf_ci.ci_upper) == (10, 4, 4)
        # CI lower should be <= point estimate <= CI upper (mostly)
        @test sum(irf_ci.ci_lower .<= irf_ci.values) > 0.5 * length(irf_ci.values)
    end

    @testset "FEVD dispatch" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        fevd_result = fevd(favar, 10)
        @test fevd_result isa FEVD{Float64}
        @test size(fevd_result.decomposition, 1) == 4  # variables
        @test size(fevd_result.decomposition, 2) == 4  # shocks
        @test size(fevd_result.decomposition, 3) == 10 # horizons
        # FEVD proportions should sum to 1 for each variable at each horizon
        for h in 1:10
            for v in 1:4
                @test isapprox(sum(fevd_result.proportions[v, :, h]), 1.0; atol=1e-10)
            end
        end
    end

    @testset "Historical decomposition dispatch" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        hd = historical_decomposition(favar)
        @test hd isa HistoricalDecomposition
    end

    @testset "Historical decomposition with explicit horizon" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        hd = historical_decomposition(favar, 50)
        @test hd isa HistoricalDecomposition
    end

    # =========================================================================
    # Task 5: Panel-Wide IRF Mapping
    # =========================================================================

    @testset "favar_panel_irf maps to N panel variables" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        irf_aug = irf(favar, 10)

        panel_irf = favar_panel_irf(favar, irf_aug)
        @test panel_irf isa ImpulseResponse{Float64}
        @test size(panel_irf.values) == (10, 30, 4)  # H x N x n_shocks
        @test length(panel_irf.variables) == 30
        @test panel_irf.variables == favar.panel_varnames
    end

    @testset "favar_panel_irf key variable override" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        irf_aug = irf(favar, 10)

        panel_irf = favar_panel_irf(favar, irf_aug)

        # Key variable IRFs should match their direct VAR IRFs
        r = favar.n_factors
        for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
            var_idx = r + k_idx  # position in augmented VAR
            for h in 1:10, j in 1:4
                @test panel_irf.values[h, panel_idx, j] == irf_aug.values[h, var_idx, j]
            end
        end
    end

    @testset "favar_panel_irf with CI" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        irf_ci = irf(favar, 10; ci_type=:bootstrap, reps=30)

        panel_irf = favar_panel_irf(favar, irf_ci)
        @test panel_irf.ci_type == :bootstrap
        @test size(panel_irf.ci_lower) == (10, 30, 4)
        @test size(panel_irf.ci_upper) == (10, 30, 4)
    end

    @testset "favar_panel_irf loadings mapping" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        irf_aug = irf(favar, 10)

        panel_irf = favar_panel_irf(favar, irf_aug)

        # For non-key variables, IRF should equal Lambda * factor_irf
        Lambda = favar.loadings
        r = favar.n_factors
        key_set = Set(favar.Y_key_indices)
        for i in 1:30
            if !(i in key_set)
                for h in 1:10, j in 1:4
                    factor_irfs = irf_aug.values[h, 1:r, j]
                    expected = dot(Lambda[i, :], factor_irfs)
                    @test isapprox(panel_irf.values[h, i, j], expected; atol=1e-10)
                end
            end
        end
    end

    # =========================================================================
    # Task 5: Panel Forecast Mapping
    # =========================================================================

    @testset "favar_panel_forecast" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        fc = forecast(favar, 5; ci_method=:none)

        panel_fc = favar_panel_forecast(favar, fc)
        @test panel_fc isa VARForecast{Float64}
        @test size(panel_fc.forecast) == (5, 30)
        @test panel_fc.varnames == favar.panel_varnames
    end

    @testset "favar_panel_forecast key variable override" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        fc = forecast(favar, 5; ci_method=:none)

        panel_fc = favar_panel_forecast(favar, fc)

        r = favar.n_factors
        for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
            var_idx = r + k_idx
            for step in 1:5
                @test panel_fc.forecast[step, panel_idx] == fc.forecast[step, var_idx]
            end
        end
    end

    @testset "favar_panel_forecast with bootstrap CI" begin
        X, _ = make_favar_data(N=30)
        favar = estimate_favar(X, [1, 5], 2, 2)
        fc = forecast(favar, 5; ci_method=:bootstrap, reps=30)

        panel_fc = favar_panel_forecast(favar, fc)
        @test size(panel_fc.ci_lower) == (5, 30)
        @test size(panel_fc.ci_upper) == (5, 30)
    end

    # =========================================================================
    # Task 6: Forecast Delegation
    # =========================================================================

    @testset "forecast dispatch" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        fc = forecast(favar, 10; ci_method=:none)
        @test fc isa VARForecast{Float64}
        @test fc.horizon == 10
        @test size(fc.forecast) == (10, 4)
        @test fc.ci_method == :none
    end

    @testset "forecast with bootstrap" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 2)

        fc = forecast(favar, 5; ci_method=:bootstrap, reps=50, conf_level=0.90)
        @test fc.ci_method == :bootstrap
        @test fc.conf_level == 0.90
        @test size(fc.ci_lower) == (5, 4)
        @test size(fc.ci_upper) == (5, 4)
    end

    # =========================================================================
    # Edge cases
    # =========================================================================

    @testset "Single factor" begin
        X, _ = make_favar_data(N=20)
        favar = estimate_favar(X, [1], 1, 1)

        @test favar.n_factors == 1
        @test favar.n_key == 1
        @test nvars(favar) == 2
        @test size(irf(favar, 5).values) == (5, 2, 2)
    end

    @testset "Many factors" begin
        X, _ = make_favar_data(T_obs=200, N=50)
        favar = estimate_favar(X, [1], 10, 1)

        @test favar.n_factors == 10
        @test nvars(favar) == 11
    end

    @testset "Single lag" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 1)

        @test nlags(favar) == 1
        @test effective_nobs(favar) == 199
    end

    @testset "Multiple lags" begin
        X, _ = make_favar_data()
        favar = estimate_favar(X, [1, 2], 2, 4)

        @test nlags(favar) == 4
        @test effective_nobs(favar) == 196
    end

    # =========================================================================
    # Task 7: Bayesian FAVAR Estimation
    # =========================================================================

    @testset "Bayesian FAVAR estimation dimensions" begin
        X, _ = make_favar_data(T_obs=150, N=20, r_true=2)
        bfavar = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=50, burnin=20)

        @test bfavar isa BayesianFAVAR{Float64}
        @test bfavar.n_factors == 2
        @test bfavar.n_key == 2
        @test bfavar.n == 4  # r + n_key
        @test bfavar.p == 1
        @test size(bfavar.B_draws, 1) == 50  # n_draws
        @test size(bfavar.B_draws, 2) == 1 + 4 * 1  # k = 1 + n * p
        @test size(bfavar.B_draws, 3) == 4  # n_var
        @test size(bfavar.Sigma_draws) == (50, 4, 4)
        @test size(bfavar.factor_draws) == (50, 150, 2)
        @test size(bfavar.loadings_draws) == (50, 20, 2)
        @test size(bfavar.X_panel) == (150, 20)
        @test size(bfavar.data) == (150, 4)
        @test length(bfavar.varnames) == 4
        @test length(bfavar.panel_varnames) == 20
        @test bfavar.Y_key_indices == [1, 5]
    end

    @testset "Bayesian FAVAR types and finite values" begin
        X, _ = make_favar_data(T_obs=100, N=15)
        bfavar = estimate_favar(X, [1, 3], 2, 1; method=:bayesian, n_draws=30, burnin=10)

        # B_draws should be finite
        @test all(isfinite, bfavar.B_draws)
        # Sigma_draws should be finite
        @test all(isfinite, bfavar.Sigma_draws)
        # Factor draws should be finite
        @test all(isfinite, bfavar.factor_draws)
        # Loadings should be finite
        @test all(isfinite, bfavar.loadings_draws)
    end

    @testset "Bayesian FAVAR with column indices" begin
        X, _ = make_favar_data(T_obs=100, N=20)
        bfavar = estimate_favar(X, [3, 10], 2, 1; method=:bayesian, n_draws=30, burnin=10)

        @test bfavar isa BayesianFAVAR{Float64}
        @test bfavar.Y_key_indices == [3, 10]
        @test bfavar.n_key == 2
    end

    @testset "Bayesian FAVAR with matrix Y_key" begin
        X, _ = make_favar_data(T_obs=100, N=20)
        Y_key = X[:, [2, 7]]
        bfavar = estimate_favar(X, Y_key, 2, 1; method=:bayesian, n_draws=30, burnin=10)

        @test bfavar isa BayesianFAVAR{Float64}
        @test bfavar.n_factors == 2
        @test bfavar.n_key == 2
    end

    @testset "Bayesian FAVAR display" begin
        X, _ = make_favar_data(T_obs=100, N=15)
        bfavar = estimate_favar(X, [1, 2], 2, 1; method=:bayesian, n_draws=30, burnin=10)

        io = IOBuffer()
        show(io, bfavar)
        output = String(take!(io))
        @test length(output) > 100
        @test occursin("Bayesian FAVAR", output)
        @test occursin("Equation", output)
        @test occursin("Draws", output)
    end

    @testset "Bayesian FAVAR default burnin" begin
        X, _ = make_favar_data(T_obs=100, N=15)
        # burnin=0 should default to 200 internally
        bfavar = estimate_favar(X, [1, 2], 2, 1; method=:bayesian, n_draws=30, burnin=0)
        @test bfavar isa BayesianFAVAR{Float64}
        @test size(bfavar.B_draws, 1) == 30
    end

    # =========================================================================
    # Task 8: Bayesian FAVAR Structural Analysis
    # =========================================================================

    @testset "Bayesian FAVAR IRF" begin
        X, _ = make_favar_data(T_obs=150, N=20, r_true=2)
        bfavar = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=60, burnin=20)

        irf_result = irf(bfavar, 10)
        @test irf_result isa BayesianImpulseResponse{Float64}
        @test irf_result.horizon == 10
        @test size(irf_result.point_estimate) == (10, 4, 4)
        @test length(irf_result.variables) == 4
        @test length(irf_result.shocks) == 4
        @test all(isfinite, irf_result.point_estimate)
    end

    @testset "Bayesian FAVAR FEVD" begin
        X, _ = make_favar_data(T_obs=150, N=20, r_true=2)
        bfavar = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=60, burnin=20)

        fevd_result = fevd(bfavar, 10)
        @test fevd_result isa BayesianFEVD{Float64}
        @test fevd_result.horizon == 10
        @test all(isfinite, fevd_result.point_estimate)
    end

    @testset "Bayesian FAVAR panel IRF" begin
        X, _ = make_favar_data(T_obs=150, N=20, r_true=2)
        bfavar = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=60, burnin=20)

        irf_aug = irf(bfavar, 10)
        panel_irf = favar_panel_irf(bfavar, irf_aug)

        @test panel_irf isa BayesianImpulseResponse{Float64}
        @test size(panel_irf.point_estimate) == (10, 20, 4)  # H x N x n_shocks
        @test length(panel_irf.variables) == 20
        @test panel_irf.variables == bfavar.panel_varnames
        @test all(isfinite, panel_irf.point_estimate)
    end

    @testset "Bayesian FAVAR panel IRF key variable override" begin
        X, _ = make_favar_data(T_obs=150, N=20, r_true=2)
        bfavar = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=60, burnin=20)

        irf_aug = irf(bfavar, 10)
        panel_irf = favar_panel_irf(bfavar, irf_aug)

        r = bfavar.n_factors
        # Key variable IRFs should match their direct VAR IRFs
        for (k_idx, panel_idx) in enumerate(bfavar.Y_key_indices)
            var_idx = r + k_idx
            for h in 1:10, j in 1:4
                @test panel_irf.point_estimate[h, panel_idx, j] ==
                      irf_aug.point_estimate[h, var_idx, j]
            end
        end
    end

    @testset "Bayesian FAVAR panel IRF loadings mapping" begin
        X, _ = make_favar_data(T_obs=150, N=20, r_true=2)
        bfavar = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=60, burnin=20)

        irf_aug = irf(bfavar, 10)
        panel_irf = favar_panel_irf(bfavar, irf_aug)

        Lambda_mean = dropdims(mean(bfavar.loadings_draws, dims=1), dims=1)
        r = bfavar.n_factors
        key_set = Set(bfavar.Y_key_indices)

        # Non-key variables should be mapped via loadings
        for i in 1:20
            if !(i in key_set)
                for h in 1:10, j in 1:4
                    factor_irfs = irf_aug.point_estimate[h, 1:r, j]
                    expected = dot(Lambda_mean[i, :], factor_irfs)
                    @test isapprox(panel_irf.point_estimate[h, i, j], expected; atol=1e-10)
                end
            end
        end
    end

    @testset "_to_bvar_posterior conversion" begin
        X, _ = make_favar_data(T_obs=100, N=15)
        bfavar = estimate_favar(X, [1, 2], 2, 1; method=:bayesian, n_draws=30, burnin=10)

        post = MacroEconometricModels._to_bvar_posterior(bfavar)
        @test post isa BVARPosterior{Float64}
        @test post.n_draws == 30
        @test post.p == 1
        @test post.n == 4  # r + n_key
        @test post.varnames == bfavar.varnames
        @test size(post.B_draws) == size(bfavar.B_draws)
        @test size(post.Sigma_draws) == size(bfavar.Sigma_draws)
    end

end  # @testset "FAVAR Tests"
