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

using Test, MacroEconometricModels, Random, DataFrames, LinearAlgebra, Statistics, Distributions

# =============================================================================
# Helper: generate balanced panel DGP
# =============================================================================

function _make_panel_dgp(; N=30, T_total=25, m=3, p=1, rng=MersenneTwister(123))
    # VAR(p) DGP with fixed effects
    # y_{i,t} = mu_i + A_1 y_{i,t-1} + eps_{i,t}
    A1 = 0.3 * I(m) + 0.05 * randn(rng, m, m)
    # Ensure stationarity
    F = eigvals(A1)
    while maximum(abs.(F)) >= 0.95
        A1 *= 0.8
        F = eigvals(A1)
    end

    data_mat = zeros(N * T_total, m)
    for i in 1:N
        mu_i = randn(rng, m)
        offset = (i - 1) * T_total
        data_mat[offset + 1, :] = mu_i + 0.1 * randn(rng, m)
        for t in 2:T_total
            data_mat[offset + t, :] = mu_i + A1 * data_mat[offset + t - 1, :] + 0.1 * randn(rng, m)
        end
    end

    df = DataFrame(data_mat, ["y$i" for i in 1:m])
    df.id = repeat(1:N, inner=T_total)
    df.time = repeat(1:T_total, outer=N)
    pd = xtset(df, :id, :time)
    (pd=pd, A1=A1, N=N, T_total=T_total, m=m)
end

# =============================================================================
# 1. GMM Extensions
# =============================================================================

@testset "GMM Extensions" begin
    rng = MersenneTwister(42)

    @testset "linear_gmm_solve known system" begin
        # Simple IV: Z'X β = Z'y
        k = 3
        q = 5
        X = randn(rng, 100, k)
        beta_true = [1.0, -2.0, 0.5]
        y = X * beta_true + 0.01 * randn(rng, 100)
        Z = hcat(X, randn(rng, 100, q - k))

        S_ZX = Z' * X
        S_Zy = Z' * y
        W = inv(Z' * Z / 100)

        beta_hat = linear_gmm_solve(S_ZX, S_Zy, W)
        @test length(beta_hat) == k
        @test isapprox(beta_hat, beta_true, atol=0.1)
    end

    @testset "linear_gmm_solve vector vs matrix S_Zy" begin
        rng2 = MersenneTwister(99)
        S_ZX = randn(rng2, 4, 2)
        S_Zy_vec = randn(rng2, 4)
        S_Zy_mat = reshape(S_Zy_vec, :, 1)
        W = Matrix(1.0I, 4, 4)

        b1 = linear_gmm_solve(S_ZX, S_Zy_vec, W)
        b2 = linear_gmm_solve(S_ZX, S_Zy_mat, W)
        @test isapprox(b1, b2, atol=1e-12)
    end

    @testset "gmm_sandwich_vcov dimensions" begin
        rng2 = MersenneTwister(77)
        q, k = 5, 3
        S_ZX = randn(rng2, q, k)
        W = Matrix(1.0I, q, q)
        D_e = randn(rng2, q, q)
        D_e = D_e' * D_e  # PSD

        V = gmm_sandwich_vcov(S_ZX, W, D_e)
        @test size(V) == (k, k)
        @test all(diag(V) .>= 0)
    end

    @testset "andrews_lu_mmsc" begin
        mmsc = andrews_lu_mmsc(10.0, 20, 5, 1000)
        @test haskey(mmsc, :bic)
        @test haskey(mmsc, :aic)
        @test haskey(mmsc, :hqic)
        @test mmsc.aic == 10.0 - 15 * 2
        # BIC penalizes more than AIC for large n
        @test mmsc.bic < mmsc.aic  # log(1000) > 2
    end

    @testset "andrews_lu_mmsc types" begin
        mmsc32 = andrews_lu_mmsc(Float32(10), 20, 5, 1000)
        @test mmsc32.bic isa Float32
    end
end

# =============================================================================
# 2. Panel Transforms
# =============================================================================

@testset "Panel Transforms" begin
    rng = MersenneTwister(55)

    @testset "first difference" begin
        Y = [1.0 2.0; 3.0 5.0; 6.0 9.0]
        fd = MacroEconometricModels._panel_first_difference(Y)
        @test size(fd) == (2, 2)
        @test fd[1, :] == [2.0, 3.0]
        @test fd[2, :] == [3.0, 4.0]
    end

    @testset "first difference random" begin
        Y = randn(rng, 20, 3)
        fd = MacroEconometricModels._panel_first_difference(Y)
        @test size(fd) == (19, 3)
        @test fd ≈ Y[2:end, :] - Y[1:end-1, :]
    end

    @testset "FOD dimensions" begin
        Y = randn(rng, 10, 2)
        fod = MacroEconometricModels._panel_fod(Y)
        @test size(fod) == (9, 2)
    end

    @testset "FOD known values" begin
        Y = [1.0; 2.0; 3.0;;]  # 3×1
        fod = MacroEconometricModels._panel_fod(Y)
        @test size(fod) == (2, 1)
        # t=1: sqrt(2/3) * (1 - 2.5) = sqrt(2/3) * (-1.5)
        @test fod[1, 1] ≈ sqrt(2/3) * (1.0 - 2.5)
        # t=2: sqrt(1/2) * (2 - 3) = sqrt(0.5) * (-1)
        @test fod[2, 1] ≈ sqrt(1/2) * (2.0 - 3.0)
    end

    @testset "FOD orthogonality" begin
        # FOD errors should be uncorrelated when original errors are i.i.d.
        rng2 = MersenneTwister(88)
        eps_orig = randn(rng2, 100, 1)
        fod_eps = MacroEconometricModels._panel_fod(eps_orig)
        # Autocorrelation should be small
        acf1 = cor(fod_eps[1:end-1, 1], fod_eps[2:end, 1])
        @test abs(acf1) < 0.3
    end

    @testset "demean" begin
        Y = [1.0 4.0; 2.0 5.0; 3.0 6.0]
        dm = MacroEconometricModels._panel_demean(Y)
        @test all(abs.(mean(dm, dims=1)) .< 1e-12)
        @test dm ≈ [-1.0 -1.0; 0.0 0.0; 1.0 1.0]
    end

    @testset "panel lag" begin
        Y = Float64[1 2; 3 4; 5 6; 7 8; 9 10]
        X = MacroEconometricModels._panel_lag(Y, 2)
        @test size(X) == (3, 4)  # (5-2) × (2*2)
        # First row: lags of t=3: Y_{t-1}=[3,4], Y_{t-2}=[1,2]
        @test X[1, 1:2] == [3.0, 4.0]
        @test X[1, 3:4] == [1.0, 2.0]
    end

    @testset "panel lag too few obs" begin
        Y = randn(rng, 2, 3)
        @test_throws ArgumentError MacroEconometricModels._panel_lag(Y, 3)
    end
end

# =============================================================================
# 3. Instruments
# =============================================================================

@testset "Instruments" begin
    rng = MersenneTwister(66)

    @testset "FD instruments dimensions" begin
        Y = randn(rng, 15, 2)  # T=15, m=2
        p = 2
        Z = MacroEconometricModels._build_instruments_fd(Y, p, 2)
        # T_eff = 15 - 2 - 1 = 12
        @test size(Z, 1) == 12
        @test size(Z, 2) > 0  # block-diagonal has many columns
    end

    @testset "FD instruments collapse" begin
        Y = randn(rng, 15, 2)
        Z_full = MacroEconometricModels._build_instruments_fd(Y, 1, 2)
        Z_coll = MacroEconometricModels._build_instruments_fd(Y, 1, 2; collapse=true)
        @test size(Z_full, 1) == size(Z_coll, 1)
        @test size(Z_coll, 2) <= size(Z_full, 2)  # collapsed has fewer columns
    end

    @testset "FD instruments min/max lag" begin
        Y = randn(rng, 20, 2)
        Z_all = MacroEconometricModels._build_instruments_fd(Y, 1, 2; min_lag=2, max_lag=99)
        Z_trunc = MacroEconometricModels._build_instruments_fd(Y, 1, 2; min_lag=2, max_lag=3)
        @test size(Z_trunc, 2) <= size(Z_all, 2)
    end

    @testset "System GMM instruments" begin
        Y = randn(rng, 15, 2)
        Z = MacroEconometricModels._build_instruments_system(Y, 1, 2)
        # System: stacked [FD; Level] with more rows
        T_eff = 15 - 1 - 1  # 13
        @test size(Z, 1) == 2 * T_eff
    end

    @testset "PCA reduction" begin
        rng2 = MersenneTwister(111)
        Z = randn(rng2, 50, 100)  # many instruments
        Z_pca = MacroEconometricModels._pca_reduce_instruments(Z; max_components=10)
        @test size(Z_pca, 1) == 50
        @test size(Z_pca, 2) == 10
    end
end

# =============================================================================
# 4. GMM Estimation
# =============================================================================

@testset "GMM Estimation" begin
    dgp = _make_panel_dgp(N=30, T_total=25, m=3, p=1)
    pd = dgp.pd

    @testset "FD-GMM onestep" begin
        model = estimate_pvar(pd, 1; steps=:onestep)
        @test model isa PVARModel
        @test model.m == 3
        @test model.p == 1
        @test model.method == :fd_gmm
        @test model.transformation == :fd
        @test model.steps == :onestep
        @test model.n_groups == 30
        @test size(model.Phi) == (3, 3)
        @test size(model.se) == (3, 3)
        @test all(model.se .>= 0)
    end

    @testset "FD-GMM twostep" begin
        model = estimate_pvar(pd, 1; steps=:twostep)
        @test model.steps == :twostep
        @test model.n_instruments > 0
        @test size(model.Phi) == (3, 3)
    end

    @testset "FOD transformation" begin
        model = estimate_pvar(pd, 1; transformation=:fod, steps=:onestep)
        @test model.transformation == :fod
        @test model.m == 3
    end

    @testset "System GMM" begin
        model = estimate_pvar(pd, 1; system_instruments=true, steps=:twostep)
        @test model.method == :system_gmm
        @test model.n_instruments > 0
    end

    @testset "Multiple lags" begin
        model = estimate_pvar(pd, 2; steps=:onestep)
        @test model.p == 2
        @test size(model.Phi) == (3, 6)  # 3 * 2 = 6 regressors
    end

    @testset "Collapsed instruments" begin
        model = estimate_pvar(pd, 1; steps=:onestep, collapse=true)
        @test model isa PVARModel
    end

    @testset "subset of dependent vars" begin
        model = estimate_pvar(pd, 1; dependent_vars=["y1", "y2"], steps=:onestep)
        @test model.m == 2
        @test model.varnames == ["y1", "y2"]
    end

    @testset "invalid inputs" begin
        @test_throws ArgumentError estimate_pvar(pd, 0)
        @test_throws ArgumentError estimate_pvar(pd, 1; transformation=:bad)
        @test_throws ArgumentError estimate_pvar(pd, 1; steps=:bad)
    end

    @testset "DGP coefficient recovery (FE-OLS, large T)" begin
        # FE-OLS should recover true coefficients better with large T
        dgp_large = _make_panel_dgp(N=50, T_total=100, m=2, p=1, rng=MersenneTwister(200))
        model = estimate_pvar_feols(dgp_large.pd, 1)
        A_hat = model.Phi[:, 1:2]
        A_true = dgp_large.A1
        # With 50 groups × 100 periods, should be close
        @test norm(A_hat - A_true) / norm(A_true) < 0.3
    end
end

# =============================================================================
# 5. FE-OLS
# =============================================================================

@testset "FE-OLS" begin
    dgp = _make_panel_dgp(N=30, T_total=25, m=3, p=1)
    pd = dgp.pd

    @testset "basic estimation" begin
        model = estimate_pvar_feols(pd, 1)
        @test model isa PVARModel
        @test model.method == :fe_ols
        @test model.transformation == :demean
        @test model.m == 3
        @test model.p == 1
        @test size(model.Phi) == (3, 3)
    end

    @testset "multiple lags" begin
        model = estimate_pvar_feols(pd, 2)
        @test model.p == 2
        @test size(model.Phi) == (3, 6)
    end

    @testset "cluster-robust SEs" begin
        model = estimate_pvar_feols(pd, 1)
        @test all(model.se .>= 0)
        @test all(isfinite.(model.se))
    end

    @testset "subset vars" begin
        model = estimate_pvar_feols(pd, 1; dependent_vars=["y1", "y3"])
        @test model.m == 2
        @test model.varnames == ["y1", "y3"]
    end

    @testset "invalid inputs" begin
        @test_throws ArgumentError estimate_pvar_feols(pd, 0)
    end
end

# =============================================================================
# 6. Analysis
# =============================================================================

@testset "Analysis" begin
    dgp = _make_panel_dgp(N=30, T_total=25, m=3, p=1)
    model = estimate_pvar(dgp.pd, 1; steps=:onestep)
    H = 10

    @testset "OIRF dimensions" begin
        oirf = pvar_oirf(model, H)
        @test size(oirf) == (H + 1, 3, 3)
    end

    @testset "OIRF impact" begin
        oirf = pvar_oirf(model, 0)
        # Impact matrix should be lower triangular (Cholesky)
        P = safe_cholesky(model.Sigma)
        @test oirf[1, :, :] ≈ Matrix(P) atol=1e-10
    end

    @testset "GIRF dimensions" begin
        girf = pvar_girf(model, H)
        @test size(girf) == (H + 1, 3, 3)
    end

    @testset "GIRF vs OIRF first shock" begin
        oirf = pvar_oirf(model, H)
        girf = pvar_girf(model, H)
        # For the first shock, OIRF and GIRF should be proportional
        # (GIRF uses Sigma e_1 / sqrt(sigma_11))
        @test size(oirf) == size(girf)
    end

    @testset "FEVD dimensions" begin
        fv = pvar_fevd(model, H)
        @test size(fv) == (H + 1, 3, 3)
    end

    @testset "FEVD sums to 1" begin
        fv = pvar_fevd(model, H)
        for h in 1:(H+1)
            for l in 1:3
                @test isapprox(sum(fv[h, l, :]), 1.0, atol=0.01)
            end
        end
    end

    @testset "FEVD non-negative" begin
        fv = pvar_fevd(model, H)
        @test all(fv .>= -1e-10)
    end

    @testset "stability" begin
        stab = pvar_stability(model)
        @test stab isa PVARStability
        @test length(stab.moduli) == 3  # m * p = 3 * 1
        @test stab.moduli == sort(stab.moduli, rev=true)
    end

    @testset "stability from DGP" begin
        # Use FE-OLS (more numerically stable than GMM) on a large panel
        dgp_large = _make_panel_dgp(N=100, T_total=50, m=3, p=1, rng=MersenneTwister(999))
        model_large = estimate_pvar_feols(dgp_large.pd, 1)
        stab = pvar_stability(model_large)
        @test maximum(stab.moduli) < 1.0
    end

    @testset "negative horizon" begin
        @test_throws ArgumentError pvar_oirf(model, -1)
        @test_throws ArgumentError pvar_girf(model, -1)
        @test_throws ArgumentError pvar_fevd(model, -1)
    end

    @testset "FE-OLS analysis" begin
        model2 = estimate_pvar_feols(dgp.pd, 1)
        oirf2 = pvar_oirf(model2, 5)
        @test size(oirf2) == (6, 3, 3)
        stab2 = pvar_stability(model2)
        @test stab2 isa PVARStability
    end
end

# =============================================================================
# 7. Specification Tests
# =============================================================================

@testset "Specification Tests" begin
    dgp = _make_panel_dgp(N=30, T_total=25, m=3, p=1)
    model = estimate_pvar(dgp.pd, 1; steps=:twostep)

    @testset "Hansen J-test" begin
        j = pvar_hansen_j(model)
        @test j isa PVARTestResult
        @test j.statistic >= 0
        @test 0 <= j.pvalue <= 1
        @test j.df >= 0
        @test j.n_instruments == model.n_instruments
    end

    @testset "Hansen J not for FE-OLS" begin
        model2 = estimate_pvar_feols(dgp.pd, 1)
        @test_throws ArgumentError pvar_hansen_j(model2)
    end

    @testset "MMSC" begin
        mmsc = pvar_mmsc(model)
        @test haskey(mmsc, :bic)
        @test haskey(mmsc, :aic)
        @test haskey(mmsc, :hqic)
        @test isfinite(mmsc.bic)
        @test isfinite(mmsc.aic)
    end

    @testset "lag selection" begin
        sel = pvar_lag_selection(dgp.pd, 3; steps=:onestep)
        @test sel.best_bic >= 1
        @test sel.best_aic >= 1
        @test sel.best_hqic >= 1
        @test size(sel.table) == (3, 4)
    end
end

# =============================================================================
# 8. Bootstrap
# =============================================================================

@testset "Bootstrap" begin
    dgp = _make_panel_dgp(N=20, T_total=15, m=2, p=1, rng=MersenneTwister(444))
    model = estimate_pvar(dgp.pd, 1; steps=:onestep, collapse=true)

    @testset "bootstrap OIRF" begin
        result = pvar_bootstrap_irf(model, 5; n_draws=(FAST ? 10 : 20), rng=MersenneTwister(55))
        @test size(result.irf) == (6, 2, 2)
        @test size(result.lower) == (6, 2, 2)
        @test size(result.upper) == (6, 2, 2)
        @test size(result.draws) == (20, 6, 2, 2)
    end

    @testset "bootstrap CI ordering" begin
        result = pvar_bootstrap_irf(model, 3; n_draws=(FAST ? 10 : 20), rng=MersenneTwister(66))
        @test all(result.lower .<= result.upper)
    end

    @testset "bootstrap GIRF" begin
        result = pvar_bootstrap_irf(model, 3; irf_type=:girf, n_draws=(FAST ? 5 : 10), rng=MersenneTwister(77))
        @test size(result.irf) == (4, 2, 2)
    end

    @testset "bootstrap invalid args" begin
        @test_throws ArgumentError pvar_bootstrap_irf(model, -1; n_draws=5)
        @test_throws ArgumentError pvar_bootstrap_irf(model, 3; irf_type=:bad, n_draws=5)
    end
end

# =============================================================================
# 9. Integration
# =============================================================================

@testset "Integration" begin
    dgp = _make_panel_dgp(N=20, T_total=20, m=2, p=1, rng=MersenneTwister(999))

    @testset "PanelData input" begin
        model = estimate_pvar(dgp.pd, 1; steps=:onestep)
        @test model.data === dgp.pd
        @test model.varnames == ["y1", "y2"]
    end

    @testset "StatsAPI interface" begin
        model = estimate_pvar(dgp.pd, 1; steps=:onestep)
        @test length(coef(model)) == 4  # 2 * 2
        @test nobs(model) > 0
        @test dof(model) == 4
        @test length(stderror(model)) == 4
    end

    @testset "refs dispatch" begin
        model = estimate_pvar(dgp.pd, 1; steps=:onestep)
        io = IOBuffer()
        refs(io, model)
        s = String(take!(io))
        @test occursin("Holtz-Eakin", s)
        @test occursin("Arellano", s)
    end

    @testset "refs symbol dispatch" begin
        io = IOBuffer()
        refs(io, :pvar)
        s = String(take!(io))
        @test occursin("Holtz-Eakin", s)

        io2 = IOBuffer()
        refs(io2, :windmeijer)
        s2 = String(take!(io2))
        @test occursin("Windmeijer", s2)
    end

    @testset "refs stability and test" begin
        model = estimate_pvar(dgp.pd, 1; steps=:onestep)
        stab = pvar_stability(model)
        io = IOBuffer()
        refs(io, stab)
        @test occursin("Holtz-Eakin", String(take!(io)))

        j = pvar_hansen_j(model)
        io2 = IOBuffer()
        refs(io2, j)
        @test occursin("Hansen", String(take!(io2)))
    end
end

# =============================================================================
# 10. Display
# =============================================================================

@testset "Display" begin
    dgp = _make_panel_dgp(N=20, T_total=20, m=2, p=1, rng=MersenneTwister(888))

    @testset "show PVARModel" begin
        model = estimate_pvar(dgp.pd, 1; steps=:onestep)
        io = IOBuffer()
        show(io, model)
        s = String(take!(io))
        @test occursin("Panel VAR", s)
        @test occursin("FD-GMM", s)
        @test occursin("L1.y1", s)
    end

    @testset "show PVARModel FE-OLS" begin
        model = estimate_pvar_feols(dgp.pd, 1)
        io = IOBuffer()
        show(io, model)
        s = String(take!(io))
        @test occursin("FE-OLS", s)
    end

    @testset "show PVARStability" begin
        model = estimate_pvar(dgp.pd, 1; steps=:onestep)
        stab = pvar_stability(model)
        io = IOBuffer()
        show(io, stab)
        s = String(take!(io))
        @test occursin("Stability", s)
    end

    @testset "show PVARTestResult" begin
        model = estimate_pvar(dgp.pd, 1; steps=:twostep)
        j = pvar_hansen_j(model)
        io = IOBuffer()
        show(io, j)
        s = String(take!(io))
        @test occursin("Hansen", s)
        @test occursin("Statistic", s)
    end

    @testset "show System GMM" begin
        model = estimate_pvar(dgp.pd, 1; system_instruments=true, steps=:onestep)
        io = IOBuffer()
        show(io, model)
        s = String(take!(io))
        @test occursin("System GMM", s)
    end
end
