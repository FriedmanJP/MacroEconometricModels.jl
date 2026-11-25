# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using StatsAPI

# Helper to simulate dynamic factor panel with q_true dynamic shocks loaded at lags 0:lag_max
# giving r_true = q_true * (lag_max + 1) static factors, plus optional AR(1) + spatial noise.
function _test_sim_dynamic_panel(rng::AbstractRNG, T_obs::Int, N::Int,
                                  q_true::Int; lag_max::Int=1,
                                  sigma_e::Float64=1.0,
                                  rho_u::Float64=0.4,
                                  rho_e::Float64=0.0,
                                  beta_spatial::Float64=0.0)
    burn = 60
    T_tot = T_obs + burn + lag_max
    if q_true > 0
        U = zeros(T_tot, q_true)
        A = rho_u * Matrix{Float64}(I, q_true, q_true)
        for t in 2:T_tot
            U[t, :] = A * U[t - 1, :] + randn(rng, q_true)
        end
        # Orthogonal-ish distinct loadings per lag for full rank r_true = q_true * (lag_max + 1)
        chi = zeros(T_obs, N)
        for ell in 0:lag_max
            Lambda_ell = randn(rng, N, q_true) .* sqrt(1.0 / (1.0 + 0.3 * ell))
            idx = (burn + lag_max + 1 - ell):(T_tot - ell)
            chi .+= U[idx, :] * Lambda_ell'
        end
    else
        chi = zeros(T_obs, N)
    end

    E = randn(rng, T_obs, N) .* sigma_e
    if rho_e != 0.0
        for t in 2:T_obs
            E[t, :] .+= rho_e .* E[t - 1, :]
        end
    end
    if beta_spatial != 0.0
        E_sp = copy(E)
        for j in 2:(N - 1)
            E_sp[:, j] .+= beta_spatial .* (E[:, j - 1] .+ E[:, j + 1])
        end
        E = E_sp
    end

    chi .+ E
end

@testset "Factor Number Selection (Onatski 2009, Ahn-Horenstein 2013, Onatski 2010)" begin

    @testset "TW_2 Critical Values & Table I Alignment" begin
        rng = MersenneTwister(101)
        X_dummy = randn(rng, 100, 30)

        # Known Onatski (2009) Table I asymptotic right-tail critical values for w = 1:7
        # 10% and 15% columns from Onatski (2009, Econometrica, Table I)
        table_i_10 = [3.33, 4.31, 4.91, 5.40, 5.77, 6.13, 6.42]
        table_i_15 = [2.75, 3.62, 4.15, 4.54, 4.89, 5.20, 5.45]

        qtable = MacroEconometricModels._FSEL_TW2_QTABLE
        pgrid = MacroEconometricModels._FSEL_TW2_PGRID
        @test size(qtable) == (15, length(pgrid))

        # Monotonicity in both probability grid and w = 1:15
        for w in 1:15
            @test issorted(qtable[w, :])
        end
        for j in 1:length(pgrid)
            @test issorted(qtable[:, j])
        end

        for w in 1:7
            res = onatski_q_test(X_dummy, 0, w)
            cv_10, cv_05, cv_01 = res.crit_vals
            @test cv_10 < cv_05 < cv_01
            @test abs(cv_10 - table_i_10[w]) < 0.045
            idx_85 = findfirst(==(0.85), pgrid)
            @test abs(qtable[w, idx_85] - table_i_15[w]) < 0.03

            # Consistency between p-value and critical values across all grid regimes
            for s in (0.1, cv_10 - 0.01, cv_10 + 0.01, cv_05 - 0.01, cv_05 + 0.01, cv_01 - 0.01, cv_01 + 0.01, 50.0)
                pv, _ = MacroEconometricModels._fsel_tw2_crit_and_pvalue(Float64(s), w)
                @test 0.0 <= pv <= 1.0
                @test (pv < 0.10) == (s > cv_10)
                @test (pv < 0.05) == (s > cv_05)
                @test (pv < 0.01) == (s > cv_01)
            end
        end

        # Tabulated w = 15 boundary
        res15 = onatski_q_test(X_dummy, 0, 15)
        @test length(res15.crit_vals) == 3
        @test issorted(res15.crit_vals)

        # Deterministic GUE tridiagonal fallback for w > 15
        res16_a = onatski_q_test(X_dummy, 0, 16; n_sim=1000)
        res16_b = onatski_q_test(X_dummy, 0, 16; n_sim=1000)
        @test res16_a.crit_vals == res16_b.crit_vals
        @test res16_a.pvalue == res16_b.pvalue
        @test issorted(res16_a.crit_vals)
        @test res16_a.crit_vals[1] > res15.crit_vals[1] * 0.9

        # Explicit seed and rng reproducibility on fallback
        res16_s1 = onatski_q_test(X_dummy, 0, 16; n_sim=800, seed=42)
        res16_s2 = onatski_q_test(X_dummy, 0, 16; n_sim=800, seed=42)
        res16_r1 = onatski_q_test(X_dummy, 0, 16; n_sim=800, rng=MersenneTwister(99))
        res16_r2 = onatski_q_test(X_dummy, 0, 16; n_sim=800, rng=MersenneTwister(99))
        @test res16_s1.crit_vals == res16_s2.crit_vals
        @test res16_r1.crit_vals == res16_r2.crit_vals
    end

    @testset "Onatski (2009) Size, Power & Sequential Selection" begin
        rng = MersenneTwister(2009)
        X = _test_sim_dynamic_panel(rng, 300, 100, 2; lag_max=1, sigma_e=1.0)

        t0 = onatski_q_test(X, 0, 6)
        t1 = onatski_q_test(X, 1, 6)
        t2 = onatski_q_test(X, 2, 6)

        @test t0 isa OnatskiQTestResult{Float64}
        @test t0.rejected_05 == true
        @test t0.pvalue < 0.01
        @test t1.rejected_05 == true
        @test t1.pvalue < 0.05
        @test t2.rejected_05 == false
        @test t2.pvalue >= 0.05

        # Keyword vs positional selector signatures agree
        q_kw, tests_kw = onatski_q_select(X; k_max=6, alpha=0.05)
        q_pos, tests_pos = onatski_q_select(X, 6; alpha=0.05)
        @test q_kw == 2
        @test q_pos == 2
        @test length(tests_kw) == 3
        @test length(tests_pos) == 3
        @test all(i -> tests_kw[i].statistic == tests_pos[i].statistic, 1:3)

        # Alternative options: n_blocks, single freq in (0, pi), Parzen kernel, unstandardized
        t_blk = onatski_q_test(X, 2, 6; n_blocks=6)
        @test t_blk.n_blocks == 6
        @test isfinite(t_blk.statistic)

        t_frq = onatski_q_test(X, 0, 6; freq=0.75)
        @test t_frq.n_blocks == 1
        @test isfinite(t_frq.statistic)

        t_prz = onatski_q_test(X, 2, 6; kernel=:parzen, bandwidth=14, standardize=false, demean=false)
        @test t_prz.bandwidth == 14
        @test isfinite(t_prz.statistic)
    end

    @testset "Monte Carlo Recovery: q_true = 2, r_true = 4 (20 reps, N=120, T=250)" begin
        rng = MersenneTwister(2026)
        n_reps = 20
        q_hits = 0
        er_hits = 0
        gr_hits = 0
        ed_hits = 0

        for _ in 1:n_reps
            X = _test_sim_dynamic_panel(rng, 250, 120, 2; lag_max=1, sigma_e=1.0)
            q_hat, _ = onatski_q_select(X; k_max=6, alpha=0.05)
            ah = ahn_horenstein(X, 10)
            ed = onatski_ed(X, 10)

            q_hits += (q_hat == 2)
            er_hits += (ah.r_er == 4)
            gr_hits += (ah.r_gr == 4)
            ed_hits += (ed.r_ed == 4)
        end

        @test q_hits / n_reps >= 0.80
        @test er_hits / n_reps >= 0.90
        @test gr_hits / n_reps >= 0.90
        @test ed_hits / n_reps >= 0.90
    end

    @testset "Factorless Noise (q_true = 0, r_true = 0) & Correlated Idiosyncratics" begin
        rng = MersenneTwister(31415)
        n_reps = 20
        q_zero = 0
        er_zero = 0
        gr_zero = 0
        ed_zero = 0

        for rep in 1:n_reps
            rho_e = rep <= 10 ? 0.0 : 0.45
            X = _test_sim_dynamic_panel(rng, 200, 80, 0; sigma_e=1.0, rho_e=rho_e)
            q_hat, _ = onatski_q_select(X; k_max=6, alpha=0.05)
            ah = ahn_horenstein(X, 8)
            ed = onatski_ed(X, 8)

            q_zero += (q_hat == 0)
            er_zero += (ah.r_er == 0)
            gr_zero += (ah.r_gr == 0)
            ed_zero += (ed.r_ed == 0)
        end

        @test q_zero / n_reps >= 0.90
        @test er_zero / n_reps >= 0.90
        @test gr_zero / n_reps >= 0.90
        @test ed_zero / n_reps >= 0.90

        # Correlated idiosyncratic static factor model (r_true = 3)
        X_corr = _test_sim_dynamic_panel(rng, 200, 100, 3; lag_max=0,
                                         sigma_e=0.9, rho_u=0.5, rho_e=0.35, beta_spatial=0.2)
        ah_corr = ahn_horenstein(X_corr, 8)
        ed_corr = onatski_ed(X_corr, 8)
        @test ah_corr.r_er == 3
        @test ah_corr.r_gr == 3
        @test ed_corr.r_ed == 3
        @test ed_corr.converged == true
        @test ed_corr.delta > 0
        @test length(ah_corr.er) == ah_corr.k_max + 1
        @test length(ah_corr.gr) == ah_corr.k_max + 1
        @test ah_corr.mock_eigenvalue > 0
        @test issorted(ah_corr.eigenvalues; rev=true)
    end

    @testset "Edge Cases, Scale Equivariance, Float32 & Int Fallbacks" begin
        rng = MersenneTwister(777)

        # Unstandardized scale equivariance in Onatski ED & AH
        X = _test_sim_dynamic_panel(rng, 180, 70, 2; lag_max=0, sigma_e=0.8)
        ed1 = onatski_ed(X, 6; standardize=false)
        ed10 = onatski_ed(10.0 .* X, 6; standardize=false)
        @test ed1.r_ed == ed10.r_ed == 2
        @test isapprox(ed10.delta, 100.0 * ed1.delta; rtol=1e-10)

        ah1 = ahn_horenstein(X, 6; standardize=false)
        ah10 = ahn_horenstein(10.0 .* X, 6; standardize=false)
        @test ah1.r_er == ah10.r_er == 2
        @test ah1.r_gr == ah10.r_gr == 2

        # Noiseless exact low-rank panel (r_true = 2, sigma_e = 0) and zero matrix
        X_clean = _test_sim_dynamic_panel(rng, 100, 40, 2; lag_max=0, sigma_e=0.0)
        ah_clean = ahn_horenstein(X_clean, 6)
        ed_clean = onatski_ed(X_clean, 6)
        q_clean, _ = onatski_q_select(X_clean, 5)
        @test ah_clean.r_er == 2
        @test ah_clean.r_gr == 2
        @test ed_clean.r_ed == 2
        @test ed_clean.converged == true
        @test q_clean == 2

        X_zero = zeros(Float64, 80, 30)
        ah_z = ahn_horenstein(X_zero, 5)
        ed_z = onatski_ed(X_zero, 5)
        q_z, _ = onatski_q_select(X_zero, 5)
        @test ah_z.r_er == 0 && ah_z.r_gr == 0
        @test ed_z.r_ed == 0
        @test q_z == 0

        # Float32 precision support
        X64 = _test_sim_dynamic_panel(rng, 200, 60, 2; lag_max=1, sigma_e=0.8)
        X32 = Float32.(X64)
        q32 = onatski_q_test(X32, 0, 5)
        qs32, tests32 = onatski_q_select(X32; k_max=5)
        ah32 = ahn_horenstein(X32, 6)
        ed32 = onatski_ed(X32, 6)

        @test q32 isa OnatskiQTestResult{Float32}
        @test q32.statistic isa Float32
        @test q32.pvalue isa Float32
        @test eltype(q32.crit_vals) == Float32
        @test eltype(q32.alpha_levels) == Float32
        @test eltype(tests32) == OnatskiQTestResult{Float32}
        @test qs32 == 2

        @test ah32 isa AhnHorensteinResult{Float32}
        @test ah32.mock_eigenvalue isa Float32
        @test eltype(ah32.eigenvalues) == Float32
        @test eltype(ah32.er) == Float32
        @test eltype(ah32.gr) == Float32
        @test ah32.r_er == 4 && ah32.r_gr == 4

        @test ed32 isa OnatskiEDResult{Float32}
        @test ed32.delta isa Float32
        @test eltype(ed32.eigenvalues) == Float32
        @test ed32.r_ed == 4

        # Int fallback
        X_int = round.(Int, 100 .* X64)
        @test onatski_q_test(X_int, 0, 5) isa OnatskiQTestResult{Float64}
        @test onatski_q_select(X_int, 5)[1] == 2
        @test ahn_horenstein(X_int, 6) isa AhnHorensteinResult{Float64}
        @test onatski_ed(X_int, 6) isa OnatskiEDResult{Float64}
    end

    @testset "StatsAPI, Display, Report, Reconstruction & Input Validation" begin
        rng = MersenneTwister(555)
        X = _test_sim_dynamic_panel(rng, 180, 50, 2; lag_max=1, sigma_e=0.9)

        qres = onatski_q_test(X, 0, 5)
        ahres = ahn_horenstein(X, 6)
        edres = onatski_ed(X, 6)

        # StatsAPI
        @test StatsAPI.pvalue(qres) == qres.pvalue
        @test StatsAPI.nobs(qres) == 180
        @test StatsAPI.dof(qres) == 5
        @test StatsAPI.nobs(ahres) == 180
        @test StatsAPI.nobs(edres) == 180

        # Show and report output
        for obj in (qres, ahres, edres)
            s = sprint(show, obj)
            @test !isempty(s)
            buf = IOBuffer()
            @test report(buf, obj) === nothing
            @test !isempty(String(take!(buf)))
        end

        # Positional serialization compatibility check (no Dict/Complex fields; positional constructor works)
        for obj in (qres, ahres, edres)
            Tobj = typeof(obj)
            fvals = [getfield(obj, f) for f in fieldnames(Tobj)]
            @test all(v -> !(v isa AbstractDict || v isa Complex), fvals)
            rebuilt = Tobj(fvals...)
            for f in fieldnames(Tobj)
                @test getfield(rebuilt, f) == getfield(obj, f)
            end
        end

        # Input validation
        X_nan = copy(X); X_nan[1, 1] = NaN
        @test_throws ArgumentError onatski_q_test(X_nan, 0, 4)
        @test_throws ArgumentError ahn_horenstein(X_nan, 4)
        @test_throws ArgumentError onatski_ed(X_nan, 4)

        @test_throws ArgumentError onatski_q_test(X, -1, 4)
        @test_throws ArgumentError onatski_q_test(X, 3, 3)
        @test_throws ArgumentError onatski_q_test(X, 0, size(X, 2))
        @test_throws ArgumentError onatski_q_test(X, 0, 4; bandwidth=1)
        @test_throws ArgumentError onatski_q_test(X, 0, 4; freq=0.0)
        @test_throws ArgumentError onatski_q_test(X, 0, 4; freq=Float64(pi))
        @test_throws ArgumentError onatski_q_test(X, 0, 4; n_blocks=0)
        @test_throws ArgumentError onatski_q_test(X, 0, 4; n_sim=10)
        @test_throws ArgumentError onatski_q_test(X, 0, 4; kernel=:invalid_kernel)

        @test_throws ArgumentError onatski_q_select(X; k_max=0)
        @test_throws ArgumentError onatski_q_select(X; k_max=size(X, 2))
        @test_throws ArgumentError onatski_q_select(X; alpha=0.0)
        @test_throws ArgumentError onatski_q_select(X; alpha=1.0)

        @test_throws ArgumentError ahn_horenstein(X, 0)
        @test_throws ArgumentError ahn_horenstein(X, size(X, 2))
        @test_throws ArgumentError onatski_ed(X, 0)
        @test_throws ArgumentError onatski_ed(X, size(X, 2))
        @test_throws ArgumentError onatski_ed(X, 4; max_iter=0)
        @test_throws ArgumentError onatski_q_test(randn(rng, 3, 20), 0, 4)
        @test_throws ArgumentError ahn_horenstein(randn(rng, 2, 20), 1)
    end
end
