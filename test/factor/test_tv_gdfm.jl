# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random
using StatsAPI

@testset "Time-Varying Locally Stationary GDFM Tests (#898)" begin

    # =========================================================================
    # 1. Time-Invariant DGP Agreement with Static-Spectrum GDFM
    # =========================================================================
    @testset "Time-invariant DGP agrees with estimate_gdfm" begin
        rng = Random.Xoshiro(89801)
        T_obs, N, q = 250, 20, 2

        # Dynamic factor DGP with lag structure: B_i(u, L) ≡ B_i(L) = Λ₀ + Λ₁ L
        burn = 50
        T_tot = T_obs + burn + 1
        U = zeros(T_tot, q)
        Phi = [0.55 0.10; 0.05 0.45]
        for t in 2:T_tot
            U[t, :] = Phi * U[t - 1, :] + randn(rng, q)
        end
        Lam0 = randn(rng, N, q)
        Lam1 = 0.5 .* randn(rng, N, q)
        chi_true = U[(burn + 2):T_tot, :] * Lam0' + U[(burn + 1):(T_tot - 1), :] * Lam1'
        X = chi_true + 0.35 .* randn(rng, T_obs, N)

        g = estimate_gdfm(X, q)
        tv = estimate_tv_gdfm(X, q; time_bandwidth=0.5)

        @test tv isa TimeVaryingGDFM{Float64}
        @test tv isa AbstractFactorModel
        @test cor(vec(tv.common_component), vec(g.common_component)) > 0.96
        @test maximum(abs, tv.common_component .+ tv.idiosyncratic .- X) < 1e-12

        # Unstandardized agreement as well
        g_u = estimate_gdfm(X, q; standardize=false)
        tv_u = estimate_tv_gdfm(X, q; time_bandwidth=0.5, standardize=false)
        @test cor(vec(tv_u.common_component), vec(g_u.common_component)) > 0.96
    end

    # =========================================================================
    # 2. Smoothly Time-Varying Loadings & Variance Share Path Recovery
    # =========================================================================
    @testset "Time-varying DGP variance share path recovery" begin
        rng = Random.Xoshiro(89802)
        T_obs, N, q = 400, 30, 1

        # Stationary unit-variance AR(1) factor
        burn = 100
        T_tot = T_obs + burn
        phi = 0.5
        f = zeros(T_tot)
        innov_sd = sqrt(1.0 - phi^2)
        for t in 2:T_tot
            f[t] = phi * f[t - 1] + innov_sd * randn(rng)
        end
        f_sub = f[(burn + 1):T_tot]

        # Loading magnitude doubles smoothly from u = 0 to u = 1: scale(u) = 0.9 * (1 + u)
        # Alternating ±1 base loadings keep per-series variance identical so standardized and
        # unstandardized analytical common-variance shares coincide cleanly.
        base_signs = [isodd(i) ? 1.0 : -1.0 for i in 1:N]
        c0 = 0.9
        sigma_xi = 1.0
        u_obs = [(t - 0.5) / T_obs for t in 1:T_obs]

        X = Matrix{Float64}(undef, T_obs, N)
        true_share = Vector{Float64}(undef, T_obs)
        for t in 1:T_obs
            u = u_obs[t]
            mag = c0 * (1.0 + u)
            v_chi = mag^2
            true_share[t] = v_chi / (v_chi + sigma_xi^2)
            for i in 1:N
                X[t, i] = mag * base_signs[i] * f_sub[t] + sigma_xi * randn(rng)
            end
        end

        tv = estimate_tv_gdfm(X, q)
        share_est = vec(sum(tv.variance_share_path; dims=2))

        @test size(tv.variance_share_path) == (T_obs, q)
        @test all(0.0 .<= share_est .<= 1.0)
        @test issorted(share_est)
        @test share_est[end] > share_est[1] + 0.15

        rmse = sqrt(mean(abs2, share_est .- true_share))
        @test rmse < 0.08

        # Also check q = 2 smoothly doubling loadings with standardize=false
        q2, N2, T2 = 2, 30, 400
        X2, true_share2 = let rng = Random.Xoshiro(89803)
            F2 = zeros(T2 + burn, q2)
            for t in 2:(T2 + burn), j in 1:q2
                F2[t, j] = 0.4 * F2[t - 1, j] + sqrt(1.0 - 0.16) * randn(rng)
            end
            F2_sub = F2[(burn + 1):end, :]
            L0 = randn(rng, N2, q2)
            L0 ./= sqrt.(sum(abs2, L0; dims=2))
            X2_loc = Matrix{Float64}(undef, T2, N2)
            ts2 = Vector{Float64}(undef, T2)
            for t in 1:T2
                u = (t - 0.5) / T2
                mag = 0.9 * (1.0 + u)
                ts2[t] = mag^2 / (mag^2 + 1.0)
                for i in 1:N2
                    X2_loc[t, i] = mag * (L0[i, 1] * F2_sub[t, 1] + L0[i, 2] * F2_sub[t, 2]) + randn(rng)
                end
            end
            X2_loc, ts2
        end
        tv2 = estimate_tv_gdfm(X2, q2; standardize=false)
        share_est2 = vec(sum(tv2.variance_share_path; dims=2))
        @test issorted(share_est2)
        @test sqrt(mean(abs2, share_est2 .- true_share2)) < 0.08

        # Rotating factor loading space: TV-GDFM (q=1) tracks orthogonal loading rotation
        # much better than time-invariant GDFM (q=1)
        T3, N3 = 300, 24
        X3, chi3_true = let rng = Random.Xoshiro(89806)
            f3 = zeros(T3 + burn)
            for t in 2:(T3 + burn)
                f3[t] = 0.6 * f3[t - 1] + sqrt(1.0 - 0.36) * randn(rng)
            end
            f3_sub = f3[(burn + 1):end]
            a_vec = [i <= div(N3, 2) ? 1.5 : 0.0 for i in 1:N3]
            b_vec = [i > div(N3, 2) ? 1.5 : 0.0 for i in 1:N3]
            c3 = Matrix{Float64}(undef, T3, N3)
            X3_loc = Matrix{Float64}(undef, T3, N3)
            for t in 1:T3
                angle = (π / 2) * ((t - 0.5) / T3)
                lam_t = cos(angle) .* a_vec .+ sin(angle) .* b_vec
                for i in 1:N3
                    c3[t, i] = lam_t[i] * f3_sub[t]
                    X3_loc[t, i] = c3[t, i] + 0.3 * randn(rng)
                end
            end
            X3_loc, c3
        end
        tv_rot = estimate_tv_gdfm(X3, 1; standardize=false)
        g_rot = estimate_gdfm(X3, 1; standardize=false)
        @test cor(vec(tv_rot.common_component), vec(chi3_true)) > 0.90
        @test sqrt(mean(abs2, tv_rot.common_component .- chi3_true)) <
              0.7 * sqrt(mean(abs2, g_rot.common_component .- chi3_true))
    end

    # =========================================================================
    # 3. Kernels, q = N Identity, Standardization & Edge Cases
    # =========================================================================
    @testset "Kernels, q = N reconstruction, and edge cases" begin
        rng = Random.Xoshiro(89804)
        T_obs, N, q = 100, 8, 2
        X = randn(rng, T_obs, N) .+ 5.0

        for tk in (:epanechnikov, :gaussian, :uniform), lk in (:bartlett, :parzen, :tukey)
            tv = estimate_tv_gdfm(X, q; time_kernel=tk, lag_kernel=lk, n_grid=12)
            @test tv.time_kernel === tk
            @test tv.lag_kernel === lk
            @test tv.n_grid == 12
            @test size(tv.eigenvalues_path) == (12, N, length(tv.frequencies))
            @test tv.common_component + tv.idiosyncratic ≈ X atol=1e-10
            @test all(isfinite, tv.common_component)
            @test all(isfinite, tv.variance_share_path)
        end

        # Exact full-rank q = N identity (projector P = I_N -> chi = X)
        for std_flag in (true, false)
            tv_full = estimate_tv_gdfm(X, N; standardize=std_flag, n_grid=8)
            @test maximum(abs, tv_full.common_component .- X) < 1e-10
            @test maximum(abs, tv_full.idiosyncratic) < 1e-10
            @test all(x -> abs(x - 1.0) < 1e-10, sum(tv_full.variance_share_path; dims=2))
        end

        # n_grid = 1 and narrow compact time_bandwidth edge fallback
        tv_g1 = estimate_tv_gdfm(X, 2; n_grid=1, time_bandwidth=1e-4, time_kernel=:uniform)
        @test tv_g1.n_grid == 1
        @test all(isfinite, tv_g1.common_component)
        @test all(isfinite, tv_g1.variance_share_path)

        # SubArray view and custom SubString varnames
        vnames = split("a b c d e f g h")
        tv_sub = estimate_tv_gdfm(@view(X[1:80, :]), 2; varnames=vnames, lag_bandwidth=4)
        @test varnames(tv_sub) == String.(vnames)
        @test tv_sub.lag_bandwidth == 4
    end

    # =========================================================================
    # 4. Precision Support (Float32 & Integer Fallback), RSER-01, StatsAPI,
    #    Show/Report, Error Validation & Method Ambiguities
    # =========================================================================
    @testset "Float32, Int fallback, RSER-01, StatsAPI, show/report & validation" begin
        rng = Random.Xoshiro(89805)
        X64 = dgp_dynamic_factors(rng; N=10, T=80, idio_sd=0.4).X
        names = ["Series_$i" for i in 1:10]
        tv = estimate_tv_gdfm(X64, 2; varnames=names)

        # StatsAPI and accessors
        @test StatsAPI.predict(tv) == tv.common_component
        @test StatsAPI.fitted(tv) == tv.common_component
        @test StatsAPI.residuals(tv) == tv.idiosyncratic
        @test StatsAPI.nobs(tv) == 80
        @test StatsAPI.dof(tv) > 0
        @test length(StatsAPI.r2(tv)) == 10
        @test all(x -> -1e-12 <= x <= 1.0 + 1e-12, StatsAPI.r2(tv))
        @test nvars(tv) == 10
        @test varnames(tv) == names
        @test common_variance_share(tv) === tv.variance_share_path
        @test size(common_variance_share(tv)) == (80, 2)

        # No manifest field
        @test !(:manifest in fieldnames(TimeVaryingGDFM))

        # RSER-01 positional constructor and _generic_construct invariant
        fnames = fieldnames(TimeVaryingGDFM)
        vals = ntuple(i -> getfield(tv, fnames[i]), length(fnames))
        @test all(v -> !(v isa AbstractDict || v isa Complex), vals)
        tv_rebuilt = TimeVaryingGDFM{Float64}(vals...)
        @test tv_rebuilt isa TimeVaryingGDFM{Float64}
        for f in fnames
            @test getfield(tv_rebuilt, f) == getfield(tv, f)
        end
        tv_generic = MacroEconometricModels._generic_construct(TimeVaryingGDFM, collect(vals))
        @test tv_generic isa TimeVaryingGDFM{Float64}
        @test tv_generic.common_component == tv.common_component

        # Float32 precision preservation
        X32 = Float32.(X64)
        tv32 = estimate_tv_gdfm(X32, 2)
        @test tv32 isa TimeVaryingGDFM{Float32}
        @test eltype(tv32.X) === Float32
        @test eltype(tv32.common_component) === Float32
        @test eltype(tv32.idiosyncratic) === Float32
        @test eltype(tv32.eigenvalues_path) === Float32
        @test eltype(tv32.frequencies) === Float32
        @test eltype(tv32.u_grid) === Float32
        @test eltype(tv32.variance_share_path) === Float32
        @test tv32.time_bandwidth isa Float32

        # Generic constructor with Float32
        vals32 = Any[getfield(tv32, f) for f in fnames]
        tv32_gen = MacroEconometricModels._generic_construct(TimeVaryingGDFM, vals32)
        @test tv32_gen isa TimeVaryingGDFM{Float32}

        # AbstractMatrix{<:Real} Int fallback
        X_int = round.(Int, 100 .* X64)
        tv_int = estimate_tv_gdfm(X_int, 2)
        @test tv_int isa TimeVaryingGDFM{Float64}

        # Base.show and report across display backends
        txt = sprint(show, tv)
        @test occursin("Time-Varying Generalized Dynamic Factor Model", txt)
        @test occursin("Common Variance Share Path", txt)
        buf = IOBuffer()
        @test report(buf, tv) === nothing
        @test !isempty(String(take!(buf)))
        redirect_stdout(devnull) do
            @test report(tv) === nothing
        end
        for bk in (:text, :latex, :html)
            s_bk = with_display_backend(bk) do
                sprint(show, tv)
            end
            @test !isempty(s_bk)
        end

        # ArgumentError validation checks
        @test_throws ArgumentError estimate_tv_gdfm(X64, 0)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 11)
        X_short_tv = randn(rng, 3, 10)
        @test_throws ArgumentError estimate_tv_gdfm(X_short_tv, 2)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; time_bandwidth=-0.1)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; time_bandwidth=Inf)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; lag_bandwidth=-1)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; n_grid=0)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; time_kernel=:bad)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; lag_kernel=:bad)
        @test_throws ArgumentError estimate_tv_gdfm(X64, 2; varnames=["only_one"])

        X_nan = copy(X64); X_nan[4, 2] = NaN
        @test_throws ArgumentError estimate_tv_gdfm(X_nan, 2)
        X_inf = copy(X64); X_inf[4, 2] = Inf
        @test_throws ArgumentError estimate_tv_gdfm(X_inf, 2)
    end
end
