# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

if !@isdefined(simulate_svar)
    include("id_dgps.jl")
end

const MEM = MacroEconometricModels

@testset "SID-13 AB-model ML" begin

    @testset "recursive_pattern structure and order condition" begin
        n = 3
        pat = recursive_pattern(n)
        @test pat isa SVARPattern
        @test size(pat.A) == (n, n)
        @test size(pat.B) == (n, n)
        @test diag(pat.A) == ones(n)
        @test istril(replace(pat.A, NaN => 0.0))
        for i in 1:n, j in i+1:n
            @test pat.A[i, j] == 0
        end
        st = check_identification(pat, n)
        @test st isa IdentificationStatus
        @test st.status === :exact
        @test st.n_overidentifying == 0
    end

    @testset "convenience constructors" begin
        n = 2
        A = [1.0 0.0; NaN 1.0]
        B = Matrix{Float64}(I, n, n)
        @test a_model_pattern(A).B ≈ I(n)
        @test b_model_pattern(B).A ≈ I(n)
        ab = ab_model_pattern(fill(NaN, n, n), fill(NaN, n, n))
        @test all(isnan, ab.A) && all(isnan, ab.B)
        bq = blanchard_quah_pattern(n)
        @test bq.A ≈ I(n)
        @test all(isnan, bq.B)
        @test bq.long_run !== nothing
        @test bq.long_run[1, 2] == 0
        @test isnan(bq.long_run[1, 1]) && isnan(bq.long_run[2, 1]) && isnan(bq.long_run[2, 2])
        st = check_identification(bq, n)
        @test st.status === :exact
    end

    @testset "recursive pattern reproduces Cholesky" begin
        rng = MersenneTwister(7421)
        Y, _ = simulate_svar([1.0 0.3; 0.4 1.0], [0.5 * Matrix{Float64}(I, 2, 2)];
                             Tobs=400, rng=rng)
        model = estimate_var(Y, 1)
        svar = estimate_svar(model, recursive_pattern(2); rng=MersenneTwister(74211))
        @test svar isa SVARModel
        @test svar.Q ≈ I(2) atol = 1e-6
        @test svar.lr_df == 0
        @test svar.lr_stat ≈ 0 atol = 1e-6
        @test svar.identification.status === :exact
        L = cholesky_factor(model)
        B0 = svar.A \ svar.B
        @test B0 ≈ L atol = 1e-5
        @test B0 * B0' ≈ model.Sigma atol = 1e-6
    end

    @testset "Blanchard–Quah long-run form ≈ identify_long_run" begin
        rng = MersenneTwister(7422)
        Y, _ = simulate_svar([1.2 0.2; 0.3 0.9], [0.4 * Matrix{Float64}(I, 2, 2)];
                             Tobs=500, rng=rng)
        model = estimate_var(Y, 1)
        svar = estimate_svar(model, blanchard_quah_pattern(2);
                             rng=MersenneTwister(74221), n_starts=3)
        Q_lr = identify_long_run(model)
        @test size(svar.Q) == (2, 2)
        @test svar.Q ≈ Q_lr atol = 1e-4
        @test svar.lr_df == 0
        @test svar.identification.status === :exact
        C1 = MEM._C1_from_B(model.B, 2, model.p)
        LR = C1 * (svar.A \ svar.B)
        @test abs(LR[1, 2]) < 1e-5
        @test LR[1, 1] > 0 && LR[2, 2] > 0
    end

    @testset "underidentified pattern throws IdentificationError" begin
        rng = MersenneTwister(7423)
        Y, _ = simulate_svar(Matrix{Float64}(I, 2, 2), [0.3 * Matrix{Float64}(I, 2, 2)];
                             Tobs=200, rng=rng)
        model = estimate_var(Y, 1)
        pat = ab_model_pattern(fill(NaN, 2, 2), Matrix{Float64}(I, 2, 2))
        st = check_identification(pat, 2)
        @test st.status === :under
        @test_throws IdentificationError estimate_svar(model, pat; rng=MersenneTwister(1))
    end

    @testset "registry flags and compute_Q" begin
        @test haskey(MEM.IDENTIFICATION_REGISTRY, :ab)
        @test !MEM._needs_residuals(:ab)
        @test !MEM._is_set_identified(:ab)
        @test !MEM._is_partial(:ab)
        rng = MersenneTwister(7424)
        Y, _ = simulate_svar([1.0 0.2; 0.3 1.0], [0.4 * Matrix{Float64}(I, 2, 2)];
                             Tobs=250, rng=rng)
        model = estimate_var(Y, 1)
        pat = recursive_pattern(2)
        Q = MEM.compute_Q(model, :ab; pattern=pat, rng=MersenneTwister(74241))
        @test Q ≈ I(2) atol = 1e-6
        @test_throws ArgumentError MEM.compute_Q(model, :ab)
    end

    @testset "irf/fevd/hd method=:ab" begin
        rng = MersenneTwister(7425)
        Y, _ = simulate_svar([1.0 0.25; 0.2 1.0], [0.45 * Matrix{Float64}(I, 2, 2)];
                             Tobs=250, rng=rng)
        model = estimate_var(Y, 1)
        pat = recursive_pattern(2)
        ir = irf(model, 6; method=:ab, pattern=pat, rng=MersenneTwister(74251))
        @test ir isa ImpulseResponse
        @test size(ir.values) == (6, 2, 2)
        @test ir.values[1, :, :] ≈ cholesky_factor(model) atol = 1e-5
        fv = fevd(model, 6; method=:ab, pattern=pat, rng=MersenneTwister(74252))
        @test fv isa FEVD
        hd = historical_decomposition(model, 20; method=:ab, pattern=pat,
                                      rng=MersenneTwister(74253))
        @test hd isa HistoricalDecomposition
    end

    @testset "report and refs" begin
        rng = MersenneTwister(7426)
        Y, _ = simulate_svar([1.0 0.2; 0.3 1.0], [0.4 * Matrix{Float64}(I, 2, 2)];
                             Tobs=200, rng=rng)
        model = estimate_var(Y, 1)
        svar = estimate_svar(model, recursive_pattern(2); rng=MersenneTwister(74261))
        buf = IOBuffer()
        report(buf, svar)
        txt = String(take!(buf))
        @test occursin("AB-Model", txt) || occursin("SVAR", txt)
        @test occursin("A[", txt) || occursin("B[", txt)
        rbuf = IOBuffer()
        refs(rbuf, svar)
        rtxt = String(take!(rbuf))
        @test occursin("Amisano", rtxt)
        @test occursin("Lütkepohl", rtxt) || occursin("Lutkepohl", rtxt)
        @test occursin("Sims", rtxt)
    end

    @testset "overidentified A-model LR" begin
        # Extra contemporaneous zero A[3,1]=0 on a recursive mixed pattern.
        n = 3
        A_true = [1.0 0.0 0.0; 0.4 1.0 0.0; 0.0 0.3 1.0]
        B_true = Diagonal([0.8, 1.1, 0.9])
        B0 = A_true \ Matrix(B_true)
        rng = MersenneTwister(7427)
        Y, _ = simulate_svar(Matrix(B0), [0.35 * Matrix{Float64}(I, n, n)];
                             Tobs=FAST ? 300 : 800, rng=rng)
        model = estimate_var(Y, 1)
        A_pat = [1.0 0.0 0.0; NaN 1.0 0.0; 0.0 NaN 1.0]
        B_pat = fill(0.0, n, n)
        B_pat[1, 1] = B_pat[2, 2] = B_pat[3, 3] = NaN
        pat = SVARPattern(A_pat, B_pat)
        st = check_identification(pat, n)
        @test st.status === :over
        @test st.n_overidentifying >= 1
        svar = estimate_svar(model, pat; rng=MersenneTwister(74271), n_starts=FAST ? 2 : 5)
        @test svar.lr_df == st.n_overidentifying || svar.lr_df >= 1
        @test svar.identification.status === :over
        @test svar.lr_pvalue > 0.01  # true restriction should not reject
        # False extra zero A[2,1]=0 (true value is 0.4) should reject at large T.
        if !FAST
            A_false = [1.0 0.0 0.0; 0.0 1.0 0.0; NaN NaN 1.0]
            pat_f = SVARPattern(A_false, B_pat)
            svar_f = estimate_svar(model, pat_f; rng=MersenneTwister(74272), n_starts=4)
            @test svar_f.lr_df >= 1
            @test svar_f.lr_pvalue < 0.05
        end
    end

    @testset "theoretical CI uses residual sample size" begin
        rng = MersenneTwister(7428)
        Y, _ = simulate_svar([1.0 0.35; 0.4 1.0], [0.5 * Matrix{Float64}(I, 2, 2)];
                             Tobs=220, rng=rng)
        model = estimate_var(Y, 1)
        A_pat = Matrix{Float64}(I, 2, 2)
        B_pat = fill(0.0, 2, 2)
        B_pat[1, 1] = B_pat[2, 2] = NaN
        pat = SVARPattern(A_pat, B_pat)
        @test check_identification(pat, 2).status === :over
        @test MEM._ab_nobs(model) == size(model.U, 1)
        T = eltype(model.Sigma)
        n = nvars(model)
        m_emptyU = VARModel(model.Y, model.p, model.B, zeros(T, 0, n),
                            model.Sigma, model.aic, model.bic, model.hqic,
                            model.varnames)
        @test MEM._ab_nobs(m_emptyU) == size(model.Y, 1) - model.p
        @test MEM._ab_nobs(m_emptyU) == effective_nobs(model)
        svar_full = estimate_svar(model, pat; n_starts=2, rng=MersenneTwister(74281))
        svar_empty = estimate_svar(m_emptyU, pat; n_starts=2, rng=MersenneTwister(74281))
        @test svar_empty.Q ≈ svar_full.Q atol = 1e-5
        L = cholesky_factor(model)
        B0 = svar_full.A \ svar_full.B
        @test B0 ≉ L atol = 1e-3
        ir = irf(model, 4; method=:ab, pattern=pat, ci_type=:theoretical,
                 reps=FAST ? 8 : 16, n_starts=1, rng=MersenneTwister(74282))
        @test ir.ci_type === :theoretical
        @test ir.values[1, :, :] ≉ L atol = 1e-3
        @test all(ir.ci_lower .<= ir.values .+ sqrt(eps(T)))
        @test all(ir.values .<= ir.ci_upper .+ sqrt(eps(T)))
    end

    @testset "non-BQ long-run throws ArgumentError" begin
        rng = MersenneTwister(7429)
        Y, _ = simulate_svar([1.0 0.2; 0.3 1.0], [0.4 * Matrix{Float64}(I, 2, 2)];
                             Tobs=180, rng=rng)
        model = estimate_var(Y, 1)
        A_mix = [1.0 0.0; NaN 1.0]
        B_mix = fill(0.0, 2, 2)
        B_mix[1, 1] = B_mix[2, 2] = NaN
        lr_mix = fill(NaN, 2, 2)
        lr_mix[1, 2] = 0.0
        pat_mix = SVARPattern(A_mix, B_mix; long_run=lr_mix)
        err = nothing
        try
            check_identification(pat_mix, 2)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("Blanchard", sprint(showerror, err))
        @test_throws ArgumentError check_identification(pat_mix, model)
        @test_throws ArgumentError estimate_svar(model, pat_mix; rng=MersenneTwister(1))
        @test_throws ArgumentError irf(model, 4; method=:ab, pattern=pat_mix)
        svar_bq = estimate_svar(model, blanchard_quah_pattern(2);
                                rng=MersenneTwister(74291), n_starts=2)
        @test svar_bq.identification.status === :exact
        @test svar_bq.Q ≈ identify_long_run(model) atol = 1e-4
    end

    if !FAST
        @testset "overidentified A-model size (gated FAST)" begin
            n = 3
            A_true = [1.0 0.0 0.0; 0.35 1.0 0.0; 0.0 0.25 1.0]
            B_true = Diagonal([1.0, 0.8, 1.2])
            B0 = A_true \ Matrix(B_true)
            A_pat = [1.0 0.0 0.0; NaN 1.0 0.0; 0.0 NaN 1.0]
            B_pat = fill(0.0, n, n)
            for i in 1:n
                B_pat[i, i] = NaN
            end
            pat = SVARPattern(A_pat, B_pat)
            pvals = Float64[]
            for s in 1:12
                rng = MersenneTwister(74300 + s)
                Y, _ = simulate_svar(Matrix(B0), [0.3 * Matrix{Float64}(I, n, n)];
                                     Tobs=600, rng=rng)
                model = estimate_var(Y, 1)
                svar = estimate_svar(model, pat; rng=MersenneTwister(74350 + s), n_starts=3)
                push!(pvals, svar.lr_pvalue)
            end
            rej = mean(pvals .< 0.05)
            @test 0.0 <= rej <= 0.35
            @test mean(pvals) > 0.2
        end
    end
end
