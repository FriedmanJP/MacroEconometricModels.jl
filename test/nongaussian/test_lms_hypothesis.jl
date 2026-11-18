# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics
using Distributions
using StatsAPI
using ForwardDiff

const MEM = MacroEconometricModels
const _suppress_warnings = MEM._suppress_warnings

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

function _sim_lms_var1_t(rng::AbstractRNG, T_obs::Int;
                         A1::Matrix{Float64}=[0.45 0.08 -0.05; -0.06 0.40 0.07; 0.04 -0.08 0.35],
                         c::Vector{Float64}=[0.1, -0.05, 0.08],
                         B::Matrix{Float64}=[1.0 0.0 0.0; -0.35 1.0 0.0; 0.22 -0.30 1.0],
                         sigma::Vector{Float64}=[1.1, 0.9, 1.25],
                         nu::Float64=5.0,
                         burn::Int=200)
    n = length(c)
    B0 = B * Diagonal(sigma)
    N = T_obs + burn
    Z = if isinf(nu)
        randn(rng, N, n)
    else
        rand(rng, TDist(nu), N, n) ./ sqrt(nu / (nu - 2.0))
    end
    Y = zeros(N, n)
    @inbounds for t in 2:N
        u = B0 * Z[t, :]
        Y[t, :] .= c .+ A1 * Y[t - 1, :] .+ u
    end
    keep = (burn + 1):N
    (Y=Y[keep, :], B=B, sigma=sigma, B0=B0, A1=A1, c=c, Z=Z[keep, :])
end

@testset "LMS (2017) Structural Hypothesis Testing & Diagnostics (#903)" begin

    @testset "1. Hypothesis Builders, LMS B ∈ B Diagonal Guard & Ordering" begin
        # Recursiveness
        h_rec = recursiveness_hypothesis(3)
        @test h_rec.df == 3
        @test h_rec.kind == :recursiveness
        @test h_rec.ordering == [1, 2, 3]
        @test size(h_rec.R_beta) == (3, 6)

        h_rec_perm = recursiveness_hypothesis(3; ordering=[3, 1, 2])
        @test h_rec_perm.df == 3
        @test h_rec_perm.ordering == [3, 1, 2]
        @test_throws ArgumentError recursiveness_hypothesis(1)
        @test_throws ArgumentError recursiveness_hypothesis(3; ordering=[1, 1, 2])
        @test_throws ArgumentError recursiveness_hypothesis(3; ordering=[1, 2])
        @test_throws ArgumentError recursiveness_hypothesis(3; ordering=:unknown_ord)

        # Impact zero & LMS B ∈ B diagonal interiority guard
        h_iz = impact_zero_hypothesis(4, [(1, 4), (2, 4), (3, 4)])
        @test h_iz.df == 3
        @test h_iz.kind == :impact_zero

        err_direct = try
            impact_zero_hypothesis(3, [(2, 2)])
            nothing
        catch e
            e
        end
        @test err_direct isa ArgumentError
        @test occursin("LMS", err_direct.msg)
        @test occursin("ordering", err_direct.msg)

        # Permutation-induced diagonal violation: (1, 2) under ordering=[2, 1, 3] hits B[1, 1]
        err_perm = try
            impact_zero_hypothesis(3, [(1, 2)]; ordering=[2, 1, 3])
            nothing
        catch e
            e
        end
        @test err_perm isa ArgumentError
        @test occursin("B ∈ B", err_perm.msg)

        # Valid off-diagonal via permutation: (2, 2) under ordering=[2, 1, 3] maps to B[2, 1]
        h_iz_perm = impact_zero_hypothesis(3, [(2, 2)]; ordering=[2, 1, 3])
        @test h_iz_perm.df == 1
        @test h_iz_perm.R_beta[1, MEM._lmshyp_beta_index(2, 1, 3)] == 1.0

        @test_throws ArgumentError impact_zero_hypothesis(3, Tuple{Int,Int}[])
        @test_throws ArgumentError impact_zero_hypothesis(3, [(1, 4)])
        @test_throws ArgumentError impact_zero_hypothesis(3, [(1, 2), (1, 2)])

        # Impact linear
        R_lin = zeros(2, 6)
        R_lin[1, 1] = 1.0; R_lin[1, 3] = -1.0  # B[2,1] - B[1,2] = 0
        R_lin[2, 5] = 1.0                       # B[1,3] = 0.2
        h_lin = impact_linear_hypothesis(R_lin, [0.0, 0.2])
        @test h_lin.df == 2
        @test h_lin.kind == :impact_linear
        @test_throws ArgumentError impact_linear_hypothesis(zeros(1, 9))     # n^2 rejected
        @test_throws ArgumentError impact_linear_hypothesis([1.0 0.0; 1.0 0.0], [0.0, 0.0]) # rank-deficient
        @test_throws ArgumentError impact_linear_hypothesis(R_lin, [0.0])   # length mismatch
        # Permutation guard on linear: column 3 in n=3 is (1, 2), which under ordering=[2, 1, 3] hits B[1, 1]
        @test_throws ArgumentError impact_linear_hypothesis(R_lin, [0.0, 0.2]; ordering=[2, 1, 3])

        # Long-run
        h_lr = long_run_hypothesis(3, [(1, 2), (1, 3), (2, 3)])
        @test h_lr.df == 3
        @test h_lr.kind == :long_run
        @test_throws ArgumentError long_run_hypothesis(1, [(1, 1)])
        @test_throws ArgumentError long_run_hypothesis(3, [(1, 1), (1, 2), (1, 3)]) # full row 0
        @test_throws ArgumentError long_run_hypothesis(3, [(1, 2), (2, 2), (3, 2)]) # full col 0

        # Horizon IRF
        h_hirf = horizon_irf_hypothesis(3, [(0, 1, 2), (1, 1, 2), (2, 2, 2)])
        @test h_hirf.df == 3
        @test h_hirf.kind == :horizon_irf
        @test_throws ArgumentError horizon_irf_hypothesis(3, [(0, 2, 2)])
        @test_throws ArgumentError horizon_irf_hypothesis(3, [(0, 1, 2)]; ordering=[2, 1, 3])
        @test_throws ArgumentError horizon_irf_hypothesis(3, [(-1, 1, 2)])
        # h >= 1 allows diagonal entry (1, 1, 1)
        @test horizon_irf_hypothesis(3, [(1, 1, 1)]).df == 1

        # Mixed SVAR hypothesis + rank checks
        h_mix = mixed_svar_hypothesis([
            impact_zero_hypothesis(3, [(1, 2)]),
            long_run_hypothesis(3, [(1, 2)]),
            horizon_irf_hypothesis(3, [(2, 2, 3)]),
        ])
        @test h_mix.df == 3
        @test h_mix.kind == :mixed

        h_mix_kw = mixed_svar_hypothesis(
            short_run = impact_zero_hypothesis(3, [(1, 3)]),
            long_run  = long_run_hypothesis(3, [(1, 2)]),
        )
        @test h_mix_kw.df == 2

        @test_throws ArgumentError mixed_svar_hypothesis(MEM._SVARHypothesisSpec[])
        @test_throws ArgumentError mixed_svar_hypothesis([
            recursiveness_hypothesis(3),
            impact_zero_hypothesis(3, [(1, 2)]),
        ])  # redundant linear
        @test_throws ArgumentError mixed_svar_hypothesis([
            impact_zero_hypothesis(3, [(1, 2)]),
            impact_linear_hypothesis([0.0 0.0 1.0 0.0 0.0 0.0], [0.5]),
        ])  # inconsistent B[1,2]=0 and B[1,2]=0.5
        @test_throws ArgumentError mixed_svar_hypothesis([
            long_run_hypothesis(3, [(1, 2)]),
            long_run_hypothesis(3, [(1, 2)]),
        ])  # duplicate long-run Jacobian rank < df
        @test_throws ArgumentError mixed_svar_hypothesis([
            impact_zero_hypothesis(2, [(1, 2)]),
            long_run_hypothesis(3, [(1, 2)]),
        ])  # mismatched n

        # Nonlinear
        h_nl = nonlinear_svar_hypothesis(th -> th[1]^2 - 1.0, 1; description="Custom test")
        @test h_nl.df == 1
        @test_throws ArgumentError nonlinear_svar_hypothesis(x -> x, 0)
    end

    @testset "2. Large-Sample Wald / LR / LM Equivalence (H₀ and Local H₁)" begin
        _suppress_warnings() do
            # (a) Under H₀: recursive lower-triangular B, large T = 1500
            sim0 = _sim_lms_var1_t(Xoshiro(90301), 1500)
            m0 = estimate_var(sim0.Y, 1)
            r3 = estimate_lms_svar(m0; distributions=:student_t, method=:three_step, seed=90302)
            r1 = estimate_lms_svar(m0; distributions=:student_t, method=:full_ml, seed=90302)

            h_rec = recursiveness_hypothesis(3)
            w0 = test_svar_hypothesis(r1, h_rec; method=:wald)
            l0 = test_svar_hypothesis(r1, h_rec; method=:lr)
            s0 = test_svar_hypothesis(r1, h_rec; method=:lm)

            @test w0 isa SVARHypothesisTest{Float64}
            @test l0 isa SVARHypothesisTest{Float64}
            @test s0 isa SVARHypothesisTest{Float64}
            @test StatsAPI.dof(w0) == 3
            @test StatsAPI.pvalue(w0) == w0.pvalue
            @test StatsAPI.nobs(w0) == size(m0.U, 1)
            @test w0.statistic >= 0 && l0.statistic >= 0 && s0.statistic >= 0
            @test w0.pvalue > 0.05 && l0.pvalue > 0.05 && s0.pvalue > 0.05
            @test l0.loglik_unrestricted >= l0.loglik_restricted - 1e-8
            @test maximum(abs, l0.g_value) < 1e-10
            @test maximum(abs, s0.g_value) < 1e-10
            @test isapprox(w0.statistic, l0.statistic; atol=1.0, rtol=0.25)
            @test isapprox(s0.statistic, l0.statistic; atol=1.0, rtol=0.25)

            # Three-step LR also valid & agrees closely
            l0_3s = test_svar_hypothesis(r3, h_rec; method=:lr)
            s0_3s = test_svar_hypothesis(r3, h_rec; method=:lm)
            @test l0_3s.pvalue > 0.05 && s0_3s.pvalue > 0.05
            @test isapprox(l0_3s.statistic, l0.statistic; atol=0.5)

            # Under-identifying partial restrictions: single zero B[2,3]=0 & linear contrast
            h_single = impact_zero_hypothesis(3, [(2, 3)])
            for meth in (:wald, :lr, :lm)
                tr = test_svar_hypothesis(r1, h_single; method=meth)
                @test tr.df == 1
                @test tr.pvalue > 0.05
            end
            # True linear contrast: B[2,1] - B[3,2] = -0.35 - (-0.30) = -0.05
            R_c = zeros(1, 6)
            R_c[1, MEM._lmshyp_beta_index(2, 1, 3)] = 1.0
            R_c[1, MEM._lmshyp_beta_index(3, 2, 3)] = -1.0
            h_lin = impact_linear_hypothesis(R_c, [-0.05])
            for meth in (:wald, :lr, :lm)
                tr = test_svar_hypothesis(r1, h_lin; method=meth)
                @test tr.pvalue > 0.05
            end

            # Covariance type options on Wald & LM
            for vt in (:hessian, :sandwich, :opg)
                wt = test_svar_hypothesis(r1, h_rec; method=:wald, vcov_type=vt)
                lt = test_svar_hypothesis(r1, h_rec; method=:lm, vcov_type=vt)
                @test wt.vcov_type == vt && lt.vcov_type == vt
                @test isfinite(wt.statistic) && isfinite(lt.statistic)
            end

            # (b) Local alternative H₁: B[1, 3] = 0.12 (asymptotic equivalence at T = 2500)
            B_loc = [1.0 0.0 0.12; -0.35 1.0 0.0; 0.22 -0.30 1.0]
            sim1 = _sim_lms_var1_t(Xoshiro(90305), 2500; B=B_loc)
            r1_loc = estimate_lms_svar(sim1.Y, 1; distributions=:student_t, method=:full_ml, seed=90306)
            w1 = test_svar_hypothesis(r1_loc, h_rec; method=:wald)
            l1 = test_svar_hypothesis(r1_loc, h_rec; method=:lr)
            s1 = test_svar_hypothesis(r1_loc, h_rec; method=:lm)
            @test w1.pvalue < 0.001 && l1.pvalue < 0.001 && s1.pvalue < 0.001
            @test isapprox(w1.statistic, l1.statistic; rtol=0.25)
            @test isapprox(s1.statistic, l1.statistic; rtol=0.25)
        end
    end

    @testset "3. Blanchard–Quah Long-Run, Horizon IRF, Mixed, Nonlinear & Ordering Permutation" begin
        _suppress_warnings() do
            # Construct VAR(1) where Θ_∞ = (I - A1)⁻¹ B is lower-triangular, while B is NON-triangular!
            B_nontri = [1.0 0.35 -0.20; -0.25 1.0 0.25; 0.15 -0.30 1.0]
            Theta_LR_true = [1.6 0.0 0.0; -0.3 1.5 0.0; 0.2 -0.25 1.4]
            A1_bq = Matrix{Float64}(I, 3, 3) - B_nontri * MEM.robust_inv(Theta_LR_true)
            @test maximum(abs.(eigvals(A1_bq))) < 0.85

            sim_bq = _sim_lms_var1_t(Xoshiro(90310), 1200; A1=A1_bq, B=B_nontri)
            r_bq = estimate_lms_svar(sim_bq.Y, 1; distributions=:student_t, method=:full_ml, seed=90311)

            # Verify joint (π, β) Jacobian has nonzero weights on BOTH π and β
            h_bq1 = long_run_hypothesis(3, [(1, 2)])
            g_bq = MEM._lmshyp_make_g_fn(h_bq1, 3, 1, [1, 2, 3])
            J_bq = ForwardDiff.jacobian(g_bq, r_bq.theta)
            n_pi = 3 * (1 + 3 * 1)
            n_beta = 6
            @test norm(J_bq[:, 1:n_pi]) > 0.1
            @test norm(J_bq[:, (n_pi + 1):(n_pi + n_beta)]) > 0.1

            # True BQ restriction Θ_∞[1, 2] = 0 passes across Wald, LR, LM
            for meth in (:wald, :lr, :lm)
                tbq = test_svar_hypothesis(r_bq, h_bq1; method=meth)
                @test tbq.pvalue > 0.05
                if meth in (:lr, :lm)
                    @test maximum(abs, tbq.g_value) < 1e-6
                end
            end

            # Impact B[1, 2] = 0.35 ≠ 0 is strongly rejected
            @test test_svar_hypothesis(r_bq, impact_zero_hypothesis(3, [(1, 2)]); method=:wald).pvalue < 0.01
            # False BQ restriction Θ_∞[2, 1] = -0.3 ≠ 0 is strongly rejected
            @test test_svar_hypothesis(r_bq, long_run_hypothesis(3, [(2, 1)]); method=:lr).pvalue < 0.01

            # Horizon IRF restriction: (A1 * B)[1, 2] = 0 while B[1, 2] = 0.40 ≠ 0
            B_h = [1.0 0.40 -0.15; -0.25 1.0 0.20; 0.15 -0.20 1.0]
            # Row 1 of A1 orthogonal to B_h[:, 2] = [0.40, 1.0, -0.20]: [0.50, -0.20, 0.0]
            A1_h = [0.50 -0.20 0.0; -0.05 0.42 0.08; 0.04 -0.06 0.38]
            @test abs((A1_h * B_h)[1, 2]) < 1e-14
            @test maximum(abs.(eigvals(A1_h))) < 0.7
            sim_h = _sim_lms_var1_t(Xoshiro(90315), 1000; A1=A1_h, B=B_h)
            r_h = estimate_lms_svar(sim_h.Y, 1; distributions=:student_t, method=:full_ml, seed=90316)

            h_irf1 = horizon_irf_hypothesis(3, [(1, 1, 2)])
            for meth in (:wald, :lr, :lm)
                th = test_svar_hypothesis(r_h, h_irf1; method=meth)
                @test th.pvalue > 0.05
            end
            # False horizon IRF (1, 1, 1) rejected
            @test test_svar_hypothesis(r_h, horizon_irf_hypothesis(3, [(1, 1, 1)]); method=:wald).pvalue < 0.001

            # Mixed SVAR hypothesis: impact B[1, 3] = 0 + long-run Θ_∞[1, 2] = 0
            # Build exact DGP with B[1, 3] = 0 AND Θ_∞[1, 2] = 0:
            B_mix = [1.0 0.30 0.0; -0.25 1.0 0.25; 0.15 -0.25 1.0]
            Theta_mix = [1.5 0.0 0.2; -0.2 1.4 0.1; 0.15 -0.2 1.3]
            A1_mix = Matrix{Float64}(I, 3, 3) - B_mix * MEM.robust_inv(Theta_mix)
            @test maximum(abs.(eigvals(A1_mix))) < 0.8
            sim_mix = _sim_lms_var1_t(Xoshiro(90320), 1000; A1=A1_mix, B=B_mix)
            r_mix = estimate_lms_svar(sim_mix.Y, 1; distributions=:student_t, method=:full_ml, seed=90321)

            h_mixed = mixed_svar_hypothesis(
                short_run = impact_zero_hypothesis(3, [(1, 3)]),
                long_run  = long_run_hypothesis(3, [(1, 2)]),
            )
            for meth in (:wald, :lr, :lm)
                tm = test_svar_hypothesis(r_mix, h_mixed; method=meth)
                @test tm.df == 2
                @test tm.pvalue > 0.05
            end

            # Nonlinear SVAR hypothesis: custom 1-arg g(theta) and 2-arg g(pi, beta) matching BQ (1, 2)
            g1_bq = th -> begin
                PiT = reshape(@view(th[1:12]), 3, 4)
                A1m = @view PiT[:, 2:4]
                Bm = MEM._lms_B_from_beta(@view(th[13:18]), 3)
                ThLR = MEM.robust_inv(Matrix{eltype(th)}(I, 3, 3) - A1m; silent=true) * Bm
                ThLR[1, 2]
            end
            g2_bq = (pi_v, beta_v) -> begin
                PiT = reshape(pi_v, 3, 4)
                A1m = @view PiT[:, 2:4]
                Bm = MEM._lms_B_from_beta(beta_v, 3)
                ThLR = MEM.robust_inv(Matrix{eltype(pi_v)}(I, 3, 3) - A1m; silent=true) * Bm
                [ThLR[1, 2]]
            end
            w_nl1 = test_svar_hypothesis(r_bq, nonlinear_svar_hypothesis(g1_bq, 1); method=:wald)
            w_nl2 = test_svar_hypothesis(r_bq, nonlinear_svar_hypothesis(g2_bq, 1); method=:wald)
            w_ref = test_svar_hypothesis(r_bq, h_bq1; method=:wald)
            @test w_nl1.statistic ≈ w_ref.statistic atol = 1e-6
            @test w_nl2.statistic ≈ w_ref.statistic atol = 1e-6
            l_nl2 = test_svar_hypothesis(r_bq, nonlinear_svar_hypothesis(g2_bq, 1); method=:lr)
            @test l_nl2.pvalue > 0.05

            # Ordering permutation test: upper-triangular B (recursive under ordering [3, 2, 1])
            B_upper = [1.0 -0.35 0.25; 0.0 1.0 -0.30; 0.0 0.0 1.0]
            sim_up = _sim_lms_var1_t(Xoshiro(90325), 800; B=B_upper)
            r_up = estimate_lms_svar(sim_up.Y, 1; distributions=:student_t, method=:full_ml, seed=90326)
            @test test_svar_hypothesis(r_up, recursiveness_hypothesis(3); method=:wald).pvalue < 0.001
            for meth in (:wald, :lr, :lm)
                t_ord = test_svar_hypothesis(r_up, recursiveness_hypothesis(3; ordering=[3, 2, 1]); method=meth)
                @test t_ord.shock_permutation == [3, 2, 1]
                @test t_ord.pvalue > 0.05
            end
            t_best = test_svar_hypothesis(r_up, recursiveness_hypothesis(3; ordering=:best_match); method=:wald)
            @test t_best.shock_permutation == [3, 2, 1]
            @test t_best.pvalue > 0.05
        end
    end

    @testset "4. Monte Carlo Size & Power (n=3, p=1, T=400, Student-t(5))" begin
        _suppress_warnings() do
            n_mc_h0 = FAST ? 25 : 50
            n_mc_bq = FAST ? 20 : 40
            n_mc_h1 = FAST ? 15 : 25
            h_rec = recursiveness_hypothesis(3)
            h_single = impact_zero_hypothesis(3, [(2, 3)])

            rej_w = 0; rej_lr = 0; rej_lm = 0
            rej_single = 0
            for rep in 1:n_mc_h0
                sim = _sim_lms_var1_t(Xoshiro(90340 + rep), 400)
                fit = estimate_lms_svar(sim.Y, 1; distributions=:student_t,
                                        method=:three_step, n_starts=2, seed=90440 + rep)
                test_svar_hypothesis(fit, h_rec; method=:wald).pvalue < 0.05 && (rej_w += 1)
                test_svar_hypothesis(fit, h_rec; method=:lr).pvalue   < 0.05 && (rej_lr += 1)
                test_svar_hypothesis(fit, h_rec; method=:lm).pvalue   < 0.05 && (rej_lm += 1)
                test_svar_hypothesis(fit, h_single; method=:wald).pvalue < 0.05 && (rej_single += 1)
            end
            @test 0.0 <= rej_w / n_mc_h0 <= 0.16
            @test 0.0 <= rej_lr / n_mc_h0 <= 0.16
            @test 0.0 <= rej_lm / n_mc_h0 <= 0.16
            @test 0.0 <= rej_single / n_mc_h0 <= 0.16

            # Blanchard–Quah long-run H₀ size check at T = 400
            B_nontri = [1.0 0.35 -0.20; -0.25 1.0 0.25; 0.15 -0.30 1.0]
            Theta_LR = [1.6 0.0 0.0; -0.3 1.5 0.0; 0.2 -0.25 1.4]
            A1_bq = Matrix{Float64}(I, 3, 3) - B_nontri * MEM.robust_inv(Theta_LR)
            h_bq = long_run_hypothesis(3, [(1, 2)])
            rej_bq_w = 0; rej_bq_lr = 0
            for rep in 1:n_mc_bq
                sim = _sim_lms_var1_t(Xoshiro(90540 + rep), 400; A1=A1_bq, B=B_nontri)
                fit = estimate_lms_svar(sim.Y, 1; distributions=:student_t,
                                        method=:full_ml, n_starts=2, seed=90640 + rep)
                test_svar_hypothesis(fit, h_bq; method=:wald).pvalue < 0.05 && (rej_bq_w += 1)
                test_svar_hypothesis(fit, h_bq; method=:lr).pvalue   < 0.05 && (rej_bq_lr += 1)
            end
            @test 0.0 <= rej_bq_w / n_mc_bq <= 0.16
            @test 0.0 <= rej_bq_lr / n_mc_bq <= 0.16

            # Power under H₁: B[1, 3] = -0.5
            B_h1 = [1.0 0.0 -0.5; -0.35 1.0 0.0; 0.22 -0.30 1.0]
            pow_w = 0; pow_lr = 0; pow_lm = 0
            for rep in 1:n_mc_h1
                sim = _sim_lms_var1_t(Xoshiro(90740 + rep), 400; B=B_h1)
                fit = estimate_lms_svar(sim.Y, 1; distributions=:student_t,
                                        method=:three_step, n_starts=2, seed=90840 + rep)
                test_svar_hypothesis(fit, h_rec; method=:wald).pvalue < 0.05 && (pow_w += 1)
                test_svar_hypothesis(fit, h_rec; method=:lr).pvalue   < 0.05 && (pow_lr += 1)
                test_svar_hypothesis(fit, h_rec; method=:lm).pvalue   < 0.05 && (pow_lm += 1)
            end
            @test pow_w / n_mc_h1 > 0.85
            @test pow_lr / n_mc_h1 > 0.85
            @test pow_lm / n_mc_h1 > 0.85
        end
    end

    @testset "5. Residual Bootstrap (:iid and :wild)" begin
        _suppress_warnings() do
            n_boot_test = FAST ? 49 : 79
            sim0 = _sim_lms_var1_t(Xoshiro(90901), 400)
            fit0 = estimate_lms_svar(sim0.Y, 1; distributions=:student_t, method=:full_ml, seed=90902)
            h_rec = recursiveness_hypothesis(3)

            b_iid_1 = test_svar_hypothesis(fit0, h_rec; method=:wald, bootstrap=:iid,
                                           n_boot=n_boot_test, seed=90903)
            b_iid_2 = test_svar_hypothesis(fit0, h_rec; method=:wald, bootstrap=:iid,
                                           n_boot=n_boot_test, seed=90903)
            @test b_iid_1.bootstrap_pvalue !== nothing
            @test b_iid_1.bootstrap_pvalue == b_iid_2.bootstrap_pvalue
            @test 0.0 <= b_iid_1.bootstrap_pvalue <= 1.0
            @test b_iid_1.bootstrap_pvalue > 0.05
            @test b_iid_1.bootstrap_scheme == :iid
            @test b_iid_1.n_boot == n_boot_test
            @test b_iid_1.manifest !== nothing && b_iid_1.manifest.seed == 90903

            b_wild = test_svar_hypothesis(fit0, h_rec; method=:lr, bootstrap=:wild,
                                          n_boot=n_boot_test, seed=90904)
            @test b_wild.bootstrap_pvalue !== nothing
            @test b_wild.bootstrap_pvalue > 0.05
            @test b_wild.bootstrap_scheme == :wild

            b_lm = test_svar_hypothesis(fit0, h_rec; method=:lm, bootstrap=:iid,
                                        n_boot=n_boot_test, seed=90905)
            @test b_lm.bootstrap_pvalue !== nothing && b_lm.bootstrap_pvalue > 0.05

            # Under H₁, bootstrap rejects
            B_h1 = [1.0 0.0 -0.65; -0.35 1.0 0.0; 0.22 -0.30 1.0]
            sim1 = _sim_lms_var1_t(Xoshiro(90910), 500; B=B_h1)
            fit1 = estimate_lms_svar(sim1.Y, 1; distributions=:student_t, method=:full_ml, seed=90911)
            b_rej_iid = test_svar_hypothesis(fit1, h_rec; method=:wald, bootstrap=:iid,
                                             n_boot=n_boot_test, seed=90912)
            b_rej_wild = test_svar_hypothesis(fit1, h_rec; method=:lr, bootstrap=:wild,
                                              n_boot=n_boot_test, seed=90913)
            b_rej_lm_iid = test_svar_hypothesis(fit1, h_rec; method=:lm, bootstrap=:iid,
                                                n_boot=n_boot_test, seed=90914)
            @test b_rej_iid.bootstrap_pvalue < 0.05
            @test b_rej_wild.bootstrap_pvalue < 0.05
            @test b_rej_lm_iid.bootstrap_pvalue < 0.05
        end
    end

    @testset "6. Residual Diagnostics (diagnose_nongaussian_svar)" begin
        _suppress_warnings() do
            # (a) Student-t(5) shocks: strong non-Gaussianity, no weak-ID warning
            sim_t = _sim_lms_var1_t(Xoshiro(90950), 500; nu=5.0)
            fit_t = estimate_lms_svar(sim_t.Y, 1; distributions=:student_t, seed=90951)
            dt = diagnose_nongaussian_svar(fit_t; lags=4)

            @test dt isa NonGaussianDiagnostics{Float64}
            @test dt.nvars == 3 && dt.nobs == 499 && dt.lags == 4
            @test StatsAPI.nobs(dt) == 499 && nvars(dt) == 3
            @test dt.n_gaussian_shocks <= 1
            @test dt.weak_id_warning == false
            @test all(dt.excess_kurtosis .> 0.3)
            @test all(dt.jb_pvals .< 0.05)
            @test dt.multivariate_jb_pval < 0.05
            @test all(isfinite, dt.lb_levels_stats) && all(isfinite, dt.lb_levels_pvals)
            @test all(isfinite, dt.lb_squares_stats) && all(isfinite, dt.lb_squares_pvals)
            @test all(isfinite, dt.lb_abs_stats) && all(isfinite, dt.lb_abs_pvals)
            @test isfinite(dt.portmanteau_stat) && dt.portmanteau_pval > 0.005
            @test all(isfinite, dt.bds_stats) && all(isfinite, dt.bds_pvals)
            @test size(dt.cross_corr_matrix) == (3, 3) && size(dt.cross_sq_corr_matrix) == (3, 3)
            @test dt.cross_moment_df == 3 && dt.cross_moment_pval > 0.005

            # (b) Gaussian shocks: >= 2 Gaussian shocks triggers weak_id_warning == true
            sim_g = _sim_lms_var1_t(Xoshiro(90960), 400; nu=Inf)
            fit_g = estimate_lms_svar(sim_g.Y, 1; distributions=:student_t, n_starts=2, seed=90961)
            dg = diagnose_nongaussian_svar(fit_g; lags=4)
            @test dg.n_gaussian_shocks >= 2
            @test dg.weak_id_warning == true

            # (c) Common ARCH / stochastic volatility shocks: detects squared correlation & dependence
            rng_arch = Xoshiro(90970)
            N_a = 600
            A1_a = [0.3 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.3]
            B_a = [1.0 0.0 0.0; -0.3 1.0 0.0; 0.2 -0.25 1.0]
            h_comm = ones(N_a)
            Z_a = zeros(N_a, 3)
            for t in 2:N_a
                h_comm[t] = 0.15 + 0.55 * (Z_a[t - 1, 1]^2) + 0.30 * h_comm[t - 1]
                Z_a[t, :] .= sqrt(h_comm[t]) .* randn(rng_arch, 3)
            end
            Y_a = zeros(N_a, 3)
            for t in 2:N_a
                Y_a[t, :] .= A1_a * Y_a[t - 1, :] .+ B_a * Z_a[t, :]
            end
            fit_a = estimate_lms_svar(Y_a[101:end, :], 1; distributions=:student_t, n_starts=2, seed=90971)
            da = diagnose_nongaussian_svar(fit_a; lags=4)
            @test any(p -> p < 0.05, da.lb_squares_pvals) || da.cross_moment_pval < 0.05
        end
    end

    @testset "7. Display (show / report), Float32 & RSER-01 Serializable Struct Invariants" begin
        _suppress_warnings() do
            sim = _sim_lms_var1_t(Xoshiro(90980), 250)
            m = estimate_var(sim.Y, 1)
            fit = estimate_lms_svar(m; distributions=:student_t, method=:full_ml, n_starts=2, seed=90981)

            h_rec = recursiveness_hypothesis(3)
            t_lr = test_svar_hypothesis(fit, h_rec; method=:lr, bootstrap=:iid, n_boot=19, seed=90982)
            s_hyp = sprint(show, t_lr)
            @test occursin("LMS (2017)", s_hyp)
            @test occursin("Bootstrap", s_hyp)
            @test occursin("Log-lik", s_hyp)
            buf1 = IOBuffer()
            report(buf1, t_lr; repro=true)
            @test occursin("Reproducibility:", String(take!(buf1)))

            d = diagnose_nongaussian_svar(fit; lags=3)
            s_diag = sprint(show, d)
            @test occursin("Non-Gaussian SVAR Residual Diagnostics", s_diag)
            @test occursin("JB", s_diag)
            buf2 = IOBuffer()
            report(buf2, d)
            @test !isempty(String(take!(buf2)))

            # Float32 end-to-end
            m32 = VARModel(Float32.(m.Y), m.p, Float32.(m.B), Float32.(m.U),
                           Float32.(m.Sigma), Float32(m.aic), Float32(m.bic),
                           Float32(m.hqic), m.varnames)
            fit32 = estimate_lms_svar(m32; distributions=:student_t, n_starts=2, seed=90983)
            @test fit32 isa LMSSVARResult{Float32}
            for meth in (:wald, :lr, :lm)
                t32 = test_svar_hypothesis(fit32, h_rec; method=meth)
                @test t32 isa SVARHypothesisTest{Float32}
                @test isfinite(t32.statistic) && isfinite(t32.pvalue)
            end
            t32_bq = test_svar_hypothesis(fit32, long_run_hypothesis(3, [(1, 2)]); method=:lr)
            @test t32_bq isa SVARHypothesisTest{Float32}
            @test isfinite(t32_bq.statistic)
            t32_b = test_svar_hypothesis(fit32, h_rec; method=:wald, bootstrap=:iid, n_boot=19, seed=90984)
            @test t32_b isa SVARHypothesisTest{Float32}
            @test isfinite(t32_b.bootstrap_pvalue)

            d32 = diagnose_nongaussian_svar(fit32; lags=3)
            @test d32 isa NonGaussianDiagnostics{Float32}
            @test all(isfinite, d32.jb_stats)

            # RSER-01 invariant checks on exported result structs
            for (obj, Tw) in ((t_lr, SVARHypothesisTest), (d, NonGaussianDiagnostics))
                @test !any(f -> fieldtype(typeof(obj), f) <: Function, fieldnames(typeof(obj)))
                args = Any[getfield(obj, i) for i in 1:nfields(obj)]
                @test Tw{Float64}(args...) isa Tw{Float64}
            end

            # Input error validation on test_svar_hypothesis & diagnose_nongaussian_svar
            @test_throws ArgumentError test_svar_hypothesis(fit, h_rec; method=:invalid)
            @test_throws ArgumentError test_svar_hypothesis(fit, h_rec; bootstrap=:block)
            @test_throws ArgumentError test_svar_hypothesis(fit, h_rec; n_boot=0)
            @test_throws ArgumentError test_svar_hypothesis(fit, h_rec; vcov_type=:bad)
            @test_throws ArgumentError test_svar_hypothesis(fit, recursiveness_hypothesis(2))
            @test_throws ArgumentError diagnose_nongaussian_svar(fit; lags=0)
        end
    end

end
