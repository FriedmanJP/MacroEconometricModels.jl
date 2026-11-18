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

function _simulate_var2_t5(rng::AbstractRNG, T_obs::Int;
                           A1::Matrix{Float64}=[0.45 0.12 -0.05; -0.08 0.40 0.10; 0.05 -0.10 0.35],
                           A2::Matrix{Float64}=[-0.15 0.05 0.04; 0.06 -0.12 -0.05; -0.04 0.08 -0.10],
                           c::Vector{Float64}=[0.2, -0.1, 0.15],
                           B_true::Matrix{Float64}=[1.0 0.25 -0.18; -0.30 1.0 0.22; 0.15 -0.28 1.0],
                           sigma_true::Vector{Float64}=[1.2, 0.85, 1.4],
                           nu::Float64=5.0,
                           burn::Int=200)
    n = length(c)
    B0_true = B_true * Diagonal(sigma_true)
    N = T_obs + burn
    Z = rand(rng, TDist(nu), N, n) ./ sqrt(nu / (nu - 2.0))
    Y = zeros(N, n)
    for t in 3:N
        u = B0_true * Z[t, :]
        y = c + A1 * Y[t - 1, :] + A2 * Y[t - 2, :] + u
        Y[t, :] .= y
    end
    keep = (burn + 1):N
    Pi_true = vcat(c', A1', A2')
    (Y=Y[keep, :], B_true=B_true, sigma_true=sigma_true, B0_true=B0_true,
     A1=A1, A2=A2, Pi_true=Pi_true, Z=Z[keep, :])
end

function _check_lms_derivatives(th::Vector{Float64}, Y_eff::Matrix{Float64},
                                X::Matrix{Float64}, dists::Vector{Symbol};
                                check_opg::Bool=false)
    sh = MEM._lms_score_hessian(th, Y_eff, X, dists)
    LT = x -> MEM._lms_avg_loglik(x, Y_eff, X, dists)
    g = ForwardDiff.gradient(LT, th)
    H = ForwardDiff.hessian(LT, th)
    @test maximum(abs, sh.score - g) < 1e-10
    @test maximum(abs, sh.hessian - H) < 1e-10
    @test issymmetric(sh.hessian)
    @test issymmetric(sh.opg)
    if check_opg
        T_eff = size(Y_eff, 1)
        J_fd = zeros(length(th), length(th))
        t_ref = Ref(1)
        lt = x -> MEM._lms_avg_loglik(x, @view(Y_eff[t_ref[]:t_ref[], :]),
                                      @view(X[t_ref[]:t_ref[], :]), dists)
        cfg_t = ForwardDiff.GradientConfig(lt, th)
        st_buf = similar(th)
        for t in 1:T_eff
            t_ref[] = t
            ForwardDiff.gradient!(st_buf, lt, th, cfg_t)
            J_fd .+= st_buf * st_buf'
        end
        J_fd ./= T_eff
        @test maximum(abs, sh.opg - J_fd) < 1e-10
    end
end

@testset "LMS (2017) Non-Gaussian SVAR (#902)" begin

    @testset "1. Complete Identification Projection (lms_project_B, LMS p. 292)" begin
        # 4x4 sequential greedy projection where column norms + remaining-column max matter
        B_canon = [
            1.0   0.25 -0.35  0.15;
           -0.40  1.0   0.30 -0.20;
            0.18 -0.45  1.0   0.28;
           -0.22  0.12 -0.32  1.0
        ]
        perm_true = [3, 1, 4, 2]
        scale_true = [-2.4, 0.75, -1.65, 3.1]
        P_true = zeros(4, 4)
        for j in 1:4
            P_true[perm_true[j], j] = 1.0
        end
        # B_in has permuted, scaled, sign-flipped columns of B_canon:
        # B_in[:, perm_true[j]] = B_canon[:, j] * scale_true[j]
        B_in = B_canon * Diagonal(scale_true) * transpose(P_true)

        proj = lms_project_B(B_in)
        @test diag(proj.B) == ones(4)
        @test proj.B ≈ B_canon atol = 1e-12
        @test proj.sigma ≈ abs.(scale_true) atol = 1e-12
        @test proj.perm == perm_true
        @test proj.signs == [-1, 1, -1, 1]
        @test B_in * proj.D1 * proj.P * proj.D2 ≈ proj.B atol = 1e-12

        H_lms = MEM._lms_H(Float64, 4)
        @test vec(proj.B) ≈ H_lms * proj.beta + vec(Matrix{Float64}(I, 4, 4)) atol = 1e-12

        # Idempotence on canonical representative
        proj2 = lms_project_B(proj.B)
        @test proj2.B ≈ proj.B atol = 1e-14
        @test proj2.perm == [1, 2, 3, 4]
        @test proj2.signs == [1, 1, 1, 1]
        @test proj2.sigma ≈ ones(4) atol = 1e-14

        # Sequential greedy elimination differs from naive global row-max:
        # Build C1 with unit-norm columns where col 3 wins row 1, col 1 wins row 2,
        # col 2 wins row 3, col 4 wins row 4.
        C_seq = [
            0.70  0.20  0.85  0.10;
            0.60  0.40  0.50  0.30;
            0.25  0.80  0.15  0.50;
            sqrt(1 - 0.70^2 - 0.60^2 - 0.25^2)  sqrt(1 - 0.20^2 - 0.40^2 - 0.80^2)  sqrt(1 - 0.85^2 - 0.50^2 - 0.15^2)  sqrt(1 - 0.10^2 - 0.30^2 - 0.50^2)
        ]
        # Rescale columns so raw magnitudes wouldn't match unit-norm magnitudes
        B_seq = C_seq * Diagonal([5.0, 2.0, -0.5, 4.0])
        p_seq = lms_project_B(B_seq)
        @test p_seq.perm == [3, 1, 2, 4]
        @test p_seq.signs == [-1, 1, 1, 1]
        @test diag(p_seq.B) == ones(4)
        @test B_seq * p_seq.D1 * p_seq.P * p_seq.D2 ≈ p_seq.B atol = 1e-12

        #LMS (2017, p. 292) non-uniqueness failure modes B1, B2, B3:
        # B1: nonzero tie in row 1 after column unit-norm scaling
        B1 = [1.0 1.0 0.2; 1.0 -1.0 0.1; 0.1 0.1 1.0]
        @test abs(det(B1)) > 0.1
        @test_throws IdentificationError lms_project_B(B1)

        # B2: zero tie in row 2 over remaining columns {1, 2}
        B2 = [0.1 0.1 1.0; 0.0 0.0 0.2; 1.0 -1.0 0.1]
        @test abs(det(B2)) > 0.01
        @test_throws IdentificationError lms_project_B(B2)

        # B3: nonsingular with unique maxima in rows 1..n-1, but zero at (n, perm[n])
        B3 = [0.9 0.1 0.8; 0.1 0.9 0.6; 0.42 0.42 0.0]
        @test abs(det(B3)) > 0.1
        @test_throws IdentificationError lms_project_B(B3)

        # Singular / non-square validation
        @test_throws IdentificationError lms_project_B([1.0 2.0; 2.0 4.0])
        @test_throws ArgumentError lms_project_B([1.0 2.0 3.0; 4.0 5.0 6.0])

        # n = 1 and Float32
        p1 = lms_project_B([-3.5;;])
        @test p1.B == [1.0;;]
        @test p1.beta == Float64[]
        @test p1.sigma == [3.5]
        @test p1.signs == [-1]

        p32 = lms_project_B(Float32[1.0 0.2; -0.3 1.0])
        @test p32.B isa Matrix{Float32}
        @test p32.sigma isa Vector{Float32}
    end

    @testset "2. Zero-Mean Unit-Variance Shock Families" begin
        ts = range(-15.0, 15.0; length=120_001)
        dt = step(ts)
        cases = [
            (:student_t, [4.2]),
            (:skewed_t, [4.5, -0.42]),
            (:mixture_normal_asym, [0.35, 0.55, -0.8]),
            (:pearson_iv, [1.5, 4.0]),
            (:pml, [-1.2, 5.0]),
            (:skew_normal, [2.3]),
            (:ged, [1.35]),
            (:laplace_generalized, [0.85]),
        ]
        for (fam, lam) in cases
            cfam = MEM._lms_canonical_dist(fam)
            Z = m1 = m2 = 0.0
            for t in ts
                x = sinh(t)
                w = exp(MEM._lms_logpdf_scalar(cfam, x, lam)) * cosh(t) * dt
                Z += w
                m1 += x * w
                m2 += x^2 * w
            end
            @test isapprox(Z, 1.0; atol=1e-6)
            @test isapprox(m1, 0.0; atol=1e-6)
            @test isapprox(m2, 1.0; atol=1e-5)
        end
    end

    @testset "3. Analytical Score & 16-Block Hessian vs ForwardDiff (< 1e-10)" begin
        rng = Xoshiro(90201)

        # (a) n = 1, p = 1 edge case
        Y1 = randn(rng, 45, 1)
        m1 = estimate_var(Y1, 1)
        Y_eff1, X1 = MEM.construct_var_matrices(m1.Y, m1.p)
        th1 = vcat(vec(Matrix(transpose(m1.B))), Float64[], [1.15], [4.5])
        _check_lms_derivatives(th1, Y_eff1, X1, [:student_t])

        # (b) n = 2, p = 1 across ALL 6 homogeneous families, including det(B) < 0
        Y2 = randn(rng, 50, 2)
        m2 = estimate_var(Y2, 1)
        Y_eff2, X2 = MEM.construct_var_matrices(m2.Y, m2.p)
        B_neg = [1.0 1.4; 1.1 1.0]  # det = -0.54 < 0
        @test det(B_neg) < 0
        beta2 = MEM._vecd0(B_neg)
        sig2 = [0.9, 1.3]
        pi2 = vec(Matrix(transpose(m2.B)))

        homog_cases = [
            (:student_t, [4.2, 6.5]),
            (:skewed_t, [4.5, 0.35, 5.2, -0.40]),
            (:mixture_normal_asym, [0.35, 0.45, -0.7, 0.65, -0.30, 0.9]),
            (:pearson_iv, [1.1, 4.0, -0.8, 5.5]),
            (:skew_normal, [1.4, -0.9]),
            (:ged, [1.25, 1.75]),
        ]
        for (fam, lam2) in homog_cases
            th2 = vcat(pi2, beta2, sig2, lam2)
            _check_lms_derivatives(th2, Y_eff2, X2, [fam, fam])
        end

        # (c) n = 3, p = 2 with heterogeneous distributions covering all 6 families
        Y3 = randn(rng, 65, 3)
        m3 = estimate_var(Y3, 2)
        Y_eff3, X3 = MEM.construct_var_matrices(m3.Y, m3.p)
        pi3 = vec(Matrix(transpose(m3.B)))
        beta3 = [0.22, -0.14, 0.31, -0.18, -0.25, 0.27]
        sig3 = [1.1, 0.85, 1.35]

        het_cases = [
            ([:student_t, :skewed_t, :mixture_normal_asym],
             [4.5, 5.0, -0.30, 0.40, 0.50, -0.60]),
            ([:pearson_iv, :skew_normal, :ged],
             [0.9, 4.2, -1.3, 1.45]),
        ]
        for (dists3, lam3) in het_cases
            th3 = vcat(pi3, beta3, sig3, lam3)
            _check_lms_derivatives(th3, Y_eff3, X3, dists3; check_opg=true)
        end
    end

    @testset "4. Student-t(5) (n=3, p=2, T=400): 3-Step vs Full ML, Slope Efficiency & Registry" begin
        _suppress_warnings() do
            sim = _simulate_var2_t5(Xoshiro(90210), 400)
            m = estimate_var(sim.Y, 2)

            r3 = estimate_lms_svar(m; distributions=:student_t, method=:three_step,
                                   se=:hessian, n_starts=5, seed=90211)
            r1 = estimate_lms_svar(m; distributions=:student_t, method=:full_ml,
                                   se=:hessian, n_starts=5, seed=90211)

            @test r3 isa LMSSVARResult{Float64}
            @test r1 isa LMSSVARResult{Float64}
            @test r3.converged
            @test r1.converged
            @test r3.method == :three_step
            @test r1.method == :full_ml
            @test r3.se_type == :hessian
            @test r3.distribution == :student_t
            @test r3.distributions == [:student_t, :student_t, :student_t]
            @test r3.se === r3.se_B0
            @test diag(r3.B) == ones(3)
            @test diag(r1.B) == ones(3)
            @test all(r3.sigma .> 0)
            @test all(r1.sigma .> 0)
            @test r3.B0 ≈ r3.B * Diagonal(r3.sigma)
            @test r1.B0 ≈ r1.B * Diagonal(r1.sigma)
            @test norm(r3.Q' * r3.Q - I) < 1e-10
            @test norm(r1.Q' * r1.Q - I) < 1e-10
            @test r3.raw_shocks * Diagonal(1 ./ r3.sigma) ≈ r3.shocks

            # Full ML dominates or matches 3-step log-likelihood, and both beat Gaussian
            @test r1.loglik >= r3.loglik - 1e-10
            @test r3.loglik > r3.loglik_gaussian
            @test r1.loglik > r1.loglik_gaussian

            # Close agreement between :three_step and :full_ml
            @test norm(r3.B0 - r1.B0) < 0.15
            @test norm(r3.B - r1.B) < 0.15
            @test norm(r3.Pi - r1.Pi) < 0.15

            # Recovery of structural B0 and B
            @test norm(r3.B - sim.B_true) / norm(sim.B_true) < 0.25
            @test norm(r1.B - sim.B_true) / norm(sim.B_true) < 0.25
            @test norm(r3.B0 - sim.B0_true) / norm(sim.B0_true) < 0.25

            # Stationary point check: full ML has zero full score; 3-step has zero γ and π scores
            Y_eff, X = MEM.construct_var_matrices(m.Y, m.p)
            sh1 = MEM._lms_score_hessian(r1.theta, Y_eff, X, r1.distributions)
            @test maximum(abs, sh1.score) < 1e-6

            # Lower VAR slope (A₁, A₂) RMSE vs OLS across Student-t(5) Monte Carlo draws
            n_mc = FAST ? 3 : 6
            rmse_ols = 0.0
            rmse_3s = 0.0
            rmse_ml = 0.0
            for rep in 1:n_mc
                s_rep = _simulate_var2_t5(Xoshiro(90220 + rep), 400)
                m_rep = estimate_var(s_rep.Y, 2)
                r3_rep = estimate_lms_svar(m_rep; distributions=:student_t, method=:three_step,
                                           n_starts=3, seed=90230 + rep)
                r1_rep = estimate_lms_svar(m_rep; distributions=:student_t, method=:full_ml,
                                           n_starts=3, seed=90230 + rep)
                slopes_true = s_rep.Pi_true[2:end, :]
                rmse_ols += sum(abs2, m_rep.B[2:end, :] - slopes_true)
                rmse_3s += sum(abs2, r3_rep.Pi[2:end, :] - slopes_true)
                rmse_ml += sum(abs2, r1_rep.Pi[2:end, :] - slopes_true)
            end
            @test sqrt(rmse_3s / n_mc) < sqrt(rmse_ols / n_mc)
            @test sqrt(rmse_ml / n_mc) < sqrt(rmse_ols / n_mc)

            # :hessian / :sandwich / :opg SEs
            r_sand = estimate_lms_svar(m; distributions=:student_t, method=:three_step,
                                       se=:sandwich, n_starts=3, seed=90211)
            r_opg = estimate_lms_svar(m; distributions=:student_t, method=:three_step,
                                      se=:opg, n_starts=3, seed=90211)
            for rr in (r3, r_sand, r_opg)
                @test all(isfinite, rr.se_B0) && all(rr.se_B0 .> 0)
                @test all(isfinite, rr.se_sigma) && all(rr.se_sigma .> 0)
                @test all(isfinite, rr.se_Pi) && all(rr.se_Pi .> 0)
                @test diag(rr.se_B) == zeros(3)
                @test all(isfinite, rr.se_B) && all(rr.se_B .>= 0)
            end
            @test isapprox(r3.se_B0, r_sand.se_B0; rtol=0.5)
            @test isapprox(r3.se_B0, r_opg.se_B0; rtol=0.5)

            # Identification registry & compute_Q + downstream IRF/FEVD
            @test haskey(MEM.IDENTIFICATION_REGISTRY, :lms)
            @test MEM._needs_residuals(:lms)
            @test !MEM._is_set_identified(:lms)
            @test !MEM._is_partial(:lms)
            @test MEM._should_match_columns(:lms)
            Q_reg = compute_Q(m, :lms; distributions=:student_t, n_starts=5, seed=90211)
            @test Q_reg ≈ r3.Q atol = 1e-10
            ir = irf(m, 6; method=:lms, seed=90211)
            @test size(ir.values) == (6, 3, 3)
            fv = fevd(m, 6; method=:lms, seed=90211)
            @test size(fv.proportions) == (3, 3, 6)
        end
    end

    @testset "5. API, Heterogeneous Families, Aliases, StatsAPI, Show & Float32" begin
        _suppress_warnings() do
            sim = _simulate_var2_t5(Xoshiro(90250), 250)
            m = estimate_var(sim.Y, 2)

            # Matrix Y, p entry point + seed reproducibility
            r_mat = estimate_lms_svar(sim.Y, 2; distributions=:student_t, n_starts=2, seed=77)
            r_mod = estimate_lms_svar(m; distributions=:student_t, n_starts=2, seed=77)
            @test r_mat.B0 ≈ r_mod.B0 atol = 1e-12
            @test r_mat.theta ≈ r_mod.theta atol = 1e-12
            @test r_mat.manifest !== nothing && r_mat.manifest.seed == 77

            # Heterogeneous vector of distributions & aliases
            r_het = estimate_lms_svar(m; distributions=[:student_t, :pml, :laplace_generalized],
                                      n_starts=2, max_iter=150, seed=88,
                                      shock_names=["S1", "S2", "S3"])
            @test r_het.distribution == :heterogeneous
            @test r_het.distributions == [:student_t, :pearson_iv, :ged]
            @test r_het.shock_names == ["S1", "S2", "S3"]
            @test length(r_het.dist_params[1]) == 1
            @test length(r_het.dist_params[2]) == 2
            @test length(r_het.dist_params[3]) == 1

            # StatsAPI interface
            @test StatsAPI.nobs(r_het) == size(m.U, 1)
            @test StatsAPI.coef(r_het) == r_het.theta
            @test StatsAPI.vcov(r_het) == r_het.vcov
            @test StatsAPI.stderror(r_het) ≈ sqrt.(max.(diag(r_het.vcov), 0.0))
            @test StatsAPI.residuals(r_het) == r_het.shocks
            @test StatsAPI.loglikelihood(r_het) == r_het.loglik
            @test StatsAPI.aic(r_het) == r_het.aic
            @test StatsAPI.bic(r_het) == r_het.bic
            @test StatsAPI.dof(r_het) == length(r_het.theta)
            @test StatsAPI.islinear(r_het) == false
            @test :se in propertynames(r_het)
            @test :distribution in propertynames(r_het)

            # show and report
            s_txt = sprint(show, r_het)
            @test occursin("Lanne–Meitz–Saikkonen", s_txt)
            @test occursin("heterogeneous", s_txt) || occursin("student_t", s_txt)
            @test occursin("LR Test", s_txt)
            buf = IOBuffer()
            report(buf, r_het)
            @test !isempty(String(take!(buf)))

            # Float32 end-to-end (VARModel{Float32} preserves Float32; raw Float32 Y delegates cleanly)
            m32 = VARModel(Float32.(m.Y), m.p, Float32.(m.B), Float32.(m.U),
                           Float32.(m.Sigma), Float32(m.aic), Float32(m.bic),
                           Float32(m.hqic), m.varnames)
            r32 = estimate_lms_svar(m32; distributions=:student_t, n_starts=2, seed=11)
            @test r32 isa LMSSVARResult{Float32}
            @test all(isfinite, r32.B0)
            @test all(isfinite, r32.se_B0)
            r32_mat = estimate_lms_svar(Float32.(sim.Y), 2; distributions=:student_t, n_starts=2, seed=11)
            @test all(isfinite, r32_mat.B0)

            # Input validation
            @test_throws ArgumentError estimate_lms_svar(m; distributions=:unknown_dist)
            @test_throws ArgumentError estimate_lms_svar(m; distributions=[:student_t, :ged])
            @test_throws ArgumentError estimate_lms_svar(m; method=:two_step)
            @test_throws ArgumentError estimate_lms_svar(m; se=:bootstrap)
            @test_throws ArgumentError estimate_lms_svar(m; n_starts=0)
            @test_throws ArgumentError estimate_lms_svar(m; shock_names=["only_one"])
        end
    end

end
