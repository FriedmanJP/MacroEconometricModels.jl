# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random
using StatsAPI
using ForwardDiff

const MEM = MacroEconometricModels
const _suppress_warnings = MEM._suppress_warnings

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "NGID-06: GMR (2020) SSVARMA(p, q) Estimation (#906)" begin

    @testset "1. Exports & Input Validation" begin
        @test isdefined(MacroEconometricModels, :estimate_ssvarma)

        Y_ok = dgp_ssvarma(; T=100, seed=90601).Y
        @test_throws ArgumentError estimate_ssvarma(Y_ok, -1, 1)
        @test_throws ArgumentError estimate_ssvarma(Y_ok, 1, -1)
        @test_throws ArgumentError estimate_ssvarma(Y_ok, 1, 1; method=:bogus)
        @test_throws ArgumentError estimate_ssvarma(Y_ok, 1, 1; distribution=:bogus)
        @test_throws ArgumentError estimate_ssvarma(Y_ok, 1, 1; se=:bogus)
        @test_throws ArgumentError estimate_ssvarma(Y_ok, 1, 1; k_iv=0)
        @test_throws ArgumentError estimate_ssvarma(Y_ok, 1, 1; n_starts=0)
        @test_throws ArgumentError estimate_ssvarma(randn(Xoshiro(1), 5, 2), 1, 1)

        Y_nan = copy(Y_ok)
        Y_nan[3, 1] = NaN
        @test_throws ArgumentError estimate_ssvarma(Y_nan, 1, 1)
    end

    @testset "2. Dual Schur-Sylvester Gradient & Hessian Verification" begin
        sim = dgp_ssvarma(; T=120, Phi=nothing, Theta=[-0.5 0.0; 1.0 -2.0],
                          C=[0.0 1.0; 1.0 0.5], dist=:student_t, nu=5.0, seed=90602)
        p0 = MEM._svarma_ml_pack(zeros(2), zeros(2, 2, 0),
                                 reshape([-0.45 0.05; 0.9 -1.9], 2, 2, 1),
                                 [0.1 1.05; 0.95 0.55], [0.5, 0.5])
        f0 = x -> MEM._svarma_ml_neg_loglik(x, sim.Y, 2, 0, 1, :student_t, 1)
        g_fd = ForwardDiff.gradient(f0, p0)
        eps_h = 1e-6
        g_cd = [(f0(p0 .+ eps_h .* [j == i ? 1.0 : 0.0 for j in 1:length(p0)]) -
                 f0(p0 .- eps_h .* [j == i ? 1.0 : 0.0 for j in 1:length(p0)])) / (2 * eps_h)
                for i in 1:length(p0)]
        @test maximum(abs, g_fd .- g_cd) < 1e-6
        H_fd = ForwardDiff.hessian(f0, p0)
        @test size(H_fd) == (length(p0), length(p0))
        @test norm(H_fd - transpose(H_fd)) < 1e-8
    end

    @testset "3. Univariate Non-Fundamental MA(1) Monte Carlo (GMR §5.1.1 Table 1)" begin
        _suppress_warnings() do
            n_reps = FAST ? 8 : 20
            for (dist_sim, dist_est) in ((:student_t, :student_t),
                                         (:mixture_normal_asym, :mixture_normal_asym))
                thetas_pml = Float64[]
                thetas_bj = Float64[]
                for rep in 1:n_reps
                    sim = dgp_ssvarma(; T=250, Phi=nothing, Theta=[-2.0;;], C=[1.0;;],
                                      dist=dist_sim, nu=5.0, seed=906100 + rep)
                    est = estimate_ssvarma(sim.Y, 0, 1; method=:pml,
                                           distribution=dist_est, n_starts=1, max_iter=150,
                                           seed=906200 + rep)
                    push!(thetas_pml, est.Theta[1, 1, 1])
                    if rep <= 8
                        bj = estimate_arima(vec(sim.Y), 0, 0, 1)
                        push!(thetas_bj, bj.theta[1])
                    end
                end
                n_nonfund = count(th -> abs(th) > 1.0, thetas_pml)
                @test n_nonfund / n_reps >= 0.80
                @test abs(median(thetas_pml) - (-2.0)) <= 0.15
                @test all(th -> abs(th) < 1.0, thetas_bj)
                @test abs(median(thetas_bj) - (-0.5)) < 0.15
            end
        end
    end

    @testset "4. Bivariate Mixed-Regime VMA(1) (GMR §5.1.2 eq. 4.27)" begin
        _suppress_warnings() do
            Theta_true = reshape([-0.5 0.0; 1.0 -2.0], 2, 2, 1)
            C_true = [0.0 1.0; 1.0 0.5]
            m_true = SVARMAModel(zeros(10, 2), nothing, Theta_true, C_true; shocks=zeros(10, 2))
            H = 12
            ir_true = irf(m_true, H).values

            sim = dgp_ssvarma(; T=300, Phi=nothing, Theta=Theta_true, C=C_true,
                              dist=:student_t, nu=5.0, seed=906302)
            est = estimate_ssvarma(sim.Y, 0, 1; method=:pml,
                                   distribution=:student_t, seed=906303)

            @test est isa SVARMAModel{Float64}
            @test est.regime === :mixed_nonfundamental
            @test est.nonfundamental_order == 1
            @test est.gmr_region === :II
            mods = sort(abs.(est.ma_eigenvalues))
            @test mods[1] < 1.0 && mods[2] > 1.0

            # Match causal structural IRFs up to shock column permutation/sign
            ir_est = irf(est, H).values
            c_id = (abs(cor(vec(ir_true[:, :, 1]), vec(ir_est[:, :, 1]))) +
                    abs(cor(vec(ir_true[:, :, 2]), vec(ir_est[:, :, 2])))) / 2
            c_sw = (abs(cor(vec(ir_true[:, :, 1]), vec(ir_est[:, :, 2]))) +
                    abs(cor(vec(ir_true[:, :, 2]), vec(ir_est[:, :, 1])))) / 2
            @test max(c_id, c_sw) > 0.92

            # Assumption A.7 normalization: C[1, :] >= 0 and non-decreasing across row 1
            @test est.C[1, 1] >= -1e-10
            @test est.C[1, 2] >= est.C[1, 1] - 1e-10
        end
    end

    @testset "5. First-Step 2SLS Correction in Σ(Ω*) vs Bootstrap Variance on SVARMA(1, 1)" begin
        _suppress_warnings() do
            Phi1 = [0.75 0.05; -0.05 0.70]
            Theta1 = [-0.45 0.0; 0.8 -2.0]
            C1 = [0.4 1.3; 1.0 -0.5]
            T_sim = 1200

            sim = dgp_ssvarma(; T=T_sim, Phi=Phi1, Theta=Theta1, C=C1,
                              dist=:mixture_normal_asym, seed=906401)
            gmm_res = MEM._svarma_gmm_estimate(sim.Y, 1, 1; k_iv=4, rng=Xoshiro(906402))
            m_gmm = gmm_res.model

            @test m_gmm.method === :gmm
            @test m_gmm.regime === :mixed_nonfundamental
            @test size(gmm_res.Q_pi) == (2 * (1 + 2 * 1), 2 * (1 + 2 * 4))
            @test size(gmm_res.Q_0, 1) == 30 && size(gmm_res.Q_0, 2) == 30
            @test isposdef(Symmetric(gmm_res.Q_0))
            @test isposdef(Symmetric(gmm_res.Omega_star))
            @test size(m_gmm.vcov) == (length(coef(m_gmm)), length(coef(m_gmm)))
            @test length(m_gmm.se) == length(coef(m_gmm))
            @test all(isfinite, m_gmm.se) && all(>=(0.0), m_gmm.se)

            # Parametric / residual bootstrap from the fitted SVARMA(1,1) model
            n_boot = FAST ? 16 : 36
            T_eff = size(m_gmm.shocks, 1)
            boot_th = Vector{Float64}[]
            boot_coef = Vector{Float64}[]
            rng = Xoshiro(906403)
            for b in 1:n_boot
                idx = rand(rng, 1:T_eff, T_sim + 120)
                eta_b = m_gmm.shocks[idx, :]
                eps_b = eta_b * transpose(m_gmm.C)
                Y_b = zeros(T_sim + 120, 2)
                for t in 2:(T_sim + 120)
                    Y_b[t, :] = m_gmm.mu + m_gmm.Phi[:, :, 1] * Y_b[t - 1, :] +
                                eps_b[t, :] + m_gmm.Theta[:, :, 1] * eps_b[t - 1, :]
                end
                Y_keep = Y_b[121:end, :]
                gb = MEM._svarma_gmm_estimate(Y_keep, 1, 1; k_iv=4, rng=Xoshiro(906500 + b))
                if gb.model.regime === :mixed_nonfundamental &&
                   norm(gb.model.C - m_gmm.C) < 1.0 &&
                   norm(gb.model.Theta - m_gmm.Theta) < 1.5
                    C0_b = gb.model.C
                    C1_b = gb.model.Theta[:, :, 1] * C0_b
                    push!(boot_th, vcat(vec(C0_b), vec(C1_b)))
                    push!(boot_coef, coef(gb.model))
                end
            end
            @test length(boot_th) >= (FAST ? 8 : 18)
            M_th = reduce(vcat, transpose.(boot_th))
            M_co = reduce(vcat, transpose.(boot_coef))
            boot_se_th = vec(std(M_th; dims=1))
            boot_se_co = vec(std(M_co; dims=1))
            ana_se_th = sqrt.(max.(diag(gmm_res.theta_vcov), 0.0))[1:8]
            ratios_th = ana_se_th ./ boot_se_th
            ratios_co = m_gmm.se ./ boot_se_co
            @test 0.5 <= median(ratios_th) <= 1.8
            @test 0.5 <= median(ratios_co) <= 1.8
            @test all(r -> 0.30 <= r <= 3.0, ratios_th)
            @test all(r -> 0.30 <= r <= 3.0, ratios_co)
        end
    end

    @testset "6. Estimator Variants (:ml, :pml, :gmm), Distributions, Float32, StatsAPI & Display" begin
        _suppress_warnings() do
            sim = dgp_ssvarma(; T=240, Phi=[0.35 0.05; -0.05 0.30],
                              Theta=[-0.5 0.0; 1.0 -2.0], C=[0.6 1.0; 0.8 0.4],
                              dist=:student_t, nu=5.0, seed=906601)

            # :ml with Hessian SEs and :pml with Sandwich & OPG SEs
            m_ml = estimate_ssvarma(sim.Y, 1, 1; method=:ml, distribution=:student_t,
                                    se=:auto, n_starts=1, max_iter=120, seed=906602)
            m_pml = estimate_ssvarma(sim.Y, 1, 1; method=:pml, distribution=:pearson_iv,
                                     se=:auto, n_starts=1, max_iter=120, seed=906602)
            m_opg = estimate_ssvarma(sim.Y, 1, 1; method=:pml, distribution=:skew_normal,
                                     se=:opg, n_starts=1, max_iter=120, seed=906602)
            m_gmm = estimate_ssvarma(sim.Y, 1, 1; method=:gmm, seed=906602)

            for m in (m_ml, m_pml, m_opg, m_gmm)
                @test m isa SVARMAModel{Float64}
                @test isfinite(m.loglik) && isfinite(m.aic) && isfinite(m.bic)
                @test size(m.vcov) == (length(coef(m)), length(coef(m)))
                @test length(m.se) == length(coef(m))
                @test all(isfinite, m.se) && all(>=(0.0), m.se)
                @test StatsAPI.vcov(m) == m.vcov
                @test StatsAPI.stderror(m) == m.se
                @test length(StatsAPI.coef(m)) == 2 + 4 + 4 + 4
            end

            # Seed reproducibility
            m_ml_rep = estimate_ssvarma(sim.Y, 1, 1; method=:ml, distribution=:student_t,
                                        se=:auto, n_starts=1, max_iter=120, seed=906602)
            @test m_ml.Theta == m_ml_rep.Theta
            @test m_ml.C == m_ml_rep.C
            @test m_ml.se == m_ml_rep.se

            # Integer matrix fallback
            Y_int = round.(Int, 10 .* sim.Y)
            m_int = estimate_ssvarma(Y_int, 1, 1; method=:gmm, seed=906603)
            @test m_int isa SVARMAModel{Float64}

            # Pure AR (q = 0) and pure MA (p = 0)
            sim_q0 = dgp_ssvarma(; T=180, Phi=[0.35 0.05; -0.05 0.30], Theta=nothing,
                                 C=[0.6 1.0; 0.8 0.4], seed=906604)
            m_q0 = estimate_ssvarma(sim_q0.Y, 1, 0; method=:pml, n_starts=1, max_iter=120, seed=906604)
            @test m_q0.p == 1 && m_q0.q == 0 && m_q0.regime === :fundamental

            # Float32 support
            Y32 = Float32.(sim.Y)
            m32_gmm = estimate_ssvarma(Y32, 1, 1; method=:gmm, seed=906605)
            m32_pml = estimate_ssvarma(Y32, 1, 1; method=:pml, n_starts=1, max_iter=80, seed=906605)
            @test m32_gmm isa SVARMAModel{Float32}
            @test m32_pml isa SVARMAModel{Float32}
            @test eltype(m32_pml.vcov) == Float32 && eltype(m32_pml.se) == Float32

            # RSER-01 23-arg constructor round-trip
            raw23 = Any[getfield(m_ml, f) for f in fieldnames(SVARMAModel)]
            m_rebuilt = SVARMAModel{Float64}(raw23...)
            @test m_rebuilt.vcov == m_ml.vcov
            @test m_rebuilt.se == m_ml.se

            # Display & report under text and LaTeX backends
            txt = sprint(report, m_ml)
            @test occursin("Strong Structural VARMA(1, 1)", txt)
            @test occursin("Structural Impact Matrix (C) Inference", txt)
            ltx = with_display_backend(:latex) do
                sprint(report, m_ml)
            end
            @test occursin("SVARMA(1, 1)", ltx)
        end
    end

    @testset "7. Canonical Empirical SSVARMA Workflows (:gnp_hamilton & Bivariate Macro)" begin
        _suppress_warnings() do
            # (a) Hamilton (1989) US real GNP growth SVARMA(2, 1) & fundamentalness check
            ts_gnp = load_example(:gnp_hamilton)
            Y_gnp = to_matrix(ts_gnp)
            est_gnp = estimate_ssvarma(Y_gnp, 2, 1; method=:pml, distribution=:student_t,
                                       n_starts=(FAST ? 1 : 2), max_iter=(FAST ? 80 : 150),
                                       varnames=["gnp_growth"], shock_names=["GNP shock"], seed=906701)
            @test est_gnp isa SVARMAModel{Float64}
            @test est_gnp.p == 2 && est_gnp.q == 1
            @test all(isfinite, est_gnp.C) && all(isfinite, est_gnp.Theta)
            f_gnp = test_fundamentalness(est_gnp; max_lag=4, n_boot=(FAST ? 19 : 49), seed=906702)
            @test f_gnp isa FundamentalnessTestResult{Float64}
            @test 0.0 <= f_gnp.pvalue <= 1.0
            @test irf(est_gnp, 8) isa ImpulseResponse{Float64}

            # (b) Bivariate Blanchard-Quah / GMR real GDP & unemployment SVARMA(2, 1) workflow
            qd = load_example(:fred_qd)
            vars_bq = ["GDPC1", "UNRATE"]
            X_raw = to_matrix(apply_tcode(qd[:, vars_bq]))
            keep = [all(isfinite, @view(X_raw[t, :])) for t in 1:size(X_raw, 1)]
            Y_all = Matrix{Float64}(X_raw[keep, :])
            T_sub = FAST ? min(size(Y_all, 1), 110) : min(size(Y_all, 1), 150)
            Y_bq = Y_all[(end - T_sub + 1):end, :]
            est_bq = estimate_ssvarma(Y_bq, 2, 1; method=:pml, distribution=:student_t,
                                      n_starts=(FAST ? 1 : 2), max_iter=(FAST ? 80 : 120),
                                      varnames=vars_bq, shock_names=["Supply", "Demand"], seed=906703)
            @test est_bq isa SVARMAModel{Float64}
            @test size(est_bq.C) == (2, 2)
            @test all(isfinite, est_bq.Theta)
            f_bq = test_fundamentalness(est_bq; max_lag=4, n_boot=(FAST ? 19 : 49), seed=906704)
            @test f_bq isa FundamentalnessTestResult{Float64}
            @test 0.0 <= f_bq.pvalue <= 1.0
            @test irf(est_bq, 8) isa ImpulseResponse{Float64}
        end
    end

end
