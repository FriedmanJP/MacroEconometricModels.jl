# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using MacroEconometricModels: dof_residual
using LinearAlgebra, Statistics, Random, Distributions, StatsAPI

@testset "Discrete-Choice Rational Inattention (#922)" begin

    # Helper to draw discrete choice in 1:J given probability vector
    function _draw_cat(rng::AbstractRNG, p::AbstractVector{<:Real})
        u = rand(rng)
        cum = 0.0
        for j in 1:length(p)
            cum += p[j]
            u <= cum && return j
        end
        length(p)
    end

    # =========================================================================
    # 1. Global KKT Optimality across random 10x20 payoff matrices & capacity
    # =========================================================================

    @testset "Global KKT Optimality (10x20 random payoffs)" begin
        rng = Random.Xoshiro(92201)
        U = randn(rng, 10, 20)
        mu_unif = fill(1.0 / 20, 20)
        mu_skew = rand(rng, 20)
        mu_skew[end] = 0.0  # zero-probability state must not break solver
        mu_skew ./= sum(mu_skew)

        for (label, mu) in [("uniform", mu_unif), ("skewed+zero", mu_skew)]
            for ω in (0.05, 0.5, 5.0)
                sol = solve_ri_discrete(U, mu; omega=ω)
                @test sol isa RIDiscreteChoiceSolution{Float64}
                @test sol.converged
                @test sol.omega == ω
                @test sum(sol.uncond_probs) ≈ 1.0 atol=1e-12
                @test all(>=(0.0), sol.uncond_probs)
                @test all(>=(0.0), sol.cond_probs)
                @test all(abs.(sum(sol.cond_probs; dims=1) .- 1.0) .< 1e-12)
                @test sol.cond_probs * sol.prior ≈ sol.uncond_probs atol=1e-12
                @test maximum(sol.kkt_multipliers) <= 1.0 + 1e-9
                @test !isempty(sol.consideration_set)
                @test maximum(abs.(sol.kkt_multipliers[sol.consideration_set] .- 1.0)) < 1e-9
                inactive = setdiff(1:10, sol.consideration_set)
                @test all(j -> sol.uncond_probs[j] == 0.0, inactive)
                @test sol.mutual_info_bits ≈ sol.mutual_info_nats / log(2) atol=1e-12
                @test sol.mutual_info_nats >= 0.0
                @test sol.net_welfare ≈ sol.expected_utility - ω * sol.mutual_info_nats atol=1e-12
            end
        end

        # Capacity mode (:bits and :nats)
        sol_bits = solve_ri_discrete(U, mu_unif; capacity=0.8, units=:bits)
        @test sol_bits.converged
        @test sol_bits.mutual_info_bits ≈ 0.8 atol=1e-10
        @test maximum(sol_bits.kkt_multipliers) <= 1.0 + 1e-9
        @test maximum(abs.(sol_bits.kkt_multipliers[sol_bits.consideration_set] .- 1.0)) < 1e-9

        sol_check = solve_ri_discrete(U, mu_unif; omega=sol_bits.omega)
        @test sol_check.uncond_probs ≈ sol_bits.uncond_probs atol=1e-10

        sol_nats = solve_ri_discrete(U, mu_unif; capacity=0.5, units=:nats)
        @test sol_nats.converged
        @test sol_nats.mutual_info_nats ≈ 0.5 atol=1e-10

        # Float32 support
        sol_f32 = solve_ri_discrete(Float32.(U), Float32.(mu_unif); omega=0.5f0)
        @test sol_f32 isa RIDiscreteChoiceSolution{Float32}
        @test sol_f32.converged
        @test maximum(sol_f32.kkt_multipliers) <= 1.0f0 + 1f-5

        # J = 1 edge case
        sol_j1 = solve_ri_discrete(reshape([1.0, 2.0, 3.0], 1, 3), [0.2, 0.3, 0.5]; omega=0.5)
        @test sol_j1.consideration_set == [1]
        @test sol_j1.uncond_probs == [1.0]
        @test sol_j1.mutual_info_nats == 0.0
        @test sol_j1.expected_utility ≈ 2.3

        # Input validation
        @test_throws ArgumentError solve_ri_discrete(U, mu_unif)
        @test_throws ArgumentError solve_ri_discrete(U, mu_unif; omega=0.5, capacity=0.5)
        @test_throws ArgumentError solve_ri_discrete(U, mu_unif; omega=-0.1)
        @test_throws ArgumentError solve_ri_discrete(U, mu_unif; capacity=10.0, units=:bits)  # > log2(10)
        @test_throws ArgumentError solve_ri_discrete(U, mu_unif; capacity=0.5, units=:ban)
        @test_throws ArgumentError solve_ri_discrete(U, ones(5); omega=0.5)
        @test_throws ArgumentError solve_ri_discrete(ones(2, 2), [0.5, 0.5]; capacity=0.1)
        @test_throws ArgumentError solve_ri_discrete([1.0 2.0], [0.5, 0.5]; capacity=0.1)
    end

    # =========================================================================
    # 2. Caplin-Dean-Leahy (2019) Analytical Reservation Cutoff ω^*
    # =========================================================================

    @testset "CDL (2019) Analytical Reservation Cutoff" begin
        R = 0.6
        U = [R R; 1.0 0.0]
        mu = [0.5, 0.5]

        # Exact cutoff ω^* solving 0.5*exp((1-R)/ω^*) + 0.5*exp(-R/ω^*) = 1
        # Binary search on ω
        lo, hi = 0.05, 5.0
        for _ in 1:80
            mid = 0.5 * (lo + hi)
            val = 0.5 * exp((1.0 - R) / mid) + 0.5 * exp(-R / mid)
            if val > 1.0
                lo = mid
            else
                hi = mid
            end
        end
        omega_star = 0.5 * (lo + hi)
        @test abs(0.5 * exp((1.0 - R) / omega_star) + 0.5 * exp(-R / omega_star) - 1.0) < 1e-12

        # Above cutoff: risky option pruned completely
        sol_hi = solve_ri_discrete(U, mu; omega=1.25 * omega_star)
        @test sol_hi.converged
        @test sol_hi.consideration_set == [1]
        @test sol_hi.uncond_probs[1] == 1.0
        @test sol_hi.uncond_probs[2] == 0.0
        @test sol_hi.kkt_multipliers[1] ≈ 1.0 atol=1e-12
        @test sol_hi.kkt_multipliers[2] < 1.0
        @test sol_hi.mutual_info_nats ≈ 0.0 atol=1e-12
        @test sol_hi.expected_utility ≈ R atol=1e-12

        # Below cutoff: both options considered
        sol_lo = solve_ri_discrete(U, mu; omega=0.75 * omega_star)
        @test sol_lo.converged
        @test sol_lo.consideration_set == [1, 2]
        @test sol_lo.uncond_probs[2] > 0.0
        @test maximum(abs.(sol_lo.kkt_multipliers .- 1.0)) < 1e-10
        @test sol_lo.mutual_info_bits > 0.0
        @test sol_lo.expected_utility > R
    end

    # =========================================================================
    # 3. Matejka-McKay (2015) Red-Bus / Blue-Bus & IIA Breakdown
    # =========================================================================

    @testset "Matejka-McKay (2015) Red-Bus / Blue-Bus & ri_iia_diagnostic" begin
        U_base = [0.6 0.6; 1.0 0.0]
        mu = [0.5, 0.5]
        ω = 0.3

        diag = ri_iia_diagnostic(U_base, mu, U_base[2, :]; omega=ω)
        @test diag isa RIIIADiagnostic{Float64}
        sol_base = diag.base_solution
        sol_exp = diag.expanded_solution

        @test sol_base.consideration_set == [1, 2]
        @test sol_exp.consideration_set == [1, 2, 3]

        # Option 1 unconditional share is invariant; duplicate option 2 splits evenly
        @test abs(sol_exp.uncond_probs[1] - sol_base.uncond_probs[1]) < 1e-10
        @test abs(sol_exp.uncond_probs[2] - sol_exp.uncond_probs[3]) < 1e-12
        @test sol_exp.uncond_probs[2] + sol_exp.uncond_probs[3] ≈ sol_base.uncond_probs[2] atol=1e-10

        # Odds ratio P(1)/P(2) doubles -> relative shift = 1.0 (IIA breakdown!)
        @test diag.uncond_odds_ratio_shift[1, 2] ≈ 1.0 atol=1e-9
        @test diag.cond_odds_ratio_max_diff ≈ 1.0 atol=1e-9
        @test diag.max_iia_violation > 0.5

        # Capacity mode in IIA diagnostic
        diag_cap = ri_iia_diagnostic(U_base, mu, U_base[2, :]; capacity=0.3, units=:bits)
        @test diag_cap.max_iia_violation > 0.5
        @test abs(diag_cap.expanded_solution.uncond_probs[2] - diag_cap.expanded_solution.uncond_probs[3]) < 1e-12

        # Strictly dominated third option induces zero IIA violation
        diag_dom = ri_iia_diagnostic(U_base, mu, [-10.0, -10.0]; omega=ω)
        @test diag_dom.max_iia_violation < 1e-10

        @test_throws ArgumentError ri_iia_diagnostic(U_base, mu, [1.0, 2.0, 3.0]; omega=ω)
    end

    # =========================================================================
    # 4. Econometric Estimator (estimate_ri_logit): Cases A, B, C, K=0, StatsAPI
    # =========================================================================

    @testset "estimate_ri_logit — Case A (base_payoff given, recovers beta0 and omega0)" begin
        rng = Random.Xoshiro(92202)
        M, J, S, K, N = 15, 3, 4, 2, 1500
        menu = randn(rng, J, S, K, M)
        base_payoff = 0.8 .* randn(rng, J, S, M)
        mu = fill(1.0 / S, S)
        beta0 = [1.0, -0.5]
        omega0 = 0.5

        choices = Vector{Int}(undef, N)
        env_id = Vector{Int}(undef, N)
        state_id = Vector{Int}(undef, N)

        sols_true = [
            solve_ri_discrete(
                base_payoff[:, :, m] .+ reshape(reshape(menu[:, :, :, m], J * S, K) * beta0, J, S),
                mu; omega=omega0
            )
            for m in 1:M
        ]

        for i in 1:N
            m = rand(rng, 1:M)
            s = rand(rng, 1:S)
            env_id[i] = m
            state_id[i] = s
            choices[i] = _draw_cat(rng, sols_true[m].cond_probs[:, s])
        end

        m = estimate_ri_logit(choices, menu, env_id;
                              state_id=state_id,
                              base_payoff=base_payoff,
                              varnames=["price", "quality"])
        @test m isa RILogitModel{Float64}
        @test m.converged
        @test m.normalize === :none
        @test dof(m) == K + 1
        @test dof_residual(m) == N - (K + 1)

        se = stderror(m)
        @test all(isfinite, se)
        @test all(>(0.0), se)
        @test all(abs.(m.beta .- beta0) .<= 2.0 .* se[1:K])
        @test abs(m.omega - omega0) <= 2.0 * se[K + 1]

        # RI improves over fixed-intercept McFadden MNL benchmark
        @test m.loglik > m.loglik_mnl
        @test m.lr_vs_mnl ≈ 2.0 * (m.loglik - m.loglik_mnl)
        @test 0.0 <= m.lr_pval <= 1.0
        @test m.aic ≈ -2.0 * m.loglik + 2.0 * dof(m)
        @test m.bic ≈ -2.0 * m.loglik + log(N) * dof(m)
        @test 0.0 < m.pseudo_r2 < 1.0

        # Sandwich / robust covariance
        m_sw = estimate_ri_logit(choices, menu, env_id;
                                 state_id=state_id, base_payoff=base_payoff,
                                 cov_type=:sandwich)
        m_rob = estimate_ri_logit(choices, menu, env_id;
                                  state_id=state_id, base_payoff=base_payoff,
                                  cov_type=:robust)
        @test coef(m_sw) ≈ coef(m) atol=1e-6
        @test vcov(m_sw) ≈ vcov(m_rob) atol=1e-8
        @test all(>(0.0), stderror(m_sw))

        # Empirical priors and explicit priors
        m_emp = estimate_ri_logit(choices, menu, env_id;
                                  state_id=state_id, priors=:empirical,
                                  base_payoff=base_payoff)
        @test m_emp.converged
        @test all(abs.(m_emp.beta .- beta0) .<= 2.5 .* stderror(m_emp)[1:K])

        priors_mat = fill(1.0 / S, S, M)
        m_mat = estimate_ri_logit(choices, menu, env_id;
                                  state_id=state_id, priors=priors_mat,
                                  base_payoff=base_payoff)
        @test coef(m_mat) ≈ coef(m) atol=1e-8
    end

    @testset "estimate_ri_logit — K=0 3D base_payoff-only methods" begin
        rng = Random.Xoshiro(92203)
        M, J, S, N = 12, 3, 4, 1200
        base_payoff = randn(rng, J, S, M)
        mu = fill(1.0 / S, S)
        omega0 = 0.6

        sols = [solve_ri_discrete(base_payoff[:, :, m], mu; omega=omega0) for m in 1:M]
        choices = Vector{Int}(undef, N)
        env_id = Vector{Int}(undef, N)
        state_id = Vector{Int}(undef, N)
        for i in 1:N
            m = rand(rng, 1:M); s = rand(rng, 1:S)
            env_id[i] = m; state_id[i] = s
            choices[i] = _draw_cat(rng, sols[m].cond_probs[:, s])
        end

        m1 = estimate_ri_logit(choices, base_payoff, env_id; state_id=state_id)
        m2 = estimate_ri_logit(choices, env_id, base_payoff; state_id=state_id)
        @test m1.converged
        @test isempty(m1.beta)
        @test length(coef(m1)) == 1
        @test dof(m1) == 1
        @test size(vcov(m1)) == (1, 1)
        @test abs(m1.omega - omega0) <= 2.0 * stderror(m1)[1]
        @test coef(m2) ≈ coef(m1)

        # Single-environment 2-arg call
        bp1 = base_payoff[:, :, 1:1]
        idx1 = findall(==(1), env_id)
        if length(idx1) >= 20
            m_s1 = estimate_ri_logit(choices[idx1], bp1; state_id=state_id[idx1])
            @test m_s1 isa RILogitModel{Float64}
            @test dof(m_s1) == 1
        end
        @test_throws ArgumentError estimate_ri_logit(choices, base_payoff)
    end

    @testset "estimate_ri_logit — Case B (:first_coef) & Case C (:omega_one)" begin
        rng = Random.Xoshiro(92204)
        M, J, S, K, N = 15, 3, 4, 2, 1500
        menu = randn(rng, J, S, K, M)
        mu = fill(1.0 / S, S)
        beta0 = [1.2, -0.6]
        omega0 = 0.6

        sols = [
            solve_ri_discrete(reshape(reshape(menu[:, :, :, m], J * S, K) * beta0, J, S), mu; omega=omega0)
            for m in 1:M
        ]
        choices = Vector{Int}(undef, N)
        env_id = Vector{Int}(undef, N)
        state_id = Vector{Int}(undef, N)
        for i in 1:N
            m = rand(rng, 1:M); s = rand(rng, 1:S)
            env_id[i] = m; state_id[i] = s
            choices[i] = _draw_cat(rng, sols[m].cond_probs[:, s])
        end

        # Case B: normalize=:first_coef -> recovers (beta0 / beta0[1], omega0 / beta0[1])
        m_b = estimate_ri_logit(choices, menu, env_id; state_id=state_id, normalize=:first_coef)
        @test m_b.converged
        @test m_b.normalize === :first_coef
        @test m_b.beta[1] == 1.0
        @test dof(m_b) == K
        se_b = stderror(m_b)
        @test se_b[1] == 0.0
        @test all(==(0.0), vcov(m_b)[1, :])
        @test all(==(0.0), vcov(m_b)[:, 1])
        @test abs(m_b.beta[2] - beta0[2] / beta0[1]) <= 2.0 * se_b[2]
        @test abs(m_b.omega - omega0 / beta0[1]) <= 2.0 * se_b[3]

        # Case C: normalize=:omega_one (default) -> recovers beta0 / omega0
        m_c = estimate_ri_logit(choices, menu, env_id; state_id=state_id)
        @test m_c.converged
        @test m_c.normalize === :omega_one
        @test m_c.omega == 1.0
        @test dof(m_c) == K
        se_c = stderror(m_c)
        @test se_c[K + 1] == 0.0
        @test all(==(0.0), vcov(m_c)[K + 1, :])
        @test all(==(0.0), vcov(m_c)[:, K + 1])
        @test all(abs.(m_c.beta .- (beta0 ./ omega0)) .<= 2.0 .* se_c[1:K])

        # Identical maximized log-likelihood across Cases B, C, and unidentified :none
        m_u = estimate_ri_logit(choices, menu, env_id; state_id=state_id, normalize=:none)
        @test m_b.loglik ≈ m_c.loglik atol=1e-6
        @test m_u.loglik ≈ m_c.loglik atol=1e-6
        @test (m_u.beta ./ m_u.omega) ≈ m_c.beta atol=1e-4

        # Unconditional (state_id=nothing) estimation
        m_uncond = estimate_ri_logit(choices, menu, env_id)
        @test m_uncond isa RILogitModel{Float64}
        @test m_uncond.state_id === nothing
        @test m_uncond.converged
        @test size(fitted(m_uncond)) == (N, J)
    end

    @testset "StatsAPI, Display, Report, and Validation" begin
        rng = Random.Xoshiro(92205)
        M, J, S, K, N = 8, 3, 3, 2, 600
        menu = randn(rng, J, S, K, M)
        base_payoff = 0.5 .* randn(rng, J, S, M)
        mu = fill(1.0 / S, S)
        sols = [
            solve_ri_discrete(
                base_payoff[:, :, m] .+ reshape(reshape(menu[:, :, :, m], J * S, K) * [0.8, -0.4], J, S),
                mu; omega=0.6
            )
            for m in 1:M
        ]
        choices = Vector{Int}(undef, N)
        env_id = Vector{Int}(undef, N)
        state_id = Vector{Int}(undef, N)
        for i in 1:N
            m = rand(rng, 1:M); s = rand(rng, 1:S)
            env_id[i] = m; state_id[i] = s
            choices[i] = _draw_cat(rng, sols[m].cond_probs[:, s])
        end

        m = estimate_ri_logit(choices, menu, env_id; state_id=state_id, base_payoff=base_payoff)

        @test nobs(m) == N
        @test loglikelihood(m) == m.loglik
        @test aic(m) == m.aic
        @test bic(m) == m.bic
        @test r2(m) == m.pseudo_r2
        @test !islinear(m)
        ci = confint(m)
        @test size(ci) == (K + 1, 2)
        @test all(ci[:, 1] .<= coef(m) .<= ci[:, 2])

        P_fit = fitted(m)
        @test size(P_fit) == (N, J)
        @test all(abs.(sum(P_fit; dims=2) .- 1.0) .< 1e-12)
        @test predict(m) ≈ P_fit
        @test predict(m; kind=:conditional) ≈ P_fit
        P_uncond = predict(m; kind=:unconditional)
        @test size(P_uncond) == (N, J)
        @test all(abs.(sum(P_uncond; dims=2) .- 1.0) .< 1e-12)

        # Out-of-sample predictions (4D and 3D)
        P_oos4 = predict(m, menu[:, :, :, 1:2], [1, 2, 1];
                         state_id=[1, 2, 3], base_payoff=base_payoff[:, :, 1:2])
        @test size(P_oos4) == (3, J)
        @test all(abs.(sum(P_oos4; dims=2) .- 1.0) .< 1e-12)

        P_oos3_cond = predict(m, menu[:, :, :, 1]; base_payoff=base_payoff[:, :, 1], kind=:conditional)
        @test size(P_oos3_cond) == (J, S)
        @test all(abs.(sum(P_oos3_cond; dims=1) .- 1.0) .< 1e-12)

        P_oos3_unc = predict(m, menu[:, :, :, 1]; base_payoff=base_payoff[:, :, 1], kind=:unconditional)
        @test length(P_oos3_unc) == J
        @test sum(P_oos3_unc) ≈ 1.0

        # Display & report
        io = IOBuffer()
        show(io, sols[1])
        @test occursin("Rational Inattention Discrete-Choice Solution", String(take!(io)))
        report(io, sols[1])
        @test occursin("KKT", String(take!(io)))

        diag = ri_iia_diagnostic(sols[1].payoffs, mu, sols[1].payoffs[1, :]; omega=0.6)
        show(io, diag)
        @test occursin("IIA Breakdown Diagnostic", String(take!(io)))
        report(io, diag)
        @test length(String(take!(io))) > 0

        show(io, m)
        out_m = String(take!(io))
        @test occursin("Rational Inattention Discrete-Choice Logit", out_m)
        @test occursin("omega", out_m)
        report(io, m)
        @test length(String(take!(io))) > 0

        # Error paths
        @test_throws ArgumentError estimate_ri_logit(choices, menu, env_id; normalize=:bad)
        @test_throws ArgumentError estimate_ri_logit(choices, menu, env_id; base_payoff=base_payoff, normalize=:omega_one)
        @test_throws ArgumentError estimate_ri_logit(choices, menu, env_id; cov_type=:bad)
        @test_throws ArgumentError estimate_ri_logit(choices, menu, env_id; priors=:empirical)
        @test_throws ArgumentError estimate_ri_logit(choices, menu, env_id; priors=ones(S, M + 1))
        @test_throws ArgumentError estimate_ri_logit(choices, menu, env_id; varnames=["only_one"])
    end

end
