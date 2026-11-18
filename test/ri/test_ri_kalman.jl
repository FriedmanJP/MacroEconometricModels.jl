# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, LinearAlgebra, Statistics, Random

@testset "Sims (2003) Steady-State RI Kalman Filter (#917)" begin

    function _check_kalman_identities(sol::SimsKalmanSolution{T}; atol=T(1e-9), check_logdet::Bool=true) where {T}
        prob = sol.problem
        n = size(prob.A, 1)
        Q = prob.B * prob.B'
        Obase = prob.H * prob.M * prob.H'

        @test sol.converged
        @test sol.iterations >= 1
        @test isapprox(sol.Sigma_prior, sol.Sigma_prior'; atol=atol)
        @test isapprox(sol.Sigma_post, sol.Sigma_post'; atol=atol)
        @test minimum(eigvals(Symmetric(Matrix(sol.Sigma_post)))) >= -atol * max(norm(sol.Sigma_post), one(T))
        @test isapprox(sol.Sigma_prior, prob.A * sol.Sigma_post * prob.A' + Q; atol=atol * max(norm(sol.Sigma_prior), one(T)))
        @test isapprox(sol.F_policy, prob.H'; atol=atol)
        @test isapprox(sol.expected_loss, tr(Obase * sol.Sigma_post); rtol=T(1e-7), atol=atol)
        @test isapprox(sol.capacity_bits, sol.capacity_nats / log(T(2)); rtol=T(1e-7), atol=atol)

        r_s = sol.signal_rank
        @test size(sol.Y_signal) == (n, r_s)
        @test size(sol.Sigma_psi) == (r_s, r_s)
        @test size(sol.K_gain) == (n, r_s)

        if r_s == 0
            @test sol.capacity_bits == zero(T)
            @test sol.capacity_nats == zero(T)
            @test isapprox(sol.Sigma_post, sol.Sigma_prior; atol=atol)
        else
            @test isapprox(sol.Y_signal' * sol.Y_signal, Matrix{T}(I, r_s, r_s); atol=atol)
            @test isdiag(sol.Sigma_psi) || isapprox(sol.Sigma_psi, Diagonal(diag(sol.Sigma_psi)); atol=atol)
            @test all(>(zero(T)), diag(sol.Sigma_psi))

            # Kalman update identity: Σ_post = (I - K Y') Σ_prior
            IKY = Matrix{T}(I, n, n) - sol.K_gain * sol.Y_signal'
            @test isapprox(sol.Sigma_post, IKY * sol.Sigma_prior; atol=atol * max(norm(sol.Sigma_prior), one(T)))
            # Joseph form
            @test isapprox(sol.Sigma_post,
                IKY * sol.Sigma_prior * IKY' + sol.K_gain * sol.Sigma_psi * sol.K_gain';
                atol=atol * max(norm(sol.Sigma_prior), one(T)))
            # Kalman gain formula: K = Σ_prior Y (Y' Σ_prior Y + Σ_ψ)^{-1}
            S_innov = sol.Y_signal' * sol.Sigma_prior * sol.Y_signal + sol.Sigma_psi
            @test isapprox(sol.K_gain, sol.Sigma_prior * sol.Y_signal / S_innov;
                atol=atol * max(norm(sol.K_gain), one(T)))

            if check_logdet && minimum(eigvals(Symmetric(Matrix(sol.Sigma_prior)))) > T(1e-12)
                I_ld = T(0.5) * (logdet(Symmetric(Matrix(sol.Sigma_prior))) - logdet(Symmetric(Matrix(sol.Sigma_post))))
                @test isapprox(sol.capacity_nats, I_ld; rtol=T(1e-7), atol=atol)
            end
        end
    end

    # =========================================================================
    # 1. Constructor & Input Validation
    # =========================================================================
    @testset "Constructor & Validation" begin
        # Scalar ergonomic construction
        p_scalar = RationalInattentionProblem(0.8, 1.2, 1.0; capacity=1.5)
        @test p_scalar isa RationalInattentionProblem{Float64}
        @test p_scalar.A == [0.8;;]
        @test p_scalar.B == [1.2;;]
        @test p_scalar.H == [1.0;;]
        @test p_scalar.M == [1.0;;]
        @test p_scalar.B_u === nothing
        @test p_scalar.beta == 1.0
        @test p_scalar.capacity == 1.5
        @test p_scalar.omega === nothing
        @test p_scalar.units === :bits
        @test p_scalar.state_names == ["x1"]
        @test p_scalar.action_names == ["u1"]
        @test p_scalar.shock_names == ["w1"]

        # Vector H, B, B_u and custom labels
        A2 = [0.7 0.1; 0.0 0.5]
        p_vec = RationalInattentionProblem(A2, [1.0, 0.5], [1.0, 2.0];
            B_u=[0.2, -0.1], beta=0.95, omega=0.4, units=:nats,
            state_names=[:y, :pi], action_names=["i"], shock_names=["e"])
        @test size(p_vec.B) == (2, 1)
        @test size(p_vec.H) == (2, 1)
        @test size(p_vec.B_u) == (2, 1)
        @test p_vec.state_names == ["y", "pi"]
        @test p_vec.action_names == ["i"]
        @test p_vec.shock_names == ["e"]

        # AbstractMatrix subtypes (Diagonal, Adjoint)
        p_diag = RationalInattentionProblem(Diagonal([0.8, 0.6]), Diagonal([1.0, 0.5]),
            [1.0 0.0; 0.0 1.0]; M=Diagonal([2.0, 1.0]), capacity=1.0)
        @test p_diag.M == [2.0 0.0; 0.0 1.0]

        # Mutual exclusivity of capacity & omega
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=1.0, omega=1.0)

        # Invalid scalars / enums
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=0.0)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=-1.0)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=Inf)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; omega=0.0)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; omega=-0.5)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; omega=NaN)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=1.0, beta=0.0)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=1.0, beta=1.05)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0; capacity=1.0, units=:bytes)

        # Dimension & matrix property errors
        @test_throws ArgumentError RationalInattentionProblem([0.8 0.1], 1.0, 1.0; capacity=1.0)
        @test_throws ArgumentError RationalInattentionProblem(A2, [1.0;;], [1.0, 0.0]; capacity=1.0)
        @test_throws ArgumentError RationalInattentionProblem(A2, [1.0, 0.0], [1.0;;]; capacity=1.0)
        @test_throws ArgumentError RationalInattentionProblem(A2, [1.0, 0.0], [1.0, 0.0];
            M=ones(2, 2), capacity=1.0)
        @test_throws ArgumentError RationalInattentionProblem(A2, [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0];
            M=[1.0 0.5; 0.0 1.0], capacity=1.0) # asymmetric M
        @test_throws ArgumentError RationalInattentionProblem(A2, [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0];
            M=[1.0 0.0; 0.0 -0.5], capacity=1.0) # indefinite M
        @test_throws ArgumentError RationalInattentionProblem(A2, [1.0, 0.0], [1.0, 0.0];
            B_u=[1.0 0.0; 0.0 1.0], capacity=1.0)
        @test_throws ArgumentError RationalInattentionProblem(0.8, 1.2, 1.0;
            capacity=1.0, state_names=["a", "b"])
        @test_throws ArgumentError RationalInattentionProblem(0.8, NaN, 1.0; capacity=1.0)

        # Invalid solver options
        @test_throws ArgumentError solve_ri_kalman(p_scalar; method=:invalid)
        @test_throws ArgumentError solve_ri_kalman(p_scalar; tol=0.0)
        @test_throws ArgumentError solve_ri_kalman(p_scalar; max_iter=0)

        # Degenerate zero-benefit / zero-shock capacity allocation error
        p_zero_B = RationalInattentionProblem(0.8, 0.0, 1.0; capacity=1.0)
        @test_throws ArgumentError solve_ri_kalman(p_zero_B)
        p_zero_M = RationalInattentionProblem(0.8, 1.2, 1.0; M=0.0, capacity=1.0)
        @test_throws ArgumentError solve_ri_kalman(p_zero_M)
    end

    # =========================================================================
    # 2. AC 1: Scalar AR(1) Analytical Benchmark (Sims 2003)
    # =========================================================================
    @testset "AC 1: Scalar AR(1) Analytical Solution" begin
        rho = 0.8
        sigma_w = 1.2
        kappa_bits = 1.5
        prob = RationalInattentionProblem(rho, sigma_w, 1.0; capacity=kappa_bits, beta=1.0, units=:bits)

        r_info = 2.0^(2 * kappa_bits) # e^{2 κ_nats} = 8.0
        Sigma_post_exact = sigma_w^2 / (r_info - rho^2)
        Sigma_prior_exact = r_info * Sigma_post_exact
        K_exact = 1.0 - 2.0^(-2 * kappa_bits)
        omega_exact = 2.0 * Sigma_post_exact / (1.0 - rho^2 / r_info)

        for meth in (:water_filling, :logdet_barrier)
            sol = solve_ri_kalman(prob; method=meth)
            @test sol isa SimsKalmanSolution{Float64}
            @test sol.method === meth
            @test sol.converged
            @test sol.signal_rank == 1
            @test abs(sol.Sigma_post[1, 1] - Sigma_post_exact) < 1e-10
            @test abs(sol.Sigma_prior[1, 1] - Sigma_prior_exact) < 1e-10
            @test abs(sol.K_gain[1, 1] - K_exact) < 1e-10
            @test abs(sol.omega - omega_exact) < 1e-10
            @test abs(sol.capacity_bits - kappa_bits) < 1e-10
            @test sol.Y_signal ≈ [1.0;;] atol=1e-12
            _check_kalman_identities(sol; atol=1e-10)
        end

        # Dual fixed-omega check recovers exact kappa and Sigma_post
        prob_om = RationalInattentionProblem(rho, sigma_w, 1.0; omega=omega_exact, beta=1.0)
        for meth in (:water_filling, :logdet_barrier)
            sol_om = solve_ri_kalman(prob_om; method=meth)
            @test abs(sol_om.Sigma_post[1, 1] - Sigma_post_exact) < 1e-10
            @test abs(sol_om.K_gain[1, 1] - K_exact) < 1e-10
            @test abs(sol_om.capacity_bits - kappa_bits) < 1e-10
        end

        # DARE cross-check: prediction covariance from _solve_dare matches Sigma_prior
        sol_wf = solve_ri_kalman(prob; method=:water_filling)
        P_dare, _, _, res_dare = MacroEconometricModels._solve_dare(
            prob.A, prob.B * prob.B', sol_wf.Y_signal', sol_wf.Sigma_psi)
        @test res_dare < 1e-12
        @test isapprox(P_dare, sol_wf.Sigma_prior; atol=1e-10)
    end

    # =========================================================================
    # 3. AC 2: Multivariate n=5, d=1 & Multi-Action Agreement
    # =========================================================================
    @testset "AC 2: Multivariate n=5, d=1 & Solver Cross-Agreement" begin
        # General 5-state system with scalar action tracking
        A5 = diagm([0.9, 0.7, 0.5, 0.3, 0.1]) +
             [0.0 0.05 0.0 0.0 0.0;
              0.0 0.0 0.05 0.0 0.0;
              0.0 0.0 0.0 0.05 0.0;
              0.0 0.0 0.0 0.0 0.05;
              0.0 0.0 0.0 0.0 0.0]
        B5 = Matrix{Float64}(I, 5, 5)
        h5 = [1.0, 0.5, 0.2, 0.1, 0.05]

        # Both capacity-constrained and omega-constrained
        for prob5 in (
            RationalInattentionProblem(A5, B5, h5; beta=0.99, capacity=1.0, units=:bits),
            RationalInattentionProblem(A5, B5, h5; beta=0.99, omega=1.5),
        )
            sol_wf = solve_ri_kalman(prob5; method=:water_filling)
            sol_lb = solve_ri_kalman(prob5; method=:logdet_barrier)

            @test sol_wf.converged && sol_lb.converged
            @test sol_wf.signal_rank == 1
            @test sol_lb.signal_rank == 1
            @test size(sol_wf.Y_signal) == (5, 1)
            @test size(sol_lb.Y_signal) == (5, 1)

            @test isapprox(sol_wf.Sigma_post, sol_lb.Sigma_post; atol=1e-7)
            @test isapprox(sol_wf.Sigma_prior, sol_lb.Sigma_prior; atol=1e-7)
            @test isapprox(sol_wf.Omega_benefit, sol_lb.Omega_benefit; atol=1e-7)
            @test isapprox(sol_wf.omega, sol_lb.omega; rtol=1e-7, atol=1e-9)
            @test isapprox(sol_wf.Y_signal, sol_lb.Y_signal; atol=1e-7)
            @test isapprox(sol_wf.K_gain, sol_lb.K_gain; atol=1e-7)
            @test isapprox(sol_wf.expected_loss, sol_lb.expected_loss; atol=1e-7)

            _check_kalman_identities(sol_wf; atol=1e-9)
            _check_kalman_identities(sol_lb; atol=1e-8)
        end

        # Multi-rank n=4, d=3 problem: test active rank > 1 agreement
        A4 = [0.85 0.1 0.0 0.0; 0.0 0.75 0.1 0.0; 0.0 0.0 0.65 0.05; 0.0 0.0 0.0 0.5]
        B4 = [1.0 0.2 0.0; 0.0 0.8 0.1; 0.0 0.0 0.7; 0.1 0.0 0.6]
        H4 = [1.0 0.2 0.0; 0.0 1.0 0.1; 0.0 0.0 1.0; 0.2 -0.1 0.3]
        M4 = [2.0 0.3 0.0; 0.3 1.5 0.1; 0.0 0.1 1.0]
        prob4 = RationalInattentionProblem(A4, B4, H4; M=M4, beta=0.95, omega=1.2)
        s4_wf = solve_ri_kalman(prob4; method=:water_filling)
        s4_lb = solve_ri_kalman(prob4; method=:logdet_barrier)
        @test s4_wf.signal_rank >= 2
        @test s4_lb.signal_rank == s4_wf.signal_rank
        @test isapprox(s4_wf.Sigma_post, s4_lb.Sigma_post; atol=1e-7)
        @test isapprox(s4_wf.Omega_benefit, s4_lb.Omega_benefit; atol=1e-7)
        _check_kalman_identities(s4_wf; atol=1e-9)
        _check_kalman_identities(s4_lb; atol=1e-8)
    end

    # =========================================================================
    # 4. AC 3: Zero-Attention Threshold
    # =========================================================================
    @testset "AC 3: Zero-Attention Threshold" begin
        A = [0.8 0.15 0.0; 0.0 0.65 0.1; 0.0 0.0 0.4]
        B = [1.0 0.2; 0.0 0.8; 0.3 0.1]
        H = [1.0 0.0; 0.4 1.0; 0.0 0.5]
        M = [1.5 0.2; 0.2 1.0]
        beta = 0.96

        Sigma_uncond = solve_lyapunov(A, B)
        Obase = H * M * H'
        Omega_eff_zero = MacroEconometricModels._dlyap(sqrt(beta) * Matrix(A'), Obase; tol=1e-13)
        F_u = eigen(Symmetric(Sigma_uncond))
        S_u_half = F_u.vectors * Diagonal(sqrt.(max.(F_u.values, 0.0))) * F_u.vectors'
        omega_zero = 2.0 * eigmax(Symmetric(S_u_half * Omega_eff_zero * S_u_half))

        # At exact threshold and strictly above threshold
        for om_test in (omega_zero, 1.05 * omega_zero, 5.0 * omega_zero)
            prob_z = RationalInattentionProblem(A, B, H; M=M, beta=beta, omega=om_test)
            for meth in (:water_filling, :logdet_barrier)
                sol_z = solve_ri_kalman(prob_z; method=meth)
                @test sol_z.converged
                @test sol_z.capacity_bits == 0.0
                @test sol_z.capacity_nats == 0.0
                @test sol_z.signal_rank == 0
                @test isempty(sol_z.K_gain) || all(iszero, sol_z.K_gain)
                @test size(sol_z.Y_signal) == (3, 0)
                @test size(sol_z.Sigma_psi) == (0, 0)
                @test isapprox(sol_z.Sigma_post, Sigma_uncond; atol=1e-10)
                @test isapprox(sol_z.Sigma_prior, Sigma_uncond; atol=1e-10)
                @test isapprox(sol_z.Omega_benefit, Omega_eff_zero; atol=1e-10)
            end
        end

        # Just below threshold: positive attention activates
        prob_active = RationalInattentionProblem(A, B, H; M=M, beta=beta, omega=0.95 * omega_zero)
        sol_act = solve_ri_kalman(prob_active; method=:water_filling)
        @test sol_act.capacity_bits > 0.0
        @test sol_act.signal_rank >= 1

        # Degenerate zero uncertainty / zero payoff under fixed omega returns zero attention cleanly
        sol_no_shock = solve_ri_kalman(RationalInattentionProblem(0.8, 0.0, 1.0; omega=1.0))
        @test sol_no_shock.converged && sol_no_shock.signal_rank == 0 && sol_no_shock.capacity_bits == 0.0
        sol_no_loss = solve_ri_kalman(RationalInattentionProblem(0.8, 1.0, 1.0; M=0.0, omega=1.0))
        @test sol_no_loss.converged && sol_no_loss.signal_rank == 0 && sol_no_loss.capacity_bits == 0.0
    end

    # =========================================================================
    # 5. AC 4: Units, Discounting, B_u, Float32, and Kalman Identities
    # =========================================================================
    @testset "AC 4: Units, β, B_u, Float32 & Identities" begin
        A = [0.75 0.1; 0.05 0.6]
        B = [1.0 0.0; 0.2 0.8]
        H = [1.0 0.3; 0.0 1.0]

        # :bits vs :nats equivalence
        cap_bits = 1.25
        p_bits = RationalInattentionProblem(A, B, H; beta=0.95, capacity=cap_bits, units=:bits)
        p_nats = RationalInattentionProblem(A, B, H; beta=0.95, capacity=cap_bits * log(2.0), units=:nats)
        s_bits = solve_ri_kalman(p_bits)
        s_nats = solve_ri_kalman(p_nats)
        @test isapprox(s_bits.Sigma_post, s_nats.Sigma_post; atol=1e-11)
        @test isapprox(s_bits.omega, s_nats.omega; atol=1e-11)
        @test isapprox(s_bits.K_gain, s_nats.K_gain; atol=1e-11)
        @test isapprox(s_bits.capacity_bits, cap_bits; atol=1e-10)
        @test isapprox(s_nats.capacity_nats, cap_bits * log(2.0); atol=1e-10)

        # β < 1 vs β = 1 comparative statics
        # Under fixed omega, higher β raises future precision shadow value -> higher attention & lower loss
        s_b05 = solve_ri_kalman(RationalInattentionProblem(A, B, H; beta=0.5, omega=1.0))
        s_b10 = solve_ri_kalman(RationalInattentionProblem(A, B, H; beta=1.0, omega=1.0))
        @test s_b10.capacity_bits > s_b05.capacity_bits
        @test s_b10.expected_loss < s_b05.expected_loss
        # Under fixed capacity in 1D, Sigma_post is invariant to β while omega increases with β
        s1_b05 = solve_ri_kalman(RationalInattentionProblem(0.8, 1.2, 1.0; beta=0.5, capacity=1.5))
        s1_b10 = solve_ri_kalman(RationalInattentionProblem(0.8, 1.2, 1.0; beta=1.0, capacity=1.5))
        @test isapprox(s1_b05.Sigma_post, s1_b10.Sigma_post; atol=1e-12)
        @test s1_b10.omega > s1_b05.omega

        # Controlled LQG with B_u !== nothing: error dynamics invariant to known past control
        B_u = [0.5 -0.2; 0.1 0.4]
        p_ctrl = RationalInattentionProblem(A, B, H; B_u=B_u, beta=0.95, omega=1.0)
        p_noctrl = RationalInattentionProblem(A, B, H; beta=0.95, omega=1.0)
        s_ctrl = solve_ri_kalman(p_ctrl)
        s_noctrl = solve_ri_kalman(p_noctrl)
        @test isapprox(s_ctrl.Sigma_post, s_noctrl.Sigma_post; atol=1e-13)
        @test isapprox(s_ctrl.Sigma_prior, s_noctrl.Sigma_prior; atol=1e-13)
        @test isapprox(s_ctrl.K_gain, s_noctrl.K_gain; atol=1e-13)
        @test s_ctrl.F_policy == Matrix{Float64}(H')
        _check_kalman_identities(s_ctrl; atol=1e-10)

        # Float32 precision support (both with default beta=1.0 and explicit Float32)
        A32 = Float32[0.8 0.1; 0.0 0.6]
        B32 = Float32[1.2 0.0; 0.0 0.9]
        H32 = Float32[1.0, 0.5]
        p32_cap = RationalInattentionProblem(A32, B32, H32; capacity=1.2f0)
        p32_om = RationalInattentionProblem(A32, B32, H32; beta=0.95f0, omega=1.5f0)
        @test p32_cap isa RationalInattentionProblem{Float32}
        @test p32_om isa RationalInattentionProblem{Float32}
        for meth in (:water_filling, :logdet_barrier)
            s32 = solve_ri_kalman(p32_cap; method=meth)
            @test s32 isa SimsKalmanSolution{Float32}
            @test eltype(s32.Sigma_post) === Float32
            @test eltype(s32.K_gain) === Float32
            @test eltype(s32.Y_signal) === Float32
            @test typeof(s32.omega) === Float32
            _check_kalman_identities(s32; atol=1f-4)

            s32_om = solve_ri_kalman(p32_om; method=meth)
            @test s32_om isa SimsKalmanSolution{Float32}
            _check_kalman_identities(s32_om; atol=1f-4)
        end
        # Compare Float32 against Float64 reference
        p64_cap = RationalInattentionProblem(Float64.(A32), Float64.(B32), Float64.(H32); capacity=1.2)
        @test isapprox(solve_ri_kalman(p32_cap).Sigma_post, Float32.(solve_ri_kalman(p64_cap).Sigma_post); atol=1e-5)
    end

    # =========================================================================
    # 6. Unstable / Unit-Root & Rank-Deficient Q Edge Cases
    # =========================================================================
    @testset "Unstable Systems & Singular Q" begin
        # 1D unstable AR(1): ρ = 1.1, insufficient vs sufficient capacity
        p_uns_bad = RationalInattentionProblem(1.1, 1.0, 1.0;
            capacity=log(1.1) / log(2.0) * 0.99, units=:bits)
        @test_throws ArgumentError solve_ri_kalman(p_uns_bad)

        p_uns_ok = RationalInattentionProblem(1.1, 1.0, 1.0; capacity=1.5, units=:bits)
        sp_exact_u = 1.0 / (2.0^3.0 - 1.1^2)
        for meth in (:water_filling, :logdet_barrier)
            su = solve_ri_kalman(p_uns_ok; method=meth)
            @test su.converged
            @test abs(su.Sigma_post[1, 1] - sp_exact_u) < 1e-10
            _check_kalman_identities(su; atol=1e-9)
        end

        # Unit root ρ = 1.0
        p_ur = RationalInattentionProblem(1.0, 1.0, 1.0; capacity=1.0, units=:bits)
        su_ur = solve_ri_kalman(p_ur)
        @test abs(su_ur.Sigma_post[1, 1] - 1.0 / 3.0) < 1e-10

        # 2D mixed unstable/stable system
        A_u2 = [1.15 0.2; 0.0 0.5]
        B_u2 = Matrix{Float64}(I, 2, 2)
        H_u2 = Matrix{Float64}(I, 2, 2)
        @test_throws ArgumentError solve_ri_kalman(
            RationalInattentionProblem(A_u2, B_u2, H_u2; capacity=log(1.15), units=:nats))
        su2_wf = solve_ri_kalman(
            RationalInattentionProblem(A_u2, B_u2, H_u2; capacity=1.5, units=:bits); method=:water_filling)
        su2_lb = solve_ri_kalman(
            RationalInattentionProblem(A_u2, B_u2, H_u2; capacity=1.5, units=:bits); method=:logdet_barrier)
        @test su2_wf.converged && su2_lb.converged
        @test isapprox(su2_wf.Sigma_post, su2_lb.Sigma_post; atol=1e-7)
        _check_kalman_identities(su2_wf; atol=1e-9)
        _check_kalman_identities(su2_lb; atol=1e-8)

        # Undetectable unstable mode under fixed omega diverges -> ArgumentError
        @test_throws ArgumentError solve_ri_kalman(
            RationalInattentionProblem([1.1 0.0; 0.0 0.5], [1.0 0.0; 0.0 1.0], [0.0, 1.0]; omega=1.0))

        # Companion AR(3) with rank-1 shock Q (m = 1 < n = 3)
        A_ar3 = [0.6 0.2 -0.05; 1.0 0.0 0.0; 0.0 1.0 0.0]
        B_ar3 = reshape([1.0, 0.0, 0.0], 3, 1)
        h_ar3 = [1.0, 0.0, 0.0]
        p_ar3 = RationalInattentionProblem(A_ar3, B_ar3, h_ar3; beta=0.98, capacity=1.0)
        s_ar3_wf = solve_ri_kalman(p_ar3; method=:water_filling)
        s_ar3_lb = solve_ri_kalman(p_ar3; method=:logdet_barrier)
        @test s_ar3_wf.converged && s_ar3_lb.converged
        @test isapprox(s_ar3_wf.Sigma_post, s_ar3_lb.Sigma_post; atol=1e-7)
        _check_kalman_identities(s_ar3_wf; atol=1e-9, check_logdet=false)
        _check_kalman_identities(s_ar3_lb; atol=1e-8, check_logdet=false)
    end

    # =========================================================================
    # 7. Show, Report & Module Ambiguities
    # =========================================================================
    @testset "Display, Report & Ambiguities" begin
        @test isempty(Test.detect_ambiguities(MacroEconometricModels))

        prob = RationalInattentionProblem([0.8 0.1; 0.0 0.6], [1.0 0.0; 0.0 0.8], [1.0, 0.5];
            capacity=1.0, state_names=["output", "inflation"], action_names=["rate"])
        sol = solve_ri_kalman(prob)
        sol_zero = solve_ri_kalman(RationalInattentionProblem(0.8, 1.0, 1.0; omega=100.0))

        s_prob = sprint(show, prob)
        r_prob = sprint(report, prob)
        s_sol = sprint(show, sol)
        r_sol = sprint(report, sol)
        r_zero = sprint(report, sol_zero)

        @test occursin("Rational Inattention Problem", s_prob)
        @test occursin("State Transition A", r_prob)
        @test occursin("Sims (2003)", s_sol)
        @test occursin("Endogenous Signal Loadings", r_sol)
        @test occursin("Sims Kalman Gain", r_sol)
        @test occursin("Signal rank", r_zero)

        # Backends (LaTeX / HTML)
        latex_out = with_display_backend(:latex) do
            sprint(report, sol)
        end
        html_out = with_display_backend(:html) do
            sprint(report, sol)
        end
        @test occursin("\\begin{tabular}", latex_out)
        @test occursin("<table", html_out)
    end
end

@testset "RI Signal Realization, State Space, IRF, FEVD & Simulation (#919)" begin

    # =========================================================================
    # 1. Exact Kalman & Mutual Information Identities
    # =========================================================================
    @testset "Kalman Update & Signal Mutual Information Identities" begin
        # 1D and multivariate multi-rank systems
        probs = (
            RationalInattentionProblem(0.85, 1.1, 1.0; beta=0.98, capacity=1.3, units=:bits),
            RationalInattentionProblem(
                [0.85 0.1 0.0 0.0; 0.0 0.75 0.1 0.0; 0.0 0.0 0.65 0.05; 0.0 0.0 0.0 0.5],
                [1.0 0.2 0.0; 0.0 0.8 0.1; 0.0 0.0 0.7; 0.1 0.0 0.6],
                [1.0 0.2 0.0; 0.0 1.0 0.1; 0.0 0.0 1.0; 0.2 -0.1 0.3];
                M=[2.0 0.3 0.0; 0.3 1.5 0.1; 0.0 0.1 1.0],
                beta=0.95, omega=1.2,
            ),
        )
        for prob in probs
            sol = solve_ri_kalman(prob)
            n = size(prob.A, 1)
            In = Matrix{Float64}(I, n, n)
            IKY = In - sol.K_gain * sol.Y_signal'
            @test isapprox(IKY * sol.Sigma_prior, sol.Sigma_post; atol=1e-10)
            S_sig = sol.Y_signal' * sol.Sigma_prior * sol.Y_signal + sol.Sigma_psi
            mi_nats = 0.5 * (logdet(Symmetric(Matrix(S_sig))) - logdet(Symmetric(Matrix(sol.Sigma_psi))))
            @test abs(mi_nats - sol.capacity_nats) < 1e-10
        end
    end

    # =========================================================================
    # 2. Joint State Space, to_statespace, to_abcd & Corner Cases
    # =========================================================================
    @testset "to_statespace, to_abcd & Corner Cases" begin
        A = [0.8 0.1; 0.0 0.6]
        B = [1.0 0.0; 0.2 0.8]
        H = [1.0, 0.5]
        B_u = [0.15, -0.05]
        prob = RationalInattentionProblem(A, B, H; B_u=B_u, beta=0.96, capacity=1.2,
            state_names=["y", "pi"], action_names=["i"], shock_names=["e_y", "e_pi"])
        sol = solve_ri_kalman(prob)
        n, m, d, r_s = 2, 2, 1, sol.signal_rank

        # Validation errors
        @test_throws ArgumentError to_statespace(sol; observables=:invalid)
        @test_throws ArgumentError to_abcd(sol; observables=:invalid)
        @test_throws ArgumentError irf(sol, 0)
        @test_throws ArgumentError fevd(sol, 0)
        @test_throws ArgumentError simulate(sol, 0)
        @test_throws ArgumentError simulate(sol, 10; burn=-1)
        @test_throws ArgumentError simulate(sol, 10; x0=[1.0])
        @test_throws ArgumentError simulate(sol, 10; xhat0=[1.0, NaN])

        # to_statespace across observables modes
        for (obs_sym, k_exp) in ((:actions, d), (:actions_and_states, n + d), (:all, 2n + d))
            ssm = to_statespace(sol; observables=obs_sym)
            @test ssm isa StateSpaceModel{Float64}
            @test ssm.n_obs == k_exp
            @test ssm.n_state == 2n
            @test size(ssm.R) == (2n, m + r_s)
            @test iszero(ssm.H)
            @test ssm.Q == Matrix{Float64}(I, m + r_s, m + r_s)
            @test ssm.init_mode === :stationary
        end

        # to_abcd matches irf(sol) exactly and matches StateSpaceModel stationary covariance
        for (obs_sym, k_exp) in ((:actions, d), (:actions_and_states, n + d), (:all, 2n + d))
            abcd = to_abcd(sol; observables=obs_sym)
            @test abcd isa ABCDSystem{Float64}
            @test nstates(abcd) == 2n
            @test nobservables(abcd) == k_exp
            @test nshocks(abcd) == m + r_s
            @test abcd.shock_names[1:m] == ["Fundamental: e_y", "Fundamental: e_pi"]
            @test abcd.shock_names[(m + 1):(m + r_s)] == ["RI Noise: s$(j)" for j in 1:r_s]

            ir_abcd = irf(abcd, 20)
            ir_sol = irf(sol, 20; observables=obs_sym)
            @test ir_sol isa ImpulseResponse{Float64}
            @test ir_sol.horizon == 20
            @test size(ir_sol.values) == (20, k_exp, m + r_s)
            @test ir_sol.variables == ir_abcd.variables
            @test ir_sol.shocks == ir_abcd.shocks
            @test isapprox(ir_sol.values, ir_abcd.values; atol=1e-12)
        end

        # Check control feedback B_u inside T_joint and analytical covariance parity
        abcd_all = to_abcd(sol; observables=:all)
        ssm_all = to_statespace(sol; observables=:all)
        @test isapprox(abcd_all.A[1:n, (n + 1):2n], B_u * sol.F_policy; atol=1e-12)
        G0_abcd = analytical_moments(abcd_all; lags=0, format=:matrices)[1]
        G0_ssm = ssm_all.Z * ssm_all.P1 * ssm_all.Z'
        @test isapprox(G0_abcd, G0_ssm; atol=1e-10)

        # r_s == 0 zero-attention corner case
        sol_zero = solve_ri_kalman(RationalInattentionProblem(0.8, 1.0, 1.0; omega=100.0))
        @test sol_zero.signal_rank == 0
        ssm_z = to_statespace(sol_zero; observables=:all)
        abcd_z = to_abcd(sol_zero; observables=:all)
        ir_z = irf(sol_zero, 10; observables=:all)
        fv_z = fevd(sol_zero, 10; observables=:actions)
        sim_z = simulate(sol_zero, 40; burn=10, seed=7)
        @test size(ssm_z.R) == (2, 1)
        @test nshocks(abcd_z) == 1
        @test size(ir_z.values) == (10, 3, 1)
        @test all(iszero, ir_z.values[:, 2:3, :]) # zero belief & action response
        @test size(fv_z.proportions) == (1, 1, 10)
        @test size(sim_z.signals) == (40, 0)
        @test size(sim_z.ri_noise) == (40, 0)
        @test all(iszero, sim_z.beliefs)
        @test all(iszero, sim_z.actions)
    end

    # =========================================================================
    # 3. Sims (2003) Inertial / Hump-Shaped IRF & RI Processing-Noise FEVD
    # =========================================================================
    @testset "Sims (2003) Inertial IRF & RI Noise FEVD Transition" begin
        rho = 0.9
        sigma_w = 1.0
        H = 25

        sol_low = solve_ri_kalman(RationalInattentionProblem(rho, sigma_w, 1.0; capacity=0.1, units=:bits))
        sol_mid = solve_ri_kalman(RationalInattentionProblem(rho, sigma_w, 1.0; capacity=0.8, units=:bits))
        sol_high = solve_ri_kalman(RationalInattentionProblem(rho, sigma_w, 1.0; capacity=20.0, units=:bits))

        ir_low = irf(sol_low, H; observables=:all)
        ir_mid = irf(sol_mid, H; observables=:all)
        ir_high = irf(sol_high, H; observables=:all)

        # Row 1 = x_t, Row 2 = xhat_{t|t}, Row 3 = u_t; Col 1 = fundamental shock, Col 2 = RI noise
        u_resp_low = ir_low.values[:, 3, 1]
        u_resp_mid = ir_mid.values[:, 3, 1]
        u_resp_high = ir_high.values[:, 3, 1]
        x_resp_high = ir_high.values[:, 1, 1]

        # At low capacity (0.1 bits): sluggish impact and strictly delayed hump-shaped peak
        @test u_resp_low[1] < 0.25
        @test argmax(u_resp_low) > 1
        @test maximum(u_resp_low) > u_resp_low[1] * 1.5

        # Monotonic impact increase toward full info across capacities
        @test u_resp_low[1] < u_resp_mid[1] < u_resp_high[1]
        @test isapprox(u_resp_high, x_resp_high; atol=1e-3)
        @test argmax(u_resp_high) == 1

        # FEVD: non-trivial RI noise share at κ=0.1 bits, vanishing (< 1e-4) at κ=20.0 bits
        fv_low = fevd(sol_low, H; observables=:actions)
        fv_mid = fevd(sol_mid, H; observables=:actions)
        fv_high = fevd(sol_high, H; observables=:actions)

        @test fv_low isa FEVD{Float64}
        @test size(fv_low.proportions) == (1, 2, H)
        for h in 1:H
            @test isapprox(sum(fv_low.proportions[1, :, h]), 1.0; atol=1e-12)
            @test isapprox(sum(fv_high.proportions[1, :, h]), 1.0; atol=1e-12)
        end

        noise_share_low = fv_low.proportions[1, 2, 1]
        noise_share_mid = fv_mid.proportions[1, 2, 1]
        noise_share_high = fv_high.proportions[1, 2, end]
        @test noise_share_low > 0.25
        @test noise_share_low > noise_share_mid > noise_share_high
        @test noise_share_high < 1e-4
    end

    # =========================================================================
    # 4. Luo (2008) Permanent-Income RI Benchmark (Excess Smoothness & Sensitivity)
    # =========================================================================
    @testset "Luo (2008) Permanent-Income RI Benchmark" begin
        # Exact unit-root permanent income: y_{p,t} = y_{p,t-1} + σ_w w_t
        sigma_w = 1.0
        kappa_bits = 0.8
        prob_pi = RationalInattentionProblem(1.0, sigma_w, 1.0; capacity=kappa_bits, units=:bits)
        sol_pi = solve_ri_kalman(prob_pi)
        K_pi = sol_pi.K_gain[1, 1] # 1 - 2^(-2κ)

        # Analytical excess smoothness of fundamental consumption change & positive excess sensitivity
        ir_pi = irf(sol_pi, 80; observables=:all)
        c_irf = ir_pi.values[:, 3, 1]
        dc_irf = vcat(c_irf[1], diff(c_irf))
        var_dc_fund = sum(abs2, dc_irf)
        @test isapprox(var_dc_fund, (K_pi / (2.0 - K_pi)) * sigma_w^2; atol=1e-10)
        @test var_dc_fund < sigma_w^2 # Excess smoothness relative to Var(Δy_p) = σ_w^2

        excess_sens_exact = K_pi * (1.0 - K_pi) * sigma_w
        @test excess_sens_exact > 0.0
        @test isapprox(c_irf[2] - c_irf[1], excess_sens_exact; atol=1e-12)

        # Simulated PIH sample: positive Cov(Δc_t, w_{t-1})
        sim_pi = simulate(sol_pi, 25_000; burn=500, seed=2008)
        dc_sim = diff(sim_pi.actions[:, 1])
        w_lag = sim_pi.fundamental_shocks[1:(end - 1), 1]
        cov_sens = cov(dc_sim, w_lag)
        @test cov_sens > 0.0
        @test isapprox(cov_sens, excess_sens_exact; atol=0.03)

        # Persistent near-unit-root permanent income (ρ = 0.95): total Var(Δc_t) < Var(Δy_p) AND Cov(Δc_t, w_{t-1}) > 0
        prob_near = RationalInattentionProblem(0.95, sigma_w, 1.0; capacity=0.6, units=:bits)
        sol_near = solve_ri_kalman(prob_near)
        sim_near = simulate(sol_near, 25_000; burn=500, seed=2009)
        dy_near = diff(sim_near.states[:, 1])
        dc_near = diff(sim_near.actions[:, 1])
        w_lag_near = sim_near.fundamental_shocks[1:(end - 1), 1]
        @test var(dc_near) < var(dy_near) # Total individual excess smoothness
        @test cov(dc_near, w_lag_near) > 0.0 # Excess sensitivity
    end

    # =========================================================================
    # 5. Stochastic Simulation Moments, Reproducibility & Float32 Support
    # =========================================================================
    @testset "Simulation Sample Moments & Float32" begin
        A = [0.75 0.1; 0.05 0.6]
        B = [1.0 0.0; 0.2 0.8]
        H = [1.0 0.3; 0.0 1.0]
        prob = RationalInattentionProblem(A, B, H; beta=0.95, capacity=1.2, units=:bits)
        sol = solve_ri_kalman(prob)

        # Seed reproducibility & unpacking
        s1 = simulate(sol, 200; burn=50, seed=919)
        s2 = simulate(sol, 200; burn=50, seed=919)
        s3 = simulate(sol, 200; burn=50, seed=920)
        states_u, beliefs_u, signals_u, actions_u, w_u, psi_u = s1
        @test states_u == s2.states && beliefs_u == s2.beliefs
        @test signals_u == s2.signals && actions_u == s2.actions
        @test w_u == s2.fundamental_shocks && psi_u == s2.ri_noise
        @test states_u != s3.states

        # Large-sample moment convergence
        sim = simulate(sol, 30_000; burn=500, seed=42)
        @test isapprox(sim.signals, sim.states * sol.Y_signal + sim.ri_noise; atol=1e-12)
        @test isapprox(sim.actions, sim.beliefs * sol.F_policy'; atol=1e-12)
        @test isapprox(cov(sim.fundamental_shocks), Matrix{Float64}(I, 2, 2); atol=0.05)
        @test isapprox(cov(sim.ri_noise), sol.Sigma_psi; atol=0.05 * max(norm(sol.Sigma_psi), 1.0))
        @test isapprox(cov(sim.states - sim.beliefs), sol.Sigma_post; atol=0.05)

        # Float32 precision support
        A32 = Float32[0.8 0.1; 0.0 0.6]
        B32 = Float32[1.0 0.0; 0.0 0.8]
        H32 = Float32[1.0, 0.5]
        p32 = RationalInattentionProblem(A32, B32, H32; beta=0.95f0, capacity=1.2f0)
        sol32 = solve_ri_kalman(p32)
        ssm32 = to_statespace(sol32; observables=:all)
        abcd32 = to_abcd(sol32; observables=:all)
        ir32 = irf(sol32, 12; observables=:all)
        fv32 = fevd(sol32, 12; observables=:actions)
        sim32 = simulate(sol32, 50; burn=20, seed=11)

        @test ssm32 isa StateSpaceModel{Float32}
        @test abcd32 isa ABCDSystem{Float32}
        @test ir32 isa ImpulseResponse{Float32}
        @test fv32 isa FEVD{Float32}
        @test eltype(sim32.states) === Float32
        @test eltype(sim32.beliefs) === Float32
        @test eltype(sim32.signals) === Float32
        @test eltype(sim32.actions) === Float32
        @test eltype(sim32.fundamental_shocks) === Float32
        @test eltype(sim32.ri_noise) === Float32
        @test isapprox(ir32.values, irf(abcd32, 12).values; atol=1f-6)

        @test isempty(Test.detect_ambiguities(MacroEconometricModels))
    end
end

