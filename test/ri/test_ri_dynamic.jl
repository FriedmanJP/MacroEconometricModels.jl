# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, LinearAlgebra, Statistics

@testset "Afrouzi-Yang (2021) Dynamic RI Transition Path (#918)" begin

    function _check_transition_laws(sol::DynamicRISolution{T}, prob::RationalInattentionProblem{T}; atol=T(1e-9)) where {T}
        n = size(prob.A, 1)
        Q = prob.B * prob.B'
        Tp = sol.T_path
        @test size(sol.Sigma_prior_path) == (n, n, Tp)
        @test size(sol.Sigma_post_path) == (n, n, Tp)
        @test size(sol.Xi_path) == (n, n, Tp)
        @test size(sol.K_gain_path) == (n, n, Tp)
        @test length(sol.omega_path) == Tp
        @test length(sol.capacity_bits_path) == Tp
        @test length(sol.capacity_nats_path) == Tp
        @test length(sol.signal_rank_path) == Tp
        @test sol.T_path == Tp
        @test sol.state_names == prob.state_names
        @test sol.action_names == prob.action_names
        @test sol.shock_names == prob.shock_names

        @test isapprox(sol.Sigma_prior_path[:, :, 1], sol.Sigma_init; atol=atol)
        for t in 1:Tp
            Sp = sol.Sigma_prior_path[:, :, t]
            Spost = sol.Sigma_post_path[:, :, t]
            Gt = sol.K_gain_path[:, :, t]
            @test isapprox(Sp, Sp'; atol=atol)
            @test isapprox(Spost, Spost'; atol=atol)
            @test minimum(eigvals(Symmetric(Matrix(Spost)))) >= -atol * max(norm(Spost), one(T))
            @test isapprox((Matrix{T}(I, n, n) - Gt) * Sp, Spost;
                atol=atol * max(norm(Sp), one(T)))
            @test isapprox(sol.capacity_bits_path[t], sol.capacity_nats_path[t] / log(T(2));
                rtol=T(1e-6), atol=atol)
            if t < Tp
                Sp_next = prob.A * Spost * prob.A' + Q
                @test isapprox(sol.Sigma_prior_path[:, :, t + 1], Sp_next;
                    atol=atol * max(norm(Sp_next), one(T)))
            end
        end
    end

    # =========================================================================
    # 1. Input Validation & Edge Cases
    # =========================================================================
    @testset "Input Validation" begin
        prob = RationalInattentionProblem([0.8 0.1; 0.0 0.6], [1.0 0.0; 0.0 0.8], [1.0, 0.5];
            beta=0.95, omega=0.5)
        S_ok = [2.0 0.0; 0.0 2.0]

        @test_throws ArgumentError solve_ri_transition(prob, [1.0;;])
        @test_throws ArgumentError solve_ri_transition(prob, [1.0 NaN; 0.0 1.0])
        @test_throws ArgumentError solve_ri_transition(prob, [1.0 0.5; 0.0 1.0]) # asymmetric
        @test_throws ArgumentError solve_ri_transition(prob, [1.0 0.0; 0.0 -0.5]) # indefinite
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; T_path=0)
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; tol=0.0)
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; max_iter=0)
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; damping=0.0)
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; damping=1.2)
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; T_path=5,
            omega_path=ones(5), capacity_path=ones(5))
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; T_path=5, omega_path=ones(4))
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; T_path=5, omega_path=[1.0, 0.0, 1.0, 1.0, 1.0])
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; T_path=5, capacity_path=ones(4))
        @test_throws ArgumentError solve_ri_transition(prob, S_ok; T_path=5, capacity_path=[1.0, -0.1, 1.0, 1.0, 1.0])

        # Degenerate zero initial prior cannot support strictly positive capacity at t = 1
        prob_cap = RationalInattentionProblem(0.8, 1.0, 1.0; capacity=1.0)
        @test_throws ArgumentError solve_ri_transition(prob_cap, [0.0;;]; T_path=10)
    end

    # =========================================================================
    # 2. Stationary Fixed-Point Check (Sigma_init = ss.Sigma_prior)
    # =========================================================================
    @testset "Stationary Fixed-Point Path" begin
        # 1D fixed-omega and fixed-capacity
        p1_om = RationalInattentionProblem(0.8, 1.2, 1.0; beta=0.98, omega=0.4)
        p1_cap = RationalInattentionProblem(0.8, 1.2, 1.0; beta=0.98, capacity=1.5, units=:bits)

        # 3D coupled system
        A3 = [0.85 0.1 0.0; 0.05 0.7 0.08; 0.0 0.0 0.5]
        B3 = [1.0 0.1; 0.0 0.8; 0.2 0.3]
        H3 = [1.0 0.0; 0.3 1.0; 0.0 0.4]
        M3 = [2.0 0.2; 0.2 1.0]
        p3_om = RationalInattentionProblem(A3, B3, H3; M=M3, beta=0.96, omega=0.8)
        p3_cap = RationalInattentionProblem(A3, B3, H3; M=M3, beta=0.96, capacity=1.5, units=:bits)

        for prob in (p1_om, p1_cap, p3_om, p3_cap)
            ss = solve_ri_kalman(prob; method=:water_filling)
            dyn = solve_ri_transition(prob, ss.Sigma_prior; T_path=40)
            @test dyn isa DynamicRISolution{Float64}
            @test dyn.converged
            @test dyn.iterations == 1
            @test dyn.residual < 1e-10
            _check_transition_laws(dyn, prob; atol=1e-10)

            G_ss = ss.K_gain * ss.Y_signal'
            for t in 1:dyn.T_path
                @test isapprox(dyn.Sigma_prior_path[:, :, t], ss.Sigma_prior; atol=1e-10)
                @test isapprox(dyn.Sigma_post_path[:, :, t], ss.Sigma_post; atol=1e-10)
                @test isapprox(dyn.K_gain_path[:, :, t], G_ss; atol=1e-10)
                @test abs(dyn.capacity_bits_path[t] - ss.capacity_bits) < 1e-10
                @test abs(dyn.capacity_nats_path[t] - ss.capacity_nats) < 1e-10
                @test abs(dyn.omega_path[t] - ss.omega) < 1e-10
                @test dyn.signal_rank_path[t] == ss.signal_rank
            end
            @test isapprox(dyn.Y_terminal, ss.Y_signal; atol=1e-10)
            @test isapprox(dyn.Sigma_psi_terminal, ss.Sigma_psi; atol=1e-10)
        end
    end

    # =========================================================================
    # 3. High & Low Initial Uncertainty Convergence + Attention Spike (Afrouzi-Yang 2021)
    # =========================================================================
    @testset "Uncertainty Shocks & Endogenous Attention Spike" begin
        # 1D fixed-omega benchmark
        p1 = RationalInattentionProblem(0.85, 1.0, 1.0; beta=0.98, omega=0.35)
        ss1 = solve_ri_kalman(p1)

        dyn1_hi = solve_ri_transition(p1, 10.0 * ss1.Sigma_prior; T_path=100)
        @test dyn1_hi.converged
        @test dyn1_hi.residual < 1e-10
        _check_transition_laws(dyn1_hi, p1; atol=1e-10)
        # Endogenous spike in attention at t=1 under high uncertainty
        @test dyn1_hi.capacity_bits_path[1] > dyn1_hi.capacity_bits_path[end]
        @test dyn1_hi.capacity_bits_path[1] > ss1.capacity_bits
        # Monotone decay toward steady state
        @test all(diff(dyn1_hi.capacity_bits_path) .<= 1e-12)
        @test isapprox(dyn1_hi.Sigma_post_path[:, :, end], ss1.Sigma_post; atol=1e-6)
        @test isapprox(dyn1_hi.Sigma_prior_path[:, :, end], ss1.Sigma_prior; atol=1e-6)
        @test abs(dyn1_hi.capacity_bits_path[end] - ss1.capacity_bits) < 1e-6

        # Low initial uncertainty
        dyn1_lo = solve_ri_transition(p1, 0.05 * ss1.Sigma_prior; T_path=100)
        @test dyn1_lo.converged
        _check_transition_laws(dyn1_lo, p1; atol=1e-10)
        @test dyn1_lo.capacity_bits_path[1] < dyn1_lo.capacity_bits_path[end]
        @test isapprox(dyn1_lo.Sigma_post_path[:, :, end], ss1.Sigma_post; atol=1e-6)
        @test isapprox(dyn1_lo.Sigma_prior_path[:, :, end], ss1.Sigma_prior; atol=1e-6)
        @test abs(dyn1_lo.capacity_bits_path[end] - ss1.capacity_bits) < 1e-6

        # Multivariate (n = 3) fixed-omega high & low initial uncertainty
        A3 = [0.85 0.12 0.0; 0.05 0.72 0.1; 0.0 0.04 0.55]
        B3 = [1.0 0.2; 0.0 0.9; 0.15 0.4]
        H3 = [1.0 0.0; 0.4 1.0; 0.0 0.6]
        M3 = [1.5 0.1; 0.1 1.0]
        p3_om = RationalInattentionProblem(A3, B3, H3; M=M3, beta=0.95, omega=0.6)
        ss3_om = solve_ri_kalman(p3_om)

        dyn3_hi = solve_ri_transition(p3_om, 10.0 * ss3_om.Sigma_prior; T_path=120)
        @test dyn3_hi.converged
        _check_transition_laws(dyn3_hi, p3_om; atol=1e-10)
        @test dyn3_hi.capacity_bits_path[1] > dyn3_hi.capacity_bits_path[end]
        @test isapprox(dyn3_hi.Sigma_post_path[:, :, end], ss3_om.Sigma_post; atol=1e-6)
        @test isapprox(dyn3_hi.Sigma_prior_path[:, :, end], ss3_om.Sigma_prior; atol=1e-6)
        @test isapprox(dyn3_hi.K_gain_path[:, :, end], ss3_om.K_gain * ss3_om.Y_signal'; atol=1e-6)

        dyn3_lo = solve_ri_transition(p3_om, 0.05 * ss3_om.Sigma_prior; T_path=120)
        @test dyn3_lo.converged
        _check_transition_laws(dyn3_lo, p3_om; atol=1e-10)
        @test dyn3_lo.capacity_bits_path[1] < dyn3_lo.capacity_bits_path[end]
        @test isapprox(dyn3_lo.Sigma_post_path[:, :, end], ss3_om.Sigma_post; atol=1e-6)
        @test isapprox(dyn3_lo.Sigma_prior_path[:, :, end], ss3_om.Sigma_prior; atol=1e-6)

        # Multivariate fixed-capacity: shadow price omega_t spikes at high uncertainty
        p3_cap = RationalInattentionProblem(A3, B3, H3; M=M3, beta=0.95, capacity=1.5, units=:bits)
        ss3_cap = solve_ri_kalman(p3_cap)
        dyn3_cap_hi = solve_ri_transition(p3_cap, 10.0 * ss3_cap.Sigma_prior; T_path=120)
        @test dyn3_cap_hi.converged
        _check_transition_laws(dyn3_cap_hi, p3_cap; atol=1e-10)
        @test dyn3_cap_hi.omega_path[1] > dyn3_cap_hi.omega_path[end]
        @test all(x -> abs(x - 1.5) < 1e-8, dyn3_cap_hi.capacity_bits_path)
        @test isapprox(dyn3_cap_hi.Sigma_post_path[:, :, end], ss3_cap.Sigma_post; atol=1e-6)
        @test abs(dyn3_cap_hi.omega_path[end] - ss3_cap.omega) < 1e-6

        dyn3_cap_lo = solve_ri_transition(p3_cap, 0.05 * ss3_cap.Sigma_prior; T_path=120)
        @test dyn3_cap_lo.converged
        @test dyn3_cap_lo.omega_path[1] < dyn3_cap_lo.omega_path[end]
        @test isapprox(dyn3_cap_lo.Sigma_post_path[:, :, end], ss3_cap.Sigma_post; atol=1e-6)
    end

    # =========================================================================
    # 4. Time-Varying Paths, Corner Cases & Float32 Support
    # =========================================================================
    @testset "Time-Varying Paths, T_path=1 & Float32" begin
        prob = RationalInattentionProblem([0.8 0.1; 0.0 0.6], [1.0 0.0; 0.0 0.8], [1.0, 0.5];
            beta=0.96, capacity=1.0, units=:bits)
        ss = solve_ri_kalman(prob)

        # Temporary attention blackout (cap = 0 for t = 1..3), then recovery
        cap_path = vcat([0.0, 0.0, 0.0], fill(1.0, 47))
        dyn_blk = solve_ri_transition(prob, ss.Sigma_prior; T_path=50, capacity_path=cap_path)
        @test dyn_blk.converged
        _check_transition_laws(dyn_blk, prob; atol=1e-10)
        @test all(dyn_blk.signal_rank_path[1:3] .== 0)
        @test all(dyn_blk.capacity_bits_path[1:3] .== 0.0)
        @test isapprox(dyn_blk.Sigma_post_path[:, :, 1], ss.Sigma_prior; atol=1e-12)
        @test isapprox(dyn_blk.Sigma_post_path[:, :, end], ss.Sigma_post; atol=1e-6)

        # Temporary shadow-cost spike in omega_path
        om_path = vcat(fill(3.0 * ss.omega, 5), fill(ss.omega, 55))
        dyn_om = solve_ri_transition(prob, ss.Sigma_prior; T_path=60, omega_path=om_path)
        @test dyn_om.converged
        @test dyn_om.capacity_bits_path[1] < dyn_om.capacity_bits_path[end]
        @test isapprox(dyn_om.Sigma_post_path[:, :, end], ss.Sigma_post; atol=1e-6)

        # T_path = 1 edge case
        dyn_1 = solve_ri_transition(prob, 2.0 * ss.Sigma_prior; T_path=1)
        @test dyn_1.converged
        _check_transition_laws(dyn_1, prob; atol=1e-10)

        # Float32 precision support
        A32 = Float32[0.8 0.1; 0.0 0.6]
        B32 = Float32[1.0 0.0; 0.0 0.8]
        H32 = Float32[1.0, 0.5]
        p32 = RationalInattentionProblem(A32, B32, H32; beta=0.96f0, omega=0.5f0)
        ss32 = solve_ri_kalman(p32)
        dyn32 = solve_ri_transition(p32, 4.0f0 * ss32.Sigma_prior; T_path=60)
        @test dyn32 isa DynamicRISolution{Float32}
        @test dyn32.converged
        @test eltype(dyn32.Sigma_prior_path) === Float32
        @test eltype(dyn32.Sigma_post_path) === Float32
        @test eltype(dyn32.Xi_path) === Float32
        @test eltype(dyn32.K_gain_path) === Float32
        @test eltype(dyn32.omega_path) === Float32
        @test eltype(dyn32.capacity_bits_path) === Float32
        @test typeof(dyn32.residual) === Float32
        @test dyn32.capacity_bits_path[1] > dyn32.capacity_bits_path[end]
        _check_transition_laws(dyn32, p32; atol=1f-4)
    end

    # =========================================================================
    # 5. Display, Report, RSER-01 Constructor & Module Ambiguities
    # =========================================================================
    @testset "Display, Report & RSER-01" begin
        prob = RationalInattentionProblem([0.8 0.1; 0.0 0.6], [1.0 0.0; 0.0 0.8], [1.0, 0.5];
            omega=0.4, state_names=["output", "inflation"], action_names=["rate"])
        ss = solve_ri_kalman(prob)
        dyn = solve_ri_transition(prob, 4.0 * ss.Sigma_prior; T_path=30)

        # RSER-01 positional constructor parity
        flds = [getfield(dyn, f) for f in fieldnames(DynamicRISolution)]
        dyn_reconstructed = DynamicRISolution{Float64}(flds...)
        for f in fieldnames(DynamicRISolution)
            @test isequal(getfield(dyn_reconstructed, f), getfield(dyn, f))
        end

        s_out = sprint(show, dyn)
        r_out = sprint(report, dyn)
        @test occursin("Dynamic RI Transition", s_out)
        @test occursin("Transition Path Summary", r_out)
        @test occursin("Initial Prior Covariance", r_out)
        @test occursin("Terminal Signal Loadings", r_out)

        latex_out = with_display_backend(:latex) do
            sprint(report, dyn)
        end
        html_out = with_display_backend(:html) do
            sprint(report, dyn)
        end
        @test occursin("\\begin{tabular}", latex_out)
        @test occursin("<table", html_out)
    end
end
