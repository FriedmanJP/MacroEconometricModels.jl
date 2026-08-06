# CF-13 (#393): the OPP statistic — WLS identity, optimality test, analytic
# scalar NK case, recommendation paths.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

@testset "OPP statistic (CF-13)" begin
    rng = MersenneTwister(20260813)

    @testset "WLS identity" begin
        H, n_s = 6, 2
        Tx1 = randn(rng, H, n_s)
        Tx2 = randn(rng, H, n_s)
        ce = PolicyCausalEffects(outcomes=[:u, :infl], Theta_x=[Tx1, Tx2])
        v1, v2 = randn(rng, H), randn(rng, H)
        fc = PolicyForecast{Float64}([:u, :infl], [v1, v2], nothing, H, "test")
        loss = policy_loss([:u, :infl], H; lambda=[2.0, 1.0], beta=0.97)
        r = opp(fc, ce, loss)
        R = vcat(Tx1, Tx2)
        W = [loss.W_x[1] zeros(H, H); zeros(H, H) loss.W_x[2]]
        EY = vcat(v1, v2)
        delta_ref = -(R' * W * R) \ (R' * W * EY)
        @test r.delta ≈ delta_ref atol = 1e-10
        @test r.gradient ≈ R' * W * EY atol = 1e-10
        @test r.loss_opp <= r.loss_base
        @test r.loss_opp < r.loss_base          # gradient != 0 here
        @test r.P_opp === nothing               # pure test: no instruments needed
        @test r.delta_draws === nothing
    end

    @testset "optimality test: gradient zero ⟺ δ zero" begin
        H, n_s = 6, 2
        Tx = randn(rng, H, n_s)
        ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx])
        # forecast orthogonal to the IRF column space (W = I): δ* ≈ 0
        v = randn(rng, H)
        v_perp = v - Tx * ((Tx' * Tx) \ (Tx' * v))
        fc0 = PolicyForecast{Float64}([:u], [v_perp], nothing, H, "opt")
        loss = policy_loss([:u], H; lambda=[1.0])
        r0 = opp(fc0, ce, loss)
        @test norm(r0.gradient) < 1e-10
        @test norm(r0.delta) < 1e-10
        @test r0.loss_opp ≈ r0.loss_base atol = 1e-12

        # forecast inside the span: exact recovery δ* ≈ d, Y_opp ≈ 0
        d = [0.7, -0.3]
        fcd = PolicyForecast{Float64}([:u], [-Tx * d], nothing, H, "spanned")
        rd = opp(fcd, ce, loss)
        @test rd.delta ≈ d atol = 1e-10
        @test norm(rd.Y_opp[1]) < 1e-10
    end

    @testset "BM scalar NK analytic case" begin
        # Static NK (BM §2): objectives (u, π), one rate shock, H = 1.
        # r_u = -1/σ · a, r_π = -κ/σ · a with a = 1/(1 + κφ/σ).
        sigma, kappa, phi = 1.0, 0.3, 1.5
        a = 1 / (1 + kappa * phi / sigma)
        r_u = -a / sigma
        r_pi = -kappa * a / sigma
        lam_u, lam_pi = 2.0, 1.0
        u_bar, pi_bar = 0.8, 0.4
        ce = MEM._suppress_warnings() do
            PolicyCausalEffects(outcomes=[:u, :infl],
                                Theta_x=[fill(r_u, 1, 1), fill(r_pi, 1, 1)])
        end
        fc = PolicyForecast{Float64}([:u, :infl], [[u_bar], [pi_bar]], nothing, 1, "2008M4")
        loss = policy_loss([:u, :infl], 1; lambda=[lam_u, lam_pi])
        r = MEM._suppress_warnings() do
            opp(fc, ce, loss)   # H = 1 square container triggers the menu warning
        end
        delta_hand = -(lam_u * r_u * u_bar + lam_pi * r_pi * pi_bar) /
                     (lam_u * r_u^2 + lam_pi * r_pi^2)
        @test r.delta[1] ≈ delta_hand atol = 1e-12
        # loss-reducing by construction
        @test r.loss_opp <= r.loss_base
    end

    @testset "recommendation paths and instrument penalties" begin
        H, n_s = 5, 2
        Tx = randn(rng, H, n_s)
        Tz = randn(rng, H, n_s)
        ce = PolicyCausalEffects(outcomes=[:u], instruments=[:rate],
                                 Theta_x=[Tx], Theta_z=[Tz])
        v = randn(rng, H)
        fc = PolicyForecast{Float64}([:u], [v], nothing, H, "t")
        p0 = randn(rng, H)
        loss = policy_loss([:u], H; lambda=[1.0])
        r = opp(fc, ce, loss; instrument_path=[:rate => p0])
        @test r.P_base[1] == p0
        @test r.P_opp[1] ≈ p0 + Tz * r.delta atol = 1e-12

        # with a smoothing penalty the solution matches the augmented formula
        sp = smoothing_penalty(H; lambda=0.4, beta=1.0, z_lag=0.2)
        loss_z = policy_loss([:u], H; lambda=[1.0], instruments=[:rate], W_z=[sp.W_z])
        rz = opp(fc, ce, loss_z; instrument_path=[:rate => p0], z_wedge=[sp.wedge_term])
        delta_ref = -(Tx' * Tx + Tz' * sp.W_z * Tz) \
                    (Tx' * v + Tz' * sp.W_z * p0 - Tz' * sp.wedge_term)
        @test rz.delta ≈ delta_ref atol = 1e-10
        @test rz.delta != r.delta
    end

    @testset "validation" begin
        H = 4
        Tx = randn(rng, H, 2)
        ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx])
        fc = PolicyForecast{Float64}([:u], [randn(rng, H)], nothing, H, "t")
        loss = policy_loss([:u], H; lambda=[1.0])
        fc5 = PolicyForecast{Float64}([:u], [randn(rng, 5)], nothing, 5, "t")
        @test_throws ArgumentError opp(fc5, ce, loss)
        @test_throws ArgumentError opp(fc, ce, policy_loss([:gdp], H; lambda=[1.0]))
        # instrument penalty without announced path
        sp = smoothing_penalty(H; lambda=0.1)
        ce_z = PolicyCausalEffects(outcomes=[:u], instruments=[:rate],
                                   Theta_x=[Tx], Theta_z=[randn(rng, H, 2)])
        loss_z = policy_loss([:u], H; lambda=[1.0], instruments=[:rate], W_z=[sp.W_z])
        @test_throws ArgumentError opp(fc, ce_z, loss_z)
        # unknown instrument in the path
        @test_throws ArgumentError opp(fc, ce_z, loss;
                                       instrument_path=[:qe => zeros(H)])
        # square-menu warning
        ce_sq = PolicyCausalEffects(outcomes=[:u], Theta_x=[randn(rng, H, H)])
        @test_logs (:warn, r"square") match_mode = :any begin
            opp(fc, ce_sq, loss)
        end
    end
end
