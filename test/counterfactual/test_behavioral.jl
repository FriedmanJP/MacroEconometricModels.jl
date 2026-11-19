# CF-09 (#389): behavioral-expectations operators on sequence-space Jacobians.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

# Fisher-menu Jacobian (forward-looking, anticipation-rich, analytic):
# J[t, s] = −φ^{−(s−t+1)} for t ≤ s, 0 after.
_fisher_J(n, phi) = [t <= s ? -phi^(-(s - t + 1)) : 0.0 for t in 1:n, s in 1:n]
# Backward AR Jacobian (anticipation-free): J[t, s] = ρ^{t−s} for t ≥ s.
_backward_J(n, rho) = [t >= s ? rho^(t - s) : 0.0 for t in 1:n, s in 1:n]

@testset "Behavioral operators (CF-09)" begin
    rng = Xoshiro(20260809)
    n = 12
    JF = _fisher_J(n, 1.5)
    JB = _backward_J(n, 0.8)
    JR = randn(rng, n, n)

    @testset "identity limits (exact)" begin
        @test cognitive_discounting(JF, 1.0) == JF
        @test sticky_expectations(JF, 0.0) == JF
        @test cognitive_discounting(JR, 1.0) == JR
        @test sticky_expectations(JR, 0.0) == JR
    end

    @testset "fake-news round trip" begin
        for J in (JF, JB, JR)
            F = MEM._fake_news_of(J)
            @test MEM._rebuild_from_fake_news(F) ≈ J atol = 1e-12
        end
        # F of the Fisher J: first row/col pinned to J's
        F = MEM._fake_news_of(JF)
        @test F[1, :] == JF[1, :]
        @test F[:, 1] == JF[:, 1]
    end

    @testset "CMW recursion agreement" begin
        for J in (JF, JB, JR), m in (0.0, 0.35, 0.65, 0.9)
            @test cognitive_discounting(J, m) ≈ MEM._cognitive_disc_cmw(J, m) atol = 1e-12
        end
        # and on a real HA fake-news Jacobian
        spec = MEM._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=80)
        ss = compute_steady_state(spec)
        J_ha = sequence_jacobian(spec, ss, :r, :C; T_horizon=25)
        @test cognitive_discounting(J_ha, 0.65) ≈ MEM._cognitive_disc_cmw(J_ha, 0.65) atol = 1e-12
    end

    @testset "m = 0 kills anticipation" begin
        J0 = cognitive_discounting(JF, 0.0)
        for s in 2:n
            # zero anticipation block
            @test all(abs.(J0[1:s-1, s]) .< 1e-14)
            # post-arrival = shifted unanticipated column
            for t in s:n
                @test J0[t, s] ≈ JF[t-s+1, 1] atol = 1e-14
            end
        end
    end

    @testset "anticipation-free J: both operators are the identity" begin
        for m in (0.0, 0.5), th in (0.3, 0.8)
            @test cognitive_discounting(JB, m) ≈ JB atol = 1e-12
            @test sticky_expectations(JB, th) ≈ JB atol = 1e-12
        end
    end

    @testset "sticky closed form on the Fisher J" begin
        th = 0.4
        Jst = sticky_expectations(JF, th)
        # anticipation entries damped by the informed share (1 − θ^t);
        # arrival-date response full; post-arrival stays zero
        for s in 1:n, t in 1:n
            expected = t < s ? JF[t, s] * (1 - th^t) :
                       t == s ? JF[t, s] : 0.0
            @test Jst[t, s] ≈ expected atol = 1e-12
        end
    end

    @testset "monotonicity in the frictions (long-end columns)" begin
        s_long = n
        norm_at(m) = norm(cognitive_discounting(JF, m)[:, s_long])
        @test norm_at(1.0) >= norm_at(0.9) >= norm_at(0.5) >= norm_at(0.0) - 1e-14
        snorm_at(th) = norm(sticky_expectations(JF, th)[1:s_long-1, s_long])
        @test snorm_at(0.0) >= snorm_at(0.4) >= snorm_at(0.8) - 1e-14
    end

    @testset "behavioral(ce)" begin
        H = 6
        Jx = _fisher_J(H, 1.5)
        Jz = _backward_J(H, 0.5)
        Dx = cat(Jx, 2 .* Jx; dims=3)
        Dz = cat(Jz, 2 .* Jz; dims=3)
        ce = PolicyCausalEffects(outcomes=[:pi], instruments=[:i],
                                 Theta_x=[Jx], Theta_z=[Jz],
                                 Theta_x_draws=[Dx], Theta_z_draws=[Dz],
                                 source=:dsge)
        bc = behavioral(ce; m=0.8, theta=0.3)
        @test bc isa PolicyCausalEffects{Float64}
        @test bc.Theta_x[1] ≈ sticky_expectations(cognitive_discounting(Jx, 0.8), 0.3) atol = 1e-12
        @test bc.Theta_x_draws[1][:, :, 2] ≈
              sticky_expectations(cognitive_discounting(2 .* Jx, 0.8), 0.3) atol = 1e-12
        @test occursin("m=0.8", bc.shock_labels[1])
        @test occursin("0.3", bc.shock_labels[1])
        @test bc.source == :dsge

        bc_ri = behavioral(ce; m=0.8, theta=0.2, ri_gain=0.45, ri_rho=0.75)
        @test bc_ri isa PolicyCausalEffects{Float64}
        @test bc_ri.Theta_x[1] ≈ rational_inattention_jacobian(
            sticky_expectations(cognitive_discounting(Jx, 0.8), 0.2), 0.45; mode=:gain, rho_news=0.75) atol = 1e-12
        @test bc_ri.Theta_z[1] ≈ Jz atol = 1e-12
        @test bc_ri.Theta_x_draws[1][:, :, 2] ≈ rational_inattention_jacobian(
            sticky_expectations(cognitive_discounting(2 .* Jx, 0.8), 0.2), 0.45; mode=:gain, rho_news=0.75) atol = 1e-12
        @test occursin("G_RI=0.45", bc_ri.shock_labels[1])
        @test occursin("ρ_RI=0.75", bc_ri.shock_labels[1])

        bc_ri_only = behavioral(ce; ri_gain=1.0)
        @test bc_ri_only.Theta_x[1] ≈ Jx atol = 1e-12
        @test bc_ri_only.Theta_z[1] ≈ Jz atol = 1e-12

        # thin containers must not be behavioralized
        thin = PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(6, 2)])
        err = try
            behavioral(thin; m=0.9)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("thin", err.msg)
    end

    @testset "rational_inattention_jacobian (RI-06)" begin
        # 1. Exact FIRE identity limits (G = 1.0 and κ = Inf)
        for J in (JF, JB, JR)
            J1 = rational_inattention_jacobian(J, 1.0; mode=:gain)
            @test J1 ≈ J atol = 1e-12
            @test J1 !== J
            Jinf = rational_inattention_jacobian(J, Inf; mode=:capacity)
            @test Jinf ≈ J atol = 1e-12
        end

        # 2. Anticipation-free (backward-looking lower-triangular) invariance
        for G in (0.15, 0.4, 0.75, 1.0), rn in (0.0, 0.5, 0.8, 1.0)
            @test rational_inattention_jacobian(JB, G; mode=:gain, rho_news=rn) ≈ JB atol = 1e-12
        end

        # 3. Capacity <-> Gain equivalence: G = 1 - 2^(-2*kappa)
        for kappa in (0.25, 0.75, 1.5, 3.0)
            G_eq = 1.0 - 2.0^(-2.0 * kappa)
            @test rational_inattention_jacobian(JF, kappa; mode=:capacity, rho_news=0.8) ≈
                  rational_inattention_jacobian(JF, G_eq; mode=:gain, rho_news=0.8) atol = 1e-12
        end

        # 4. Nesting sticky expectations at rho_news = 1.0 on forward Fisher Jacobian
        for G in (0.25, 0.6, 0.85)
            @test rational_inattention_jacobian(JF, G; mode=:gain, rho_news=1.0) ≈
                  sticky_expectations(JF, 1.0 - G) atol = 1e-12
        end

        # 5. Micro Jumps, Macro Humps on both synthetic forward and HA Huggett Jacobians
        J_fwd = _fisher_J(15, 1.05)
        dr_f = [0.95^(s - 1) for s in 1:15]
        re_f = abs.(J_fwd * dr_f)
        ri_f = abs.(rational_inattention_jacobian(J_fwd, 0.20; mode=:gain, rho_news=0.7) * dr_f)
        @test argmax(re_f) == 1
        @test argmax(ri_f) > 1

        spec = MEM._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=80)
        ss = compute_steady_state(spec)
        J_ha = sequence_jacobian(spec, ss, :r, :C; T_horizon=25)
        J_ha_ri = rational_inattention_jacobian(J_ha, 0.25; mode=:gain, rho_news=0.8)
        @test abs(J_ha_ri[1, 5]) < 0.25 * abs(J_ha[1, 5])
        @test argmax(abs.(J_ha_ri[:, 5])) == 5
        @test issorted(abs.(J_ha_ri[1:5, 5]))
        dr_ha = [0.85^(s - 1) for s in 1:size(J_ha, 2)]
        re_ha = abs.(J_ha * dr_ha)
        ri_ha = abs.(J_ha_ri * dr_ha)
        @test ri_ha[1] < 0.5 * re_ha[1]
        @test ri_ha[2] / ri_ha[1] > re_ha[2] / re_ha[1]

        # 6. Monotonicity in gain G and news persistence rho_news on long anticipation columns
        s_col = n
        gnorm(g) = norm(rational_inattention_jacobian(JF, g; mode=:gain, rho_news=0.8)[1:s_col-1, s_col])
        @test gnorm(1.0) >= gnorm(0.7) >= gnorm(0.35) >= gnorm(0.1) - 1e-14
        rnorm(rn) = norm(rational_inattention_jacobian(JF, 0.4; mode=:gain, rho_news=rn)[1:s_col-1, s_col])
        @test rnorm(1.0) >= rnorm(0.8) >= rnorm(0.4) >= rnorm(0.0) - 1e-14

        # 7. Float32 precision & ambiguities
        JF32 = Float32.(JF)
        JB32 = Float32.(JB)
        @test eltype(rational_inattention_jacobian(JF32, 0.5; mode=:gain)) === Float32
        @test rational_inattention_jacobian(JB32, 0.5f0; mode=:gain) ≈ JB32 atol = 1f-5
    end

    @testset "validation" begin
        @test_throws ArgumentError cognitive_discounting(JF, 1.2)
        @test_throws ArgumentError cognitive_discounting(JF, -0.1)
        @test_throws ArgumentError sticky_expectations(JF, 1.5)
        @test_throws DimensionMismatch cognitive_discounting(ones(3, 4), 0.5)
        @test_throws ArgumentError rational_inattention_jacobian(JF, -0.1; mode=:gain)
        @test_throws ArgumentError rational_inattention_jacobian(JF, 1.2; mode=:gain)
        @test_throws ArgumentError rational_inattention_jacobian(JF, -0.5; mode=:capacity)
        @test_throws ArgumentError rational_inattention_jacobian(JF, 0.5; mode=:bad)
        @test_throws ArgumentError rational_inattention_jacobian(JF, 0.5; rho_news=1.2)
        @test_throws ArgumentError rational_inattention_jacobian(JF, 0.5; rho_news=-0.1)
        @test_throws DimensionMismatch rational_inattention_jacobian(ones(3, 4), 0.5)
        ce_small = PolicyCausalEffects(outcomes=[:pi], instruments=[:i],
                                       Theta_x=[_fisher_J(4, 1.5)], Theta_z=[_backward_J(4, 0.5)],
                                       source=:dsge)
        @test_throws ArgumentError behavioral(ce_small; ri_gain=1.2)
        @test_throws ArgumentError behavioral(ce_small; ri_gain=0.5, ri_rho=-0.1)
    end
end
