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
    rng = MersenneTwister(20260809)
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

    @testset "validation" begin
        @test_throws ArgumentError cognitive_discounting(JF, 1.2)
        @test_throws ArgumentError cognitive_discounting(JF, -0.1)
        @test_throws ArgumentError sticky_expectations(JF, 1.5)
        @test_throws DimensionMismatch cognitive_discounting(ones(3, 4), 0.5)
    end
end
