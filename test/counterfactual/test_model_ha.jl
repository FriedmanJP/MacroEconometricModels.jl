# CF-08 (#388): public sequence-space Jacobians + HA GE policy causal effects.
using Test
using LinearAlgebra
using MacroEconometricModels

const MEM = MacroEconometricModels

# Small Huggett economy (shared across testsets; SS solve is the slow part).
const CF08_SPEC = MEM._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=120)
const CF08_SS = compute_steady_state(CF08_SPEC)

@testset "HA sequence-space policy effects (CF-08)" begin

    @testset "sequence_jacobian public wrapper" begin
        Th = 40
        J = sequence_jacobian(CF08_SPEC, CF08_SS, :r, :C; T_horizon=Th)
        @test size(J) == (Th, Th)
        # behavior-preserving: identical to the internal fake-news Jacobian
        J_ref = MEM._ssj_jacobian(CF08_SS, CF08_SPEC.individual, CF08_SPEC.grid,
                                  CF08_SPEC.income, :r, :C; T_horizon=Th)
        @test J == J_ref
        # anticipation entries above the diagonal (t < s) are nonzero
        @test maximum(abs, J[1:5, 10]) > 0
        @test_throws ArgumentError sequence_jacobian(CF08_SPEC, CF08_SS, :z, :C)
    end

    @testset "one-asset-only guard" begin
        spec2 = load_ha_example(:two_asset_hank)
        err = try
            sequence_jacobian(spec2, CF08_SS, :r, :C; T_horizon=10)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("one-asset", err.msg)
    end

    @testset "administered-rate closure" begin
        H, Th = 8, 60
        ce = policy_causal_effects(CF08_SPEC, CF08_SS;
                                   outcomes=[:cons => :C, :assets => :A],
                                   instruments=[:rate => :r],
                                   H=H, T_horizon=Th)
        @test ce isa PolicyCausalEffects{Float64}
        @test is_square(ce)
        @test ce.source == :hank
        # rate follows the wedge one-for-one
        @test ce.Theta_z[1] == Matrix{Float64}(I, H, H)
        # outcome map is the truncated fake-news Jacobian
        J_C = sequence_jacobian(CF08_SPEC, CF08_SS, :r, :C; T_horizon=Th)
        @test ce.Theta_x[1] == J_C[1:H, 1:H]
        # +m is a hike: impact consumption response is negative
        @test ce.Theta_x[1][1, 1] < 0
        # anticipation: consumption moves before the wedge arrives
        @test abs(ce.Theta_x[1][1, 5]) > 0
    end

    @testset "market closure: Huggett rate-wedge neutrality" begin
        H, Th = 6, 60
        ce = policy_causal_effects(CF08_SPEC, CF08_SS;
                                   outcomes=[:cons => :C],
                                   instruments=[:rate => :r],
                                   H=H, T_horizon=Th, rule_closure=:market)
        # zero-net-supply GE: the market rate offsets the wedge one-for-one,
        # so effective-rate and outcome responses vanish (neutrality theorem)
        @test maximum(abs, ce.Theta_z[1]) < 1e-6
        @test maximum(abs, ce.Theta_x[1]) < 1e-4
    end

    @testset "market closure requires :huggett" begin
        ks = load_ha_example(:krusell_smith)
        err = try
            policy_causal_effects(ks, CF08_SS; outcomes=[:cons => :C],
                                  H=4, T_horizon=60, rule_closure=:market)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("administered", err.msg)
    end

    @testset "horizon guards" begin
        @test_logs (:warn, r"truncation-edge") match_mode = :any begin
            policy_causal_effects(CF08_SPEC, CF08_SS; outcomes=[:cons => :C],
                                  H=8, T_horizon=20)
        end
        @test_throws ArgumentError policy_causal_effects(CF08_SPEC, CF08_SS;
                                                         outcomes=[:cons => :C],
                                                         H=30, T_horizon=20)
        @test_throws ArgumentError policy_causal_effects(CF08_SPEC, CF08_SS;
                                                         outcomes=[:cons => :C],
                                                         H=0, T_horizon=20)
        @test_throws ArgumentError policy_causal_effects(CF08_SPEC, CF08_SS;
                                                         outcomes=[:cons => :C],
                                                         instruments=[:rate => :w],
                                                         H=4, T_horizon=60)
        @test_throws ArgumentError policy_causal_effects(CF08_SPEC, CF08_SS;
                                                         outcomes=[:cons => :C],
                                                         H=4, T_horizon=60,
                                                         rule_closure=:nominal)
    end
end
