# CF-10 (#390): McKay-Wolf rule counterfactuals — hand-derived oracle,
# self-rule fixed point, exact square enforcement, bands, spanning warnings.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

# NK spec with a demand shock (baseline driver) and a policy shock, closed by
# i = φπ·π + eps_i. Used for exact-square menu tests.
const CF10_NK = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
    endogenous: π, y, i
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
    i[t] = φπ * π[t] + eps_i[t]
end

@testset "MW rule counterfactuals (CF-10)" begin

    @testset "hand-derived thin oracle" begin
        H = 2
        # exactly enforceable thin problem
        ce = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                 Theta_x=[reshape([-0.5, -0.25], 2, 1)],
                                 Theta_z=[reshape([1.0, 0.5], 2, 1)])
        base = MEM.BaselinePath{Float64}([:x], [:z], [[1.0, 0.5]], [[0.0, 0.0]],
                                         nothing, nothing, H, "demand")
        rule = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=[:z])
        pc = policy_counterfactual(base, ce, rule)
        # nu* = -(M'M)^-1 M'b with M = Theta_x, b = x_base: hand value 2.0
        @test pc.nu ≈ [2.0] atol = 1e-10
        @test pc.x_cf[1] ≈ [0.0, 0.0] atol = 1e-10
        @test pc.z_cf[1] ≈ [2.0, 1.0] atol = 1e-10
        @test pc.rel_residual < 1e-10
        @test pc.spanned

        # non-enforceable variant: hand-derived LS values
        ce2 = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                  Theta_x=[reshape([-0.5, -0.5], 2, 1)],
                                  Theta_z=[reshape([1.0, 1.0], 2, 1)])
        pc2 = @test_logs (:warn, r"not enforceable") match_mode = :any begin
            policy_counterfactual(base, ce2, rule)
        end
        @test pc2.nu ≈ [1.5] atol = 1e-10
        @test pc2.error_path ≈ [0.25, -0.25] atol = 1e-10
        @test pc2.rel_residual ≈ norm([0.25, -0.25]) / norm([1.0, 0.5]) atol = 1e-10
        @test !pc2.spanned
    end

    @testset "exact square enforcement (CF-07 menu)" begin
        H = 8
        ce = policy_news_matrix(CF10_NK, :eps_i,
                                [:infl => :π, :ygap => :y], [:rate => :i]; H=H)
        sol = solve(CF10_NK)
        ir = irf(sol, H)
        base = baseline_path(ir, "eps_d", [:infl => "π", :ygap => "y"],
                             [:rate => "i"]; H=H)

        # self-rule fixed point: the spec's own Taylor rule => nu ≈ 0
        r_self = taylor_rule(H; rho=0.0, phi_pi=1.5, phi_y=0.0,
                             outcomes=[:infl, :ygap], instruments=[:rate])
        pc_self = policy_counterfactual(base, ce, r_self)
        @test norm(pc_self.nu) < 1e-8
        @test pc_self.x_cf[1] ≈ base.x[1] atol = 1e-8
        @test pc_self.z_cf[1] ≈ base.z[1] atol = 1e-8

        # peg / strict targeting / NGDP all hold exactly with the square menu
        for rule in (rate_peg_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]),
                     inflation_target_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]),
                     ngdp_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]))
            pc = policy_counterfactual(base, ce, rule)
            @test norm(pc.error_path) < 1e-8
            @test pc.spanned
        end
        # the peg really pegs; strict targeting really zeroes inflation
        pc_peg = policy_counterfactual(base, ce,
                                       rate_peg_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]))
        @test norm(pc_peg.z_cf[1]) < 1e-8
        pc_pi = policy_counterfactual(base, ce,
                                      inflation_target_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]))
        @test norm(pc_pi.x_cf[1]) < 1e-8
        # NGDP: pi_t + y_t - y_{t-1} = 0 along the counterfactual
        pc_n = policy_counterfactual(base, ce,
                                     ngdp_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]))
        pi_cf, y_cf = pc_n.x_cf[1], pc_n.x_cf[2]
        @test abs(pi_cf[1] + y_cf[1]) < 1e-8
        for t in 2:H
            @test abs(pi_cf[t] + y_cf[t] - y_cf[t-1]) < 1e-8
        end

        # thin 2-shock sub-menu: larger residual, honestly reported
        ce_thin = PolicyCausalEffects(outcomes=[:infl, :ygap], instruments=[:rate],
                                      Theta_x=[ce.Theta_x[1][:, 1:2], ce.Theta_x[2][:, 1:2]],
                                      Theta_z=[ce.Theta_z[1][:, 1:2]])
        pc_thin = MEM._suppress_warnings() do
            policy_counterfactual(base, ce_thin,
                                  rate_peg_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]))
        end
        @test pc_thin.rel_residual > pc_peg.rel_residual
    end

    @testset "bands from posterior draws" begin
        rng = MersenneTwister(10)
        A = [0.5 0.1 0.0; 0.0 0.4 0.1; 0.1 0.0 0.3]
        Y = zeros(220, 3)
        for t in 2:220
            Y[t, :] = A * Y[t-1, :] + randn(rng, 3)
        end
        post = estimate_bvar(Y, 2; n_draws=120)
        bir = irf(post, 10)
        H = 8
        ce = policy_causal_effects(bir, [3], [:x1 => 1, :x2 => 2], [:z => 3]; H=H)
        base = baseline_path(bir, 1, [:x1 => 1, :x2 => 2], [:z => 3]; H=H)
        rule = inflation_target_rule(H; pi_var=:x1, outcomes=[:x1, :x2], instruments=[:z])

        pc = MEM._suppress_warnings() do
            policy_counterfactual(base, ce, rule; baseline_draws=:match)
        end
        @test pc.n_draws_used + pc.n_draws_failed == MEM.n_draws(ce)
        @test pc.n_draws_used > 0
        @test size(pc.x_bands[1]) == (H, 3)
        # monotone bands
        @test all(pc.x_bands[1][:, 1] .<= pc.x_bands[1][:, 2] .+ 1e-12)
        @test all(pc.x_bands[1][:, 2] .<= pc.x_bands[1][:, 3] .+ 1e-12)
        @test pc.rel_residual_bands !== nothing

        # :fixed runs too
        pc_f = MEM._suppress_warnings() do
            policy_counterfactual(base, ce, rule; baseline_draws=:fixed)
        end
        @test pc_f.n_draws_used > 0

        # constant-replicated draws collapse bands onto the point
        Hs, ns = 4, 1
        Tx = reshape([-0.5, -0.25, -0.1, 0.0], Hs, ns)
        Tz = reshape([1.0, 0.5, 0.2, 0.1], Hs, ns)
        Dx = cat(Tx, Tx, Tx; dims=3)
        Dz = cat(Tz, Tz, Tz; dims=3)
        ce_c = PolicyCausalEffects(outcomes=[:x1], instruments=[:z],
                                   Theta_x=[Tx], Theta_z=[Tz],
                                   Theta_x_draws=[Dx], Theta_z_draws=[Dz])
        base_c = MEM.BaselinePath{Float64}([:x1], [:z], [[1.0, 0.5, 0.2, 0.1]],
                                           [[0.0, 0.0, 0.0, 0.0]], nothing, nothing,
                                           Hs, "toy")
        rule_c = inflation_target_rule(Hs; pi_var=:x1, outcomes=[:x1], instruments=[:z])
        pc_c = MEM._suppress_warnings() do
            policy_counterfactual(base_c, ce_c, rule_c)
        end
        for q in 1:3
            @test pc_c.x_bands[1][:, q] ≈ pc_c.x_cf[1] atol = 1e-10
        end

        # :match with mismatched draw counts errors
        base_m = MEM.BaselinePath{Float64}([:x1], [:z], [[1.0, 0.5, 0.2, 0.1]],
                                           [[0.0, 0.0, 0.0, 0.0]],
                                           [ones(Hs, 2)], [zeros(Hs, 2)], Hs, "toy")
        @test_throws ArgumentError policy_counterfactual(base_m, ce_c, rule_c;
                                                         baseline_draws=:match)
    end

    @testset "validation" begin
        H = 4
        ce = PolicyCausalEffects(outcomes=[:x], Theta_x=[ones(H, 1)])
        base = MEM.BaselinePath{Float64}([:x], Symbol[], [[1.0, 0.5, 0.2, 0.1]],
                                         Vector{Float64}[], nothing, nothing, H, "d")
        rule_bad = inflation_target_rule(H; pi_var=:pi, outcomes=[:pi], instruments=Symbol[])
        # rule references a symbol the inputs lack
        err = try
            policy_counterfactual(base, ce, rule_bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(":pi", err.msg)
        # H mismatch
        rule5 = inflation_target_rule(5; pi_var=:x, outcomes=[:x], instruments=Symbol[])
        @test_throws ArgumentError policy_counterfactual(base, ce, rule5)
        # draws=:on without draws
        rule4 = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=Symbol[])
        @test_throws ArgumentError MEM._suppress_warnings() do
            policy_counterfactual(base, ce, rule4; draws=:on)
        end
    end
end
