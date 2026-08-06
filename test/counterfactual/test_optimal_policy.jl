# CF-11 (#391): optimal-policy projection, loss accounting, implied rule.
using Test
using LinearAlgebra
using MacroEconometricModels

const MEM = MacroEconometricModels

const CF11_NK = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
    endogenous: π, y, i
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
    i[t] = φπ * π[t] + eps_i[t]
end

# Deterministic thin container + baseline for property tests.
function _cf11_thin(H)
    Tx1 = [-(0.6^t) * (0.8 + 0.1k) for t in 1:H, k in 1:2]
    Tx2 = [-(0.5^t) * (1.1 - 0.2k) for t in 1:H, k in 1:2]
    Tz = [(0.7^t) * (1.0 + 0.05k) for t in 1:H, k in 1:2]
    ce = PolicyCausalEffects(outcomes=[:infl, :ygap], instruments=[:rate],
                             Theta_x=[Tx1, Tx2], Theta_z=[Tz])
    base = MEM.BaselinePath{Float64}([:infl, :ygap], [:rate],
                                     [[0.9^t for t in 1:H], [0.5 * 0.85^t for t in 1:H]],
                                     [[0.1 * 0.9^t for t in 1:H]],
                                     nothing, nothing, H, "demand")
    return ce, base
end

@testset "Optimal-policy projection (CF-11)" begin

    @testset "scalar analytic case" begin
        H = 4
        theta = reshape([-0.5, -0.4, -0.3, -0.2], H, 1)
        xb = [1.0, 0.8, 0.6, 0.4]
        ce = PolicyCausalEffects(outcomes=[:x], Theta_x=[theta])
        base = MEM.BaselinePath{Float64}([:x], Symbol[], [xb], Vector{Float64}[],
                                         nothing, nothing, H, "d")
        loss = policy_loss([:x], H; lambda=[1.0])
        pc = optimal_policy(base, ce, loss)
        nu_ref = -(theta' * theta) \ (theta' * xb)
        @test pc.nu ≈ nu_ref atol = 1e-10
        @test pc.x_cf[1] ≈ xb + theta * nu_ref atol = 1e-10
        @test pc.loss_cf <= pc.loss_base
        @test pc.foc_norm < 1e-10
        @test pc.rule_name == loss.name
    end

    @testset "FOC ≈ 0 across loss templates" begin
        H = 6
        ce, base = _cf11_thin(H)
        sp = smoothing_penalty(H; lambda=0.2, beta=0.98, z_lag=0.3)
        for (loss, wz) in ((policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25], beta=0.99), nothing),
                           (ait_loss(H; pi_var=:infl, y_var=:ygap), nothing),
                           (policy_loss([:infl, :ygap], H; lambda=[1.0, 0.5],
                                        instruments=[:rate], W_z=[sp.W_z]), [sp.wedge_term]))
            pc = optimal_policy(base, ce, loss; z_wedge=wz)
            @test pc.foc_norm < 1e-8
            @test pc.loss_cf <= pc.loss_base + 1e-12
        end
    end

    @testset "optimality within the span (vs CF-10 rule templates)" begin
        H = 6
        ce, base = _cf11_thin(H)
        loss = policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25], beta=0.99)
        pc_opt = optimal_policy(base, ce, loss)
        for rule in (rate_peg_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]),
                     inflation_target_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]),
                     ngdp_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]),
                     taylor_rule(H; outcomes=[:infl, :ygap], instruments=[:rate]))
            pc_rule = MEM._suppress_warnings() do
                policy_counterfactual(base, ce, rule)
            end
            L_rule = MEM._policy_loss_value(loss,
                                            [pc_rule.x_cf[1], pc_rule.x_cf[2]],
                                            Vector{Float64}[])
            @test pc_opt.loss_cf <= L_rule + 1e-10
        end
    end

    @testset "smoothing linear term: augmented normal equations by hand" begin
        H = 3
        Tx = reshape([-0.5, -0.4, -0.3], H, 1)
        Tz = reshape([1.0, 0.6, 0.3], H, 1)
        xb = [1.0, 0.7, 0.4]
        zb = [0.0, 0.0, 0.0]
        ce = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                 Theta_x=[Tx], Theta_z=[Tz])
        base = MEM.BaselinePath{Float64}([:x], [:z], [xb], [zb],
                                         nothing, nothing, H, "d")
        sp = smoothing_penalty(H; lambda=0.5, beta=1.0, z_lag=0.4)
        loss = policy_loss([:x], H; lambda=[1.0], instruments=[:z], W_z=[sp.W_z])
        pc = optimal_policy(base, ce, loss; z_wedge=[sp.wedge_term])
        Wx = Matrix{Float64}(I, H, H)
        nu_ref = -(Tx' * Wx * Tx + Tz' * sp.W_z * Tz) \
                 (Tx' * Wx * xb + Tz' * sp.W_z * zb - Tz' * sp.wedge_term)
        @test pc.nu ≈ nu_ref atol = 1e-10
        # without the wedge the answer differs (the linear term binds)
        pc0 = optimal_policy(base, ce, loss)
        @test !(pc0.nu ≈ pc.nu)
    end

    @testset "square case: optimal_rule closes the circle" begin
        H = 8
        ce = policy_news_matrix(CF11_NK, :eps_i,
                                [:infl => :π, :ygap => :y], [:rate => :i]; H=H)
        sol = solve(CF11_NK)
        base = baseline_path(irf(sol, H), "eps_d",
                             [:infl => "π", :ygap => "y"], [:rate => "i"]; H=H)
        loss = policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25], beta=0.99)
        pc_opt = optimal_policy(base, ce, loss)
        @test pc_opt.foc_norm < 1e-8

        rule = optimal_rule(ce, loss)
        @test rule isa PolicyRule{Float64}
        @test occursin("optimal targeting", rule.name)
        pc_rule = MEM._suppress_warnings() do
            policy_counterfactual(base, ce, rule)
        end
        @test pc_rule.x_cf[1] ≈ pc_opt.x_cf[1] atol = 1e-8
        @test pc_rule.x_cf[2] ≈ pc_opt.x_cf[2] atol = 1e-8
        @test pc_rule.z_cf[1] ≈ pc_opt.z_cf[1] atol = 1e-8

        # thin container refuses to produce a targeting rule
        ce_thin, _ = _cf11_thin(4)
        loss4 = policy_loss([:infl, :ygap], 4; lambda=[1.0, 1.0])
        @test_throws ArgumentError optimal_rule(ce_thin, loss4)
    end

    @testset "draw propagation" begin
        H = 4
        Tx = reshape([-0.5, -0.4, -0.3, -0.2], H, 1)
        Tz = reshape([1.0, 0.6, 0.3, 0.1], H, 1)
        Dx = cat(Tx, 1.1 .* Tx, 0.9 .* Tx; dims=3)
        Dz = cat(Tz, 1.1 .* Tz, 0.9 .* Tz; dims=3)
        ce = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                 Theta_x=[Tx], Theta_z=[Tz],
                                 Theta_x_draws=[Dx], Theta_z_draws=[Dz])
        base = MEM.BaselinePath{Float64}([:x], [:z], [[1.0, 0.8, 0.6, 0.4]],
                                         [[0.0, 0.0, 0.0, 0.0]],
                                         nothing, nothing, H, "d")
        loss = policy_loss([:x], H; lambda=[1.0])
        pc = optimal_policy(base, ce, loss)
        @test pc.n_draws_used == 3
        @test pc.n_draws_failed == 0
        @test size(pc.x_bands[1]) == (H, 3)
        @test all(pc.x_bands[1][:, 1] .<= pc.x_bands[1][:, 2] .+ 1e-12)
        @test all(pc.x_bands[1][:, 2] .<= pc.x_bands[1][:, 3] .+ 1e-12)
    end

    @testset "validation" begin
        H = 4
        ce, base = _cf11_thin(H)
        @test_throws ArgumentError optimal_policy(base, ce,
                                                  policy_loss([:gdp], H; lambda=[1.0]))
        @test_throws ArgumentError optimal_policy(base, ce,
                                                  policy_loss([:infl], H + 1; lambda=[1.0]))
        # z_wedge without loss instruments
        @test_throws ArgumentError optimal_policy(base, ce,
                                                  policy_loss([:infl], H; lambda=[1.0]);
                                                  z_wedge=[ones(H)])
    end
end
