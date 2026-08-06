# CF-18 (#398): historical/conditional counterfactuals via forecast revisions.
# The decisive test is the end-to-end linear oracle: the revision recursion
# must reproduce a direct simulation under the alternative rule exactly.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

const CF18_H = 16

function _cf18_spec(phi_pi)
    if phi_pi == 1.5
        return @dsge begin
            parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
            endogenous: π, y, i
            exogenous: eps_i, eps_d
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
            i[t] = φπ * π[t] + eps_i[t]
        end
    else
        return @dsge begin
            parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 3.0
            endogenous: π, y, i
            exogenous: eps_i, eps_d
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
            i[t] = φπ * π[t] + eps_i[t]
        end
    end
end

# simulate y_t = G1 y_{t-1} + impact ε_t from a zero initial state
function _cf18_sim(sol, E)
    Tn = size(E, 1)
    n = size(sol.G1, 1)
    Y = zeros(Tn, n)
    for t in 1:Tn
        prev = t == 1 ? zeros(n) : Y[t-1, :]
        Y[t, :] = sol.G1 * prev + sol.impact * E[t, :]
    end
    return Y
end

# exact VAR(1) representation of a linear DSGE solution
function _cf18_varmodel(sol, Y)
    n = size(sol.G1, 1)
    B = vcat(zeros(1, n), Matrix(sol.G1'))
    Sigma = sol.impact * sol.impact' + 1e-12 * I
    VARModel(Y, 1, B, zeros(size(Y, 1) - 1, n), Matrix(Sigma), 0.0, 0.0, 0.0,
             ["π", "y", "i"])
end

@testset "Historical counterfactuals (CF-18)" begin
    rng = MersenneTwister(20260818)

    @testset "revision alignment on a trended AR(1)" begin
        c, phi = 0.7, 0.6                     # nonzero intercept: must cancel
        Tn, H = 40, 8
        u = randn(rng, Tn)
        y = zeros(Tn)
        y[1] = c / (1 - phi)
        for t in 2:Tn
            y[t] = c + phi * y[t-1] + u[t]
        end
        B = reshape([c, phi], 2, 1)
        t0 = 20
        d = MEM._forecast_revisions(B, reshape(y, Tn, 1), t0, 1, H)
        for r in 1:H
            @test d[r, 1] ≈ phi^(r - 1) * u[t0] atol = 1e-12
        end
    end

    @testset "end-to-end linear oracle" begin
        spec_A = _cf18_spec(1.5)
        spec_At = _cf18_spec(3.0)
        sol_A = solve(spec_A)
        sol_At = solve(spec_At)
        t1, t2, T_all = 4, 12, 14
        E = zeros(T_all, 2)
        E[t1:t2, 2] = randn(rng, t2 - t1 + 1)      # demand shocks only, from t1
        Y_A = _cf18_sim(sol_A, E)
        Y_At = _cf18_sim(sol_At, E)                 # zero state before t1: exact comparator
        m = _cf18_varmodel(sol_A, Y_A)
        ce = policy_news_matrix(spec_A, :eps_i, [:pi => :π, :y => :y],
                                [:rate => :i]; H=CF18_H)
        rule_At = taylor_rule(CF18_H; rho=0.0, phi_pi=3.0, phi_y=0.0,
                              pi_var=:pi, y_var=:y,
                              outcomes=[:pi, :y], instruments=[:rate])
        ch = counterfactual_history(m, Y_A, t1:t2, ce, rule_At;
                                    outcomes=[:pi => "π", :y => "y"],
                                    instruments=[:rate => "i"], H=CF18_H)
        @test ch isa CounterfactualHistory{Float64}
        for (j, t) in enumerate(t1:t2)
            @test ch.cf[j, 1] ≈ Y_At[t, 1] atol = 1e-6   # π
            @test ch.cf[j, 2] ≈ Y_At[t, 2] atol = 1e-6   # y
            @test ch.cf[j, 3] ≈ Y_At[t, 3] atol = 1e-6   # i
            @test ch.realized[j, 1] ≈ Y_A[t, 1] atol = 1e-12
        end
        @test maximum(ch.rel_residual) < 1e-6            # square menu: exact enforcement

        # rule-satisfied case: the generating rule reproduces the data
        rule_A = taylor_rule(CF18_H; rho=0.0, phi_pi=1.5, phi_y=0.0,
                             pi_var=:pi, y_var=:y,
                             outcomes=[:pi, :y], instruments=[:rate])
        ch0 = counterfactual_history(m, Y_A, t1:t2, ce, rule_A;
                                     outcomes=[:pi => "π", :y => "y"],
                                     instruments=[:rate => "i"], H=CF18_H)
        @test ch0.cf ≈ ch0.realized atol = 1e-8

        # consistency of the two entry points at the first date
        pcf = counterfactual_forecast(m, Y_A[1:t1, :], ce, rule_At;
                                      outcomes=[:pi => "π", :y => "y"],
                                      instruments=[:rate => "i"], H=CF18_H)
        @test pcf.x_cf[1][1] ≈ ch.cf[1, 1] atol = 1e-8
        @test pcf.x_cf[2][1] ≈ ch.cf[1, 2] atol = 1e-8
        @test pcf.z_cf[1][1] ≈ ch.cf[1, 3] atol = 1e-8
        # add-back contract: :forecast == :none + the pre-existing trajectory
        # (this NK model is purely forward-looking, G1 ≈ 0, so the carry itself
        # is ~0 — assert the identity through the forecast helper instead)
        pcf_l = counterfactual_forecast(m, Y_A[1:t2, :], ce, rule_At;
                                        outcomes=[:pi => "π", :y => "y"],
                                        instruments=[:rate => "i"], H=CF18_H)
        pcf0 = counterfactual_forecast(m, Y_A[1:t2, :], ce, rule_At;
                                       outcomes=[:pi => "π", :y => "y"],
                                       instruments=[:rate => "i"], H=CF18_H,
                                       add_back=:none)
        carry = MEM._var_forecast_path(m.B, Y_A[1:t2-1, :], 1, CF18_H)
        @test pcf_l.x_cf[1] ≈ pcf0.x_cf[1] + carry[:, 1] atol = 1e-10
        @test pcf_l.x_base[1] ≈ pcf0.x_base[1] + carry[:, 1] atol = 1e-10
        # and on a persistent scalar model the carry is genuinely nonzero
        cp, phip, Hs = 0.5, 0.6, 6
        Tn = 20
        ys = zeros(Tn)
        ys[1] = cp / (1 - phip)
        us = randn(rng, Tn)
        for t in 2:Tn
            ys[t] = cp + phip * ys[t-1] + us[t]
        end
        m1 = VARModel(reshape(ys, Tn, 1), 1, reshape([cp, phip], 2, 1),
                      zeros(Tn - 1, 1), reshape([1.0], 1, 1), 0.0, 0.0, 0.0, ["y"])
        ce1 = PolicyCausalEffects(outcomes=[:y], Theta_x=[reshape(-(0.5 .^ (1:Hs)), Hs, 1)])
        r1 = inflation_target_rule(Hs; pi_var=:y, outcomes=[:y], instruments=Symbol[])
        pf_f = MEM._suppress_warnings() do
            counterfactual_forecast(m1, reshape(ys, Tn, 1), ce1, r1;
                                    outcomes=[:y => "y"], H=Hs)
        end
        pf_n = MEM._suppress_warnings() do
            counterfactual_forecast(m1, reshape(ys, Tn, 1), ce1, r1;
                                    outcomes=[:y => "y"], H=Hs, add_back=:none)
        end
        @test !(pf_f.x_cf[1] ≈ pf_n.x_cf[1])
    end

    @testset "wedge threading bookkeeping" begin
        spec_A = _cf18_spec(1.5)
        sol_A = solve(spec_A)
        t1, t2, T_all = 4, 8, 10
        E = zeros(T_all, 2)
        E[t1:t2, 2] = randn(rng, t2 - t1 + 1)
        Y_A = _cf18_sim(sol_A, E)
        m = _cf18_varmodel(sol_A, Y_A)
        ce = policy_news_matrix(spec_A, :eps_i, [:pi => :π, :y => :y],
                                [:rate => :i]; H=CF18_H)
        sp = smoothing_penalty(CF18_H; lambda=0.2, beta=0.99)
        loss = policy_loss([:pi, :y], CF18_H; lambda=[1.0, 0.25],
                           instruments=[:rate], W_z=[sp.W_z])
        seen = Float64[]
        builder = z -> begin
            push!(seen, z)
            smoothing_penalty(CF18_H; lambda=0.2, beta=0.99, z_lag=z).wedge_term
        end
        ch = counterfactual_history(m, Y_A, t1:t2, ce, loss;
                                    outcomes=[:pi => "π", :y => "y"],
                                    instruments=[:rate => "i"], H=CF18_H,
                                    wedge_builder=builder)
        @test length(seen) == t2 - t1 + 1
        @test seen[1] == 0.0
        for j in 2:length(seen)
            @test seen[j] ≈ ch.cf[j-1, 3] atol = 1e-12   # previous CF instrument level
        end
    end

    @testset "draw propagation" begin
        spec_A = _cf18_spec(1.5)
        sol_A = solve(spec_A)
        t1, t2, T_all = 4, 7, 9
        E = zeros(T_all, 2)
        E[t1:t2, 2] = randn(rng, t2 - t1 + 1)
        Y_A = _cf18_sim(sol_A, E)
        m = _cf18_varmodel(sol_A, Y_A)
        ce0 = policy_news_matrix(spec_A, :eps_i, [:pi => :π, :y => :y],
                                 [:rate => :i]; H=CF18_H)
        rep(M) = cat(M, M, M; dims=3)
        ce = PolicyCausalEffects(outcomes=ce0.outcomes, instruments=ce0.instruments,
                                 Theta_x=ce0.Theta_x, Theta_z=ce0.Theta_z,
                                 Theta_x_draws=[rep(M) for M in ce0.Theta_x],
                                 Theta_z_draws=[rep(M) for M in ce0.Theta_z],
                                 shock_labels=ce0.shock_labels, source=:dsge)
        rule = taylor_rule(CF18_H; rho=0.0, phi_pi=3.0, phi_y=0.0,
                           pi_var=:pi, y_var=:y,
                           outcomes=[:pi, :y], instruments=[:rate])
        ch = counterfactual_history(m, Y_A, t1:t2, ce, rule;
                                    outcomes=[:pi => "π", :y => "y"],
                                    instruments=[:rate => "i"], H=CF18_H)
        @test ch.n_draws_used == 3
        @test ch.cf_bands !== nothing
        for q in 1:3
            @test ch.cf_bands[:, :, q] ≈ ch.cf atol = 1e-10   # replicated draws collapse
        end
    end

    @testset "validation" begin
        spec_A = _cf18_spec(1.5)
        sol_A = solve(spec_A)
        Y = _cf18_sim(sol_A, zeros(8, 2))
        m = _cf18_varmodel(sol_A, Y)
        ce = policy_news_matrix(spec_A, :eps_i, [:pi => :π]; H=4)
        rule = inflation_target_rule(4; pi_var=:pi, outcomes=[:pi], instruments=Symbol[])
        maps = (outcomes=[:pi => "π"], instruments=Pair{Symbol,Int}[])
        # window longer than H − 1
        @test_throws ArgumentError counterfactual_history(m, Y, 2:8, ce, rule;
                                                          outcomes=maps.outcomes, H=4)
        # revision origin needs t > p
        @test_throws ArgumentError counterfactual_history(m, Y, 1:2, ce, rule;
                                                          outcomes=maps.outcomes, H=4)
        # dates length mismatch
        @test_throws ArgumentError counterfactual_history(m, Y, 3:4, ce, rule;
                                                          outcomes=maps.outcomes, H=4,
                                                          dates=["a"])
        # wedge_builder with a rule policy
        @test_throws ArgumentError counterfactual_history(m, Y, 3:4, ce, rule;
                                                          outcomes=maps.outcomes, H=4,
                                                          wedge_builder=z -> zeros(4))
        # H mismatch with the container
        @test_throws ArgumentError counterfactual_forecast(m, Y[1:4, :], ce, rule;
                                                           outcomes=maps.outcomes, H=5)
    end
end
