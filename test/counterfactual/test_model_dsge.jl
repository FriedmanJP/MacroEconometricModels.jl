# CF-07 (#387): model-implied news-shock menus from linear DSGE.
using Test
using LinearAlgebra
using MacroEconometricModels

const MEM = MacroEconometricModels

# Backward-looking toy: news columns must be exact time-shifts of column 1.
const CF07_AR1 = @dsge begin
    parameters: ρ = 0.8
    endogenous: y
    exogenous: eps_i
    y[t] = ρ * y[t-1] + eps_i[t]
end

# Fisher-equation model with Taylor closure i = φπ + ε:
#   φ·π[t] + ε_total[t] = E[π[t+1]]  ⇒  π[t] = (1/φ)·π[t+1] − (1/φ)·ε_total[t].
# Forward solution: a shock announced at 1, hitting at date c, gives
#   π_t = −φ^{−(c−t+1)} for t ≤ c and 0 after — a full analytic news menu.
const CF07_FISHER = @dsge begin
    parameters: φ = 1.5
    endogenous: π
    exogenous: eps_i
    π[t] = (1 / φ) * π[t+1] - (1 / φ) * eps_i[t]
end

const CF07_FISHER_INDET = @dsge begin
    parameters: φ = 0.5
    endogenous: π
    exogenous: eps_i
    π[t] = (1 / φ) * π[t+1] - (1 / φ) * eps_i[t]
end

# 3-eq NK for the closure-dependence assertion.
function _cf07_nk(phi_pi)
    if phi_pi == 1.5
        return @dsge begin
            parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
            endogenous: π, y, i
            exogenous: eps_i
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - σ * (i[t] - π[t+1])
            i[t] = φπ * π[t] + eps_i[t]
        end
    else
        return @dsge begin
            parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 3.0
            endogenous: π, y, i
            exogenous: eps_i
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - σ * (i[t] - π[t+1])
            i[t] = φπ * π[t] + eps_i[t]
        end
    end
end

@testset "Model-implied news menus (CF-07)" begin

    @testset "backward-looking: news columns are time shifts" begin
        H = 6
        ce = policy_news_matrix(CF07_AR1, :eps_i, [:y => :y]; H=H)
        @test ce isa PolicyCausalEffects{Float64}
        @test is_square(ce)
        @test ce.source == :dsge
        Θ = ce.Theta_x[1]
        for s in 1:H, t in 1:H
            expected = t >= s ? 0.8^(t - s) : 0.0
            @test Θ[t, s] ≈ expected atol = 1e-10
        end
    end

    @testset "Fisher analytic forward menu" begin
        H = 5
        φ = 1.5
        ce = policy_news_matrix(CF07_FISHER, :eps_i, [:pi => :π]; H=H)
        Θ = ce.Theta_x[1]
        for c in 1:H, t in 1:H
            expected = t <= c ? -φ^(-(c - t + 1)) : 0.0
            @test Θ[t, c] ≈ expected atol = 1e-8
        end
        # column 1 equals the ordinary IRF to the policy shock exactly
        sol = solve(CF07_FISHER)
        ir = irf(sol, H)
        si = findfirst(==("eps_i"), ir.shocks)
        @test Θ[:, 1] ≈ ir.values[1:H, 1, si] atol = 1e-12
        # anticipation: inflation moves BEFORE the shock arrives in news columns
        @test abs(Θ[1, 4]) > 0
        @test abs(Θ[2, 4]) > abs(Θ[1, 4])   # builds up toward arrival
    end

    @testset "closure rules produce different menus" begin
        H = 4
        ce_a = policy_news_matrix(_cf07_nk(1.5), :eps_i, [:pi => :π, :y => :y], [:i => :i]; H=H)
        ce_b = policy_news_matrix(_cf07_nk(3.0), :eps_i, [:pi => :π, :y => :y], [:i => :i]; H=H)
        @test !(ce_a.Theta_x[1] ≈ ce_b.Theta_x[1])
        # anticipated columns move inflation before the rate-arrival date
        @test abs(ce_a.Theta_x[1][1, 3]) > 1e-10
        @test length(ce_a.Theta_z) == 1
    end

    @testset "chunked menus match monolithic" begin
        H = 6
        ce_m = policy_news_matrix(CF07_AR1, :eps_i, [:y => :y]; H=H)
        ce_c = policy_news_matrix(CF07_AR1, :eps_i, [:y => :y]; H=H, chunk=2)
        @test ce_m.Theta_x[1] ≈ ce_c.Theta_x[1] atol = 1e-12
        ce_f = policy_news_matrix(CF07_FISHER, :eps_i, [:pi => :π]; H=5, chunk=3)
        ce_fm = policy_news_matrix(CF07_FISHER, :eps_i, [:pi => :π]; H=5)
        @test ce_f.Theta_x[1] ≈ ce_fm.Theta_x[1] atol = 1e-10
    end

    @testset "H = 1 edge (no news, unanticipated column only)" begin
        ce = policy_news_matrix(CF07_AR1, :eps_i, [:y => :y]; H=1)
        @test size(ce.Theta_x[1]) == (1, 1)
        @test ce.Theta_x[1][1, 1] ≈ 1.0 atol = 1e-10
    end

    @testset "indeterminate closure errors" begin
        err = try
            policy_news_matrix(CF07_FISHER_INDET, :eps_i, [:pi => :π]; H=4)
            nothing
        catch e
            e
        end
        @test err isa Exception
        if err isa ArgumentError
            @test occursin("determinacy", err.msg)
        end
    end

    @testset "validation" begin
        @test_throws ArgumentError policy_news_matrix(CF07_AR1, :nope, [:y => :y]; H=4)
        @test_throws ArgumentError policy_news_matrix(CF07_AR1, :eps_i, [:y => :z]; H=4)
        @test_throws ArgumentError policy_news_matrix(CF07_AR1, :eps_i, [:y => :y]; H=0)
        @test_throws ArgumentError policy_news_matrix(CF07_AR1, :eps_i, [:y => :y];
                                                      H=4, solver=:perturbation)
        @test_throws ArgumentError policy_news_matrix(CF07_AR1, :eps_i,
                                                      Pair{Symbol,Symbol}[]; H=4)
    end
end
