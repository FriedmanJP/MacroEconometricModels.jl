# CF-19 (#399): spanning + forecast-sufficiency diagnostics.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

@testset "Spanning + forecast sufficiency (CF-19)" begin
    rng = MersenneTwister(20260819)

    @testset "spanning: spanned case" begin
        H = 8
        Tx = randn(rng, H, H)
        Tz = randn(rng, H, H)
        ce_full = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                      Theta_x=[Tx], Theta_z=[Tz])
        ce_emp = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                     Theta_x=[Tx[:, 1:2]], Theta_z=[Tz[:, 1:2]])
        # base engineered so the exact nu* loads only on news dims 1:2
        d = [0.8, -0.4]
        base = MEM.BaselinePath{Float64}([:x], [:z], [-Tx[:, 1:2] * d],
                                         [-Tz[:, 1:2] * d], nothing, nothing, H, "eng")
        rule = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=[:z])
        sd = spanning_diagnostic(base, ce_emp, ce_full, rule)
        @test sd isa SpanningDiagnostic{Float64}
        @test sd.gap[1] < 1e-8
        @test sd.gap_rel[1] < 1e-8
        @test sd.loading_inside ≈ 1.0 atol = 1e-8
        @test sd.spanned
        @test sd.rel_residual_emp < 1e-8
        @test sd.x_cf_emp[1] ≈ sd.x_cf_full[1] atol = 1e-8
    end

    @testset "spanning: unspanned case (persistent baseline)" begin
        H = 12
        # empirical columns are transitory; the baseline is persistent
        Tx_emp = hcat([-(0.3^t) for t in 1:H], [-(0.4^t) * 0.5 for t in 1:H])
        Tz_emp = hcat([(0.3^t) for t in 1:H], [(0.4^t) for t in 1:H])
        Tx_full = randn(rng, H, H) .* 0.3 .+ hcat(Tx_emp, zeros(H, H - 2))
        Tz_full = randn(rng, H, H) .* 0.3 .+ hcat(Tz_emp, zeros(H, H - 2))
        ce_full = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                      Theta_x=[Tx_full], Theta_z=[Tz_full])
        ce_emp = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                     Theta_x=[Tx_emp], Theta_z=[Tz_emp])
        base = MEM.BaselinePath{Float64}([:x], [:z], [[0.95^t for t in 1:H]],
                                         [zeros(H)], nothing, nothing, H, "persistent")
        rule = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=[:z])
        sd = MEM._suppress_warnings() do
            spanning_diagnostic(base, ce_emp, ce_full, rule)
        end
        @test sd.gap_rel[1] > 0.15
        @test sd.loading_inside < 0.95
        @test !sd.spanned
        @test 0.0 <= sd.loading_inside <= 1.0 + 1e-10
        @test sd.rel_residual_emp > 0.05     # the thin solve reports its failure
    end

    @testset "spanning: draw bands + validation" begin
        H = 6
        Tx = randn(rng, H, H)
        Tz = randn(rng, H, H)
        rep3(M) = cat(M, 1.05 .* M, 0.95 .* M; dims=3)
        ce_full = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                      Theta_x=[Tx], Theta_z=[Tz],
                                      Theta_x_draws=[rep3(Tx)], Theta_z_draws=[rep3(Tz)])
        ce_emp = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                     Theta_x=[Tx[:, 1:2]], Theta_z=[Tz[:, 1:2]],
                                     Theta_x_draws=[rep3(Tx[:, 1:2])],
                                     Theta_z_draws=[rep3(Tz[:, 1:2])])
        base = MEM.BaselinePath{Float64}([:x], [:z], [randn(rng, H)], [zeros(H)],
                                         nothing, nothing, H, "b")
        rule = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=[:z])
        sd = MEM._suppress_warnings() do
            spanning_diagnostic(base, ce_emp, ce_full, rule; n_sim=40,
                                rng=MersenneTwister(2))
        end
        @test sd.bands_gap !== nothing
        @test size(sd.bands_gap[1]) == (H, 3)

        # mismatched containers error
        ce_bad = PolicyCausalEffects(outcomes=[:other], instruments=[:z],
                                     Theta_x=[Tx[:, 1:2]], Theta_z=[Tz[:, 1:2]])
        @test_throws ArgumentError spanning_diagnostic(base, ce_bad, ce_full, rule)
        # thin ce_full rejected
        @test_throws ArgumentError spanning_diagnostic(base, ce_emp, ce_emp, rule)
    end

    @testset "forecast sufficiency: invertible case" begin
        spec = @dsge begin
            parameters: ρ = 0.8
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + ε[t]
        end
        sol = solve(spec)
        fs = forecast_sufficiency(sol, [:y]; H=20)
        @test fs isa ForecastSufficiency{Float64}
        @test fs.invertible
        @test maximum(abs.(fs.fev_ratio .- 1)) < 1e-8
        @test all(fs.fev_ratio .>= 1 - 1e-8)
    end

    @testset "forecast sufficiency: non-invertible but sufficient" begin
        # unobserved slow-moving state with tiny variance: 2 shocks, 1 observable
        spec = @dsge begin
            parameters: ρs = 0.98, ρx = 0.5, σs = 0.001
            endogenous: s, x
            exogenous: eps_s, eps_x
            s[t] = ρs * s[t-1] + σs * eps_s[t]
            x[t] = ρx * x[t-1] + s[t] + eps_x[t]
        end
        sol = solve(spec)
        fs = forecast_sufficiency(sol, [:x]; H=30)
        @test !fs.invertible                       # 2 shocks, 1 innovation
        @test maximum(fs.fev_ratio) < 1.05         # ...but forecasts barely suffer
        @test all(fs.fev_ratio .>= 1 - 1e-8)
        @test fs.one_step_ratio[1] > 1 + 1e-6

        # cranking up the hidden state's variance breaks sufficiency
        spec2 = @dsge begin
            parameters: ρs = 0.98, ρx = 0.5, σs = 1.0
            endogenous: s, x
            exogenous: eps_s, eps_x
            s[t] = ρs * s[t-1] + σs * eps_s[t]
            x[t] = ρx * x[t-1] + s[t] + eps_x[t]
        end
        fs2 = forecast_sufficiency(solve(spec2), [:x]; H=30)
        @test maximum(fs2.fev_ratio) > maximum(fs.fev_ratio)

        # validation
        @test_throws ArgumentError forecast_sufficiency(sol, [:nope]; H=10)
        @test_throws ArgumentError forecast_sufficiency(sol, Symbol[]; H=10)
        @test_throws ArgumentError forecast_sufficiency(sol, [:x]; H=0)
    end
end
