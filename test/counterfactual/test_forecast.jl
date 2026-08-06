# CF-05 (#385): forecast draw retention + PolicyForecast gap container —
# store_draws on the BVAR side, gap transforms, SEP-style simulation,
# min_sd flooring, quarterly interpolation.
using Test
using LinearAlgebra
using Random
using Statistics
using MacroEconometricModels

const MEM = MacroEconometricModels

function _cf05_data(rng; T_obs=250, n=3)
    A = [0.5 0.1 0.0; 0.0 0.4 0.1; 0.1 0.0 0.3][1:n, 1:n]
    Y = zeros(T_obs, n)
    for t in 2:T_obs
        Y[t, :] = A * Y[t-1, :] + randn(rng, n)
    end
    return Y
end

@testset "Forecast adapters (CF-05)" begin
    rng = MersenneTwister(20260805)
    Y = _cf05_data(rng)

    @testset "BVAR store_draws retention" begin
        post = estimate_bvar(Y, 2; n_draws=300)
        fc0 = forecast(post, 8; rng=MersenneTwister(1))
        @test fc0._draws === nothing            # default: zero behavior change

        fc = forecast(post, 8; store_draws=true, rng=MersenneTwister(1))
        @test fc._draws isa Array{Float64,3}
        @test size(fc._draws, 2) == 8
        @test size(fc._draws, 3) == 3
        # quantile-consistency: stored bands are exact quantiles of the stored draws
        for hi in (1, 5), j in 1:3
            d = fc._draws[:, hi, j]
            @test quantile(d, 0.025) ≈ fc.ci_lower[hi, j] atol = 1e-12
            @test quantile(d, 0.975) ≈ fc.ci_upper[hi, j] atol = 1e-12
        end
    end

    @testset "VAR bootstrap draws present (pre-existing #524 path)" begin
        m = estimate_var(Y, 2)
        fc = forecast(m, 6; reps=60, rng=MersenneTwister(2))
        @test fc._draws isa Array{Float64,3}
        @test size(fc._draws, 2) == 6
    end

    @testset "gap transform from package forecasts" begin
        m = estimate_var(Y, 2)
        fc = forecast(m, 10; reps=50, rng=MersenneTwister(3))
        pf = policy_forecast(fc, [:infl => 1, :ygap => 2];
                             targets=[:infl => 2.0], H=8, origin="2021Q2")
        @test pf isa PolicyForecast{Float64}
        @test pf.H == 8
        @test pf.origin == "2021Q2"
        # inflation shifted by exactly -2 in point and every draw
        @test pf.values[1] == fc.forecast[1:8, 1] .- 2.0
        @test pf.values[2] == fc.forecast[1:8, 2]
        @test pf.draws[1][3, 7] == fc._draws[7, 3, 1] - 2.0
        @test pf.draws[2][3, 7] == fc._draws[7, 3, 2]
        @test MEM.n_draws(pf) == size(fc._draws, 1)

        # length-H target path
        tgt = collect(0.1:0.1:0.8)
        pf2 = policy_forecast(fc, [:infl => 1]; targets=[:infl => tgt], H=8)
        @test pf2.values[1] == fc.forecast[1:8, 1] .- tgt

        # no draws when the source forecast has none
        post = estimate_bvar(Y, 2; n_draws=100)
        fcb = forecast(post, 8)
        pfb = policy_forecast(fcb, [:infl => 1])
        @test pfb.draws === nothing

        # H beyond the forecast horizon errors with the re-run hint
        err = try
            policy_forecast(fc, [:infl => 1]; H=11)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("longer horizon", err.msg)
        # unknown target symbol
        @test_throws ArgumentError policy_forecast(fc, [:infl => 1]; targets=[:gdp => 2.0])
    end

    @testset "external SEP-style route" begin
        H = 6
        vals = [collect(range(1.0, 0.0; length=H)), fill(-0.5, H)]
        sds = [fill(0.4, H), fill(0.2, H)]
        pf = policy_forecast([:infl, :ygap], vals; sd=sds, rho=0.9,
                             n_draws=6000, rng=MersenneTwister(4), H=H)
        @test pf.values[1] == vals[1]
        @test MEM.n_draws(pf) == 6000

        # sample covariance reproduces Sigma_i = sd_j sd_k rho^|j-k| (5% tolerance)
        E1 = pf.draws[1] .- vals[1]
        S_hat = E1 * E1' ./ 6000
        S_ref = [0.4 * 0.4 * 0.9^abs(j - k) for j in 1:H, k in 1:H]
        @test maximum(abs.(S_hat .- S_ref)) < 0.05 * maximum(S_ref)

        # outcomes uncorrelated under :independent
        c12 = cor(vec(pf.draws[1][1, :]), vec(pf.draws[2][1, :]))
        @test abs(c12) < 0.05

        # user block covariance route: perfectly correlated outcomes
        n_x = 2
        base = [0.09 * 0.95^abs(j - k) for j in 1:H, k in 1:H]
        Sig = [base base; base base] + 1e-10 * I
        pfc = policy_forecast([:infl, :ygap], vals; cross_corr=Sig,
                              n_draws=4000, rng=MersenneTwister(5), H=H)
        cc = cor(vec(pfc.draws[1][2, :]), vec(pfc.draws[2][2, :]))
        @test cc > 0.99

        # validation
        @test_throws ArgumentError policy_forecast([:infl], vals; sd=sds, H=H)
        @test_throws ArgumentError policy_forecast([:infl, :ygap], vals; H=H)  # sd missing
        @test_throws ArgumentError policy_forecast([:infl, :ygap], vals; sd=sds,
                                                   cross_corr=:block, H=H)
    end

    @testset "min_sd flooring" begin
        H = 4
        vals = [zeros(H)]
        sds = [[0.3, 0.3, 0.0, 0.0]]      # long-run pinned at target, zero sd
        pf = @test_logs (:warn, r"floored 2 dispersion entries") match_mode = :any begin
            policy_forecast([:infl], vals; sd=sds, n_draws=2000,
                            rng=MersenneTwister(6), H=H, min_sd=0.05)
        end
        @test std(pf.draws[1][3, :]) > 0.03   # floored, not degenerate
    end

    @testset "interp_to_quarterly" begin
        annual = [2.0, 3.0, 1.0]
        q = interp_to_quarterly(annual, 12)
        @test length(q) == 12
        @test q[4] == 2.0                  # anchors at quarters 4k
        @test q[8] == 3.0
        @test q[12] == 1.0
        @test all(q[1:3] .== 2.0)          # left endpoint held
        @test q[6] ≈ 0.5 * (2.0 + 3.0)     # midpoint linear
        @test interp_to_quarterly(annual, 16)[13:16] == fill(1.0, 4)  # right hold
        @test_throws ArgumentError interp_to_quarterly(Float64[], 4)
        @test_throws ArgumentError interp_to_quarterly(annual, 0)
    end
end
