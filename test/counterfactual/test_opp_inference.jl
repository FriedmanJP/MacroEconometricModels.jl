# CF-14 (#394): OPP inference — two-source simulation, 60/75/90 bands,
# reversed polarity, failed-draw accounting.
using Test
using LinearAlgebra
using Random
using Statistics
using MacroEconometricModels

const MEM = MacroEconometricModels

# Container/forecast factories with controllable multiplicative noise.
function _cf14_ce(H, n_s, Tx0; noises=nothing)
    if noises === nothing
        return PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx0])
    end
    D = cat((Tx0 .* (1 + e) for e in noises)...; dims=3)
    return PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx0], Theta_x_draws=[D])
end
function _cf14_fc(H, v0; noises=nothing)
    d = noises === nothing ? nothing : [hcat((v0 .* (1 + e) for e in noises)...)]
    return MEM.PolicyForecast{Float64}([:u], [v0], d, H, "t")
end

@testset "OPP inference (CF-14)" begin
    rng = MersenneTwister(20260814)
    H, n_s = 6, 2
    Tx0 = randn(rng, H, n_s)
    v0 = randn(rng, H)
    loss = policy_loss([:u], H; lambda=[1.0])

    @testset "degenerate draws collapse bands to the point" begin
        zs = zeros(4)
        ce = _cf14_ce(H, n_s, Tx0; noises=zs)
        fc = _cf14_fc(H, v0; noises=zs)
        r = estimate_opp(fc, ce, loss; n_sim=50, rng=MersenneTwister(1))
        r_pt = opp(fc, ce, loss)
        @test r.delta ≈ r_pt.delta atol = 1e-12
        @test r.delta_plugin ≈ r_pt.delta atol = 1e-12
        for l in (0.60, 0.75, 0.90)
            @test r.bands[l][:, 1] ≈ r_pt.delta atol = 1e-12
            @test r.bands[l][:, 2] ≈ r_pt.delta atol = 1e-12
            @test r.reject[l] == (r_pt.delta .!= 0)
        end
        @test r.n_failed == 0
    end

    @testset "variance decomposition under independence" begin
        noises = 0.05 .* randn(MersenneTwister(2), 60)
        ceR = _cf14_ce(H, n_s, Tx0; noises=noises)
        fcY = _cf14_fc(H, v0; noises=0.05 .* randn(MersenneTwister(3), 60))
        ce0 = _cf14_ce(H, n_s, Tx0; noises=zeros(60))
        fc0 = _cf14_fc(H, v0; noises=zeros(60))
        kw = (; n_sim=4000)
        vR = var(estimate_opp(fc0, ceR, loss; kw..., rng=MersenneTwister(4)).delta_draws[1, :])
        vY = var(estimate_opp(fcY, ce0, loss; kw..., rng=MersenneTwister(5)).delta_draws[1, :])
        vB = var(estimate_opp(fcY, ceR, loss; kw..., rng=MersenneTwister(6)).delta_draws[1, :])
        @test isapprox(vB, vR + vY; rtol=0.35)   # small-noise linearity, MC tolerance
    end

    @testset "paired vs independent under perfect correlation" begin
        noises = 0.3 .* randn(MersenneTwister(7), 40)
        ce = _cf14_ce(H, n_s, Tx0; noises=noises)
        fc = _cf14_fc(H, v0; noises=noises)     # SAME noise: delta_d constant when paired
        rp = estimate_opp(fc, ce, loss; independent=false, rng=MersenneTwister(8))
        wp = rp.bands[0.90][1, 2] - rp.bands[0.90][1, 1]
        @test wp < 1e-10                        # paired: (1+e) cancels exactly
        ri = estimate_opp(fc, ce, loss; independent=true, n_sim=2000,
                          rng=MersenneTwister(9))
        wi = ri.bands[0.90][1, 2] - ri.bands[0.90][1, 1]
        @test wi > 1e-3                         # independent mixing does not cancel

        # mismatched counts error under pairing
        fc_bad = _cf14_fc(H, v0; noises=zeros(10))
        @test_throws ArgumentError estimate_opp(fc_bad, ce, loss; independent=false)
    end

    @testset "failed-draw accounting" begin
        D = cat(Tx0, zeros(H, n_s), Tx0; dims=3)   # draw 2 is rank-0
        ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx0], Theta_x_draws=[D])
        fc = _cf14_fc(H, v0)
        r = MEM._suppress_warnings() do
            estimate_opp(fc, ce, loss; n_sim=300, rng=MersenneTwister(10))
        end
        @test r.n_failed >= 1
        @test size(r.delta_draws, 2) + r.n_failed == 300
    end

    @testset "reject flags" begin
        d_true = [0.8, -0.5]
        fc_span = _cf14_fc(H, -Tx0 * d_true; noises=1e-6 .* randn(MersenneTwister(11), 50))
        ce_n = _cf14_ce(H, n_s, Tx0; noises=1e-6 .* randn(MersenneTwister(12), 50))
        r = estimate_opp(fc_span, ce_n, loss; n_sim=800, rng=MersenneTwister(13))
        for l in (0.60, 0.75, 0.90)
            @test all(r.reject[l])
        end
        @test r.delta ≈ d_true atol = 1e-3

        fc_zero = _cf14_fc(H, zeros(H); noises=0.1 .* randn(MersenneTwister(14), 50))
        r0 = estimate_opp(fc_zero, ce_n, loss; n_sim=800, rng=MersenneTwister(15))
        for l in (0.60, 0.75, 0.90)
            @test !any(r0.reject[l])
        end
    end

    @testset "median vs plugin under symmetric noise" begin
        ce = _cf14_ce(H, n_s, Tx0; noises=0.02 .* randn(MersenneTwister(16), 200))
        fc = _cf14_fc(H, v0)
        r = MEM._suppress_warnings() do
            estimate_opp(fc, ce, loss; n_sim=4000, rng=MersenneTwister(17))
        end
        @test r.delta ≈ r.delta_plugin rtol = 0.05
        @test r.delta != r.delta_plugin           # but not identical
    end

    @testset "one-sided sources + validation" begin
        ce_n = _cf14_ce(H, n_s, Tx0; noises=0.05 .* randn(MersenneTwister(18), 30))
        fc0 = _cf14_fc(H, v0)
        @test_logs (:info, r"forecast has no draws") match_mode = :any begin
            estimate_opp(fc0, ce_n, loss; n_sim=100, rng=MersenneTwister(19))
        end
        ce0 = _cf14_ce(H, n_s, Tx0)
        @test_throws ArgumentError estimate_opp(fc0, ce0, loss; rng=MersenneTwister(20))
        @test_throws ArgumentError estimate_opp(fc0, ce_n, loss; levels=(0.0, 0.9),
                                                rng=MersenneTwister(21))
        @test_throws ArgumentError estimate_opp(fc0, ce_n, loss; n_sim=1,
                                                rng=MersenneTwister(22))
    end
end
