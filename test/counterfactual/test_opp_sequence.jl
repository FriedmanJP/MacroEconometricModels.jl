# CF-16 (#396): OPP sequences, time-consistency decomposition, λ sensitivity,
# robust weights.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

function _cf16_fc(H, v)
    MEM.PolicyForecast{Float64}([:u], [Vector{Float64}(v)], nothing, H, "d")
end

@testset "OPP sequences (CF-16)" begin
    rng = MersenneTwister(20260816)
    H, n_s = 6, 2
    Tx = randn(rng, H, n_s)
    ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx])
    loss = policy_loss([:u], H; lambda=[1.0])

    @testset "constant-ce identity" begin
        fcs = [_cf16_fc(H, randn(rng, H)) for _ in 1:4]
        sq = opp_sequence(fcs, ce, loss)
        @test size(sq.delta) == (n_s, 4)
        # constant D: pref == 0, delta_tc == delta
        @test all(abs.(sq.pref_part) .< 1e-12)
        @test sq.delta_tc ≈ sq.delta atol = 1e-12
        # exact three-part identity (the aging term is real under truncation —
        # deviation from the issue's two-term statement, see file header)
        for t in 2:4
            rev = sq.delta[:, t] - sq.delta[:, t-1]
            parts = sq.news_part[:, t] + sq.pref_part[:, t] + sq.aging_part[:, t]
            @test rev ≈ parts atol = 1e-10
        end
        @test sq.dates == ["t1", "t2", "t3", "t4"]
    end

    @testset "no-revision forecasts attribute zero to news" begin
        v1 = randn(rng, H)
        fc1 = _cf16_fc(H, v1)
        fc2 = MEM._shift_forecast(fc1)          # date-2 forecast = aged date-1
        fc3 = MEM._shift_forecast(fc2)
        sq = opp_sequence([fc1, fc2, fc3], ce, loss)
        @test all(abs.(sq.news_part[:, 2:3]) .< 1e-12)
        # revisions are pure aging
        for t in 2:3
            @test sq.delta[:, t] - sq.delta[:, t-1] ≈ sq.aging_part[:, t] atol = 1e-10
        end
    end

    @testset "time-varying operator: hand-checked decomposition" begin
        Tx2 = 1.5 .* Tx
        ce2 = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx2])
        v1, v2 = randn(rng, H), randn(rng, H)
        fc1, fc2 = _cf16_fc(H, v1), _cf16_fc(H, v2)
        sq = opp_sequence([fc1, fc2], ce, loss; ce_by_date=[ce, ce2])
        D1 = -inv(Tx' * Tx) * Tx'
        D2 = -inv(Tx2' * Tx2) * Tx2'
        vs = vcat(v1[2:end], 0.0)
        @test sq.news_part[:, 2] ≈ D2 * (v2 - vs) atol = 1e-10
        @test sq.pref_part[:, 2] ≈ (D2 - D1) * vs atol = 1e-10
        @test sq.aging_part[:, 2] ≈ D1 * (vs - v1) atol = 1e-10
        @test sq.delta_tc[:, 2] ≈ sq.delta[:, 2] - sq.pref_part[:, 2] atol = 1e-12
    end

    @testset "missing dates become NaN columns" begin
        fcs = Any[_cf16_fc(H, randn(rng, H)), missing, _cf16_fc(H, randn(rng, H))]
        sq = @test_logs (:warn, r"skipped") match_mode = :any begin
            opp_sequence(fcs, ce, loss; dates=["a", "b", "c"])
        end
        @test all(isnan, sq.delta[:, 2])
        @test all(isfinite, sq.delta[:, 1])
        @test all(isfinite, sq.delta[:, 3])
        @test sq.dates == ["a", "b", "c"]
    end

    @testset "bands per date" begin
        noises = 0.05 .* randn(MersenneTwister(2), 30)
        Dx = cat((Tx .* (1 + e) for e in noises)...; dims=3)
        ce_n = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx], Theta_x_draws=[Dx])
        fcs = [_cf16_fc(H, randn(rng, H)) for _ in 1:3]
        sq = opp_sequence(fcs, ce_n, loss; n_sim=200, rng=MersenneTwister(3))
        @test sq.bands !== nothing
        @test size(sq.bands[0.9]) == (n_s, 2, 3)
        for t in 1:3, k in 1:n_s
            @test sq.bands[0.9][k, 1, t] <= sq.delta[k, t] <= sq.bands[0.9][k, 2, t]
        end
        @test size(sq.reject[0.6]) == (n_s, 3)
    end

    @testset "λ sensitivity: monotone dependence" begin
        # 2 objectives, 1 shock: δ(λ) = −(a'u + λ b'π)/(a'a + λ b'b);
        # with u = a, π = −b: δ(λ) = −(|a|² − λ|b|²)/(|a|² + λ|b|²), increasing in λ.
        a = randn(rng, H)
        b = randn(rng, H)
        ce2 = PolicyCausalEffects(outcomes=[:u, :infl],
                                  Theta_x=[reshape(a, H, 1), reshape(b, H, 1)])
        fc = MEM.PolicyForecast{Float64}([:u, :infl], [copy(a), -copy(b)], nothing, H, "d")
        grid = [0.2, 0.5, 1.0, 2.0]
        seqs = opp_sensitivity([fc], ce2, H; lambda_grid=grid,
                               build_loss=l -> policy_loss([:u, :infl], H; lambda=[1.0, l]))
        deltas = [s.delta[1, 1] for s in seqs]
        @test all(diff(deltas) .> 0)
        for (l, d) in zip(grid, deltas)
            @test d ≈ -(dot(a, a) - l * dot(b, b)) / (dot(a, a) + l * dot(b, b)) atol = 1e-10
        end
    end

    @testset "robust_weights recovers the generating λ" begin
        lam_star = 0.7
        a = randn(rng, H)
        b = randn(rng, H)
        ce2 = PolicyCausalEffects(outcomes=[:u, :infl],
                                  Theta_x=[reshape(a, H, 1), reshape(b, H, 1)])
        # forecasts optimal under lam_star: R'W(λ*)·EY = 0
        R = vcat(a, b)
        W(l) = Diagonal(vcat(ones(H), l .* ones(H)))
        fcs = MEM.PolicyForecast{Float64}[]
        for _ in 1:6
            y = randn(rng, 2H)
            # project so that R'W(λ*)·y = 0 exactly (optimal under λ*)
            y -= W(lam_star) * R * ((R' * W(lam_star) * W(lam_star) * R) \
                                    (R' * W(lam_star) * y))
            push!(fcs, MEM.PolicyForecast{Float64}([:u, :infl],
                                                   [y[1:H], y[H+1:2H]], nothing, H, "d"))
        end
        grid = [0.3, 0.5, 0.7, 1.0, 1.5]
        out = robust_weights(l -> opp_sequence(fcs, ce2,
                                               policy_loss([:u, :infl], H; lambda=[1.0, l])),
                             grid)
        @test out.theta_hat == lam_star
        @test out.criterion_path[3] < minimum(out.criterion_path[[1, 2, 4, 5]])
    end

    @testset "validation" begin
        fcs = [_cf16_fc(H, randn(rng, H))]
        @test_throws ArgumentError opp_sequence(fcs, ce, loss; dates=["a", "b"])
        @test_throws ArgumentError opp_sequence(MEM.PolicyForecast{Float64}[], ce, loss)
        @test_throws ArgumentError opp_sensitivity(fcs, ce, H + 1;
                                                   lambda_grid=[1.0],
                                                   build_loss=l -> loss)
        @test_throws ArgumentError robust_weights(l -> fill(NaN, 2, 2), [1.0])
    end
end
