# CF-12 (#392): second-moment (Wold) counterfactuals.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

# Synthetic square Wold: 3 mapped variables (x1, x2, z), 3 innovations.
function _cf12_wold(H; rho=0.6, zero_instrument=true, ndraws=0)
    Theta = zeros(H, 3, 3)
    for j in 1:3, v in 1:3, h in 1:H
        zero_instrument && v == 3 && continue     # instrument row identically 0
        Theta[h, v, j] = (v == j ? 1.0 : 0.2) * rho^(h - 1)
    end
    draws = ndraws == 0 ? nothing :
            cat((Theta for _ in 1:ndraws)...; dims=4)
    MEM.WoldRepresentation{Float64}(Theta, Matrix{Float64}(I, 3, 3),
                                    ["x1", "x2", "z"], draws)
end

function _cf12_ce(H; ndraws=0)
    Tx1 = [-(0.6^t) * (0.8 + 0.1k) for t in 1:H, k in 1:2]
    Tx2 = [-(0.5^t) * (1.1 - 0.2k) for t in 1:H, k in 1:2]
    Tz = [(0.7^t) * (1.0 + 0.05k) for t in 1:H, k in 1:2]
    kw = ndraws == 0 ? (;) :
         (; Theta_x_draws=[cat((Tx1 for _ in 1:ndraws)...; dims=3),
                           cat((Tx2 for _ in 1:ndraws)...; dims=3)],
          Theta_z_draws=[cat((Tz for _ in 1:ndraws)...; dims=3)])
    PolicyCausalEffects(outcomes=[:x1, :x2], instruments=[:z],
                        Theta_x=[Tx1, Tx2], Theta_z=[Tz]; kw...)
end

const CF12_MAPS = (outcomes=[:x1 => 1, :x2 => "x2"], instruments=[:z => 3])

@testset "Second-moment counterfactuals (CF-12)" begin

    @testset "baseline identity vs Lyapunov (estimated VAR)" begin
        rng = MersenneTwister(12)
        A = [0.5 0.2; 0.1 0.4]
        Y = zeros(500, 2)
        for t in 2:500
            Y[t, :] = A * Y[t-1, :] + randn(rng, 2)
        end
        m = estimate_var(Y, 1)
        H = 200
        w = wold_representation(m; H=H)
        A1 = Matrix(m.B[2:end, :]')
        Sig_y = reshape((Matrix{Float64}(I, 4, 4) - kron(A1, A1)) \ vec(m.Sigma), 2, 2)
        ce = PolicyCausalEffects(outcomes=[:y1], instruments=[:y2],
                                 Theta_x=[zeros(H, 1)], Theta_z=[ones(H, 1)])
        # do-nothing rule on a zero-effect container: baseline preserved
        rule = custom_rule([zeros(H, H)], [zeros(H, H)];
                           outcomes=[:y1], instruments=[:y2], name="null")
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, rule;
                                   outcomes=[:y1 => 1], instruments=[:y2 => 2],
                                   warn_invertibility=false)
        end
        @test cm.Sigma_base[1, 1] ≈ Sig_y[1, 1] atol = 1e-6
        @test cm.Sigma_base[2, 2] ≈ Sig_y[2, 2] atol = 1e-6
        @test cm.Sigma_base ≈ cm.Sigma_cf atol = 1e-12   # null rule: b == 0, nu == 0
    end

    @testset "do-nothing rule leaves moments unchanged" begin
        H = 40
        w = _cf12_wold(H)                       # instrument row is identically zero
        ce = _cf12_ce(H)
        peg = rate_peg_rule(H; outcomes=[:x1, :x2], instruments=[:z])
        cm = counterfactual_moments(w, ce, peg;
                                    outcomes=CF12_MAPS.outcomes,
                                    instruments=CF12_MAPS.instruments,
                                    warn_invertibility=false)
        @test cm.Sigma_cf ≈ cm.Sigma_base atol = 1e-10
        @test cm.sd_cf ≈ cm.sd_base atol = 1e-10
    end

    @testset "rotation invariance" begin
        H = 40
        w = _cf12_wold(H; zero_instrument=false)
        ce = _cf12_ce(H)
        rng = MersenneTwister(3)
        Q = Matrix(qr(randn(rng, 3, 3)).Q)
        Theta_rot = similar(w.Theta)
        for h in 1:H
            Theta_rot[h, :, :] = w.Theta[h, :, :] * Q
        end
        w_rot = MEM.WoldRepresentation{Float64}(Theta_rot, w.Sigma_u, w.varnames, nothing)
        loss = policy_loss([:x1, :x2], H; lambda=[1.0, 0.5], beta=0.98)
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, loss; outcomes=CF12_MAPS.outcomes,
                                   instruments=CF12_MAPS.instruments,
                                   warn_invertibility=false)
        end
        cm_rot = MEM._suppress_warnings() do
            counterfactual_moments(w_rot, ce, loss; outcomes=CF12_MAPS.outcomes,
                                   instruments=CF12_MAPS.instruments,
                                   warn_invertibility=false)
        end
        @test cm.Sigma_cf ≈ cm_rot.Sigma_cf atol = 1e-10
        @test cm.Sigma_base ≈ cm_rot.Sigma_base atol = 1e-10
    end

    @testset "optimal policy weakly reduces the targeted variance" begin
        H = 40
        w = _cf12_wold(H; zero_instrument=false)
        ce = _cf12_ce(H)
        loss = policy_loss([:x1], H; lambda=[1.0])
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, loss; outcomes=CF12_MAPS.outcomes,
                                   instruments=CF12_MAPS.instruments,
                                   warn_invertibility=false)
        end
        @test cm.sd_cf[1] <= cm.sd_base[1] + 1e-12
        @test cm.policy_name == loss.name
    end

    @testset "tail-share warning on short H" begin
        H = 12
        w = _cf12_wold(H; rho=0.99)              # persistent: mass in the tail
        ce = _cf12_ce(H)
        peg = rate_peg_rule(H; outcomes=[:x1, :x2], instruments=[:z])
        @test_logs (:warn, r"tail_share") match_mode = :any begin
            counterfactual_moments(w, ce, peg; outcomes=CF12_MAPS.outcomes,
                                   instruments=CF12_MAPS.instruments,
                                   warn_invertibility=false)
        end
    end

    @testset "frequency bands" begin
        H = 30
        # white noise: Theta_1 = I, rest 0 -> band variance proportional to width
        Theta = zeros(H, 2, 2)
        Theta[1, :, :] = Matrix{Float64}(I, 2, 2)
        w = MEM.WoldRepresentation{Float64}(Theta, Matrix{Float64}(I, 2, 2),
                                            ["a", "b"], nothing)
        ce = PolicyCausalEffects(outcomes=[:a], instruments=[:b],
                                 Theta_x=[zeros(H, 1)], Theta_z=[ones(H, 1)])
        rule = custom_rule([zeros(H, H)], [zeros(H, H)];
                           outcomes=[:a], instruments=[:b], name="null")
        cm_bc = counterfactual_moments(w, ce, rule;
                                       outcomes=[:a => 1], instruments=[:b => 2],
                                       frequencies=:business_cycle,
                                       warn_invertibility=false)
        lo, hi = 2pi / 32, 2pi / 6
        @test cm_bc.Sigma_base[1, 1] ≈ (hi - lo) / pi atol = 1e-6
        @test cm_bc.freq_band !== nothing
        cm_all = counterfactual_moments(w, ce, rule;
                                        outcomes=[:a => 1], instruments=[:b => 2],
                                        frequencies=(0.0, Float64(pi)),
                                        warn_invertibility=false)
        @test cm_all.Sigma_base[1, 1] ≈ 1.0 atol = 1e-6   # Parseval: full band = time sum
        @test_throws ArgumentError counterfactual_moments(w, ce, rule;
                                                          outcomes=[:a => 1],
                                                          instruments=[:b => 2],
                                                          frequencies=(2.0, 1.0),
                                                          warn_invertibility=false)
    end

    @testset "draw propagation on sd_cf" begin
        H = 30
        w = _cf12_wold(H; zero_instrument=false, ndraws=3)
        ce = _cf12_ce(H; ndraws=3)
        loss = policy_loss([:x1, :x2], H; lambda=[1.0, 0.5])
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, loss; outcomes=CF12_MAPS.outcomes,
                                   instruments=CF12_MAPS.instruments,
                                   draw_source=:both, warn_invertibility=false)
        end
        @test size(cm.sd_cf_bands) == (3, 3)
        # replicated draws: bands collapse onto sd_cf
        for q in 1:3
            @test cm.sd_cf_bands[:, q] ≈ cm.sd_cf atol = 1e-10
        end
        # mismatched counts error under :both
        ce2 = _cf12_ce(H; ndraws=2)
        @test_throws ArgumentError counterfactual_moments(w, ce2, loss;
                                                          outcomes=CF12_MAPS.outcomes,
                                                          instruments=CF12_MAPS.instruments,
                                                          draw_source=:both,
                                                          warn_invertibility=false)
    end

    @testset "validation + invertibility reminder" begin
        H = 20
        w = _cf12_wold(H)
        ce = _cf12_ce(H)
        peg = rate_peg_rule(H; outcomes=[:x1, :x2], instruments=[:z])
        # wrong outcome coverage
        @test_throws ArgumentError counterfactual_moments(w, ce, peg;
                                                          outcomes=[:x1 => 1],
                                                          instruments=CF12_MAPS.instruments,
                                                          warn_invertibility=false)
        # Wold horizon too short
        ce_long = _cf12_ce(H + 10)
        @test_throws ArgumentError counterfactual_moments(w, ce_long, peg;
                                                          outcomes=CF12_MAPS.outcomes,
                                                          instruments=CF12_MAPS.instruments,
                                                          warn_invertibility=false)
        # the invertibility reminder fires when not suppressed
        @test_logs (:warn, r"invertibility") match_mode = :any begin
            counterfactual_moments(w, ce, peg; outcomes=CF12_MAPS.outcomes,
                                   instruments=CF12_MAPS.instruments)
        end
    end
end
