# CF-15 (#395): constrained OPP — ZLB floors, analytic QP check, projection
# fallback, multistart on nonconvex pledges, infeasibility.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

# 1 outcome, 1 instrument, 2 shock columns; forecast chosen so the
# unconstrained recommendation cuts the rate deep below zero.
function _cf15_setup(; H=6, rng=MersenneTwister(15))
    Tx = randn(rng, H, 2)
    Tz = hcat(ones(H), fill(0.5, H))          # deterministic rate loadings
    ce = PolicyCausalEffects(outcomes=[:u], instruments=[:rate],
                             Theta_x=[Tx], Theta_z=[Tz])
    v = -Tx * [-2.0, 1.0]                     # spanned: delta_u = [-2, 1]
    # => P_opp = 0.5 + (-2 + 0.5) = -1.0 < 0 at every horizon: the ZLB binds
    fc = MEM.PolicyForecast{Float64}([:u], [v], nothing, H, "t")
    loss = policy_loss([:u], H; lambda=[1.0])
    p0 = fill(0.5, H)
    return ce, fc, loss, p0, Tx, Tz
end

@testset "Constrained OPP (CF-15)" begin

    @testset "inactive constraints reproduce the unconstrained OPP" begin
        ce, fc, loss, p0, _, _ = _cf15_setup()
        r_u = opp(fc, ce, loss; instrument_path=[:rate => p0])
        out = constrained_opp(fc, ce, loss, [zlb_constraint(floor=-1e6)];
                              instrument_path=[:rate => p0])
        @test out.result.delta ≈ r_u.delta atol = 1e-8
        @test out.warm_start_feasible
        @test isempty(out.binding)
        @test out.method_used == :slsqp
        @test out.kkt_residual < 1e-6
    end

    @testset "binding ZLB" begin
        ce, fc, loss, p0, _, Tz = _cf15_setup()
        r_u = opp(fc, ce, loss; instrument_path=[:rate => p0])
        @test minimum(r_u.P_opp[1]) < 0.0      # the unconstrained rec violates the ZLB
        out = constrained_opp(fc, ce, loss, [zlb_constraint()];
                              instrument_path=[:rate => p0])
        r_c = out.result
        @test minimum(r_c.P_opp[1]) >= -1e-8
        @test r_c.loss_opp >= r_u.loss_opp - 1e-10   # constraint can only cost
        @test !out.warm_start_feasible
        @test !isempty(out.binding)
        @test out.kkt_residual < 1e-5
    end

    @testset "analytic equality-constrained QP" begin
        ce, fc, loss, p0, Tx, Tz = _cf15_setup()
        r_u = opp(fc, ce, loss; instrument_path=[:rate => p0])
        # single-horizon floor chosen to bind
        h0 = argmin(r_u.P_opp[1])
        fl = 0.0
        out = constrained_opp(fc, ce, loss,
                              [MEM.PathFloorConstraint{Float64}(:rate, fl, h0:h0)];
                              instrument_path=[:rate => p0])
        a = Tz[h0, :]
        Q = Tx' * Tx                          # R'WR with W = I
        ctil = fl - p0[h0]
        du = r_u.delta
        corr = (Q \ a) * ((a' * (Q \ a)) \ (a' * du - ctil))
        delta_ref = du - vec(corr)
        @test out.result.delta ≈ delta_ref atol = 1e-6
        @test occursin("h=$h0", out.binding[1])
    end

    @testset "projection fallback: labeled, warned, weakly worse" begin
        ce, fc, loss, p0, _, _ = _cf15_setup()
        out_s = constrained_opp(fc, ce, loss, [zlb_constraint()];
                                instrument_path=[:rate => p0])
        out_p = @test_logs (:warn, r"NOT the constrained optimum") match_mode = :any begin
            constrained_opp(fc, ce, loss, [zlb_constraint()];
                            instrument_path=[:rate => p0], method=:projection)
        end
        @test out_p.method_used == :projection
        @test out_p.result.loss_opp >= out_s.result.loss_opp - 1e-10
        # FunctionConstraints are rejected under :projection
        fcon = MEM.FunctionConstraint((d, p) -> [1.0], 1, "always ok")
        @test_throws ArgumentError constrained_opp(fc, ce, loss, [fcon];
                                                   instrument_path=[:rate => p0],
                                                   method=:projection)
    end

    @testset "multistart on a nonconvex constraint" begin
        ce, fc, loss, p0, _, _ = _cf15_setup()
        # feasible set: OUTSIDE the circle of radius 3 around delta_u — nonconvex
        r_u = opp(fc, ce, loss; instrument_path=[:rate => p0])
        du = r_u.delta
        ring = MEM.FunctionConstraint((d, p) -> [sum(abs2, d .- du) - 9.0], 1, "ring")
        # a start on the far side converges to a poor local optimum
        out_bad = constrained_opp(fc, ce, loss, [ring];
                                  instrument_path=[:rate => p0],
                                  delta0=du .+ [3.5, 0.0], multistart=1,
                                  rng=MersenneTwister(1))
        out_multi = constrained_opp(fc, ce, loss, [ring];
                                    instrument_path=[:rate => p0],
                                    delta0=du .+ [3.5, 0.0], multistart=8,
                                    rng=MersenneTwister(2))
        @test out_multi.result.loss_opp <= out_bad.result.loss_opp + 1e-10
        @test sum(abs2, out_multi.result.delta .- du) >= 9.0 - 1e-6  # on/outside the ring
    end

    @testset "infeasible floor errors with the constraint named" begin
        # zero instrument loadings: no delta can lift the path to the floor
        H = 6
        rng = MersenneTwister(16)
        Tx = randn(rng, H, 2)
        ce = PolicyCausalEffects(outcomes=[:u], instruments=[:rate],
                                 Theta_x=[Tx], Theta_z=[zeros(H, 2)])
        fc = MEM.PolicyForecast{Float64}([:u], [randn(rng, H)], nothing, H, "t")
        loss = policy_loss([:u], H; lambda=[1.0])
        err = try
            constrained_opp(fc, ce, loss, [zlb_constraint(floor=1.0)];
                            instrument_path=[:rate => fill(0.5, H)])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("rate", err.msg)
        @test occursin("infeasible", lowercase(err.msg))
    end

    @testset "inference with the constrained solve per draw" begin
        rng = MersenneTwister(155)
        H = 6
        Tx = randn(rng, H, 2)
        Tz = 0.5 .* randn(rng, H, 2) .+ 1.0
        noises = 0.05 .* randn(MersenneTwister(3), 25)
        Dx = cat((Tx .* (1 + e) for e in noises)...; dims=3)
        Dz = cat((Tz .* (1 + e) for e in noises)...; dims=3)
        ce = PolicyCausalEffects(outcomes=[:u], instruments=[:rate],
                                 Theta_x=[Tx], Theta_z=[Tz],
                                 Theta_x_draws=[Dx], Theta_z_draws=[Dz])
        v = -Tx * [2.0, -1.0]
        fc = MEM.PolicyForecast{Float64}([:u], [v], nothing, H, "t")
        loss = policy_loss([:u], H; lambda=[1.0])
        p0 = fill(0.5, H)
        out = MEM._suppress_warnings() do
            constrained_opp(fc, ce, loss, [zlb_constraint()];
                            instrument_path=[:rate => p0],
                            n_sim=60, rng=MersenneTwister(4))
        end
        @test out.result.delta_draws !== nothing
        @test size(out.result.delta_draws, 1) == 2
        @test haskey(out.result.bands, 0.9)
        @test out.result.n_failed + size(out.result.delta_draws, 2) == 60
    end

    @testset "validation" begin
        ce, fc, loss, p0, _, _ = _cf15_setup()
        @test_throws ArgumentError constrained_opp(fc, ce, loss, MEM.OPPConstraint[];
                                                   instrument_path=[:rate => p0])
        @test_throws ArgumentError constrained_opp(fc, ce, loss, [zlb_constraint()])
        @test_throws ArgumentError constrained_opp(fc, ce, loss,
                                                   [zlb_constraint(instrument=:qe)];
                                                   instrument_path=[:rate => p0])
        @test_throws ArgumentError constrained_opp(fc, ce, loss, [zlb_constraint()];
                                                   instrument_path=[:rate => p0],
                                                   method=:newton)
    end
end
