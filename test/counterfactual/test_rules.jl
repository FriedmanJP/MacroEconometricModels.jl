# CF-02 (#382): rule templates and loss builders — exact banded structure,
# truncation edges, and the smoothing-penalty expansion identity.
using Test
using LinearAlgebra
using MacroEconometricModels

const MEM = MacroEconometricModels

@testset "Rule templates + loss builders (CF-02)" begin
    H = 8

    @testset "_lag_shift" begin
        L = MEM._lag_shift(Float64, 4)
        @test L == [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0]
        @test L * collect(1.0:4.0) == [0.0, 1.0, 2.0, 3.0]   # (L·z)_t = z_{t-1}
    end

    @testset "rate peg + rate target" begin
        r = rate_peg_rule(H)
        @test r isa PolicyRule{Float64}
        @test r.A_z[1] == Matrix{Float64}(I, H, H)
        @test all(A -> A == zeros(H, H), r.A_x)
        @test r.wedge == zeros(H)
        @test r.name == "rate peg"

        path = collect(0.1:0.1:0.8)
        rt = rate_target_rule(H, path)
        @test rt.A_z[1] == Matrix{Float64}(I, H, H)
        @test rt.wedge == path

        @test_throws ArgumentError rate_target_rule(H, ones(H + 1))
        @test_throws ArgumentError rate_peg_rule(H; instruments=[:rate, :qe])
    end

    @testset "strict targeting rules" begin
        ri = inflation_target_rule(H)
        @test ri.A_x[1] == Matrix{Float64}(I, H, H)   # :infl block
        @test ri.A_x[2] == zeros(H, H)                # :ygap block
        @test ri.A_z[1] == zeros(H, H)

        # symbol matching, not position: outcomes reordered
        ri2 = inflation_target_rule(H; outcomes=[:ygap, :infl])
        @test ri2.A_x[1] == zeros(H, H)
        @test ri2.A_x[2] == Matrix{Float64}(I, H, H)

        ry = output_gap_rule(H)
        @test ry.A_x[2] == Matrix{Float64}(I, H, H)
        @test ry.A_x[1] == zeros(H, H)

        @test_throws ArgumentError inflation_target_rule(H; pi_var=:cpi)
        @test_throws ArgumentError output_gap_rule(H; y_var=:gdp)
    end

    @testset "NGDP rule" begin
        r = ngdp_rule(H)
        @test r.A_x[1] == Matrix{Float64}(I, H, H)
        Ay = r.A_x[2]
        # row 1 is e1' (truncation edge: only the contemporaneous term)
        @test Ay[1, :] == [1.0; zeros(H - 1)]
        # row t >= 2: pi_t + y_t - y_{t-1}
        for t in 2:H
            expected = zeros(H)
            expected[t] = 1.0
            expected[t-1] = -1.0
            @test Ay[t, :] == expected
        end
        @test_throws ArgumentError ngdp_rule(H; pi_var=:infl, y_var=:infl)
    end

    @testset "Taylor rule" begin
        rho, phi_pi, phi_y, z_lag = 0.5, 1.5, 1.0, 0.3
        r = taylor_rule(H; rho=rho, phi_pi=phi_pi, phi_y=phi_y, z_lag=z_lag)
        Az = r.A_z[1]
        # I - rho*L: unit diagonal, -rho subdiagonal, row 1 truncated
        @test Az[1, :] == [1.0; zeros(H - 1)]
        for t in 2:H
            @test Az[t, t] == 1.0
            @test Az[t, t-1] == -rho
        end
        @test r.A_x[1] == -(1 - rho) * phi_pi * Matrix{Float64}(I, H, H)
        @test r.A_x[2] == -(1 - rho) * phi_y * Matrix{Float64}(I, H, H)
        @test r.wedge[1] == rho * z_lag
        @test all(r.wedge[2:end] .== 0.0)
        @test occursin("taylor", r.name)

        # rho = 0 reduces to the static rule
        r0 = taylor_rule(H; rho=0.0, phi_pi=phi_pi, phi_y=phi_y, z_lag=z_lag)
        @test r0.A_z[1] == Matrix{Float64}(I, H, H)
        @test r0.A_x[1] == -phi_pi * Matrix{Float64}(I, H, H)
        @test r0.wedge == zeros(H)

        # CMW parameterization reachable
        rc = taylor_rule(H; rho=0.85, phi_pi=2.0, phi_y=0.25)
        @test rc.A_z[1][2, 1] == -0.85
    end

    @testset "custom rule" begin
        r = custom_rule([ones(H, H)], [zeros(H, H)];
                        outcomes=[:infl], instruments=[:rate], name="mine")
        @test r isa PolicyRule{Float64}
        @test r.name == "mine"
        # CF-01 validation still applies through the escape hatch
        @test_throws ArgumentError custom_rule([ones(H, H + 1)], Matrix{Float64}[];
                                               outcomes=[:infl], instruments=Symbol[])
    end

    @testset "weight_matrix" begin
        lam, bet = 0.5, 0.97
        W = weight_matrix(H; lambda=lam, beta=bet)
        @test W isa Matrix{Float64}
        @test isdiag(W)
        @test [W[t, t] for t in 1:H] ≈ [lam * bet^(t - 1) for t in 1:H]
        W32 = weight_matrix(Float32, H)
        @test W32 isa Matrix{Float32}
        @test W32 == Matrix{Float32}(I, H, H)
    end

    @testset "policy_loss" begin
        l = policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25], beta=0.99)
        @test l isa PolicyLoss{Float64}
        @test l.W_x[1] == weight_matrix(H; lambda=1.0, beta=0.99)
        @test l.W_x[2] == weight_matrix(H; lambda=0.25, beta=0.99)
        @test l.W_z === nothing
        @test l.lambda == [1.0, 0.25]
        @test_throws ArgumentError policy_loss([:infl, :ygap], H; lambda=[1.0])

        # attach an instrument penalty
        sp = smoothing_penalty(H; lambda=0.1)
        l2 = policy_loss([:infl], H; lambda=[1.0], instruments=[:rate], W_z=[sp.W_z])
        @test l2.W_z[1] == sp.W_z
    end

    @testset "AIT loss" begin
        Pi_bar = MEM._ait_averaging_matrix(Float64, H; delta=0.1, K=3)
        # all rows sum to 1 (truncated windows renormalize)
        @test all(abs.(sum(Pi_bar, dims=2) .- 1.0) .< 1e-14)
        # row 1 puts all weight on entry 1; row t has min(K+1, t) nonzeros
        @test Pi_bar[1, 1] == 1.0
        for t in 1:H
            @test count(!iszero, Pi_bar[t, :]) == min(3 + 1, t)
        end
        # window weights decay with the lag
        @test Pi_bar[5, 5] > Pi_bar[5, 4] > Pi_bar[5, 3] > Pi_bar[5, 2]

        l = ait_loss(H)
        @test l isa PolicyLoss{Float64}
        @test l.outcomes == [:infl, :ygap]
        @test l.beta ≈ 1 / 1.01
        @test l.W_x[2] == weight_matrix(H; lambda=1.0, beta=1 / 1.01)
        @test occursin("AIT", l.name)

        # delta -> huge: averaging collapses onto current inflation, W_pi -> (l_avg+l_t)*W
        lbig = ait_loss(H; delta=1e6)
        W = weight_matrix(H; beta=1 / 1.01)
        @test lbig.W_x[1] ≈ (0.6 + 0.4) * W atol = 1e-12
    end

    @testset "smoothing_penalty expansion identity" begin
        lam, bet, z_lag = 0.7, 0.95, 0.4
        sp = smoothing_penalty(H; lambda=lam, beta=bet, z_lag=z_lag)
        @test sp.wedge_term ≈ [lam * z_lag; zeros(H - 1)] atol = 1e-14

        # z'W_z z - 2 wedge_term'z + lam*z_lag^2 == lam * sum(beta^(t-1) (z_t - z_{t-1})^2)
        for z in (collect(1.0:H), [(-1.0)^t * sqrt(t) for t in 1:H], zeros(H))
            direct = lam * sum(bet^(t - 1) * (z[t] - (t == 1 ? z_lag : z[t-1]))^2 for t in 1:H)
            quadform = z' * sp.W_z * z - 2 * sp.wedge_term' * z + lam * z_lag^2
            @test quadform ≈ direct atol = 1e-12
        end

        # no initial condition: wedge_term vanishes
        sp0 = smoothing_penalty(H; lambda=lam, beta=bet)
        @test sp0.wedge_term == zeros(H)
    end
end
