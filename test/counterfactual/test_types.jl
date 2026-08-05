# CF-01 (#381): counterfactual core types — construction, validation, accessors.
using Test
using MacroEconometricModels

const MEM = MacroEconometricModels

@testset "Counterfactual core types (CF-01)" begin
    H = 12
    n_s = 2
    ndr = 5
    Tx = [ones(H, n_s), 2 .* ones(H, n_s)]   # 2 outcomes
    Tz = [3 .* ones(H, n_s)]                 # 1 instrument
    Dx = [ones(H, n_s, ndr), ones(H, n_s, ndr)]
    Dz = [ones(H, n_s, ndr)]

    @testset "construction happy path" begin
        # thin, no draws
        ce = PolicyCausalEffects(outcomes=[:pi, :y], instruments=[:i],
                                 Theta_x=Tx, Theta_z=Tz, source=:var)
        @test ce isa PolicyCausalEffects{Float64}
        @test ce isa MEM.AbstractCounterfactual
        @test ce.H == H
        @test ce.shock_labels == ["shock 1", "shock 2"]
        @test ce.Theta_x_draws === nothing

        # thin, with draws
        ce_d = PolicyCausalEffects(outcomes=[:pi, :y], instruments=[:i],
                                   Theta_x=Tx, Theta_z=Tz,
                                   Theta_x_draws=Dx, Theta_z_draws=Dz, source=:bvar)
        @test MEM.n_draws(ce_d) == ndr
        @test ce_d.Theta_x_draws[1] isa Array{Float64,3}

        # square (n_s == H), no instruments (pure OPP shape)
        ce_sq = PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(H, H)], source=:dsge)
        @test is_square(ce_sq)
        @test isempty(ce_sq.instruments)
        @test isempty(ce_sq.Theta_z)

        # AbstractMatrix (views) + Int inputs convert to Matrix{Float64}
        vTx = [view(ones(Int, H + 2, n_s), 1:H, 1:n_s), ones(Int, H, n_s)]
        ce_v = PolicyCausalEffects(outcomes=[:pi, :y], Theta_x=vTx)
        @test ce_v isa PolicyCausalEffects{Float64}
        @test ce_v.Theta_x[1] isa Matrix{Float64}

        # Float32 inputs stay Float32
        ce32 = PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(Float32, H, n_s)])
        @test ce32 isa PolicyCausalEffects{Float32}

        # custom shock labels
        ce_l = PolicyCausalEffects(outcomes=[:pi, :y], Theta_x=Tx,
                                   shock_labels=["mp surprise", "ff4 futures"])
        @test ce_l.shock_labels == ["mp surprise", "ff4 futures"]
    end

    @testset "validation errors" begin
        # empty Theta_x
        @test_throws ArgumentError PolicyCausalEffects(outcomes=Symbol[],
                                                       Theta_x=Matrix{Float64}[])
        # Theta_x count != outcomes count
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi], Theta_x=Tx)
        # instruments declared but Theta_z missing
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y],
                                                       instruments=[:i], Theta_x=Tx)
        # ragged Theta_x dims
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y],
                                                       Theta_x=[ones(H, n_s), ones(H + 1, n_s)])
        # Theta_z dims mismatch
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y], instruments=[:i],
                                                       Theta_x=Tx, Theta_z=[ones(H, n_s + 1)])
        # shock_labels wrong length
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y], Theta_x=Tx,
                                                       shock_labels=["only one"])
        # draws vector wrong length
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y], Theta_x=Tx,
                                                       Theta_x_draws=[ones(H, n_s, ndr)])
        # draws entry wrong (H, n_s)
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y], Theta_x=Tx,
                                                       Theta_x_draws=[ones(H, n_s + 1, ndr), ones(H, n_s, ndr)])
        # inconsistent n_draws across x and z draws
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y], instruments=[:i],
                                                       Theta_x=Tx, Theta_z=Tz,
                                                       Theta_x_draws=Dx,
                                                       Theta_z_draws=[ones(H, n_s, ndr + 1)])
        # unknown source tag
        @test_throws ArgumentError PolicyCausalEffects(outcomes=[:pi, :y], Theta_x=Tx,
                                                       source=:carrier_pigeon)
    end

    @testset "PolicyRule construction + validation" begin
        r = PolicyRule(outcomes=[:pi, :y], instruments=[:i],
                       A_x=[ones(H, H), zeros(H, H)], A_z=[ones(H, H)])
        @test r isa PolicyRule{Float64}
        @test r.wedge == zeros(H)
        @test r.name == "custom"

        # instrument-only rule (e.g. a peg) with explicit wedge
        r2 = PolicyRule(outcomes=Symbol[], instruments=[:i],
                        A_x=Matrix{Float64}[], A_z=[ones(H, H)],
                        wedge=fill(0.5, H), name="peg")
        @test r2.name == "peg"
        @test r2.wedge == fill(0.5, H)

        # non-square A_x
        @test_throws ArgumentError PolicyRule(outcomes=[:pi], instruments=Symbol[],
                                              A_x=[ones(H, H + 1)], A_z=Matrix{Float64}[])
        # wedge wrong length
        @test_throws ArgumentError PolicyRule(outcomes=[:pi], instruments=Symbol[],
                                              A_x=[ones(H, H)], A_z=Matrix{Float64}[],
                                              wedge=ones(H + 1))
        # A_x count != outcomes count
        @test_throws ArgumentError PolicyRule(outcomes=[:pi], instruments=Symbol[],
                                              A_x=Matrix{Float64}[], A_z=Matrix{Float64}[])
        # nothing to derive H from
        @test_throws ArgumentError PolicyRule(outcomes=Symbol[], instruments=Symbol[],
                                              A_x=Matrix{Float64}[], A_z=Matrix{Float64}[])
    end

    @testset "PolicyLoss construction + validation" begin
        l = PolicyLoss(outcomes=[:pi, :y], instruments=[:i],
                       W_x=[ones(H, H), ones(H, H)], W_z=[ones(H, H)],
                       lambda=[1.0, 0.5], beta=0.99, name="dual mandate")
        @test l isa PolicyLoss{Float64}
        @test l.beta == 0.99
        @test l.lambda == [1.0, 0.5]

        # defaults: no W_z, unit lambda, beta = 1
        l2 = PolicyLoss(outcomes=[:pi], W_x=[ones(H, H)])
        @test l2.W_z === nothing
        @test l2.lambda == [1.0]
        @test l2.beta == 1.0

        # PSD-singular W is accepted by design (AIT-style weights)
        l3 = PolicyLoss(outcomes=[:pi], W_x=[zeros(H, H)])
        @test l3 isa PolicyLoss{Float64}

        # W_x count != outcomes count
        @test_throws ArgumentError PolicyLoss(outcomes=[:pi, :y], W_x=[ones(H, H)])
        # non-square W_x
        @test_throws ArgumentError PolicyLoss(outcomes=[:pi], W_x=[ones(H, H + 1)])
        # W_z count != instruments count
        @test_throws ArgumentError PolicyLoss(outcomes=[:pi], instruments=[:i],
                                              W_x=[ones(H, H)],
                                              W_z=[ones(H, H), ones(H, H)])
        # lambda wrong length
        @test_throws ArgumentError PolicyLoss(outcomes=[:pi], W_x=[ones(H, H)],
                                              lambda=[1.0, 2.0])
        # beta out of range
        @test_throws ArgumentError PolicyLoss(outcomes=[:pi], W_x=[ones(H, H)], beta=0.0)
        @test_throws ArgumentError PolicyLoss(outcomes=[:pi], W_x=[ones(H, H)], beta=1.5)
        # no outcomes
        @test_throws ArgumentError PolicyLoss(outcomes=Symbol[], W_x=Matrix{Float64}[])
    end

    @testset "accessors" begin
        ce_thin = PolicyCausalEffects(outcomes=[:pi, :y], instruments=[:i],
                                      Theta_x=Tx, Theta_z=Tz,
                                      Theta_x_draws=Dx, Theta_z_draws=Dz, source=:bvar)
        ce_square = PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(H, H)], source=:dsge)
        @test !is_square(ce_thin)
        @test is_square(ce_square)
        @test MEM.n_shocks(ce_thin) == n_s
        @test MEM.n_shocks(ce_square) == H
        @test MEM.n_draws(ce_thin) == ndr
        @test MEM.n_draws(ce_square) == 0
        @test eltype(ce_thin) == Float64
        @test eltype(PolicyCausalEffects{Float32}) == Float32
        @test eltype(PolicyRule{Float64}) == Float64
        @test eltype(PolicyLoss{Float64}) == Float64
    end

    @testset "show smoke test" begin
        ce = PolicyCausalEffects(outcomes=[:pi, :y], instruments=[:i],
                                 Theta_x=Tx, Theta_z=Tz,
                                 Theta_x_draws=Dx, Theta_z_draws=Dz, source=:bvar)
        s = sprint(show, ce)
        @test occursin("PolicyCausalEffects{Float64}", s)
        @test occursin("2 outcomes", s)
        @test occursin("1 instrument", s)
        @test occursin("H=$H", s)
        @test occursin("n_s=$n_s (thin)", s)
        @test occursin("$ndr draws", s)
        @test occursin("source=:bvar", s)

        sq = sprint(show, PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(H, H)]))
        @test occursin("(square)", sq)
        @test occursin("0 draws", sq)

        r = PolicyRule(outcomes=[:pi], instruments=[:i],
                       A_x=[ones(H, H)], A_z=[ones(H, H)], name="taylor")
        @test occursin("taylor", sprint(show, r))
        @test occursin("H=$H", sprint(show, r))

        l = PolicyLoss(outcomes=[:pi], W_x=[ones(H, H)], name="strict inflation")
        @test occursin("strict inflation", sprint(show, l))
    end
end
