using Test, MacroEconometricModels, LinearAlgebra, SparseArrays

# ── hard-coded economies (no RNG) ────────────────────────────────────────────

"""Central finite-difference Hessian of bf_equilibrium dlogY w.r.t. dlogA."""
function _fd_hessian(net; ε=1e-4)
    n = net.n
    H = zeros(n, n)
    for j in 1:n, k in 1:n
        function y(dj, dk)
            dA = zeros(n)
            dA[j] += dj
            dA[k] += dk
            return bf_equilibrium(net; dlogA=dA, tol=1e-12).dlogY
        end
        H[j, k] = (y(ε, ε) - y(ε, -ε) - y(-ε, ε) + y(-ε, -ε)) / (4ε^2)
    end
    return H
end

"""Balanced 3-sector 2-factor IO table (hard-coded)."""
function _three_sector_io()
    Z = [50.0 80.0 40.0;
         60.0 100.0 50.0;
         30.0 70.0 40.0]
    Y = reshape([200.0, 300.0, 150.0], 3, 1)
    x = vec(sum(Z; dims=2)) .+ vec(Y)
    va_tot = x .- vec(sum(Z; dims=1))
    va = vcat(0.6 .* va_tot', 0.4 .* va_tot')
    return IOData(Z, Y, va; sectors=["A", "B", "C"], va_cats=["L", "K"])
end

@testset "B&F generalized Hessian (IO2-B2)" begin
    io = load_example(:wiot)

    # ── Oracle 3a: 2-sector single-factor FD ──────────────────────────────────
    @testset "Oracle 3a: 2-sec 1-factor FD" begin
        net = production_network(io; theta=2.0, sigma=0.9)
        H = baqaee_farhi(net).second_order
        H_fd = _fd_hessian(net)
        @test maximum(abs.(H .- H_fd)) < 1e-5
        @test maximum(abs.(H .- H')) < 1e-8
    end

    # ── Oracle 3b: 3-sector 2-factor FD ───────────────────────────────────────
    @testset "Oracle 3b: 3-sec 2-factor FD" begin
        io3 = _three_sector_io()
        net = production_network(io3; theta=0.5, sigma=0.9, factors=:va_cats)
        @test net.n == 3 && net.F == 2
        H = baqaee_farhi(net).second_order
        H_fd = _fd_hessian(net)
        @test maximum(abs.(H .- H_fd)) < 1e-5
        @test maximum(abs.(H .- H')) < 1e-8
    end

    # ── Oracle 3c: two-nest FD ────────────────────────────────────────────────
    @testset "Oracle 3c: two-nest FD" begin
        net = production_network(io; nests=:two, theta=0.5, epsilon=0.8,
                                 eta=1.0, sigma=0.9)
        H = baqaee_farhi(net).second_order
        H_fd = _fd_hessian(net)
        @test maximum(abs.(H .- H_fd)) < 1e-5
        @test size(H) == (2, 2)
    end

    # ── Oracle 5: freeze legacy baqaee_farhi(io; theta=2, sigma=0.9) ──────────
    @testset "Oracle 5: legacy baqaee_farhi freeze" begin
        bf = baqaee_farhi(io; theta=2.0, sigma=0.9)
        @test bf.first_order ≈ [0.4878048780487805, 0.975609756097561] atol=1e-12
        # Frozen Hessian from pre-IO2 scalar intermediate-only formula
        H_frozen = [0.2360607146203021  -0.1888485716962417;
                    -0.1888485716962417   0.1510788573569934]
        @test bf.second_order ≈ H_frozen atol=1e-12
        @test bf.second_order ≈ bf.second_order' atol=1e-12
        bf_cd = baqaee_farhi(io)
        @test all(abs.(bf_cd.second_order) .<= 1e-10)
    end

    # ── Oracle 6: uniform-θ consistency ───────────────────────────────────────
    @testset "Oracle 6: uniform-θ consistency" begin
        # Scalar θ vs constant vector θ on ProductionNetwork must match exactly
        net_s = production_network(io; theta=2.0, sigma=0.9)
        net_v = production_network(io; theta=[2.0, 2.0], sigma=0.9)
        Hs = baqaee_farhi(net_s).second_order
        Hv = baqaee_farhi(net_v).second_order
        @test Hs ≈ Hv atol=1e-12

        # First-order (Hulten) matches the legacy Domar path
        bf_leg = baqaee_farhi(io; theta=2.0, sigma=0.9)
        bf_net = baqaee_farhi(net_s)
        @test bf_net.first_order ≈ bf_leg.first_order atol=1e-12

        # Cobb-Douglas: both paths give zero Hessian
        @test all(abs.(baqaee_farhi(production_network(io)).second_order) .<= 1e-10)
        @test all(abs.(baqaee_farhi(io).second_order) .<= 1e-10)

        # Note: full standard-form H (CES over intermediates + factors) differs
        # from the legacy intermediate-only scalar formula by design (B&F 2019
        # §4 includes primary factors in Ω). Oracle 3 licenses the standard-form
        # numbers against the exact solver; oracle 5 freezes the legacy path.
        @test maximum(abs.(bf_net.second_order .- bf_leg.second_order)) > 1e-3
    end

    # ── Oracle 7: structural invariants ───────────────────────────────────────
    @testset "Oracle 7: structural invariants" begin
        net = production_network(io; theta=0.3, sigma=0.8, factors=:va_cats,
                                 nests=:two)
        bf = baqaee_farhi(net)
        H = bf.second_order
        @test H ≈ H' atol=1e-8
        @test bf.lambda[1] ≈ 1 atol=1e-12
        @test sum(bf.Lambda) ≈ 1 atol=1e-12
        @test length(bf.first_order) == net.n

        # Ψ factor columns sum to ~1 node-wise (CRS homogeneity)
        sys = MacroEconometricModels._bf_local_system(net)
        Ψ_F = sys.Ψ_F
        @test all(abs.(vec(sum(Ψ_F; dims=2)) .- 1) .< 1e-8)

        # Elasticities shapes
        e = bf.elasticities
        @test e !== nothing
        @test size(e.dlogw_dlogA) == (net.F, net.n)
        @test size(e.dlogp_dlogA) == (net.n, net.n)
        @test size(e.dlambda_dlogA) == (net.n, net.n)
        @test e.dlambda_dlogA ≈ H atol=1e-14

        # Single-factor: dlogw/dlogA ≡ 0
        net1 = production_network(io; theta=2.0)
        e1 = bf_elasticities(net1)
        @test all(abs.(e1.dlogw_dlogA) .< 1e-12)
    end

    # ── bf_quadratic without forming H ────────────────────────────────────────
    @testset "bf_quadratic" begin
        io3 = _three_sector_io()
        net = production_network(io3; theta=0.5, sigma=0.9, factors=:va_cats)
        H = baqaee_farhi(net).second_order
        v = [0.1, -0.05, 0.02]
        @test bf_quadratic(net, v) ≈ dot(v, H * v) atol=1e-12

        # Single-factor path
        net1 = production_network(io; theta=2.0, sigma=0.9)
        H1 = baqaee_farhi(net1).second_order
        v1 = [0.1, -0.2]
        @test bf_quadratic(net1, v1) ≈ dot(v1, H1 * v1) atol=1e-12
    end

    # ── hessian=:none / :auto ─────────────────────────────────────────────────
    @testset "hessian options" begin
        net = production_network(io; theta=2.0)
        bf0 = baqaee_farhi(net; hessian=:none, elasticities=false)
        @test isempty(bf0.second_order)
        @test bf0.elasticities === nothing
        bf1 = baqaee_farhi(net; hessian=:full)
        @test size(bf1.second_order) == (2, 2)
        bf2 = baqaee_farhi(net; hessian=:auto)
        @test size(bf2.second_order) == (2, 2)
    end

    # ── bf_shock_curve ────────────────────────────────────────────────────────
    @testset "bf_shock_curve" begin
        net = production_network(io; theta=0.5, sigma=0.9)
        sc = bf_shock_curve(net, 1; range=(-0.2, 0.2), points=5)
        @test sc isa BFShockCurve
        @test length(sc.shocks) == 5
        @test sc.sector_index == 1
        # At zero shock all three series are ~0
        iz = findfirst(iszero, sc.shocks)
        @test iz !== nothing
        @test abs(sc.exact[iz]) < 1e-10
        @test abs(sc.hulten[iz]) < 1e-14
        @test abs(sc.second_order[iz]) < 1e-14
        # Complements (θ<1): exact curve more concave than Hulten for large |shock|
        # At negative shock end, exact ≤ Hulten + tol (losses amplified)
        @test sc.exact[1] <= sc.hulten[1] + 1e-6

        sc2 = bf_shock_curve(net, "Manufacturing"; points=3)
        @test sc2.sector_index == 2
    end

    # ── show / report / plot ──────────────────────────────────────────────────
    @testset "show / report / plot" begin
        net = production_network(io; theta=2.0, sigma=0.9)
        bf = baqaee_farhi(net)
        e = bf.elasticities
        sc = bf_shock_curve(net, 1; points=5, range=(-0.1, 0.1))
        for obj in (bf, e, sc)
            s = sprint(show, obj)
            @test !isempty(s)
            mktemp() do _, ioh
                redirect_stdout(ioh) do
                    report(obj)
                end
            end
            p = plot_result(obj)
            @test p isa PlotOutput
        end
    end
end
