using Test, MacroEconometricModels, LinearAlgebra, SparseArrays

# ── hard-coded economies (no RNG) ────────────────────────────────────────────

"""Horizontal 2-sector economy: no intermediates, labor only, equal size."""
function _horizontal_io()
    # x = [1, 1], Z = 0, Y = [1, 1], va = [1, 1], GDP = 2, λ = [0.5, 0.5]
    Z = zeros(2, 2)
    Y = reshape([1.0, 1.0], 2, 1)
    va = reshape([1.0, 1.0], 1, 2)
    return IOData(Z, Y, va; sectors=["A", "B"])
end

"""Two-sector IO with intermediates (wiot-like but explicit)."""
function _two_sector_io()
    # Miller-Blair style small table
    Z = [150.0 500.0; 200.0 100.0]
    Y = reshape([350.0, 1700.0], 2, 1)   # so x = [1000, 2000]
    va = reshape([650.0, 1400.0], 1, 2)
    return IOData(Z, Y, va; sectors=["Ag", "Manuf"])
end

@testset "B&F 2020 wedges" begin
    io = load_example(:wiot)

    # ── μ ≡ 1 reproduces efficient solver ────────────────────────────────────
    @testset "μ ≡ 1 matches efficient path" begin
        net0 = production_network(io; theta=0.5, sigma=0.9)
        net1 = production_network(io; theta=0.5, sigma=0.9, mu=1.0)
        netv = production_network(io; theta=0.5, sigma=0.9, mu=[1.0, 1.0])
        @test net0.mu == ones(net0.M)
        @test net1.lambda ≈ net0.lambda atol=0
        @test net1.lambda_rev ≈ net0.lambda atol=0
        @test netv.lambda_rev ≈ net0.lambda atol=0
        @test cost_based_domar(net0) ≈ revenue_based_domar(net0) atol=1e-14

        dA = [0.1, -0.05]
        eq0 = bf_equilibrium(net0; dlogA=dA)
        eq1 = bf_equilibrium(net1; dlogA=dA)
        eqv = bf_equilibrium(netv; dlogA=dA)
        @test eq0.converged && eq1.converged && eqv.converged
        @test eq1.dlogY ≈ eq0.dlogY atol=0
        @test eqv.dlogY ≈ eq0.dlogY atol=1e-14
        @test eq1.p ≈ eq0.p atol=1e-14
        @test eq1.w ≈ eq0.w atol=1e-14
        @test eq1.lambda ≈ eq0.lambda atol=1e-14
        @test eq1.lambda_cost ≈ eq0.lambda atol=1e-14
        @test eq0.profit_share ≈ 0 atol=1e-12
        @test abs(eq0.allocative) < 1e-8   # first-order AE ≈ 0 when μ≡1
    end

    # ── cost vs revenue Domar with base markups ──────────────────────────────
    @testset "cost vs revenue Domar with μ > 1" begin
        μ = [1.2, 1.5]
        net = production_network(io; mu=μ)
        @test all(net.mu[1:2] .≈ μ)          # single-nest: M = n
        λ̃ = cost_based_domar(net)
        λ  = revenue_based_domar(net)
        @test λ̃ ≈ [net.io.x[i] / sum(net.io.va) for i in 1:2] atol=1e-12
        # With μ > 1, revenue Domar on producers drops (profits leak from cost rows)
        # and factor revenue share < 1
        Λr = net.lambda_rev[net.M+2:end]
        Λc = net.lambda[net.M+2:end]
        @test sum(Λc) ≈ 1 atol=1e-12
        @test sum(Λr) < 1 - 1e-8
        # profit share of GDP = 1 − Σ Λ_rev
        profit = 1 - sum(Λr)
        @test profit > 0
        # Ω̃ rows still sum to 1; revenue producer rows sum to 1/μ
        Ωc = Matrix(net.Omega)
        Ωr = Matrix(MacroEconometricModels._bf_revenue_omega(
            net.Omega, net.mu, net.M, net.F))
        @test all(abs.(sum(Ωc[1:1+net.M, :]; dims=2) .- 1) .< 1e-12)
        @test abs(sum(Ωr[2, :]) - 1 / μ[1]) < 1e-12
        @test abs(sum(Ωr[3, :]) - 1 / μ[2]) < 1e-12
        # zero-shock equilibrium recovers base
        eq = bf_equilibrium(net)
        @test eq.converged
        @test eq.dlogY ≈ 0 atol=1e-9
        @test all(abs.(eq.p .- 1) .< 1e-8)
        @test eq.profit_share ≈ profit atol=1e-8
    end

    # ── Theorem 1 small-shock FD ─────────────────────────────────────────────
    @testset "Theorem 1: small-shock FD vs formula" begin
        μ = [1.3, 1.1]
        net = production_network(io; theta=0.5, sigma=1.5, mu=μ)
        λ̃ = cost_based_domar(net)
        ε = 1e-5

        # productivity shock to sector 1
        eq_p = bf_equilibrium(net; dlogA=[ε, 0.0])
        eq_m = bf_equilibrium(net; dlogA=[-ε, 0.0])
        fd_A = (eq_p.dlogY - eq_m.dlogY) / (2ε)
        # Theorem 1: dlogY/dlogA_k = λ̃_k − Σ_f Λ̃_f (dlogΛ_f / dlogA_k)
        # Use the equilibrium's technology + allocative first-order split
        thm1_A = (eq_p.technology - eq_m.technology +
                  eq_p.allocative - eq_m.allocative) / (2ε)
        @test fd_A ≈ thm1_A atol=1e-4
        # pure technology piece equals λ̃_1
        @test (eq_p.technology - eq_m.technology) / (2ε) ≈ λ̃[1] atol=1e-10

        # markup shock to sector 2
        eqμ_p = bf_equilibrium(net; dlogmu=[0.0, ε])
        eqμ_m = bf_equilibrium(net; dlogmu=[0.0, -ε])
        fd_μ = (eqμ_p.dlogY - eqμ_m.dlogY) / (2ε)
        thm1_μ = (eqμ_p.technology - eqμ_m.technology +
                  eqμ_p.allocative - eqμ_m.allocative) / (2ε)
        @test fd_μ ≈ thm1_μ atol=1e-4
        # pure technology is zero for pure markup shocks
        @test eqμ_p.technology ≈ 0 atol=1e-14
    end

    # ── Horizontal economy closed form (B&F 2020 §2.4) ───────────────────────
    @testset "horizontal economy markup example" begin
        io_h = _horizontal_io()
        μ = [1.2, 1.5]
        σ = 2.0
        net = production_network(io_h; sigma=σ, theta=1.0, mu=μ)
        λ = revenue_based_domar(net)
        λ̃ = cost_based_domar(net)
        # horizontal: no intermediates ⇒ λ̃ = λ on goods
        @test λ̃ ≈ λ atol=1e-12
        @test λ ≈ [0.5, 0.5] atol=1e-12

        dY_dA_cf, dY_dμ_cf = MacroEconometricModels._bf_horizontal_elasticities(λ, μ, σ)

        ε = 1e-5
        for k in 1:2
            dAp = zeros(2); dAp[k] = ε
            dAm = zeros(2); dAm[k] = -ε
            yp = bf_equilibrium(net; dlogA=dAp).dlogY
            ym = bf_equilibrium(net; dlogA=dAm).dlogY
            @test (yp - ym) / (2ε) ≈ dY_dA_cf[k] atol=1e-4

            dμp = zeros(2); dμp[k] = ε
            dμm = zeros(2); dμm[k] = -ε
            yp = bf_equilibrium(net; dlogmu=dμp).dlogY
            ym = bf_equilibrium(net; dlogmu=dμm).dlogY
            @test (yp - ym) / (2ε) ≈ dY_dμ_cf[k] atol=1e-4
        end

        # equal markups ⇒ AE terms vanish for A-shocks (gap = 0)
        net_eq = production_network(io_h; sigma=σ, mu=[1.4, 1.4])
        λe = revenue_based_domar(net_eq)
        dY_dA_e, dY_dμ_e = MacroEconometricModels._bf_horizontal_elasticities(
            λe, [1.4, 1.4], σ)
        @test dY_dA_e ≈ λe atol=1e-12          # recovers Hulten
        @test all(abs.(dY_dμ_e) .< 1e-12)      # equal μ ⇒ no AE from μ shocks either
    end

    # ── two-nest wedges ──────────────────────────────────────────────────────
    @testset "two-nest with wedges" begin
        net = production_network(io; nests=:two, theta=0.5, epsilon=0.8,
                                 eta=1.0, sigma=0.9, mu=[1.25, 1.1])
        @test net.M == 6
        # only outer nodes carry markup
        @test net.mu[1] ≈ 1.25
        @test net.mu[2] ≈ 1.1
        @test all(net.mu[3:6] .≈ 1.0)
        eq = bf_equilibrium(net; dlogA=[0.05, -0.05])
        @test eq.converged
        @test eq.profit_share > 0
        w = bf_wedge_decomp(net; dlogA=[0.05, -0.05])
        @test w isa BFWedgeDecomp
        @test w.dlogY ≈ eq.dlogY atol=1e-14
        @test w.technology ≈ eq.technology atol=1e-14
        @test w.allocative ≈ eq.allocative atol=1e-14
    end

    # ── report / show / plot / refs smoke ────────────────────────────────────
    @testset "display & plot smoke" begin
        net = production_network(io; mu=[1.2, 1.0])
        eq = bf_equilibrium(net; dlogA=[0.1, 0.0], dlogmu=[0.0, 0.05])
        w = bf_wedge_decomp(net; dlogA=[0.1, 0.0])
        @test occursin("wedge", sprint(show, net)) || occursin("μ", sprint(show, net))
        @test occursin("allocative", sprint(show, eq))
        @test occursin("Theorem 1", sprint(show, w))
        @test occursin("Baqaee", sprint(refs, w))
        p1 = plot_result(net)
        p2 = plot_result(eq)
        p3 = plot_result(w)
        @test p1 isa PlotOutput
        @test p2 isa PlotOutput
        @test p3 isa PlotOutput
        report(w)
    end

    # ── Cobb-Douglas + wedges: productivity shocks have no AE (B&F §4) ───────
    @testset "Cobb-Douglas: productivity shocks leave allocation fixed" begin
        # Under CD, allocation matrix X is independent of A ⇒ dlogΛ from dlogA
        # is only the mechanical profit/sales revaluation… For μ fixed and CD,
        # B&F note that productivity shocks do not reallocate resources.
        # With single factor and CD, exact dlogY = technology + allocative (FO)
        # and for pure A-shocks the AE term is second-order small.
        net = production_network(io; theta=1.0, sigma=1.0, mu=[1.3, 1.1])
        ε = 1e-5
        eq_p = bf_equilibrium(net; dlogA=[ε, 0.0])
        eq_m = bf_equilibrium(net; dlogA=[-ε, 0.0])
        # FO allocative change per unit dlogA should be near 0 under CD
        d_alloc = (eq_p.allocative - eq_m.allocative) / (2ε)
        @test abs(d_alloc) < 1e-3
        λ̃ = cost_based_domar(net)
        fd = (eq_p.dlogY - eq_m.dlogY) / (2ε)
        @test fd ≈ λ̃[1] atol=1e-3
    end

    # ── mu validation ────────────────────────────────────────────────────────
    @testset "mu validation" begin
        @test_throws ArgumentError production_network(io; mu=[0.5, 1.0])
        @test_throws ArgumentError production_network(io; mu=[1.0])  # length
    end
end
