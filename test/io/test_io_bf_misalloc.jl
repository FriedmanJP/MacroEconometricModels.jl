using Test, MacroEconometricModels, LinearAlgebra

# ── hard-coded economies (no RNG) ────────────────────────────────────────────

"""Horizontal 2-sector economy: no intermediates, labor only, equal size."""
function _horizontal_io()
    # x = [1, 1], Z = 0, Y = [1, 1], va = [1, 1], GDP = 2, λ = [0.5, 0.5]
    Z = zeros(2, 2)
    Y = reshape([1.0, 1.0], 2, 1)
    va = reshape([1.0, 1.0], 1, 2)
    return IOData(Z, Y, va; sectors=["A", "B"])
end

"""Balanced 3-sector 2-factor IO table (hard-coded, non-proportional factor intensities).

A is labor-heavy, C is capital-heavy; column VA still sums to residual value added.
Proportional L/K shares make Prop 5 eq. (17) line 2 vanish, so M4b would not license
the multi-factor block.
"""
function _three_sector_io()
    Z = [50.0 80.0 40.0;
         60.0 100.0 50.0;
         30.0 70.0 40.0]
    Y = reshape([200.0, 300.0, 150.0], 3, 1)
    # va_tot = [230, 260, 160]; L/K shares (0.8/0.2, 0.5/0.5, 0.3/0.7)
    va = [184.0 130.0  48.0;
           46.0 130.0 112.0]
    return IOData(Z, Y, va; sectors=["A", "B", "C"], va_cats=["L", "K"])
end

"""Central 4-point Hessian of `bf_equilibrium` `dlogY` w.r.t. `dlogmu`."""
function _fd_hessian_mu(net; ε=1e-4)
    n = net.n
    H = zeros(n, n)
    for j in 1:n, k in 1:n
        function y(dj, dk)
            dμ = zeros(n)
            dμ[j] += dj
            dμ[k] += dk
            return bf_equilibrium(net; dlogmu=dμ, tol=1e-14).dlogY
        end
        H[j, k] = (y(ε, ε) - y(ε, -ε) - y(-ε, ε) + y(-ε, -ε)) / (4ε^2)
    end
    return H
end

@testset "B&F 2020 Prop 5 misallocation" begin
    @testset "Prop 5 horizontal Harberger" begin
        io = _horizontal_io()
        μ = [1.2, 1.5]
        σ = 2.0
        net = production_network(io; sigma=σ, theta=1.0, mu=μ)
        m = bf_misallocation(net; point=:efficient)
        v = log.(μ)
        λ = [0.5, 0.5]
        # L ≈ (1/2) σ Var_λ(v)   (B&F 2020 eq. 19)
        mean_v = dot(λ, v)
        L_hk = 0.5 * σ * sum(λ .* (v .- mean_v).^2)
        @test m.second_order ≈ L_hk atol=1e-12
        @test m.first_order ≈ 0 atol=1e-12          # envelope at μ=1
        @test size(m.H_mu) == (2, 2)
        @test m.second_order ≈ -0.5 * dot(v, m.H_mu * v) atol=1e-12
    end

    @testset "μ≡1 ⇒ L=0 and H_μ well-defined" begin
        net = production_network(load_example(:wiot); theta=0.5, sigma=0.9)
        m = bf_misallocation(net)
        @test m.distance ≈ 0 atol=1e-12
        @test m.second_order ≈ 0 atol=1e-12
        @test all(iszero, m.delta_logmu)
    end

    @testset "Leontief (all θ=0) ⇒ H_μ = 0" begin
        io = load_example(:wiot)
        net = production_network(io; theta=0.0, sigma=0.0, mu=[1.3, 1.1])
        m = bf_misallocation(net; point=:efficient)
        @test maximum(abs, m.H_mu) < 1e-12
        @test m.second_order ≈ 0 atol=1e-12
    end

    @testset "Oracle: FD Hessian in log μ vs analytic H_μ (2sec 1fac)" begin
        net = production_network(load_example(:wiot); theta=0.5, sigma=0.9, mu=1.0)
        H = bf_misallocation(net; point=:efficient).H_mu
        H_fd = _fd_hessian_mu(net; ε=1e-4)
        @test maximum(abs.(H .- H_fd)) < 1e-5
        @test maximum(abs.(H .- H')) < 1e-8
    end

    @testset "Oracle: FD Hessian in log μ vs analytic H_μ (3sec 2fac)" begin
        io = _three_sector_io()
        net = production_network(io; theta=0.5, sigma=0.9, factors=:va_cats, mu=1.0)
        H = bf_misallocation(net; point=:efficient).H_mu
        H_fd = _fd_hessian_mu(net; ε=1e-4)
        @test maximum(abs.(H .- H_fd)) < 1e-5
        @test maximum(abs.(H .- H')) < 1e-8
    end

    @testset "Oracle: FD Hessian in log μ vs analytic H_μ (two-nest 2fac)" begin
        net = production_network(load_example(:wiot); nests=:two, theta=0.5,
                                 epsilon=0.8, eta=1.0, sigma=0.9,
                                 factors=:va_cats, mu=1.0)
        @test net.n == 2 && net.F == 2
        H = bf_misallocation(net; point=:efficient).H_mu
        H_fd = _fd_hessian_mu(net; ε=1e-4)
        @test maximum(abs.(H .- H_fd)) < 1e-5
        @test maximum(abs.(H .- H')) < 1e-8
    end

    @testset "bf_wedge_quadratic matches v'H v" begin
        net = production_network(_three_sector_io(); theta=0.5, sigma=0.9, factors=:va_cats)
        H = bf_misallocation(net).H_mu
        v = [0.1, -0.05, 0.02]
        @test bf_wedge_quadratic(net, v) ≈ dot(v, H * v) atol=1e-12
    end

    @testset "small-μ quadratic vs exact L" begin
        io = load_example(:wiot)
        μ = [1.02, 1.03]                         # small — second-order must bite
        net = production_network(io; theta=0.5, sigma=1.5, mu=μ)
        m = bf_misallocation(net; point=:efficient)
        @test m.distance > 0                     # markups reduce Y
        @test abs(m.distance - m.second_order) < 5e-5
    end

    @testset "vertical / acyclic: exact L ≈ 0 even if μ ≠ 1" begin
        # One viable allocation: Z = 0, each sector buys only labor, household
        # Leontief (σ=0) so no substitution — Corollary 2 style.
        io = IOData(zeros(2,2), reshape([1.0,1.0],2,1), reshape([1.0,1.0],1,2);
                    sectors=["A","B"])
        net = production_network(io; sigma=0.0, theta=1.0, mu=[1.4, 1.8])
        m = bf_misallocation(net)
        @test abs(m.distance) < 1e-8
        @test abs(m.second_order) < 1e-8
    end

    @testset ":observed is local curvature; exact L independent of point" begin
        net = production_network(load_example(:wiot); theta=0.5, sigma=0.9, mu=[1.2, 1.1])
        m_e = bf_misallocation(net; point=:efficient)
        m_o = bf_misallocation(net; point=:observed)
        @test m_e.distance ≈ m_o.distance atol=1e-12
        @test m_e.first_order ≈ 0 atol=1e-12
        @test m_o.point === :observed
        v = m_o.delta_logmu
        @test bf_wedge_quadratic(net, v; point=:observed) ≈ dot(v, m_o.H_mu * v) atol=1e-12
        @test size(m_o.H_mu) == (2, 2)
        m_none = bf_misallocation(net; hessian=:none)
        @test size(m_none.H_mu) == (0, 0)
        @test m_none.second_order ≈ m_e.second_order atol=1e-12
    end

    @testset ":observed first_order matches horizontal dlogY/dlogμ" begin
        io = _horizontal_io()
        μ = [1.2, 1.5]
        σ = 2.0
        net = production_network(io; sigma=σ, theta=1.0, mu=μ)
        m = bf_misallocation(net; point=:observed)
        λ = [0.5, 0.5]
        s = sum(λ ./ μ)
        g = [λ[k] * σ * ((1 / μ[k]) / s - 1) for k in 1:2]
        @test m.first_order ≈ dot(g, log.(μ)) atol=1e-12
        @test m.distance ≈ bf_misallocation(net).distance atol=1e-12
    end
end
