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
end
