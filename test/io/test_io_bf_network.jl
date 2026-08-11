using Test, MacroEconometricModels, LinearAlgebra, SparseArrays

@testset "ProductionNetwork calibration" begin
    io = load_example(:wiot)

    @testset "single nest, single factor" begin
        net = production_network(io)
        @test net isa ProductionNetwork
        @test net.n == 2
        @test net.M == 2
        @test net.F == 1
        @test net.nests === :single
        N = 1 + net.M + net.F
        @test size(net.Omega) == (N, N)
        @test length(net.theta) == 1 + net.M
        @test length(net.lambda) == N
        @test length(net.factor_supplies) == net.F
        @test length(net.parent) == N
        @test length(net.outer_nodes) == net.n

        # structural invariants
        Ω = Matrix(net.Omega)
        @test all(abs.(sum(Ω[1:1+net.M, :]; dims=2) .- 1) .< 1e-12)  # HH+producers rows sum to 1
        @test all(abs.(sum(Ω[net.M+2:N, :]; dims=2)) .< 1e-12)       # factor rows zero
        @test net.lambda[1] ≈ 1 atol=1e-12
        @test sum(net.lambda[net.M+2:N]) ≈ 1 atol=1e-12

        # Domar on real sectors matches sales/GDP
        gdp = sum(io.va)
        λ_sec = [net.lambda[g] for g in net.outer_nodes]
        @test λ_sec ≈ io.x ./ gdp atol=1e-12

        # row-orientation share check: producer 1 on supplier 1 = Z[1,1]/x[1]
        # node of producer 1 is 2; supplier 1 is node 2
        @test Ω[2, 2] ≈ io.Z[1, 1] / io.x[1] atol=1e-12
        @test Ω[2, 3] ≈ io.Z[2, 1] / io.x[1] atol=1e-12
        @test Ω[2, 4] ≈ sum(io.va[:, 1]) / io.x[1] atol=1e-12
    end

    @testset "multi-factor via :va_cats" begin
        net = production_network(io; factors=:va_cats)
        @test net.F == 2
        @test net.lambda[1] ≈ 1 atol=1e-12
        @test sum(net.lambda[net.M+2:end]) ≈ 1 atol=1e-12
        # factor Domars = factor income / GDP
        gdp = sum(io.va)
        Λ = net.lambda[net.M+2:end]
        @test Λ ≈ vec(sum(io.va; dims=2)) ./ gdp atol=1e-12
    end

    @testset "two-nest structure" begin
        net = production_network(io; nests=:two, theta=0.5, epsilon=0.8, eta=1.0)
        @test net.M == 3 * net.n
        @test net.nests === :two
        N = 1 + net.M + net.F
        Ω = Matrix(net.Omega)
        @test all(abs.(sum(Ω[1:1+net.M, :]; dims=2) .- 1) .< 1e-12)
        @test net.lambda[1] ≈ 1 atol=1e-12
        @test sum(net.lambda[net.M+2:N]) ≈ 1 atol=1e-12
        # outer-node Domars still equal sales/GDP
        gdp = sum(io.va)
        λ_out = [net.lambda[g] for g in net.outer_nodes]
        @test λ_out ≈ io.x ./ gdp atol=1e-10
        # parent map: fictitious nodes point to real sectors
        @test net.parent[1] == 0
        @test all(net.parent[g] > 0 for g in 2:1+net.M)
        @test all(net.parent[net.M+2:N] .== 0)
        # theta length and assignment: [σ, ε…, θ…, η…]
        @test length(net.theta) == 1 + net.M
        @test net.theta[1] ≈ 1.0          # default sigma
        @test all(net.theta[2:3] .≈ 0.8)  # epsilon on outers
        @test all(net.theta[4:5] .≈ 0.5)  # theta on inter bundles
        @test all(net.theta[6:7] .≈ 1.0)  # eta on VA bundles
    end

    @testset "heterogeneous elasticities + factor matrix" begin
        V = [300.0 1000.0; 350.0 400.0]   # same as wiot va
        net = production_network(io; theta=[0.2, 1.5], sigma=0.9,
                                 factors=V)
        @test net.F == 2
        @test net.theta[1] ≈ 0.9
        @test net.theta[2] ≈ 0.2
        @test net.theta[3] ≈ 1.5
    end

    @testset "negative-share clipping" begin
        # Inject a small negative intermediate flow; keep column balance via VA.
        Z = [150.0 500.0; 200.0 100.0]
        Y = reshape([350.0, 1700.0], 2, 1)
        va = reshape([650.0, 1400.0], 1, 2)
        Zneg = copy(Z)
        Zneg[2, 1] = -5.0          # small negative
        # rebalance: reduce VA so colsum still = x? Constructor checks balance.
        # Use check=false on IOData and on production_network.
        x = [1000.0, 2000.0]
        va_adj = reshape(x .- vec(sum(Zneg; dims=1)), 1, 2)
        io_neg = IOData(Zneg, Y, va_adj; check=false)
        net = production_network(io_neg; check=false)
        Ω = Matrix(net.Omega)
        @test all(Ω .>= -1e-15)
        @test all(abs.(sum(Ω[1:1+net.M, :]; dims=2) .- 1) .< 1e-10)
    end

    @testset "rejects :custom nests" begin
        @test_throws ArgumentError production_network(io; nests=:custom)
    end

    @testset "show / report smoke" begin
        net = production_network(io)
        s = sprint(show, net)
        @test occursin("ProductionNetwork", s)
        mktemp() do _, ioh
            redirect_stdout(ioh) do
                report(net)
            end
        end
    end
end
