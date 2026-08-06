# CF-03 (#383): weighted policy-projection kernel — LS/exact/min-norm regimes,
# weighting (incl. PSD-singular), linear term, degenerate inputs, and the
# permanent sign-error property test.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

@testset "Policy-projection kernel (CF-03)" begin
    rng = MersenneTwister(20260805)

    @testset "unweighted-ls" begin
        for _ in 1:5
            M = randn(rng, 50, 3)
            b = randn(rng, 50)
            res = MEM._policy_projection(M, b)
            nu_ref = -((M' * M) \ (M' * b))
            @test res.nu ≈ nu_ref atol = 1e-10
            @test res.rank == 3
            @test !res.deficient
            @test res.method_used == :ls
            @test res.error_path ≈ M * res.nu + b atol = 1e-14
            @test res.rel_residual ≈ norm(res.error_path) / norm(b) atol = 1e-12
        end
    end

    @testset "weighted" begin
        M = randn(rng, 50, 3)
        b = randn(rng, 50)

        # W = I equals unweighted
        res_I = MEM._policy_projection(M, b; W=Matrix{Float64}(I, 50, 50))
        res_0 = MEM._policy_projection(M, b)
        @test res_I.nu ≈ res_0.nu atol = 1e-10

        # random SPD W matches the explicit normal-equation formula
        Q0 = qr(randn(rng, 50, 50)).Q
        lam = 0.1 .+ rand(rng, 50)
        W = Matrix(Q0 * Diagonal(lam) * Q0')
        W = (W + W') / 2
        res_W = MEM._policy_projection(M, b; W=W)
        nu_ref = -((M' * W * M) \ (M' * W * b))
        @test res_W.nu ≈ nu_ref atol = 1e-8

        # PSD-singular rank-1 W runs without throwing, matches the reduced problem
        w = randn(rng, 50)
        W1 = w * w'
        res_1 = @test_logs (:warn, r"rank-deficient") match_mode = :any begin
            MEM._policy_projection(M, b; W=W1)
        end
        a = M' * w
        nu_ref1 = -a * (w' * b) / (a' * a)   # min-norm solution of the rank-1 problem
        @test res_1.nu ≈ nu_ref1 atol = 1e-8
        @test res_1.deficient
    end

    @testset "exact-square" begin
        H = 40
        M = randn(rng, H, H)
        b = randn(rng, H)
        res = MEM._policy_projection(M, b)
        @test res.method_used == :exact
        @test norm(res.error_path) < 1e-10
        @test res.rel_residual < 1e-10
        @test res.rank == H

        # exact solve is W-independent
        Wd = Matrix(Diagonal(0.5 .+ rand(rng, H)))
        res_w = MEM._policy_projection(M, b; W=Wd)
        @test res_w.nu ≈ res.nu atol = 1e-8

        # near-singular square warns and falls back to :ls
        Mns = copy(M)
        Mns[:, 2] = Mns[:, 1] + 1e-12 * randn(rng, H)
        res_ns = @test_logs (:warn, r"ill-conditioned") match_mode = :any begin
            MEM._policy_projection(Mns, b)
        end
        @test res_ns.method_used in (:ls, :ls_minnorm)

        # :ls forces least squares even for square M
        res_ls = MEM._policy_projection(M, b; method=:ls)
        @test res_ls.method_used == :ls
        @test res_ls.nu ≈ res.nu atol = 1e-8
    end

    @testset "rank-deficient" begin
        M = randn(rng, 50, 3)
        M[:, 3] = M[:, 1]   # duplicated column
        b = randn(rng, 50)
        res = @test_logs (:warn, r"rank-deficient") match_mode = :any begin
            MEM._policy_projection(M, b)
        end
        @test res.deficient
        @test res.method_used == :ls_minnorm
        @test res.rank == 2
        @test res.nu ≈ pinv(M) * (-b) atol = 1e-8
    end

    @testset "linear-term" begin
        M = randn(rng, 50, 3)
        b = randn(rng, 50)
        c = randn(rng, 3)
        Q0 = qr(randn(rng, 50, 50)).Q
        W = Matrix(Q0 * Diagonal(0.1 .+ rand(rng, 50)) * Q0')
        W = (W + W') / 2

        res = MEM._policy_projection(M, b; W=W, c=c)
        @test M' * W * (M * res.nu + b) + c ≈ zeros(3) atol = 1e-8

        # unweighted with linear term
        res_u = MEM._policy_projection(M, b; c=c)
        @test M' * (M * res_u.nu + b) + c ≈ zeros(3) atol = 1e-8

        # square M + c routes through :ls, never :exact
        Msq = randn(rng, 6, 6)
        res_sq = MEM._policy_projection(Msq, randn(rng, 6); c=randn(rng, 6))
        @test res_sq.method_used == :ls
    end

    @testset "degenerate" begin
        M = randn(rng, 30, 2)
        res = MEM._policy_projection(M, zeros(30))
        @test res.nu == zeros(2)
        @test res.rel_residual == 0.0
        @test res.error_path == zeros(30)
    end

    @testset "property: projection never worsens the loss" begin
        for W in (nothing, Matrix{Float64}(I, 50, 50) .* (0.1 .+ rand(rng, 50)))
            M = randn(rng, 50, 3)
            b = randn(rng, 50)
            Weff = W === nothing ? Matrix{Float64}(I, 50, 50) : W
            res = MEM._policy_projection(M, b; W=W)
            loss(nu) = (M * nu + b)' * Weff * (M * nu + b)
            lstar = loss(res.nu)
            @test lstar <= loss(zeros(3)) + 1e-10
            for _ in 1:20
                @test lstar <= loss(randn(rng, 3)) + 1e-10
            end
        end
    end

    @testset "input validation" begin
        M = randn(rng, 10, 2)
        b = randn(rng, 10)
        @test_throws ArgumentError MEM._policy_projection(M, randn(rng, 9))
        @test_throws ArgumentError MEM._policy_projection(M, b; W=ones(9, 9))
        @test_throws ArgumentError MEM._policy_projection(M, b; c=ones(3))
        @test_throws ArgumentError MEM._policy_projection(M, b; method=:qp)
        @test_throws ArgumentError MEM._policy_projection(M, b; method=:exact)          # non-square
        Msq = randn(rng, 4, 4)
        @test_throws ArgumentError MEM._policy_projection(Msq, randn(rng, 4);
                                                          method=:exact, c=ones(4))     # exact + c
        # all-negative W has no positive eigenvalue
        @test_throws ArgumentError MEM._policy_projection(M, b; W=-Matrix{Float64}(I, 10, 10))
    end
end
