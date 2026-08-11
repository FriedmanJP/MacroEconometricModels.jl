using Test, MacroEconometricModels

@testset "RAS fixed point of already-balanced table" begin
    A = [2.0 1.0; 1.0 3.0]
    u = vec(sum(A, dims=2))
    v = vec(sum(A, dims=1))
    r = ras(A, u, v)
    @test r.converged
    @test r.method === :ras
    @test r.X ≈ A atol=1e-10
    @test maximum(abs, r.residual_u) < 1e-10
    @test maximum(abs, r.residual_v) < 1e-10
end

@testset "RAS hits target margins" begin
    # Hard-coded prior and targets (no RNG)
    A0 = [2.0 1.0; 1.0 2.0]
    u  = [4.0, 5.0]
    v  = [3.0, 6.0]
    @test sum(u) ≈ sum(v)
    r = ras(A0, u, v; tol=1e-12)
    @test r.converged
    @test vec(sum(r.X, dims=2)) ≈ u atol=1e-10
    @test vec(sum(r.X, dims=1)) ≈ v atol=1e-10
    # Structure-preserving: zeros stay zero
    A0z = [2.0 0.0; 1.0 3.0]
    rz = ras(A0z, [2.0, 4.0], [2.5, 3.5]; tol=1e-12)
    @test rz.X[1, 2] == 0.0
    @test vec(sum(rz.X, dims=2)) ≈ [2.0, 4.0] atol=1e-8
    @test vec(sum(rz.X, dims=1)) ≈ [2.5, 3.5] atol=1e-8
end

@testset "RAS rejects negatives and inconsistent margins" begin
    @test_throws ArgumentError ras([-1.0 2.0; 1.0 1.0], [1.0, 2.0], [0.0, 3.0])
    @test_throws ArgumentError ras([1.0 1.0; 1.0 1.0], [3.0, 3.0], [2.0, 2.0])  # 6 ≠ 4
end

@testset "GRAS fixed point of already-balanced signed table" begin
    # Already satisfies margins (including a negative entry)
    A = [-1.0 2.0; 3.0 -1.0]
    u = vec(sum(A, dims=2))   # [1, 2]
    v = vec(sum(A, dims=1))   # [2, 1]
    r = gras(A, u, v)
    @test r.converged
    @test r.method === :gras
    @test r.X ≈ A atol=1e-10
end

@testset "GRAS sign-preserving margin match (J&O-style)" begin
    # Archetypal mixed-sign prior (structure of the Junius–Oosterhaven / Lemelin
    # net-trade toy: positives and a pure-negative column entry). Hard-coded;
    # oracle is margin identities + sign preservation, not a published digit table.
    A0 = [7.0  3.0  5.0  -3.0;
          2.0  9.0  8.0   1.0;
         -2.0  0.0  2.0   1.0]
    u = [15.0, 18.0, 2.0]
    v = [8.0, 10.0, 14.0, 3.0]
    @test sum(u) ≈ sum(v)
    r = gras(A0, u, v; tol=1e-12)
    @test r.converged
    @test vec(sum(r.X, dims=2)) ≈ u atol=1e-9
    @test vec(sum(r.X, dims=1)) ≈ v atol=1e-9
    # Sign preservation (and zeros stay zero)
    for i in eachindex(A0)
        if A0[i] == 0
            @test r.X[i] == 0
        else
            @test sign(r.X[i]) == sign(A0[i])
        end
    end
    # All-nonnegative GRAS reduces to RAS margins
    P = [2.0 1.0; 1.0 2.0]
    uP, vP = [4.0, 5.0], [3.0, 6.0]
    rg = gras(P, uP, vP)
    rr = ras(P, uP, vP)
    @test rg.X ≈ rr.X rtol=1e-8
end

@testset "balance(io) fixed point and repair" begin
    io = load_example(:wiot)
    # Already-balanced Miller–Blair table is a fixed point
    b = balance(io; method=:ras)
    @test b.Z ≈ io.Z atol=1e-10
    @test b.x ≈ io.x atol=1e-10
    # Row and column identities hold
    @test b.x ≈ vec(sum(b.Z, dims=2)) .+ vec(sum(b.Y, dims=2)) atol=1e-10
    @test b.x ≈ vec(sum(b.Z, dims=1)) .+ vec(sum(b.va, dims=1)) atol=1e-10

    # Perturb Z while holding the published x fixed → breaks both identities
    Zb = copy(io.Z)
    Zb[1, 1] += 50.0
    io_bad = IOData(Zb, io.Y, io.x; va=io.va, sectors=io.sectors,
                    fd_cats=io.fd_cats, va_cats=io.va_cats, check=false)
    @test_throws ArgumentError IOData(Zb, io.Y, io.x; va=io.va, sectors=io.sectors,
                                      check=true)
    fixed = balance(io_bad; method=:ras)
    @test fixed.x ≈ vec(sum(fixed.Z, dims=2)) .+ vec(sum(fixed.Y, dims=2)) atol=1e-8
    @test fixed.x ≈ vec(sum(fixed.Z, dims=1)) .+ vec(sum(fixed.va, dims=1)) atol=1e-8
end

@testset "balance with GRAS on signed intermediate block" begin
    # Tiny table with a negative intermediate flow (inventory drawdown style)
    Z = [10.0 -2.0; 4.0 8.0]
    Y = reshape([12.0, 10.0], 2, 1)
    va = reshape([8.0, 14.0], 1, 2)
    x = [20.0, 22.0]   # rowsum(Z)+Y = [20,22]; colsum(Z)+va = [22,20] — inconsistent cols
    # Build a consistent signed table: choose x and va so both identities hold
    # row: x = [10-2+12, 4+8+10] = [20, 22]
    # col need: x_j = colsum(Z)_j + va_j ⇒ va = x - colsum(Z) = [20-14, 22-6] = [6, 16]
    va_ok = reshape([6.0, 16.0], 1, 2)
    io = IOData(Z, Y, va_ok; sectors=["A", "B"], check=true)
    b = balance(io; method=:gras)
    @test b.Z ≈ Z atol=1e-10
    # Unbalance Z and repair
    Z2 = copy(Z); Z2[2, 2] += 3.0
    io_bad = IOData(Z2, Y, io.x; va=va_ok, sectors=["A", "B"], check=false)
    fixed = balance(io_bad; method=:gras)
    @test fixed.x ≈ vec(sum(fixed.Z, dims=2)) .+ vec(sum(fixed.Y, dims=2)) atol=1e-8
    @test fixed.x ≈ vec(sum(fixed.Z, dims=1)) .+ vec(sum(fixed.va, dims=1)) atol=1e-8
    # Negative cell keeps its sign
    @test fixed.Z[1, 2] < 0
end

@testset "RAS/GRAS show, report, plot, refs" begin
    A0 = [2.0 1.0; 1.0 2.0]
    r = ras(A0, [4.0, 5.0], [3.0, 6.0])
    @test !isempty(sprint(show, r))
    mktemp() do _path, ioh
        redirect_stdout(ioh) do
            report(r)
        end
    end
    p = plot_result(r)
    @test p isa PlotOutput
    @test !isempty(sprint(refs, r))
end
