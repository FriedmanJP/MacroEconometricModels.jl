using Test, MacroEconometricModels, LinearAlgebra

@testset "Hypothetical extraction" begin
    io = load_example(:wiot)
    r = hypothetical_extraction(io, 1)
    @test r.extracted == [1]
    @test r.total_loss > 0
    @test r.total_loss ≈ sum(r.sector_loss) atol=1e-8
    @test r.mode === :complete
    @test r.share ≈ 1.0
    # Legacy oracle (Miller & Blair 2-sector complete extraction of Agriculture).
    @test r.total_loss ≈ 1210.5263157894735 atol=1e-6
    @test r.sector_loss ≈ [1000.0, 210.52631578947353] atol=1e-6
    @test r.loss_pct_go ≈ r.total_loss / sum(io.x) atol=1e-12
    @test r.loss_pct_gdp ≈ r.total_loss / sum(io.va) atol=1e-12

    # extracting by name matches extracting by index
    rn = hypothetical_extraction(io, "Agriculture")
    @test rn.total_loss ≈ r.total_loss
    # extracting multiple sectors loses at least as much as extracting one
    rall = hypothetical_extraction(io, [1, 2])
    @test rall.total_loss >= r.total_loss - 1e-8
    # Full extraction of all sectors ≈ total gross output.
    @test rall.total_loss ≈ sum(io.x) atol=1e-6
end

@testset "Extraction variants (Dietzenbacher–Lahr)" begin
    io = load_example(:wiot)
    rc = hypothetical_extraction(io, 1; mode=:complete)
    rb = hypothetical_extraction(io, 1; mode=:backward)
    rf = hypothetical_extraction(io, 1; mode=:forward)
    rp = hypothetical_extraction(io, 1; mode=:partial, share=0.5)

    # All modes report positive losses for a real sector.
    @test rc.total_loss > 0
    @test rb.total_loss > 0
    @test rf.total_loss > 0
    @test rp.total_loss > 0

    # Backward / forward sever only one side → weakly smaller loss than complete.
    @test rb.total_loss <= rc.total_loss + 1e-8
    @test rf.total_loss <= rc.total_loss + 1e-8

    # Partial share=0.5 is strictly between zero and complete.
    @test rp.total_loss < rc.total_loss - 1e-8
    @test rp.share ≈ 0.5
    @test rp.mode === :partial

    # share=1 under :partial equals :complete.
    rp1 = hypothetical_extraction(io, 1; mode=:partial, share=1.0)
    @test rp1.total_loss ≈ rc.total_loss atol=1e-8

    # share=1 under :complete is the legacy path (already pinned above).
    @test rc.mode === :complete

    @test_throws ArgumentError hypothetical_extraction(io, 1; mode=:foo)
    @test_throws ArgumentError hypothetical_extraction(io, 1; share=0.0)
    @test_throws ArgumentError hypothetical_extraction(io, 1; share=1.5)
end

@testset "Extraction of a whole MRIO region" begin
    # Balanced 2-region × 2-sector toy table (column orientation).
    # Regions North, South; sectors A, B within each (block layout).
    Z = [10.0  4.0  1.0  0.0;
          3.0 12.0  0.0  2.0;
          1.0  0.0  8.0  3.0;
          0.0  2.0  2.0 10.0]
    Y = reshape([12.0, 13.0, 11.0, 14.0], 4, 1)
    # Column VA so x = colsum(Z)+va = rowsum(Z)+Y.
    x_row = vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2))
    va = reshape(x_row .- vec(sum(Z, dims=1)), 1, 4)
    io = IOData(Z, Y, va;
                sectors=["A", "B", "A", "B"],
                regions=["North", "South"],
                fd_cats=["fd"], va_cats=["va"])
    @test length(io.x) == 4

    rN = hypothetical_extraction(io, nothing; region="North")
    @test rN.extracted == [1, 2]
    @test rN.total_loss > 0

    # Bare region name (not a sector label) also works.
    rS = hypothetical_extraction(io, "South")
    @test rS.extracted == [3, 4]
end
