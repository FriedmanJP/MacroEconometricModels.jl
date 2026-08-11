using Test, MacroEconometricModels, LinearAlgebra
using MacroEconometricModels: nsectors, nregions

# ── helpers: hard-coded KWW (2014) Section 2.4 examples ──────────────────────

"KWW Example 1: Country A exports intermediates; B re-exports as final (no VA)."
function _kww_example1()
    # Z = [50 50; 0 0], Y = [30 20; 50 0], va = [100 0], x = [150, 50]
    Z = [50.0 50.0; 0.0 0.0]
    Y = [30.0 20.0; 50.0 0.0]
    va = reshape([100.0, 0.0], 1, 2)
    IOData(Z, Y, va;
           sectors=["A_goods", "B_goods"],
           regions=["A", "B"],
           fd_cats=["A_fd", "B_fd"],
           va_cats=["VA"], check=true)
end

"KWW Example 2: USA exports intermediates; CHN exports finals that embed US VA."
function _kww_example2()
    # Z = [100 50; 0 50], Y = [30 20; 70 80], va = [100 100], x = [200, 200]
    Z = [100.0 50.0; 0.0 50.0]
    Y = [30.0 20.0; 70.0 80.0]
    va = reshape([100.0, 100.0], 1, 2)
    IOData(Z, Y, va;
           sectors=["USA_e", "CHN_e"],
           regions=["USA", "CHN"],
           fd_cats=["USA_fd", "CHN_fd"],
           va_cats=["VA"], check=true)
end

"3-country 1-sector toy MRIO for aggregate / footprint / third-country DVA."
function _three_country()
    # Balanced table: each country has GO=100, va=60, domestic Z=20,
    # intermediate exports 10 to each of the other two, final domestic 40,
    # final exports 10 to each of the other two.
    # Row for country i: Z_i1+Z_i2+Z_i3 + Y_i1+Y_i2+Y_i3 = 100
    #   Z_ii=20, Z_ij=10 (j≠i) → intermediate row sum = 40
    #   Y_ii=40, Y_ij=10 (j≠i) → final row sum = 60 → x=100
    # Column: Z_1i+Z_2i+Z_3i + va_i = 20+10+10+60 = 100
    Z = [20.0 10.0 10.0;
         10.0 20.0 10.0;
         10.0 10.0 20.0]
    Y = [40.0 10.0 10.0;
         10.0 40.0 10.0;
         10.0 10.0 40.0]
    va = reshape([60.0, 60.0, 60.0], 1, 3)
    IOData(Z, Y, va;
           sectors=["R1_s", "R2_s", "R3_s"],
           regions=["R1", "R2", "R3"],
           fd_cats=["R1_fd", "R2_fd", "R3_fd"],
           va_cats=["VA"], check=true)
end

@testset "MRIO region_block & bilateral trade" begin
    io = _kww_example1()
    @test nregions(io) == 2
    @test nsectors(io) == 1
    @test region_block(io, "A", "A") ≈ [50.0] atol=1e-12
    @test region_block(io, "A", "B") ≈ [50.0] atol=1e-12
    @test region_block(io, "B", "A") ≈ [0.0] atol=1e-12
    @test region_block(io, 1, 2) ≈ region_block(io, "A", "B")

    bt = bilateral_trade(io, "A", "B")
    @test bt.intermediate ≈ 50.0 atol=1e-12
    @test bt.final ≈ 20.0 atol=1e-12
    @test bt.total ≈ 70.0 atol=1e-12
    @test bt.by_sector ≈ [70.0] atol=1e-12

    btBA = bilateral_trade(io, "B", "A")
    @test btBA.total ≈ 50.0 atol=1e-12
    @test btBA.intermediate ≈ 0.0 atol=1e-12
    @test btBA.final ≈ 50.0 atol=1e-12

    @test gross_exports(io, "A") ≈ [70.0] atol=1e-12
    @test gross_exports(io, "B") ≈ [50.0] atol=1e-12
    @test region_indices(io, "A") == [1]
    @test region_indices(io, 2) == [2]
    @test_throws ArgumentError region_block(io, "Z", "A")
end

@testset "KWW Example 1 oracle (Section 2.4)" begin
    # Paper: Country A GE=70, DVA=20, RDV=50, FVA=0, PDC=0
    #        Country B GE=50, DVA=0,  RDV=0,  FVA=50, PDC=0
    #        VS_A=0%, VS_B=100%
    io = _kww_example1()
    edA = export_decomposition(io, "A")
    edB = export_decomposition(io, "B")

    @test edA.gross_exports ≈ 70.0 atol=1e-10
    @test edA.dva ≈ 20.0 atol=1e-10
    @test edA.rdv ≈ 50.0 atol=1e-10
    @test edA.fva ≈ 0.0 atol=1e-10
    @test edA.pdc ≈ 0.0 atol=1e-10
    @test edA.dva + edA.rdv + edA.fva + edA.pdc ≈ edA.gross_exports atol=1e-10
    @test edA.vax_ratio ≈ 20.0 / 70.0 atol=1e-10

    @test edB.gross_exports ≈ 50.0 atol=1e-10
    @test edB.dva ≈ 0.0 atol=1e-10
    @test edB.rdv ≈ 0.0 atol=1e-10
    @test edB.fva ≈ 50.0 atol=1e-10
    @test edB.pdc ≈ 0.0 atol=1e-10
    @test edB.dva + edB.rdv + edB.fva + edB.pdc ≈ edB.gross_exports atol=1e-10

    vsA = vertical_specialization(io, "A")
    vsB = vertical_specialization(io, "B")
    @test vsA.vs ≈ 0.0 atol=1e-10
    @test vsA.vs_share ≈ 0.0 atol=1e-10
    @test vsA.domestic_content ≈ 70.0 atol=1e-10
    @test vsB.vs ≈ 50.0 atol=1e-10
    @test vsB.vs_share ≈ 1.0 atol=1e-10
    @test vsB.domestic_content ≈ 0.0 atol=1e-10
end

@testset "KWW Example 2 oracle (USA–CHN electronics)" begin
    # Paper: both GE=70, both DVA≈46.67; USA RDV≈23.33 FVA=0;
    #        CHN RDV=0 FVA≈23.33; VAX identical.
    io = _kww_example2()
    edU = export_decomposition(io, "USA")
    edC = export_decomposition(io, "CHN")

    @test edU.gross_exports ≈ 70.0 atol=1e-10
    @test edC.gross_exports ≈ 70.0 atol=1e-10
    @test edU.dva ≈ 140 / 3 atol=1e-8          # 46.666…
    @test edC.dva ≈ 140 / 3 atol=1e-8
    @test edU.rdv ≈ 70 / 3 atol=1e-8           # 23.333…
    @test edC.rdv ≈ 0.0 atol=1e-10
    @test edU.fva ≈ 0.0 atol=1e-10
    @test edC.fva ≈ 70 / 3 atol=1e-8
    @test edU.pdc ≈ 0.0 atol=1e-10
    @test edC.pdc ≈ 0.0 atol=1e-10
    @test edU.dva + edU.rdv + edU.fva + edU.pdc ≈ 70.0 atol=1e-10
    @test edC.dva + edC.rdv + edC.fva + edC.pdc ≈ 70.0 atol=1e-10
    @test edU.vax_ratio ≈ edC.vax_ratio atol=1e-10

    vsU = vertical_specialization(io, "USA")
    vsC = vertical_specialization(io, "CHN")
    @test vsU.vs ≈ 0.0 atol=1e-10
    @test vsC.vs ≈ 70 / 3 atol=1e-8
end

@testset "KWW adding-up identity (3-country)" begin
    io = _three_country()
    for r in io.regions
        ed = export_decomposition(io, r)
        @test ed.dva + ed.rdv + ed.fva + ed.pdc ≈ ed.gross_exports atol=1e-8
        @test ed.gross_exports ≈ 40.0 atol=1e-10   # 10+10 intermediate + 10+10 final
        # Third-country DVA route (term 3) is active with G=3
        @test ed.terms[3] > 0
    end
end

@testset "aggregate regions and sectors" begin
    io = _three_country()
    # Collapse R2 and R3 into "ROW"
    agg = aggregate(io; region_map=Dict("R2" => "ROW", "R3" => "ROW"))
    @test nregions(agg) == 2
    @test Set(agg.regions) == Set(["R1", "ROW"])
    @test length(agg.x) == 2
    @test sum(agg.x) ≈ sum(io.x) atol=1e-10
    @test sum(agg.Z) ≈ sum(io.Z) atol=1e-10
    @test sum(agg.Y) ≈ sum(io.Y) atol=1e-10
    @test sum(agg.va) ≈ sum(io.va) atol=1e-10

    # Extensions carried along
    add_extension!(io, "CO2", [1.0 2.0 3.0]; stressors=["CO2"], unit=["kt"])
    agg2 = aggregate(io; region_map=Dict("R2" => "ROW", "R3" => "ROW"))
    @test haskey(agg2.extensions, "CO2")
    @test sum(agg2.extensions["CO2"].F) ≈ 6.0 atol=1e-12

    # Sector aggregation on a 2-sector × 2-region table
    Z = [10.0 5.0  2.0 1.0;
          4.0 20.0 1.0 3.0;
          2.0 1.0 10.0 5.0;
          1.0 3.0  4.0 20.0]
    Y = [30.0 5.0  4.0 1.0;
         40.0 10.0 2.0 3.0;
          4.0 1.0 30.0 5.0;
          2.0 3.0 40.0 10.0]
    # Build va from column balance so the table balances
    x_row = vec(sum(Z; dims=2)) .+ vec(sum(Y; dims=2))
    va_row = x_row .- vec(sum(Z; dims=1))
    va = reshape(va_row, 1, 4)
    io4 = IOData(Z, Y, va;
                 sectors=["N_ag", "N_mfg", "S_ag", "S_mfg"],
                 regions=["N", "S"],
                 fd_cats=["N_c", "N_g", "S_c", "S_g"],
                 va_cats=["VA"], check=true)
    # sector types from first block: N_ag, N_mfg — map by those labels
    # _sector_types uses first ns labels: ["N_ag", "N_mfg"]
    # Map both types to "all" via the actual first-block names
    sec_types = MacroEconometricModels._sector_types(io4)
    smap = Dict(sec_types[1] => "all", sec_types[2] => "all")
    aggs = aggregate(io4; sector_map=smap)
    @test nsectors(aggs) == 1
    @test nregions(aggs) == 2
    @test sum(aggs.x) ≈ sum(io4.x) atol=1e-10

    # No-op identity
    id = aggregate(io)
    @test id.Z == io.Z
end

@testset "Regional footprints sum to global totals" begin
    io = _three_country()
    F = [10.0 20.0 30.0]
    add_extension!(io, "CO2", F; stressors=["CO2"], unit=["kt"])
    fp = footprint(io, "CO2"; by=:region)
    @test fp isa RegionalFootprintResult
    @test size(fp.production) == (1, 3)
    @test size(fp.consumption) == (1, 3)
    @test sum(fp.production) ≈ sum(F) atol=1e-10
    # Consumption-based global total = production total when F_Y = 0
    @test sum(fp.consumption) ≈ sum(F) atol=1e-8
    @test fp.production[1, :] ≈ [10.0, 20.0, 30.0] atol=1e-12

    # Single-region: production ≈ consumption
    wiot = load_example(:wiot)
    fp1 = footprint(wiot, "CO2"; by=:region)
    @test size(fp1.production, 2) == 1
    @test sum(fp1.production) ≈ sum(fp1.consumption) atol=1e-8
end

@testset "report / show / plot / refs for MRIO types" begin
    io = _kww_example1()
    ed = export_decomposition(io, "A")
    vs = vertical_specialization(io, "A")
    add_extension!(io, "CO2", [10.0 0.0]; stressors=["CO2"], unit=["kt"])
    rfp = footprint(io, "CO2"; by=:region)

    for obj in (ed, vs, rfp)
        @test !isempty(sprint(show, obj))
        report(obj)
        @test !isempty(sprint(refs, obj))
    end

    for obj in (ed, vs, rfp)
        p = plot_result(obj)
        @test p isa PlotOutput
    end
end

@testset "single-region edge cases" begin
    wiot = load_example(:wiot)
    @test_throws ArgumentError export_decomposition(wiot, "nope")
    ed = export_decomposition(wiot)               # G=1 → zeros
    @test ed.gross_exports == 0
    @test ed.dva == 0
    vs = vertical_specialization(wiot)
    @test vs.vs == 0
    @test all(region_block(wiot, "total", "total") .== wiot.Z)
end
