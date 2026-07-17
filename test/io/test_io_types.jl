using Test, MacroEconometricModels
using MacroEconometricModels: nsectors, nregions, _invdiag

@testset "IOData construction & validation" begin
    Z = [150.0 500.0; 200.0 100.0]
    Y = reshape([350.0, 1700.0], 2, 1)
    va = [300.0 1000.0; 350.0 400.0]          # 2 va categories × 2 sectors
    io = IOData(Z, Y, va; sectors=["Agriculture", "Manufacturing"],
                fd_cats=["final_demand"], va_cats=["compensation", "other_va"])
    @test nsectors(io) == 2
    @test nregions(io) == 1
    @test io.x ≈ [1000.0, 2000.0]              # x = rowsum(Z)+rowsum(Y)
    @test _invdiag([2.0, 0.0]) == [0.5, 0.0]   # guarded reciprocal

    # accounting identity violation is rejected
    @test_throws ArgumentError IOData([1.0 2.0; 3.0 4.0], reshape([1.0, 1.0], 2, 1),
                                      reshape([0.0, 0.0], 1, 2); check=true)

    # x-form constructor derives a single value-added row
    io2 = IOData(Z, Y, [1000.0, 2000.0])
    @test io2.x ≈ [1000.0, 2000.0]
    @test size(io2.va) == (1, 2)
end
