using Test, MacroEconometricModels, LinearAlgebra, Statistics

@testset "Network statistics" begin
    io = load_example(:wiot)
    ns = network_stats(io)

    # Domar weights and Herfindahl.
    λ = domar_weights(io)
    @test ns.domar ≈ λ atol=1e-12
    @test ns.herfindahl ≈ sum(abs2, λ) atol=1e-12
    @test ns.herfindahl > 0

    # Multipliers match Type I output multipliers; dispersion is their sd.
    om = multipliers(io; kind=:output, type=:I)
    @test ns.multipliers ≈ om.values atol=1e-10
    @test ns.multiplier_dispersion ≈ std(om.values; corrected=true) atol=1e-12

    # Upstreamness / downstreamness match baqaee_farhi (same L-based defs).
    bf = baqaee_farhi(io)
    @test ns.upstreamness ≈ bf.upstreamness atol=1e-12
    @test ns.downstreamness ≈ bf.downstreamness atol=1e-12
    @test ns.downstreamness ≈ ns.multipliers atol=1e-12

    # APL identities: diagonal entries ≥ 1 when self-linkages exist;
    # off-diagonals positive when L_ij > 0; H = L(L−I) recovers num.
    L = leontief_inverse(io)
    H = L * (L - I)
    for i in 1:2, j in 1:2
        denom = L[i, j] - (i == j ? 1.0 : 0.0)
        if abs(denom) > 1e-14
            @test ns.apl[i, j] ≈ H[i, j] / denom atol=1e-10
            @test ns.apl[i, j] > 0
        else
            @test ns.apl[i, j] == 0
        end
    end

    # Weighted degrees of A.
    A = technical_coefficients(io)
    @test ns.in_degree ≈ vec(sum(A, dims=2)) atol=1e-12
    @test ns.out_degree ≈ vec(sum(A, dims=1)) atol=1e-12

    @test length(ns.sectors) == 2
end
