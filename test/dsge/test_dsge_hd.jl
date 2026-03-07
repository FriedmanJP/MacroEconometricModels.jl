# Test DSGE Historical Decomposition
using Test
using MacroEconometricModels
using LinearAlgebra
using Random

@testset "DSGE Historical Decomposition" begin

@testset "KalmanSmootherResult type exists" begin
    # Verify the type is exported and has correct supertype
    @test KalmanSmootherResult <: MacroEconometricModels.AbstractAnalysisResult

    # Verify constructor works
    n_s, n_sh, T_obs = 3, 2, 10
    r = KalmanSmootherResult{Float64}(
        zeros(n_s, T_obs), zeros(n_s, n_s, T_obs), zeros(n_sh, T_obs),
        zeros(n_s, T_obs), zeros(n_s, n_s, T_obs),
        zeros(n_s, T_obs), zeros(n_s, n_s, T_obs), -100.0
    )
    @test r isa KalmanSmootherResult{Float64}
    @test size(r.smoothed_states) == (3, 10)
    @test size(r.smoothed_shocks) == (2, 10)
    @test r.log_likelihood == -100.0
end

end  # outer testset
