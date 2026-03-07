# Test DSGE Historical Decomposition
using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics: cor

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

@testset "RTS smoother log-likelihood matches _kalman_loglikelihood" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)
    observables = [:y, :pi_var, :r]
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H)

    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, 50; rng=rng)
    data_matrix = Matrix{Float64}(sim_data' .- sol.spec.steady_state)

    smoother_result = dsge_smoother(ss, data_matrix)
    kf_ll = MacroEconometricModels._kalman_loglikelihood(ss, data_matrix)
    @test smoother_result.log_likelihood ≈ kf_ll atol=1e-8
end

@testset "RTS smoother recovers known shocks — univariate" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    T_obs = 100
    rng = Random.MersenneTwister(123)
    true_shocks = randn(rng, T_obs)
    shock_matrix = reshape(true_shocks, T_obs, 1)
    sim_data = simulate(sol, T_obs; shock_draws=shock_matrix)

    observables = [:y]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, [1e-6])
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)
    data_matrix = Matrix{Float64}(sim_data' .- sol.spec.steady_state)

    result = dsge_smoother(ss, data_matrix)
    recovered = result.smoothed_shocks[1, :]
    @test cor(recovered, true_shocks) > 0.95
end

@testset "RTS smoother recovers known shocks — 3-variable" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)

    T_obs = 100
    rng = Random.MersenneTwister(99)
    true_shocks = randn(rng, T_obs, 3)
    sim_data = simulate(sol, T_obs; shock_draws=true_shocks)

    observables = [:y, :pi_var, :r]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, [1e-6, 1e-6, 1e-6])
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)
    data_matrix = Matrix{Float64}(sim_data' .- sol.spec.steady_state)

    result = dsge_smoother(ss, data_matrix)
    for j in 1:3
        @test cor(result.smoothed_shocks[j, :], true_shocks[:, j]) > 0.90
    end
end

end  # outer testset
