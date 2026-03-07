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

@testset "Linear DSGE historical decomposition" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)

    T_obs = 80
    rng = Random.MersenneTwister(55)
    sim_data = simulate(sol, T_obs; rng=rng)
    observables = [:y, :pi_var, :r]

    hd = historical_decomposition(sol, sim_data, observables)

    @test hd isa HistoricalDecomposition{Float64}
    @test size(hd.contributions) == (T_obs, 3, 3)
    @test size(hd.initial_conditions) == (T_obs, 3)
    @test size(hd.actual) == (T_obs, 3)
    @test hd.method == :dsge_linear
    @test verify_decomposition(hd; tol=0.1)
end

@testset "Linear DSGE HD — single variable" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    T_obs = 60
    rng = Random.MersenneTwister(77)
    sim_data = simulate(sol, T_obs; rng=rng)
    hd = historical_decomposition(sol, sim_data, [:y])
    @test size(hd.contributions, 3) == 1
    @test verify_decomposition(hd; tol=0.05)
end

@testset "Linear DSGE HD — states=:all" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5
        endogenous: y, pi_var
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    end
    sol = solve(spec)

    T_obs = 40
    rng = Random.MersenneTwister(33)
    sim_data = simulate(sol, T_obs; rng=rng)
    hd = historical_decomposition(sol, sim_data, [:y, :pi_var]; states=:all)
    @test size(hd.contributions, 2) == 2
end

# =============================================================================
# Particle smoother tests
# =============================================================================

@testset "Particle smoother helper — _categorical_draw" begin
    rng = Random.MersenneTwister(42)
    w = [0.1, 0.3, 0.6]
    counts = zeros(Int, 3)
    for _ in 1:10000
        idx = MacroEconometricModels._categorical_draw(w, rng)
        counts[idx] += 1
    end
    # Should roughly match the weights
    @test counts[3] > counts[2] > counts[1]
    @test counts[3] / 10000 > 0.5
end

@testset "Particle smoother helper — _nonlinear_transition" begin
    spec = @dsge begin
        parameters: rho = 0.7
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    spec = compute_steady_state(spec)
    psol = perturbation_solver(spec; order=1)

    # Build NSS
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, [:y], [1e-4])
    nss = MacroEconometricModels._build_nonlinear_state_space(psol, Z, d, H_mat)

    nx = length(nss.state_indices)
    xf_prev = [0.5]
    xs_prev = zeros(nx)
    xrd_prev = zeros(nx)
    eps_t = [0.3]

    xf_new, xs_new, xrd_new, x_total, y_t = MacroEconometricModels._nonlinear_transition(
        nss, xf_prev, xs_prev, xrd_prev, eps_t)

    # For order 1: xf_new = 0.7 * 0.5 + 0.3 = 0.65
    @test xf_new[1] ≈ 0.65 atol=1e-10
    @test x_total[1] ≈ 0.65 atol=1e-10
end

@testset "Particle smoother — linear DGP matches RTS" begin
    spec = @dsge begin
        parameters: rho = 0.7
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)
    spec = compute_steady_state(spec)

    T_obs = 30
    rng = Random.MersenneTwister(88)
    sim_data = simulate(sol, T_obs; rng=rng)
    observables = [:y]

    # Build nonlinear state space (order 1 perturbation)
    psol = perturbation_solver(spec; order=1)
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, [1e-4])
    nss = MacroEconometricModels._build_nonlinear_state_space(psol, Z, d, H_mat)

    # Build linear for comparison
    Z2, d2, H2 = MacroEconometricModels._build_observation_equation(spec, observables, [1e-4])
    ss = MacroEconometricModels._build_state_space(sol, Z2, d2, H2)
    data_matrix = Matrix{Float64}(sim_data' .- sol.spec.steady_state)

    rts_result = dsge_smoother(ss, data_matrix)
    pf_result = dsge_particle_smoother(nss, data_matrix; N=2000, N_back=200,
                                        rng=Random.MersenneTwister(42))

    @test pf_result isa KalmanSmootherResult{Float64}
    @test size(pf_result.smoothed_states, 1) == 1
    @test size(pf_result.smoothed_states, 2) == T_obs
    @test size(pf_result.smoothed_shocks) == (1, T_obs)

    # Correlation between particle smoother and RTS smoother should be high
    @test cor(vec(pf_result.smoothed_states), vec(rts_result.smoothed_states)) > 0.85
end

@testset "Particle smoother — 2-variable linear DGP" begin
    spec = @dsge begin
        parameters: rho_y = 0.7, rho_pi = 0.5
        endogenous: y, pi_var
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    end
    sol = solve(spec)
    spec = compute_steady_state(spec)

    T_obs = 30
    rng = Random.MersenneTwister(101)
    sim_data = simulate(sol, T_obs; rng=rng)

    psol = perturbation_solver(spec; order=1)
    observables = [:y, :pi_var]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, [1e-4, 1e-4])
    nss = MacroEconometricModels._build_nonlinear_state_space(psol, Z, d, H_mat)
    data_matrix = Matrix{Float64}(sim_data' .- sol.spec.steady_state)

    pf_result = dsge_particle_smoother(nss, data_matrix; N=1500, N_back=150,
                                        rng=Random.MersenneTwister(55))

    @test size(pf_result.smoothed_states) == (2, T_obs)
    @test size(pf_result.smoothed_shocks) == (2, T_obs)
    # Smoothed states should be non-trivial (not all zero)
    @test norm(pf_result.smoothed_states) > 0.1
end

# =============================================================================
# Nonlinear DSGE HD tests
# =============================================================================

@testset "Nonlinear DSGE HD — 1st-order perturbation" begin
    spec = @dsge begin
        parameters: rho = 0.7
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    spec = compute_steady_state(spec)
    psol = perturbation_solver(spec; order=1)

    T_obs = 25
    rng = Random.MersenneTwister(44)
    sim_data = simulate(psol, T_obs; rng=rng)
    observables = [:y]

    hd = historical_decomposition(psol, sim_data, observables;
                                   N=1000, N_back=100,
                                   rng=Random.MersenneTwister(42))

    @test hd isa HistoricalDecomposition{Float64}
    @test size(hd.contributions) == (T_obs, 1, 1)
    @test hd.method == :dsge_nonlinear

    # Approximate identity: total contributions + IC should correlate with actual
    total = sum(hd.contributions[:, 1, :]; dims=2)[:, 1] .+ hd.initial_conditions[:, 1]
    @test cor(total, hd.actual[:, 1]) > 0.85
end

@testset "Nonlinear DSGE HD — 2nd-order perturbation" begin
    spec = @dsge begin
        parameters: rho_y = 0.7, rho_pi = 0.5
        endogenous: y, pi_var
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    end
    spec = compute_steady_state(spec)
    psol = perturbation_solver(spec; order=2)

    T_obs = 30
    rng = Random.MersenneTwister(66)
    sim_data = simulate(psol, T_obs; rng=rng)
    observables = [:y, :pi_var]

    hd = historical_decomposition(psol, sim_data, observables;
                                   N=500, N_back=50,
                                   rng=Random.MersenneTwister(42))

    @test hd isa HistoricalDecomposition{Float64}
    @test size(hd.contributions) == (T_obs, 2, 2)
    @test hd.method == :dsge_nonlinear

    # Approximate identity for each variable
    for i in 1:2
        total = sum(hd.contributions[:, i, :]; dims=2)[:, 1] .+ hd.initial_conditions[:, i]
        @test cor(total, hd.actual[:, i]) > 0.85
    end
end

end  # outer testset
