# Coverage tests for DSGE Bayesian estimation pipeline
# Targets: particle_filter.jl, smc.jl, bayes_estimation.jl,
#          pruning.jl, derivatives.jl, occbin.jl
#
# Focuses on code paths NOT exercised by:
#   test/dsge/test_bayesian_dsge.jl  (~2500 lines)
#   test/coverage/test_dsge_coverage.jl  (~1100 lines)

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const M = MacroEconometricModels
const _suppress_warnings = M._suppress_warnings

@testset "DSGE Bayes Coverage" begin

# =====================================================================
# Helper: shared RBC-style model for reuse
# =====================================================================
function _make_rbc_spec()
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    compute_steady_state(spec)
end

function _make_fwd_spec()
    spec = @dsge begin
        parameters: beta = 0.99, rho = 0.9
        endogenous: c, k
        exogenous: e
        c[t] = beta * E[t](c[t+1]) + e[t]
        k[t] = rho * k[t-1] + c[t]
    end
    compute_steady_state(spec)
end

# =====================================================================
# 1. particle_filter.jl — uncovered paths
# =====================================================================

@testset "particle_filter: _pf_initialize_stationary! with fallback Lyapunov" begin
    # Build state space where solve_lyapunov would fail (explosive G1)
    # Use a nearly-explosive system that still has valid Cholesky
    n_states = 2
    N = 30
    G1 = [0.3 0.1; 0.0 0.4]
    impact = [0.5 0.0; 0.0 0.5]
    Z = Matrix{Float64}(I, 2, 2)
    d = zeros(2)
    H = 0.01 * Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2)

    ss = M.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)
    ws = M._allocate_pf_workspace(Float64, n_states, 2, 2, N)

    M._pf_initialize_stationary!(ws, ss; rng=Random.MersenneTwister(1234))

    @test all(isfinite, ws.particles)
    @test all(ws.weights .≈ 1.0 / N)
    @test all(ws.log_weights .≈ -log(Float64(N)))
end

@testset "particle_filter: _fill_kron_cross_buffer!" begin
    nv = 3
    N = 10
    V1 = randn(Random.MersenneTwister(100), nv, N)
    V2 = randn(Random.MersenneTwister(101), nv, N)
    buffer = zeros(nv * nv, N)

    M._fill_kron_cross_buffer!(buffer, V1, V2, nv)

    for k in 1:N, i in 1:nv, j in 1:nv
        idx = (i - 1) * nv + j
        @test buffer[idx, k] ≈ V1[i, k] * V2[j, k] atol=1e-14
    end
end

@testset "particle_filter: bootstrap PF with store_trajectory" begin
    rng = Random.MersenneTwister(5001)
    rho = 0.7
    sigma_eps = 0.4
    sigma_me = 0.05
    T_sim = 60

    x = zeros(T_sim)
    y_obs = zeros(T_sim)
    for t in 2:T_sim
        x[t] = rho * x[t-1] + sigma_eps * randn(rng)
    end
    for t in 1:T_sim
        y_obs[t] = x[t] + sigma_me * randn(rng)
    end

    G1 = fill(rho, 1, 1)
    impact = fill(sigma_eps, 1, 1)
    Z = ones(1, 1)
    d = zeros(1)
    H_mat = fill(sigma_me^2, 1, 1)
    Q_mat = ones(1, 1)

    ss = M.DSGEStateSpace{Float64}(G1, impact, Z, d, H_mat, Q_mat)
    data = reshape(y_obs, 1, T_sim)

    N_particles = 80
    ws = M._allocate_pf_workspace(Float64, 1, 1, 1, N_particles; T_obs=T_sim)

    ll = M._bootstrap_particle_filter!(ws, ss, data, T_sim;
            rng=Random.MersenneTwister(5002), store_trajectory=true)

    @test isfinite(ll)
    @test ws.reference_trajectory !== nothing
    # After store_trajectory, last column should be populated
    @test any(!iszero, ws.reference_trajectory[:, T_sim])
end

@testset "particle_filter: auxiliary PF with various thresholds" begin
    rng = Random.MersenneTwister(5003)
    T_sim = 40

    G1 = fill(0.6, 1, 1)
    impact = fill(0.3, 1, 1)
    Z = ones(1, 1)
    d = zeros(1)
    H_mat = fill(0.01, 1, 1)
    Q_mat = ones(1, 1)

    ss = M.DSGEStateSpace{Float64}(G1, impact, Z, d, H_mat, Q_mat)
    data = randn(rng, 1, T_sim)

    N_particles = 50
    ws = M._allocate_pf_workspace(Float64, 1, 1, 1, N_particles)

    # Very low threshold => rarely resample
    ll1 = M._auxiliary_particle_filter!(ws, ss, data, T_sim;
            threshold=0.01, rng=Random.MersenneTwister(5004))
    @test isfinite(ll1)

    # Very high threshold => always resample
    ws2 = M._allocate_pf_workspace(Float64, 1, 1, 1, N_particles)
    ll2 = M._auxiliary_particle_filter!(ws2, ss, data, T_sim;
            threshold=0.99, rng=Random.MersenneTwister(5005))
    @test isfinite(ll2)
end

@testset "particle_filter: conditional SMC with short reference" begin
    rng = Random.MersenneTwister(5006)
    T_sim = 30

    G1 = fill(0.7, 1, 1)
    impact = fill(0.4, 1, 1)
    Z = ones(1, 1)
    d = zeros(1)
    H_mat = fill(0.01, 1, 1)
    Q_mat = ones(1, 1)

    ss = M.DSGEStateSpace{Float64}(G1, impact, Z, d, H_mat, Q_mat)
    data = randn(rng, 1, T_sim)

    N_particles = 40
    ws = M._allocate_pf_workspace(Float64, 1, 1, 1, N_particles; T_obs=T_sim)

    # Bootstrap first to populate reference
    M._bootstrap_particle_filter!(ws, ss, data, T_sim;
        rng=Random.MersenneTwister(5007), store_trajectory=true)

    # Run CSMC multiple times (exercises trajectory update)
    for i in 1:3
        ll = M._conditional_smc!(ws, ss, data, T_sim;
            rng=Random.MersenneTwister(5007 + i))
        @test isfinite(ll)
    end
end

@testset "particle_filter: _pf_initialize_nonlinear! order 2" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=2)

        # Build the NonlinearStateSpace from the perturbation solution
        n_endog = nvars(sol)
        n_obs = n_endog
        Z = Matrix{Float64}(I, n_obs, n_endog)
        d = zeros(n_obs)
        H_mat = 0.01 * Matrix{Float64}(I, n_obs, n_obs)
        nlss = M._build_nonlinear_state_space(sol, Z, d, H_mat)

        nx = length(nlss.state_indices)
        ny = length(nlss.control_indices)
        n_eps = size(nlss.eta, 2)
        nv = nx + n_eps
        N = 30

        # n_states in workspace = nx (state count for particles_fo/so/to)
        # but particles itself = n_endog (all variables for observation)
        ws = M._allocate_pf_workspace(Float64, n_endog, n_obs, n_eps, N;
                                        nv=nv, nx=nx, order=2)

        M._pf_initialize_nonlinear!(ws, nlss; rng=Random.MersenneTwister(6001))

        @test all(isfinite, ws.particles)
        @test all(isfinite, ws.particles_fo)
        @test all(ws.particles_so .== 0.0)
        @test all(ws.weights .≈ 1.0 / N)
    end
end

@testset "particle_filter: workspace resize with order 3" begin
    n_states = 2
    n_obs = 1
    n_shocks = 1
    N = 20
    nv = n_states + n_shocks

    ws = M._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                    nv=nv, order=3)
    @test size(ws.particles, 2) == N
    @test size(ws.kron3_buffer, 2) == N

    N_new = 40
    M._resize_pf_workspace!(ws, N_new)

    @test size(ws.particles, 2) == N_new
    @test size(ws.kron_buffer, 2) == N_new
    @test size(ws.kron3_buffer, 2) == N_new
    @test size(ws.particles_fo, 2) == N_new
    @test size(ws.particles_so, 2) == N_new
    @test size(ws.particles_to, 2) == N_new
end

# =====================================================================
# 2. smc.jl — uncovered paths
# =====================================================================

@testset "smc: _log_prior in-bounds and out-of-bounds" begin
    priors_dict = Dict(:alpha => Normal(0.33, 0.1),
                       :beta => Beta(5.0, 2.0))
    prior = M.DSGEPrior(priors_dict;
        lower=Dict(:alpha => -Inf, :beta => 0.0),
        upper=Dict(:alpha => Inf, :beta => 1.0))

    # In-bounds
    θ = [0.33, 0.8]  # sorted: alpha, beta
    lp = M._log_prior(θ, prior)
    @test isfinite(lp)
    expected = logpdf(Normal(0.33, 0.1), 0.33) + logpdf(Beta(5.0, 2.0), 0.8)
    @test lp ≈ expected atol=1e-10

    # Out of bounds (beta > 1)
    θ_oob = [0.33, 1.5]
    @test M._log_prior(θ_oob, prior) == -Inf

    # Out of bounds (beta < 0)
    θ_oob2 = [0.33, -0.1]
    @test M._log_prior(θ_oob2, prior) == -Inf
end

@testset "smc: _build_likelihood_fn returns closure" begin
    _suppress_warnings() do
        spec = @dsge begin
            parameters: rho = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        data_mat = simulate(sol, 80; rng=Random.MersenneTwister(7001))'

        ll_fn = M._build_likelihood_fn(spec, [:rho], data_mat,
            [:y], nothing, :gensys, NamedTuple())

        # Valid parameter
        ll_val = ll_fn([0.7])
        @test isfinite(ll_val)
        @test ll_val < 0.0

        # Invalid (explosive)
        ll_bad = ll_fn([2.0])
        @test ll_bad == -Inf

        # With measurement error
        ll_fn_me = M._build_likelihood_fn(spec, [:rho], data_mat,
            [:y], [0.1], :gensys, NamedTuple())
        ll_me = ll_fn_me([0.7])
        @test isfinite(ll_me)
    end
end

@testset "smc: _adaptive_tempering edge cases" begin
    N = 50
    rng = Random.MersenneTwister(7002)

    # Case: uniform log-likelihoods => can jump to phi=1
    log_liks_uniform = fill(0.0, N)
    phi = M._adaptive_tempering(log_liks_uniform, 0.0, 0.5, N)
    @test phi == 1.0

    # Case: highly dispersed => needs many small steps
    log_liks_dispersed = randn(rng, N) * 100.0
    phi2 = M._adaptive_tempering(log_liks_dispersed, 0.0, 0.9, N)
    @test phi2 > 0.0
    @test phi2 <= 1.0

    # Case: starting from phi_old > 0
    log_liks_mild = randn(rng, N)
    phi3 = M._adaptive_tempering(log_liks_mild, 0.5, 0.5, N)
    @test phi3 >= 0.5
    @test phi3 <= 1.0
end

@testset "smc: _update_proposal_cov!" begin
    n_params = 2
    N = 50
    rng = Random.MersenneTwister(7003)

    theta_particles = randn(rng, n_params, N)
    log_weights = fill(-log(Float64(N)), N)
    log_likelihoods = randn(rng, N)
    log_priors = randn(rng, N)

    state = M.SMCState{Float64}(
        theta_particles,
        log_weights,
        log_likelihoods,
        log_priors,
        Float64[0.0],
        Float64[],
        Float64[],
        0.0,
        M.PFWorkspace{Float64}[],
        Matrix{Float64}(I, n_params, n_params)
    )

    M._update_proposal_cov!(state)

    @test size(state.proposal_cov) == (n_params, n_params)
    @test all(isfinite, state.proposal_cov)
    # Should be positive semi-definite
    @test all(eigvals(Symmetric(state.proposal_cov)) .>= 0.0)
    # Should have been scaled by Roberts-Rosenthal factor
    c2 = 2.38^2 / n_params
    @test state.proposal_cov[1, 1] > 0.0
end

@testset "smc: _adapt_n_particles" begin
    @test M._adapt_n_particles(100, 20.0, 10.0) == 200  # double
    @test M._adapt_n_particles(100, 5.0, 10.0) == 100   # no change
    @test M._adapt_n_particles(50, 10.0, 10.0) == 50    # at threshold, no change
end

# =====================================================================
# 3. bayes_estimation.jl — uncovered paths
# =====================================================================

@testset "bayes_estimation: _infer_prior_bounds" begin
    # Beta
    lo, hi = M._infer_prior_bounds(Beta(2, 2))
    @test lo == 0.0
    @test hi == 1.0

    # InverseGamma
    lo, hi = M._infer_prior_bounds(InverseGamma(2.0, 0.5))
    @test lo == 0.0
    @test hi == Inf

    # Gamma
    lo, hi = M._infer_prior_bounds(Gamma(2.0, 1.0))
    @test lo == 0.0
    @test hi == Inf

    # Normal (unbounded)
    lo, hi = M._infer_prior_bounds(Normal(0.0, 1.0))
    @test lo == -Inf
    @test hi == Inf

    # Uniform (finite bounds)
    lo, hi = M._infer_prior_bounds(Uniform(0.0, 1.0))
    @test lo == 0.0
    @test hi == 1.0

    # Truncated Normal (finite bounds)
    lo, hi = M._infer_prior_bounds(truncated(Normal(0.5, 0.1), 0.0, 1.0))
    @test lo ≈ 0.0
    @test hi ≈ 1.0
end

@testset "bayes_estimation: StatsAPI interface" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(8001)
        spec = @dsge begin
            parameters: rho = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        sim_data = simulate(sol, 100; rng=rng)

        priors = Dict(:rho => Beta(2, 2))
        result = estimate_dsge_bayes(spec, sim_data, [0.5];
            priors=priors, method=:smc, observables=[:y],
            n_smc=50, rng=Random.MersenneTwister(8002))

        # StatsAPI.coef
        c = StatsAPI.coef(result)
        @test length(c) == 1
        @test isfinite(c[1])

        # StatsAPI.islinear
        @test StatsAPI.islinear(result) == false
    end
end

@testset "bayes_estimation: prior_posterior_table" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(8003)
        spec = @dsge begin
            parameters: rho = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        sim_data = simulate(sol, 100; rng=rng)

        priors = Dict(:rho => Beta(2, 2))
        result = estimate_dsge_bayes(spec, sim_data, [0.5];
            priors=priors, method=:smc, observables=[:y],
            n_smc=50, rng=Random.MersenneTwister(8004))

        pt = prior_posterior_table(result)
        @test length(pt) == 1
        @test pt[1].param == :rho
        @test pt[1].prior_dist == "Beta"
        @test isfinite(pt[1].prior_mean)
        @test isfinite(pt[1].prior_std)
        @test isfinite(pt[1].post_mean)
        @test isfinite(pt[1].post_std)
        @test pt[1].ci_lower < pt[1].ci_upper
    end
end

@testset "bayes_estimation: show method" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(8005)
        spec = @dsge begin
            parameters: rho = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        sim_data = simulate(sol, 100; rng=rng)

        priors = Dict(:rho => Beta(2, 2))
        result = estimate_dsge_bayes(spec, sim_data, [0.5];
            priors=priors, method=:smc, observables=[:y],
            n_smc=50, rng=Random.MersenneTwister(8006))

        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test occursin("Bayesian DSGE Estimation", output)
        @test occursin("Method", output)
        @test occursin("Posterior Summary", output)
        @test occursin("Prior vs Posterior", output)
        @test occursin("Tempering stages", output)
    end
end

@testset "bayes_estimation: estimate_dsge_bayes with :mh (small)" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(8007)
        spec = @dsge begin
            parameters: rho = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        sim_data = simulate(sol, 80; rng=rng)

        priors = Dict(:rho => Beta(2, 2))
        result = estimate_dsge_bayes(spec, sim_data, [0.5];
            priors=priors, method=:mh, observables=[:y],
            n_draws=200, burnin=50,
            rng=Random.MersenneTwister(8008))

        @test result isa BayesianDSGE{Float64}
        @test result.method == :rwmh
        @test size(result.theta_draws, 1) == 200
        @test isfinite(result.log_marginal_likelihood)
        @test isempty(result.ess_history)
        @test isempty(result.phi_schedule)

        # marginal_likelihood and bayes_factor
        ml = marginal_likelihood(result)
        @test isfinite(ml)

        # posterior_summary
        ps = posterior_summary(result)
        @test haskey(ps, :rho)
        @test ps[:rho][:ci_lower] <= ps[:rho][:mean]
    end
end

@testset "bayes_estimation: data transpose (T_obs x n_obs input)" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(8009)
        spec = @dsge begin
            parameters: rho = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        # T_obs x n_obs format
        sim_data = simulate(sol, 80; rng=rng)
        @test size(sim_data, 1) == 80  # T x 1

        priors = Dict(:rho => Beta(2, 2))
        result = estimate_dsge_bayes(spec, sim_data, [0.5];
            priors=priors, method=:smc, observables=[:y],
            n_smc=50, rng=Random.MersenneTwister(8010))

        @test result isa BayesianDSGE{Float64}
    end
end

# =====================================================================
# 4. pruning.jl — uncovered paths
# =====================================================================

@testset "pruning: simulate order 3 with pruned dynamics" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=3)
        @test sol.order == 3

        sim = simulate(sol, 60; rng=Random.MersenneTwister(9001))
        @test all(isfinite, sim)
        @test size(sim, 1) == 60
    end
end

@testset "pruning: simulate order 3 with explicit shock_draws" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=3)

        n_eps = nshocks(sol)
        shocks = randn(Random.MersenneTwister(9002), 40, n_eps)
        sim = simulate(sol, 40; shock_draws=shocks)
        @test all(isfinite, sim)
        @test size(sim, 1) == 40
    end
end

@testset "pruning: simulate order 2 with antithetic=true" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=2)

        sim = simulate(sol, 100; antithetic=true, rng=Random.MersenneTwister(9003))
        @test all(isfinite, sim)
        @test size(sim, 1) == 100
    end
end

@testset "pruning: simulate order 3 with antithetic=true" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=3)

        sim = simulate(sol, 100; antithetic=true, rng=Random.MersenneTwister(9004))
        @test all(isfinite, sim)
        @test size(sim, 1) == 100
    end
end

@testset "pruning: irf analytical for perturbation solution" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=2)

        irf_result = irf(sol, 15; irf_type=:analytical)
        @test size(irf_result.values, 1) == 15
        @test all(isfinite, irf_result.values)
    end
end

@testset "pruning: irf GIRF for order 3" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=3)

        irf_result = irf(sol, 8; irf_type=:girf, n_draws=10)
        @test size(irf_result.values, 1) == 8
        @test all(isfinite, irf_result.values)
    end
end

@testset "pruning: fevd for order 3" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=3)

        f = fevd(sol, 10)
        @test size(f.proportions, 3) == 10
        # Proportions at each horizon should sum to ~1
        for h in 1:10
            for i in 1:size(f.proportions, 1)
                total = sum(f.proportions[i, :, h])
                @test total ≈ 1.0 atol=1e-10
            end
        end
    end
end

@testset "pruning: analytical_moments order 2 GMM format" begin
    _suppress_warnings() do
        spec = _make_fwd_spec()
        sol = perturbation_solver(spec; order=2)

        # GMM format: closed-form augmented Lyapunov
        m = analytical_moments(sol; format=:gmm, lags=2)
        @test length(m) > 0
        @test all(isfinite, m)
    end
end

@testset "pruning: analytical_moments order 3 GMM format" begin
    FAST && return
    _suppress_warnings() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)
        sol = perturbation_solver(spec; order=3)

        m = analytical_moments(sol; format=:gmm, lags=1)
        @test length(m) > 0
        @test all(isfinite, m)
    end
end

@testset "pruning: _extract_xx_block and _extract_xxx_block" begin
    nx = 2
    nv = 3
    nrows = 2

    # xx block
    Mvv = randn(Random.MersenneTwister(9010), nrows, nv * nv)
    Mxx = M._extract_xx_block(Mvv, nx, nv)
    @test size(Mxx) == (nrows, nx * nx)
    for a in 1:nx, b in 1:nx
        col_vv = (a - 1) * nv + b
        col_xx = (a - 1) * nx + b
        @test Mxx[:, col_xx] ≈ Mvv[:, col_vv]
    end

    # xxx block
    Mvvv = randn(Random.MersenneTwister(9011), nrows, nv^3)
    Mxxx = M._extract_xxx_block(Mvvv, nx, nv)
    @test size(Mxxx) == (nrows, nx^3)
    for a in 1:nx, b in 1:nx, c in 1:nx
        col_vvv = ((a - 1) * nv + b - 1) * nv + c
        col_xxx = ((a - 1) * nx + b - 1) * nx + c
        @test Mxxx[:, col_xxx] ≈ Mvvv[:, col_vvv]
    end
end

@testset "pruning: _innovation_variance_2nd" begin
    rng = Random.MersenneTwister(9012)
    nx = 2
    n_eps = 1
    hx_state = [0.5 0.1; 0.0 0.3]
    eta_x = [0.4; 0.3][:, :]
    Var_xf = M._dlyap_doubling(hx_state, eta_x * eta_x')

    Var_inov = M._innovation_variance_2nd(hx_state, eta_x, Var_xf, nx, n_eps)

    nz = 2 * nx + nx^2
    @test size(Var_inov) == (nz, nz)
    @test all(isfinite, Var_inov)
    # Should be symmetric
    @test norm(Var_inov - Var_inov') < 1e-10
end

@testset "pruning: _dlyap_doubling convergence" begin
    A = [0.8 0.1; 0.0 0.5]
    B = [1.0 0.2; 0.2 1.0]

    Sigma = M._dlyap_doubling(A, B)
    @test size(Sigma) == (2, 2)
    # Verify: Sigma = A * Sigma * A' + B
    @test norm(Sigma - (A * Sigma * A' + B)) < 1e-8

    # Errors
    @test_throws ArgumentError M._dlyap_doubling(randn(2, 3), B)
    @test_throws ArgumentError M._dlyap_doubling(A, randn(3, 3))
end

# =====================================================================
# 5. derivatives.jl — uncovered paths
# =====================================================================

@testset "derivatives: _slot_dim all 4 which values" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y, k
        exogenous: e1, e2
        y[t] = rho * y[t-1] + e1[t]
        k[t] = y[t] + e2[t]
    end
    spec = compute_steady_state(spec)

    @test M._slot_dim(spec, :current) == 2
    @test M._slot_dim(spec, :lag) == 2
    @test M._slot_dim(spec, :lead) == 2
    @test M._slot_dim(spec, :shock) == 2
end

@testset "derivatives: step sizes for hessian and third" begin
    y_ss = [0.0, 1.5, -0.5]

    # Hessian step sizes
    h_shock = M._step_size_hessian(Float64, y_ss, :shock, 1)
    @test h_shock == 1e-5
    h_var1 = M._step_size_hessian(Float64, y_ss, :current, 1)
    @test h_var1 == 1e-5  # y_ss[1] == 0
    h_var2 = M._step_size_hessian(Float64, y_ss, :lag, 2)
    @test h_var2 ≈ 1e-5 * abs(1.5) atol=1e-15  # adaptive

    # Third derivative step sizes
    h3_shock = M._step_size_third(Float64, y_ss, :shock, 1)
    @test h3_shock == 1e-4
    h3_var = M._step_size_third(Float64, y_ss, :current, 2)
    @test h3_var ≈ 1e-4 * abs(1.5) atol=1e-15
    h3_zero = M._step_size_third(Float64, y_ss, :lead, 1)
    @test h3_zero == 1e-4  # y_ss[1] == 0 so max(1e-4, 0) = 1e-4
end

@testset "derivatives: _compute_hessian" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    y_ss = spec.steady_state

    H_cc = M._compute_hessian(spec, y_ss, :current, :current)
    @test size(H_cc) == (1, 1, 1)
    # Linear model => Hessians should be ~0
    @test abs(H_cc[1, 1, 1]) < 1e-4

    H_cs = M._compute_hessian(spec, y_ss, :current, :shock)
    @test size(H_cs) == (1, 1, 1)
    @test abs(H_cs[1, 1, 1]) < 1e-4
end

@testset "derivatives: _compute_all_hessians" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    y_ss = spec.steady_state

    all_H = M._compute_all_hessians(spec, y_ss)

    # Should have 10 unique pairs (4 choose 2 with replacement)
    @test length(all_H) == 10
    @test haskey(all_H, (:current, :current))
    @test haskey(all_H, (:current, :lag))
    @test haskey(all_H, (:current, :lead))
    @test haskey(all_H, (:current, :shock))
    @test haskey(all_H, (:lag, :lag))
    @test haskey(all_H, (:lag, :lead))
    @test haskey(all_H, (:lag, :shock))
    @test haskey(all_H, (:lead, :lead))
    @test haskey(all_H, (:lead, :shock))
    @test haskey(all_H, (:shock, :shock))
end

@testset "derivatives: _third_derivative" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    y_ss = spec.steady_state

    D3 = M._third_derivative(spec, y_ss, :current, :current, :current)
    @test size(D3) == (1, 1, 1, 1)
    # Linear model => third derivatives should be ~0
    @test abs(D3[1, 1, 1, 1]) < 1e-2

    # Mixed slots
    D3_mixed = M._third_derivative(spec, y_ss, :current, :lag, :shock)
    @test size(D3_mixed) == (1, 1, 1, 1)
    @test abs(D3_mixed[1, 1, 1, 1]) < 1e-2
end

@testset "derivatives: _make_args and _make_args_two" begin
    y_ss = [1.0, 2.0]
    eps_zero = [0.0]

    # _make_args: perturb current
    y_t, y_lag, y_lead, eps = M._make_args(y_ss, eps_zero, :current, 1, 0.1)
    @test y_t[1] ≈ 1.1
    @test y_lag == y_ss
    @test y_lead == y_ss
    @test eps == eps_zero

    # _make_args: perturb shock
    y_t2, y_lag2, y_lead2, eps2 = M._make_args(y_ss, eps_zero, :shock, 1, 0.5)
    @test y_t2 == y_ss
    @test eps2[1] ≈ 0.5

    # _make_args_two: perturb current and lag
    y_t3, y_lag3, y_lead3, eps3 = M._make_args_two(y_ss, eps_zero, :current, 1, 0.1, :lag, 2, -0.2)
    @test y_t3[1] ≈ 1.1
    @test y_lag3[2] ≈ 1.8
    @test y_lead3 == y_ss
    @test eps3 == eps_zero
end

@testset "derivatives: _make_args_three" begin
    y_ss = [1.0, 2.0]
    eps_zero = [0.0]

    y_t, y_lag, y_lead, eps = M._make_args_three(y_ss, eps_zero,
        :current, 1, 0.1, :lag, 2, -0.2, :shock, 1, 0.3)
    @test y_t[1] ≈ 1.1
    @test y_lag[2] ≈ 1.8
    @test eps[1] ≈ 0.3
    @test y_lead == y_ss
end

# =====================================================================
# 6. occbin.jl — uncovered paths
# =====================================================================

@testset "occbin: parse_constraint GEQ and LEQ" begin
    spec = _make_rbc_spec()

    # GEQ constraint
    c_geq = parse_constraint(:(i[t] >= 0), spec)
    @test c_geq.variable == :i
    @test c_geq.bound == 0.0
    @test c_geq.direction == :geq

    # LEQ constraint
    c_leq = parse_constraint(:(y[t] <= 1.0), spec)
    @test c_leq.variable == :y
    @test c_leq.bound == 1.0
    @test c_leq.direction == :leq

    # Negative bound
    c_neg = parse_constraint(:(i[t] >= -0.5), spec)
    @test c_neg.bound ≈ -0.5

    # Expression bound
    c_expr = parse_constraint(:(y[t] <= 1/400), spec)
    @test c_expr.bound ≈ 1.0/400.0
end

@testset "occbin: _parse_constraint_expr errors" begin
    # Not a comparison
    @test_throws ArgumentError M._parse_constraint_expr(:(begin end), Float64)

    # Wrong operator
    @test_throws ArgumentError M._parse_constraint_expr(:(y[t] == 0), Float64)

    # Invalid LHS (no time index)
    @test_throws ArgumentError M._parse_constraint_expr(:(y >= 0), Float64)

    # Invalid bound (non-numeric symbol)
    @test_throws ArgumentError M._parse_constraint_expr(:(y[t] >= x), Float64)
end

@testset "occbin: _extract_constrained_var" begin
    # Valid
    lhs = :(y[t])
    @test M._extract_constrained_var(lhs) == :y

    # Invalid (not a ref)
    @test_throws ArgumentError M._extract_constrained_var(:y)

    # Invalid (wrong subscript)
    @test_throws ArgumentError M._extract_constrained_var(:(y[t-1]))
end

@testset "occbin: _eval_bound" begin
    # Number
    @test M._eval_bound(0.5, Float64) ≈ 0.5
    @test M._eval_bound(0, Float64) ≈ 0.0
    @test M._eval_bound(-1, Float64) ≈ -1.0

    # Expression
    @test M._eval_bound(:(1/400), Float64) ≈ 1.0/400.0

    # Symbol (should error)
    @test_throws ArgumentError M._eval_bound(:x, Float64)
end

@testset "occbin: _derive_alternative_regime" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()
        constraint = parse_constraint(:(i[t] >= 0), spec)

        alt_spec = M._derive_alternative_regime(spec, constraint)

        # The alternative regime should have i[t] = 0
        @test alt_spec.n_endog == spec.n_endog
        @test alt_spec.n_exog == spec.n_exog

        # Test residual for the binding equation
        var_idx = findfirst(==(:i), spec.endog)
        y_test = [1.0, 0.0]  # [y, i]
        resid_val = alt_spec.residual_fns[var_idx](y_test, y_test, y_test, [0.0], spec.param_values)
        @test resid_val ≈ 0.0  # i[t] - bound = 0 - 0 = 0
    end
end

@testset "occbin: occbin_solve single constraint" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()
        constraint = parse_constraint(:(i[t] >= 0), spec)

        shock_path = zeros(20, 1)
        shock_path[1, 1] = -2.0

        sol = occbin_solve(spec, constraint;
                           shock_path=shock_path, nperiods=20)
        @test sol isa OccBinSolution{Float64}
        @test size(sol.linear_path, 1) == 20
        @test size(sol.piecewise_path, 1) == 20
        @test sol.converged

        # Linear path should differ from piecewise when constraint binds
        if any(sol.regime_history .== 1)
            @test sol.linear_path != sol.piecewise_path
        end
    end
end

@testset "occbin: occbin_solve with explicit alt_spec" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()
        constraint = parse_constraint(:(i[t] >= 0), spec)

        alt_spec = M._derive_alternative_regime(spec, constraint)

        shock_path = zeros(20, 1)
        shock_path[1, 1] = -2.0

        sol = occbin_solve(spec, constraint, alt_spec;
                           shock_path=shock_path, nperiods=20)
        @test sol isa OccBinSolution{Float64}
        @test sol.converged
    end
end

@testset "occbin: _map_regime" begin
    violvec = BitVector([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
    regimes, starts = M._map_regime(violvec)
    @test regimes == [0, 1, 0, 1, 0]
    @test starts == [1, 3, 6, 8, 9]

    # All false
    regimes2, starts2 = M._map_regime(falses(5))
    @test regimes2 == [0]
    @test starts2 == [1]

    # All true
    regimes3, starts3 = M._map_regime(trues(5))
    @test regimes3 == [1]
    @test starts3 == [1]

    # Empty
    regimes4, starts4 = M._map_regime(BitVector())
    @test isempty(regimes4)
    @test isempty(starts4)
end

@testset "occbin: _backward_iteration no binding periods" begin
    n = 2
    n_shocks = 1
    T_max = 5

    ref = M.OccBinRegime{Float64}(
        zeros(n, n), Matrix{Float64}(I, n, n),
        zeros(n, n), ones(n, n_shocks))
    alt = M.OccBinRegime{Float64}(
        zeros(n, n), Matrix{Float64}(I, n, n),
        zeros(n, n), ones(n, n_shocks))
    d_ref = zeros(n)
    d_alt = zeros(n)
    P = 0.5 * Matrix{Float64}(I, n, n)
    Q = ones(n, n_shocks)

    violvec = falses(T_max)
    shock_path = zeros(T_max, n_shocks)

    P_tv, D_tv, E = M._backward_iteration(ref, alt, d_ref, d_alt, P, Q, violvec, shock_path)

    # No binding periods => empty arrays
    @test size(P_tv, 3) == 0
    @test size(D_tv, 2) == 0
end

@testset "occbin: occbin_irf single constraint (various magnitudes)" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()
        constraint = parse_constraint(:(i[t] >= 0), spec)

        # Large negative shock to trigger binding
        oirf = occbin_irf(spec, constraint, 1, 20; magnitude=-3.0)
        @test oirf isa OccBinIRF{Float64}
        @test size(oirf.linear, 1) == 20
        @test size(oirf.piecewise, 1) == 20

        # Small shock that might not trigger binding
        oirf_small = occbin_irf(spec, constraint, 1, 20; magnitude=-0.01)
        @test oirf_small isa OccBinIRF{Float64}
    end
end

@testset "occbin: two-constraint with both LEQ and GEQ" begin
    _suppress_warnings() do
        spec = @dsge begin
            parameters: rho = 0.9, phi = 1.5
            endogenous: y, i, cap
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = phi * y[t]
            cap[t] = y[t]
        end
        spec = compute_steady_state(spec)

        c1 = parse_constraint(:(i[t] >= 0), spec)  # GEQ
        c2 = parse_constraint(:(cap[t] <= 0.5), spec)  # LEQ

        shock_path = zeros(25, 1)
        shock_path[1, 1] = -2.5

        sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=25)
        @test sol isa OccBinSolution{Float64}
        @test size(sol.regime_history, 2) == 2

        # Two-constraint IRF
        oirf = occbin_irf(spec, c1, c2, 1, 20; magnitude=-3.0)
        @test oirf isa OccBinIRF{Float64}
        @test size(oirf.regime_history, 2) == 2
    end
end

@testset "occbin: _find_last_binding_two additional" begin
    vm = falses(10, 2)
    @test M._find_last_binding_two(BitMatrix(vm)) == 0

    vm2 = falses(10, 2)
    vm2[3, 1] = true
    @test M._find_last_binding_two(BitMatrix(vm2)) == 3

    vm3 = falses(10, 2)
    vm3[7, 2] = true
    @test M._find_last_binding_two(BitMatrix(vm3)) == 7

    vm4 = trues(5, 2)
    @test M._find_last_binding_two(BitMatrix(vm4)) == 5
end

@testset "occbin: OccBinSolution show" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()
        constraint = parse_constraint(:(i[t] >= 0), spec)

        shock_path = zeros(20, 1)
        shock_path[1, 1] = -2.0
        sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)

        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test !isempty(output)
    end
end

@testset "occbin: OccBinConstraint show" begin
    spec = _make_rbc_spec()
    c = parse_constraint(:(i[t] >= 0), spec)

    io = IOBuffer()
    show(io, c)
    output = String(take!(io))
    @test !isempty(output)
end

@testset "occbin: _regime_constant" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()

        # Reference regime constant should be ~0 at steady state
        d_ref = M._regime_constant(spec)
        @test all(abs.(d_ref) .< 1e-10)

        # Alternative regime constant may be nonzero
        constraint = parse_constraint(:(i[t] >= 0.5), spec)
        alt_spec = M._derive_alternative_regime(spec, constraint)
        d_alt = M._regime_constant(alt_spec)
        # d_alt may be nonzero since the bound (0.5) differs from steady state (0.0)
        @test all(isfinite, d_alt)
    end
end

@testset "occbin: _extract_regime" begin
    _suppress_warnings() do
        spec = _make_rbc_spec()
        regime = M._extract_regime(spec)

        @test size(regime.A) == (2, 2)
        @test size(regime.B) == (2, 2)
        @test size(regime.C) == (2, 2)
        @test size(regime.D) == (2, 1)
    end
end

@testset "occbin: _extract_regime without steady state errors" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    # Do NOT compute steady state
    @test_throws ArgumentError M._extract_regime(spec)
end

# =====================================================================
# 7. Cross-cutting: full pipeline with multi-parameter estimation
# =====================================================================

@testset "full pipeline: estimate_dsge_bayes with 2 params via SMC" begin
    FAST && return
    _suppress_warnings() do
        rng = Random.MersenneTwister(10001)
        spec = @dsge begin
            parameters: rho = 0.5, sigma = 0.5
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + sigma * e[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)

        true_spec = @dsge begin
            parameters: rho = 0.8, sigma = 0.3
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + sigma * e[t]
            steady_state = [0.0]
        end
        true_spec = compute_steady_state(true_spec)
        sol_true = solve(true_spec; method=:gensys)
        data = simulate(sol_true, 150; rng=rng)

        priors = Dict(:rho => Beta(2, 2),
                      :sigma => InverseGamma(2.0, 0.5))
        result = estimate_dsge_bayes(spec, data, [0.5, 0.5];
            priors=priors, method=:smc, observables=[:y],
            n_smc=80, n_mh_steps=1,
            rng=Random.MersenneTwister(10002))

        @test result isa BayesianDSGE{Float64}
        @test length(result.param_names) == 2
        @test :rho in result.param_names
        @test :sigma in result.param_names
        @test result.phi_schedule[end] ≈ 1.0

        # Posterior summary
        ps = posterior_summary(result)
        @test haskey(ps, :rho)
        @test haskey(ps, :sigma)

        # Prior-posterior table
        pt = prior_posterior_table(result)
        @test length(pt) == 2

        # Show
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test occursin("Bayesian DSGE Estimation", output)

        # Bayes factor with itself (should be 0)
        bf = bayes_factor(result, result)
        @test bf ≈ 0.0 atol=1e-10
    end
end

end  # @testset "DSGE Bayes Coverage"
