using Test, MacroEconometricModels, DataFrames, Distributions, Random, Statistics
using StatsAPI: coef, vcov, residuals, predict, nobs, stderror, confint, r2
using LinearAlgebra

@testset "Panel Covariance" begin
    # Setup: small panel N=10, T=20
    rng = Random.MersenneTwister(42)
    N_g = 10; T_p = 20; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    alpha = repeat(randn(rng, N_g), inner=T_p)
    y = alpha .+ 1.5 .* x1 .- 0.8 .* x2 .+ 0.3 .* randn(rng, n)

    df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1, :x2])

    # Extract internals for covariance testing
    X_dm = m.X  # stored X is original, but we can test via the model
    k = length(coef(m))

    @testset "Entity cluster SEs" begin
        se = stderror(m)
        @test length(se) == 2
        @test all(se .> 0)
    end

    @testset "Two-way cluster SEs" begin
        m2 = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:twoway)
        se2 = stderror(m2)
        @test length(se2) == 2
        @test all(se2 .> 0)
        # Two-way SEs should generally differ from entity-only
        @test se2 != stderror(m)
    end

    @testset "Driscoll-Kraay SEs" begin
        m3 = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:driscoll_kraay)
        se3 = stderror(m3)
        @test length(se3) == 2
        @test all(se3 .> 0)
    end

    @testset "OLS SEs" begin
        m4 = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:ols)
        se4 = stderror(m4)
        @test length(se4) == 2
        @test all(se4 .> 0)
    end

    @testset "Cluster vs classical SEs differ" begin
        m_ols = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:ols)
        m_clust = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:cluster)
        se_ols = stderror(m_ols)
        se_clust = stderror(m_clust)
        # Clustered SEs should generally be larger with group effects
        @test se_clust != se_ols
    end
end

@testset "estimate_xtreg - Fixed Effects" begin
    @testset "Coefficient recovery with entity FE" begin
        rng = Random.MersenneTwister(123)
        N_g = 50; T_p = 20; n = N_g * T_p
        beta_true = [1.5, -0.8]

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        alpha = repeat(randn(rng, N_g) .* 2.0, inner=T_p)
        y = alpha .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2])

        # Coefficients should be close to true values
        @test abs(coef(m)[1] - 1.5) < 0.1
        @test abs(coef(m)[2] - (-0.8)) < 0.1

        # R-squared
        @test m.r2_within > 0.8
        @test m.r2_between >= 0.0
        @test m.r2_overall >= 0.0

        # Dimensions
        @test m.n_groups == 50
        @test nobs(m) == 1000
        @test m.method == :fe
        @test m.twoway == false
        @test length(m.group_effects) == 50
    end

    @testset "between-R² not degenerate to 1.0 (B3/T172)" begin
        rng = Random.MersenneTwister(271)
        N_g = 20; T_p = 8; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x = repeat(randn(rng, N_g), inner=T_p) .+ 0.5 .* randn(rng, n)  # between + within
        alpha = repeat(randn(rng, N_g), inner=T_p)                     # entity fixed effect
        y = 1.0 .* x .+ alpha .+ 0.3 .* randn(rng, n)
        df = DataFrame(id=ids, t=ts, x=x, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x])
        # between/overall R² must NOT be trivially 1.0 (the α_i-absorption bug set it =1)
        @test 0 < m.r2_between < 1 - 1e-6
        @test 0 < m.r2_overall < 1 - 1e-6
        @test m.r2_within > 0
        # independent Stata-form pin: between = cor(ȳ_i, x̄_i)² (scale-invariant in β̂)
        y_g = [mean(y[ids .== i]) for i in 1:N_g]
        x_g = [mean(x[ids .== i]) for i in 1:N_g]
        @test isapprox(m.r2_between, cor(y_g, x_g)^2; atol=1e-6)
    end

    @testset "Two-way FE" begin
        rng = Random.MersenneTwister(456)
        N_g = 30; T_p = 15; n = N_g * T_p
        beta_true = [2.0, -1.0]

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        alpha_i = repeat(randn(rng, N_g) .* 1.5, inner=T_p)
        gamma_t = repeat(randn(rng, T_p) .* 0.5, N_g)
        y = alpha_i .+ gamma_t .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.3 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; twoway=true)

        # Coefficients should be close to true values
        @test abs(coef(m)[1] - 2.0) < 0.15
        @test abs(coef(m)[2] - (-1.0)) < 0.15
        @test m.twoway == true
        @test m.r2_within > 0.8
    end

    @testset "StatsAPI interface" begin
        rng = Random.MersenneTwister(789)
        N_g = 20; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1])

        @test length(coef(m)) == 1
        @test size(vcov(m)) == (1, 1)
        @test length(residuals(m)) == n
        @test length(predict(m)) == n
        @test nobs(m) == n
        @test length(stderror(m)) == 1
        @test stderror(m)[1] > 0

        ci = confint(m)
        @test size(ci) == (1, 2)
        @test ci[1, 1] < coef(m)[1] < ci[1, 2]

        @test r2(m) == m.r2_within
    end

    @testset "Display output" begin
        rng = Random.MersenneTwister(101)
        N_g = 10; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1])

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Fixed Effects", output)
        @test occursin("R-sq. within", output)
        @test occursin("Groups", output)
        @test occursin("x1", output)
    end

    @testset "Variance components" begin
        rng = Random.MersenneTwister(202)
        N_g = 40; T_p = 25; n = N_g * T_p

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g) .* 3.0, inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1])

        @test m.sigma_e > 0
        @test m.sigma_u > 0
        @test 0 <= m.rho <= 1
        # With large entity effects (sigma_u=3) and small noise (sigma_e=0.5),
        # rho should be large
        @test m.rho > 0.5
    end

    @testset "Input validation" begin
        rng = Random.MersenneTwister(303)
        N_g = 5; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        y = randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; model=:pooled)
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; cov_type=:invalid)
        @test_throws ArgumentError estimate_xtreg(pd, :nonexistent, [:x1])
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:nonexistent])
    end
end

@testset "estimate_xtreg — Random Effects" begin
    @testset "Coefficient recovery" begin
        # N=50, T=20, uncorrelated alpha_i with X
        rng = Random.MersenneTwister(5001)
        N_g = 50; T_p = 20; n = N_g * T_p
        beta_true = [1.5, -0.8]

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        # Random effects uncorrelated with regressors
        alpha = repeat(randn(rng, N_g) .* 2.0, inner=T_p)
        y = alpha .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)

        @test m.method == :re
        @test abs(coef(m)[1] - 1.5) < 0.15
        @test abs(coef(m)[2] - (-0.8)) < 0.15
        @test length(coef(m)) == 2
        @test nobs(m) == n
        @test m.n_groups == N_g
    end

    @testset "Variance components" begin
        rng = Random.MersenneTwister(5002)
        N_g = 50; T_p = 20; n = N_g * T_p

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g) .* 3.0, inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:re)

        @test m.sigma_u > 0
        @test m.sigma_e > 0
        @test m.theta !== nothing
        @test m.theta > 0
        @test 0 <= m.rho <= 1
        @test m.rho > 0.5  # large entity effects
    end

    @testset "R-squared variants" begin
        rng = Random.MersenneTwister(5003)
        N_g = 40; T_p = 15; n = N_g * T_p

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g) .* 1.5, inner=T_p)
        y = alpha .+ 2.0 .* x1 .+ 0.3 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:re)

        @test 0 <= m.r2_within <= 1
        @test 0 <= m.r2_between <= 1
        @test 0 <= m.r2_overall <= 1
    end

    @testset "Display output" begin
        rng = Random.MersenneTwister(5004)
        N_g = 10; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:re)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Random Effects", output)
        @test occursin("theta", output)
    end
end

@testset "estimate_xtreg — First Differences" begin
    @testset "Coefficient recovery" begin
        rng = Random.MersenneTwister(6001)
        N_g = 50; T_p = 20; n = N_g * T_p
        beta_true = [1.5, -0.8]

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        # Persistent X (random walk component) to ensure FD works well
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        alpha = repeat(randn(rng, N_g) .* 2.0, inner=T_p)
        y = alpha .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fd)

        @test m.method == :fd
        @test abs(coef(m)[1] - 1.5) < 0.2
        @test abs(coef(m)[2] - (-0.8)) < 0.2
        @test length(coef(m)) == 2
        # n_obs should be NT - N (one obs dropped per group)
        @test nobs(m) == N_g * (T_p - 1)
    end

    @testset "Handles time gaps" begin
        # Panel with a gap in time
        rng = Random.MersenneTwister(6002)
        N_g = 10; n_obs = N_g * 5
        ids = repeat(1:N_g, inner=5)
        # Time periods with a gap: 1,2,3,5,6 (skip 4)
        ts = repeat([1,2,3,5,6], N_g)
        x1 = randn(rng, n_obs)
        alpha = repeat(randn(rng, N_g), inner=5)
        y = alpha .+ 1.0 .* x1 .+ 0.3 .* randn(rng, n_obs)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:fd)

        # Should have N_g * 3 obs (only consecutive: 1->2, 2->3, 5->6)
        @test nobs(m) == N_g * 3
        @test m.method == :fd
    end

    @testset "Display output" begin
        rng = Random.MersenneTwister(6003)
        N_g = 10; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:fd)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("First-Difference", output)
    end
end

@testset "estimate_xtreg — Between" begin
    @testset "Coefficient recovery" begin
        # Between variation: time-invariant component drives identification
        rng = Random.MersenneTwister(7001)
        N_g = 100; T_p = 10; n = N_g * T_p
        beta_true = [2.0, -1.0]

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        # X with large between variation (group-level component)
        x1_between = repeat(randn(rng, N_g) .* 3.0, inner=T_p)
        x1_within = randn(rng, n) .* 0.5
        x1 = x1_between .+ x1_within
        x2_between = repeat(randn(rng, N_g) .* 3.0, inner=T_p)
        x2_within = randn(rng, n) .* 0.5
        x2 = x2_between .+ x2_within
        y = beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:between)

        @test m.method == :between
        @test abs(coef(m)[1] - 2.0) < 0.3
        @test abs(coef(m)[2] - (-1.0)) < 0.3
        @test length(coef(m)) == 2
        # n_obs = N (number of groups)
        @test nobs(m) == N_g
        @test m.n_groups == N_g
    end

    @testset "R-squared" begin
        rng = Random.MersenneTwister(7002)
        N_g = 80; T_p = 10; n = N_g * T_p

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = repeat(randn(rng, N_g) .* 2.0, inner=T_p) .+ randn(rng, n) .* 0.3
        y = 1.5 .* x1 .+ 0.3 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:between)

        @test m.r2_between > 0.5
    end

    @testset "Display output" begin
        rng = Random.MersenneTwister(7003)
        N_g = 20; T_p = 5; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        y = 1.0 .* x1 .+ randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:between)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Between", output)
    end
end

@testset "estimate_xtreg — CRE (Mundlak)" begin
    @testset "CRE slopes approximate FE slopes" begin
        rng = Random.MersenneTwister(8001)
        N_g = 50; T_p = 20; n = N_g * T_p
        beta_true = [1.5, -0.8]

        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        # Correlated effects (alpha depends on mean X)
        x1_means = repeat([mean(x1[((i-1)*T_p+1):(i*T_p)]) for i in 1:N_g], inner=T_p)
        alpha = repeat(randn(rng, N_g) .* 1.0, inner=T_p) .+ 0.5 .* x1_means
        y = alpha .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)

        m_fe = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
        m_cre = estimate_xtreg(pd, :y, [:x1, :x2]; model=:cre)

        @test m_cre.method == :cre
        # CRE original slopes should approximate FE slopes
        @test abs(coef(m_cre)[1] - coef(m_fe)[1]) < 0.3
        @test abs(coef(m_cre)[2] - coef(m_fe)[2]) < 0.3
    end

    @testset "Variable names include mean variables" begin
        rng = Random.MersenneTwister(8002)
        N_g = 20; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        y = alpha .+ 1.0 .* x1 .- 0.5 .* x2 .+ randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:cre)

        @test length(m.varnames) == 4
        @test m.varnames == ["x1", "x2", "x1_mean", "x2_mean"]
        @test length(coef(m)) == 4
    end

    @testset "Theta and variance components" begin
        rng = Random.MersenneTwister(8003)
        N_g = 40; T_p = 15; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g) .* 2.0, inner=T_p)
        y = alpha .+ 1.5 .* x1 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:cre)

        @test m.theta !== nothing
        @test m.theta > 0
        @test m.sigma_u > 0
        @test m.sigma_e > 0
        @test 0 <= m.rho <= 1
    end

    @testset "Display output" begin
        rng = Random.MersenneTwister(8004)
        N_g = 10; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        y = alpha .+ 1.0 .* x1 .+ randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:cre)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Correlated RE", output)
        @test occursin("x1_mean", output)
    end
end

@testset "estimate_xtreg -- Arellano-Bond" begin
    rng = Random.MersenneTwister(9001)
    N_g = 100; T_p = 20; n = N_g * T_p

    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)

    # Dynamic panel DGP: y_t = 0.3*y_{t-1} + 0.5*x_t + alpha_i + error
    alpha = repeat(randn(rng, N_g) .* 0.5, inner=T_p)
    x = randn(rng, n)
    y = zeros(n)
    for g in 1:N_g
        base = (g - 1) * T_p
        y[base + 1] = alpha[base + 1] + 0.5 * x[base + 1] + 0.3 * randn(rng)
        for t in 2:T_p
            y[base + t] = 0.3 * y[base + t - 1] + 0.5 * x[base + t] + alpha[base + t] + 0.3 * randn(rng)
        end
    end

    df = DataFrame(id=ids, t=ts, x=x, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtreg(pd, :y, [:x]; model=:ab)
    @test m isa PanelRegModel{Float64}
    @test m.method == :ab
    @test length(coef(m)) >= 2  # L.y + x
    @test length(m.varnames) >= 2
    @test m.varnames[1] == "L.y"
    @test m.varnames[2] == "x"
    # Coefficients should be near DGP (0.3, 0.5)
    @test abs(coef(m)[1] - 0.3) < 0.15
    @test abs(coef(m)[2] - 0.5) < 0.15

    # (T083) Full GMM coefficient covariance (was diagonal-only)
    V = vcov(m)
    @test size(V) == (length(coef(m)), length(coef(m)))
    @test isapprox(V, V')                                  # symmetric
    @test isapprox(diag(V), stderror(m) .^ 2)              # SEs unchanged (diagonal)
    @test any(abs.(V - Diagonal(diag(V))) .> 0)            # off-diagonals nonzero — the fix
    # joint Wald now uses the off-diagonals ⇒ differs from the diagonal-only version
    W_full = coef(m)' * inv(V) * coef(m)
    W_diag = sum(coef(m) .^ 2 ./ diag(V))
    @test !isapprox(W_full, W_diag; rtol=1e-6)

    # (T083) Arellano-Bond serial-correlation diagnostics: reject AR(1), not AR(2)
    d = m.dynamic_diagnostics
    @test d !== nothing
    @test d.ar1_p < 0.05          # FD of iid error is MA(1) ⇒ negative AR(1) ⇒ reject
    @test abs(d.ar1) > 2
    @test d.ar2_p > 0.05          # zero lag-2 autocovariance ⇒ do NOT reject
    @test abs(d.ar2) < 2.5
    ar2 = arellano_bond_ar_test(m; order=2)
    @test ar2.statistic == d.ar2 && ar2.pvalue == d.ar2_p
    @test arellano_bond_ar_test(m; order=1).statistic == d.ar1

    # (T083) Hansen J overidentification
    @test d.hansen_df == d.n_instruments - length(coef(m))
    @test d.hansen_df > 0
    @test 0 <= d.hansen_p <= 1
    @test d.hansen_p > 0.01       # valid instruments in this correctly-specified DGP

    # (T083) report surfaces the diagnostics
    io = IOBuffer(); show(io, m); s = String(take!(io))
    @test occursin("AR(2)", s)
    @test occursin("Hansen", s)
end

@testset "estimate_xtreg -- Blundell-Bond" begin
    rng = Random.MersenneTwister(9002)
    N_g = 100; T_p = 20; n = N_g * T_p

    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)

    alpha = repeat(randn(rng, N_g) .* 0.5, inner=T_p)
    x = randn(rng, n)
    y = zeros(n)
    for g in 1:N_g
        base = (g - 1) * T_p
        y[base + 1] = alpha[base + 1] + 0.5 * x[base + 1] + 0.3 * randn(rng)
        for t in 2:T_p
            y[base + t] = 0.3 * y[base + t - 1] + 0.5 * x[base + t] + alpha[base + t] + 0.3 * randn(rng)
        end
    end

    df = DataFrame(id=ids, t=ts, x=x, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtreg(pd, :y, [:x]; model=:bb)
    @test m isa PanelRegModel{Float64}
    @test m.method == :bb
    @test length(coef(m)) >= 2
    @test m.varnames[1] == "L.y"
    @test m.varnames[2] == "x"
    # Coefficients should be near DGP (0.3, 0.5)
    @test abs(coef(m)[1] - 0.3) < 0.15
    @test abs(coef(m)[2] - 0.5) < 0.15

    # (T083) full vcov + diagnostics also populated for Blundell-Bond
    V = vcov(m)
    @test size(V) == (length(coef(m)), length(coef(m)))
    @test any(abs.(V - Diagonal(diag(V))) .> 0)
    d = m.dynamic_diagnostics
    @test d !== nothing
    @test d.ar1_p < 0.05
    @test d.ar2_p > 0.05
    @test d.hansen_df == d.n_instruments - length(coef(m))
end

# =============================================================================
# T089 (#188): M-33 absorbed-FE cluster dof, M-35 between cov_type warning
# =============================================================================

@testset "T089: panel cluster dof + between cov_type warning" begin

    @testset "M-33: n_absorbed scales the cluster correction" begin
        rng = Random.MersenneTwister(18933)
        N_g = 8; T_p = 12; n = N_g * T_p; k = 2
        X = randn(rng, n, k)
        resid = randn(rng, n)
        groups = repeat(1:N_g, inner=T_p)
        time_ids = repeat(1:T_p, N_g)
        XtXinv = Matrix(MacroEconometricModels.robust_inv(X' * X))

        V0 = MacroEconometricModels._panel_cluster_vcov(X, resid, XtXinv, groups)
        n_abs = 5
        V1 = MacroEconometricModels._panel_cluster_vcov(X, resid, XtXinv, groups; n_absorbed=n_abs)

        # The correction rescales the whole matrix by (n-k)/(n-k-n_absorbed)
        ratio = (n - k) / (n - k - n_abs)
        @test V1 ≈ V0 .* ratio atol = 1e-12
        # Every clustered SE strictly increases by sqrt(ratio)
        @test all(sqrt.(diag(V1)) .≈ sqrt.(diag(V0)) .* sqrt(ratio))

        # Threaded through _panel_vcov (:cluster branch)
        V2 = MacroEconometricModels._panel_vcov(X, resid, XtXinv, groups, time_ids, :cluster;
                                                n_absorbed=n_abs)
        @test V2 ≈ V1 atol = 1e-12

        # Default is inert
        V3 = MacroEconometricModels._panel_vcov(X, resid, XtXinv, groups, time_ids, :cluster)
        @test V3 ≈ V0 atol = 1e-12

        # Time-cluster variant takes the same kwarg
        Vt0 = MacroEconometricModels._panel_time_cluster_vcov(X, resid, XtXinv, time_ids)
        Vt1 = MacroEconometricModels._panel_time_cluster_vcov(X, resid, XtXinv, time_ids; n_absorbed=n_abs)
        @test Vt1 ≈ Vt0 .* ratio atol = 1e-12
    end

    @testset "M-35: between estimator warns when cov_type is ignored" begin
        rng = Random.MersenneTwister(18935)
        N_g = 12; T_p = 6; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = repeat(randn(rng, N_g), inner=T_p) .+ 0.5 .* randn(rng, n)
        y = 2.0 .* x1 .+ repeat(randn(rng, N_g), inner=T_p) .+ 0.3 .* randn(rng, n)
        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        # Default cov_type=:cluster is silently classical -> now warns
        m_warn = @test_logs (:warn, r"between estimator uses classical") estimate_xtreg(
            pd, :y, [:x1]; model=:between)
        # Explicit :ols does not warn
        m_ols = @test_logs estimate_xtreg(pd, :y, [:x1]; model=:between, cov_type=:ols)
        # Both return the same (classical) covariance
        @test vcov(m_warn) ≈ vcov(m_ols) atol = 1e-12
    end

end

# =============================================================================
# T090 (#189) SUB-1: group-index map equivalence (perf refactor, exact)
# =============================================================================

@testset "T090 SUB-1: _group_index_map == findall" begin
    rng = Random.MersenneTwister(19001)
    ids = rand(rng, [3, 7, 1, 12, 5], 500)  # unsorted, repeated group labels
    gmap = MacroEconometricModels._group_index_map(ids)
    @test sort(collect(keys(gmap))) == sort(unique(ids))
    for g in unique(ids)
        @test gmap[g] == findall(==(g), ids)  # exact: ascending order preserved
    end

    # End-to-end: FE via estimate_xtreg equals a manual findall-based within-demean OLS
    N_g = 15; T_p = 8; n = N_g * T_p
    groups = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    y = repeat(randn(rng, N_g), inner=T_p) .+ 1.3 .* x1 .+ 0.4 .* randn(rng, n)
    df = DataFrame(id=groups, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)
    m_fe = estimate_xtreg(pd, :y, [:x1]; model=:fe, cov_type=:ols)

    # manual pre-change algorithm (findall per group)
    y_dm = similar(y); x_dm = similar(x1)
    for g in 1:N_g
        idx = findall(==(g), groups)
        y_dm[idx] = y[idx] .- mean(y[idx])
        x_dm[idx] = x1[idx] .- mean(x1[idx])
    end
    beta_manual = (x_dm' * x_dm) \ (x_dm' * y_dm)
    @test coef(m_fe)[1] ≈ beta_manual atol = 1e-12
end

# =============================================================================
# T272 (#371): high-dimensional fixed-effect absorption (alternating projections)
# =============================================================================

@testset "T272: HDFE absorption (alternating projections)" begin
    MEM = MacroEconometricModels

    # Dense dummy matrix for a level-coded vector (oracle only — the whole point
    # of `absorb_fe` is never to build one of these).
    _dumm(c, G) = [Float64(c[i] == g) for i in 1:length(c), g in 1:G]

    rng = Random.MersenneTwister(11)
    Ng, Tp = 25, 12
    n = Ng * Tp
    ids = repeat(1:Ng, inner=Tp)
    ts = repeat(1:Tp, Ng)
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    y = repeat(randn(rng, Ng), inner=Tp) .+ 1.5 .* x1 .- 0.8 .* x2 .+ 0.4 .* randn(rng, n)
    df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
    df.ind = Float64.((df.id .% 5) .+ 1.0)      # nested inside entity
    df.shock = Float64.((df.t .% 3) .+ 1.0)     # NOT nested inside entity
    pd = xtset(df, :id, :t)

    @testset "one-way absorption == existing one-way FE" begin
        m_fe = estimate_xtreg(pd, :y, [:x1, :x2])
        m_ab = estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity])

        # Coefficients, SEs, variance components and dof must all match the
        # dedicated within path — absorbing entity IS the within transformation.
        @test coef(m_ab) ≈ coef(m_fe) atol = 1e-12
        @test stderror(m_ab) ≈ stderror(m_fe) atol = 1e-12
        @test m_ab.sigma_e ≈ m_fe.sigma_e atol = 1e-12
        @test m_ab.r2_within ≈ m_fe.r2_within atol = 1e-12
        @test dof_residual(m_ab) == dof_residual(m_fe)

        @test m_ab.hdfe !== nothing
        @test m_ab.hdfe.n_absorbed == Ng                # exactly G levels
        @test m_ab.hdfe.n_levels == [Ng]
        @test m_ab.hdfe.converged
        # Entity FE nested in the entity cluster ⟹ charged 0 against the cluster
        # dof, which is what reproduces the plain FE standard errors above.
        @test m_ab.hdfe.n_absorbed_cluster == 0
    end

    @testset "two-way absorption == twoway=true (balanced)" begin
        m_tw = estimate_xtreg(pd, :y, [:x1, :x2]; twoway=true)
        m_a2 = estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity, :time])

        @test coef(m_a2) ≈ coef(m_tw) atol = 1e-12
        @test m_a2.sigma_e ≈ m_tw.sigma_e atol = 1e-12
        # A balanced panel is one mobility group, so the dummy rank is N+T-1 and
        # the residual dof reproduces the `n - N - k - T + 1` used by :twoway.
        @test m_a2.hdfe.n_components == 1
        @test m_a2.hdfe.n_absorbed == Ng + Tp - 1
        @test dof_residual(m_a2) == n - 2 - (Ng + Tp - 1)

        # `twoway=true` now routes through this very path, so the two are the
        # SAME estimator and agree bit-for-bit — including the cluster dof, which
        # charges the T-1 non-nested time parameters.
        @test m_a2.hdfe.n_absorbed_cluster == Tp - 1
        @test stderror(m_a2) ≈ stderror(m_tw) atol = 1e-14
        @test m_tw.hdfe !== nothing
        @test m_tw.hdfe.absorb == [:entity, :time]
        @test m_tw.twoway                       # the flag survives for display/API
    end

    @testset "multi-way absorption == explicit-dummy OLS" begin
        rng3 = Random.MersenneTwister(7)
        n3 = 600
        firm = rand(rng3, 1:20, n3)
        yr = rand(rng3, 1:8, n3)
        ind = rand(rng3, 1:5, n3)
        X3 = randn(rng3, n3, 2)
        y3 = X3 * [1.2, -0.6] .+ randn(rng3, 20)[firm] .+ randn(rng3, 8)[yr] .+
             randn(rng3, 5)[ind] .+ 0.5 .* randn(rng3, n3)

        D3 = hcat(_dumm(firm, 20), _dumm(yr, 8), _dumm(ind, 5))
        b_dummy = pinv(hcat(X3, D3)) * y3          # min-norm OLS with dummies
        a3 = absorb_fe(y3, X3, [firm, yr, ind])
        b_map = a3.X \ a3.y

        @test a3.converged
        @test b_map ≈ b_dummy[1:2] atol = 1e-10
        # Reported absorbed parameters == the true rank of the dummy design here.
        @test a3.n_absorbed == rank(hcat(X3, D3)) - 2
        @test a3.n_levels == [20, 8, 5]

        # The absorbed data is orthogonal to the full dummy span (convergence)...
        @test maximum(abs, D3' * a3.y) < 1e-6
        @test maximum(abs, D3' * a3.X) < 1e-6
        # ...and what was removed lies exactly in that span (exact, by
        # construction: every sweep and every extrapolation stays in x₀+span(D)).
        r = y3 .- a3.y
        @test norm(r .- D3 * (pinv(D3) * r)) < 1e-10
    end

    @testset "coefficients invariant to FE-dimension ordering" begin
        rng4 = Random.MersenneTwister(7)
        n4 = 400
        d1 = rand(rng4, 1:15, n4)
        d2 = rand(rng4, 1:9, n4)
        d3 = rand(rng4, 1:6, n4)
        X4 = randn(rng4, n4, 2)
        y4 = X4 * [0.7, 1.4] .+ randn(rng4, 15)[d1] .+ randn(rng4, 9)[d2] .+
             randn(rng4, 6)[d3] .+ 0.5 .* randn(rng4, n4)

        base = absorb_fe(y4, X4, [d1, d2, d3])
        b_base = base.X \ base.y
        for perm in ([3, 1, 2], [2, 3, 1], [3, 2, 1], [2, 1, 3])
            a = absorb_fe(y4, X4, [d1, d2, d3][perm])
            @test a.converged
            @test (a.X \ a.y) ≈ b_base atol = 1e-9
        end
    end

    @testset "connected components (mobility groups)" begin
        # Two islands: workers 1-10 only ever meet firms 1-3, workers 11-20 only
        # firms 4-6. The dummy design then has rank G₁+G₂-2, not G₁+G₂-1.
        wk = vcat(repeat(1:10, inner=6), repeat(11:20, inner=6))
        fm = vcat([1 + (i % 3) for i in 1:60], [4 + (i % 3) for i in 1:60])
        nn = length(wk)
        Xd = randn(Random.MersenneTwister(3), nn, 1)
        yd = Xd * [1.0] .+ randn(Random.MersenneTwister(4), nn)

        ad = absorb_fe(yd, Xd, [wk, fm])
        Dd = hcat(_dumm(wk, 20), _dumm(fm, 6))
        @test ad.n_components == 2
        @test ad.n_absorbed == rank(Dd)             # exact, == 24
        @test ad.n_absorbed == 20 + 6 - 2
        @test ad.n_absorbed != 20 + 6 - 1           # the naive count is wrong
        @test (ad.X \ ad.y) ≈ (pinv(hcat(Xd, Dd)) * yd)[1:1] atol = 1e-9

        # A fully connected design collapses to one group.
        wk2 = repeat(1:10, inner=6)
        fm2 = [1 + (i % 6) for i in 1:60]
        a2 = absorb_fe(randn(Random.MersenneTwister(6), 60), zeros(60, 0), [wk2, fm2])
        @test a2.n_components == 1
        @test a2.n_absorbed == 10 + 6 - 1
    end

    @testset "unbalanced two-way matches dummy OLS (twoway= and absorb= alike)" begin
        # y - ȳᵢ - ȳₜ + ȳ is the two-way within transformation only on a BALANCED
        # panel; on an unbalanced one it is a different, biased estimator (it put
        # the x2 coefficient 2.1e-3 away from the dummy-OLS truth here). Both
        # entry points now use alternating projections, which are exact either way.
        rng8 = Random.MersenneTwister(21)
        dfu = df[rand(rng8, n) .> 0.25, :]
        pdu = xtset(dfu, :id, :t)
        Xu = Matrix{Float64}(dfu[:, [:x1, :x2]])
        yu = Vector{Float64}(dfu.y)
        Du = hcat(_dumm(dfu.id, Ng), _dumm(dfu.t, Tp))
        b_dummy = (pinv(hcat(Xu, Du)) * yu)[1:2]

        mu_ab = estimate_xtreg(pdu, :y, [:x1, :x2]; absorb=[:entity, :time])
        mu_tw = estimate_xtreg(pdu, :y, [:x1, :x2]; twoway=true)

        @test coef(mu_ab) ≈ b_dummy atol = 1e-10
        @test coef(mu_tw) ≈ b_dummy atol = 1e-10          # was off by 2.1e-3
        @test coef(mu_tw) ≈ coef(mu_ab) atol = 1e-12
        @test mu_ab.hdfe.n_absorbed == rank(Du)
        @test mu_tw.hdfe.n_absorbed == rank(Du)
        # The residual dof now uses the true dummy rank (N + T - components), not
        # the balanced-panel shortcut n - N - T + 1.
        @test dof_residual(mu_tw) == size(Xu, 1) - 2 - rank(Du)
    end

    @testset "acceleration: same fixed point, far fewer sweeps" begin
        # Sparse worker-firm mobility — each worker meets only two firms — is the
        # design that makes plain alternating projections crawl.
        worker = repeat(1:150, inner=6)
        frm = [1 + ((w - 1) ÷ 3 + (j % 2)) % 50 for w in 1:150 for j in 1:6]
        nw = length(worker)
        Xw = randn(Random.MersenneTwister(99), nw, 1)
        yw = Xw * [2.0] .+ randn(Random.MersenneTwister(98), 150)[worker] .+
             randn(Random.MersenneTwister(97), 50)[frm] .+
             0.3 .* randn(Random.MersenneTwister(96), nw)

        Dw = hcat(_dumm(worker, 150), _dumm(frm, 50))
        b_true = (pinv(hcat(Xw, Dw)) * yw)[1]

        a_on = absorb_fe(yw, Xw, [worker, frm]; accel=true)
        a_off = absorb_fe(yw, Xw, [worker, frm]; accel=false)

        @test a_on.converged
        @test (a_on.X \ a_on.y)[1] ≈ b_true atol = 1e-8
        # Un-accelerated projections exhaust the default budget on this design
        # and land four orders of magnitude further from the truth.
        @test !a_off.converged
        @test abs((a_off.X \ a_off.y)[1] - b_true) > 100 * abs((a_on.X \ a_on.y)[1] - b_true)
        @test a_on.sweeps < a_off.sweeps

        # Given enough budget the un-accelerated loop reaches the same point.
        a_long = absorb_fe(yw, Xw, [worker, frm]; accel=false, maxiter=20_000)
        @test a_long.converged
        @test (a_long.X \ a_long.y)[1] ≈ (a_on.X \ a_on.y)[1] atol = 1e-7
    end

    @testset "nested-in-cluster dof accounting" begin
        m_nested = estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity, :ind])
        # `ind` is a coarsening of `entity`: given entity FE it adds no free
        # parameters, and the components count detects that automatically.
        @test m_nested.hdfe.n_levels == [Ng, 5]
        @test m_nested.hdfe.marginal == [Ng, 0]
        @test m_nested.hdfe.n_absorbed == Ng
        @test m_nested.hdfe.n_absorbed_cluster == 0
        @test coef(m_nested) ≈ coef(estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity])) atol = 1e-12

        m_free = estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity, :shock])
        @test m_free.hdfe.n_levels == [Ng, 3]
        @test m_free.hdfe.marginal == [Ng, 2]
        @test m_free.hdfe.n_absorbed == Ng + 2
        @test m_free.hdfe.n_absorbed_cluster == 2      # shock crosses entities

        # Unit tests of the nesting predicate itself.
        @test MEM._hdfe_nested_in([1, 1, 2, 2], 2, [1, 1, 2, 2])
        @test MEM._hdfe_nested_in([1, 1, 2, 2], 2, [1, 1, 1, 1])
        @test !MEM._hdfe_nested_in([1, 2, 1, 2], 2, [1, 1, 2, 2])
    end

    @testset "absorb_fe mechanics" begin
        # y-only absorption (zero-column X) equals the plain within transform.
        a0 = absorb_fe(y, zeros(n, 0), [ids])
        @test size(a0.X) == (n, 0)
        @test a0.n_absorbed == Ng
        y_manual = y .- [mean(y[ids .== g]) for g in ids]
        @test a0.y ≈ y_manual atol = 1e-12

        # Level ids may be of any type; they are dense-ranked internally.
        str_ids = ["g$(g)" for g in ids]
        a_str = absorb_fe(y, hcat(x1), [str_ids])
        @test a_str.y ≈ y_manual atol = 1e-12
        @test a_str.n_levels == [Ng]

        # Float32 end-to-end.
        a32 = absorb_fe(Float32.(y), Float32.(hcat(x1, x2)), [ids, ts])
        @test eltype(a32.y) === Float32
        @test a32.converged

        # Integer input promotes to Float64.
        a_int = absorb_fe([1, 2, 3, 4], reshape([1, 0, 1, 0], 4, 1), [[1, 1, 2, 2]])
        @test eltype(a_int.y) === Float64

        # `converged` is honest about a starved budget.
        worker = repeat(1:150, inner=6)
        frm = [1 + ((w - 1) ÷ 3 + (j % 2)) % 50 for w in 1:150 for j in 1:6]
        Xs = randn(Random.MersenneTwister(1), length(worker), 1)
        a_short = absorb_fe(Xs[:, 1], Xs, [worker, frm]; maxiter=3, accel=false)
        @test !a_short.converged
        @test a_short.iterations == 3
        @test a_short.change > 1e-8

        # Internal helpers.
        codes, G = MEM._hdfe_codes([10.0, 20.0, 10.0, 30.0])
        @test codes == [1, 2, 1, 3] && G == 3
        sets = MEM._hdfe_index_sets(codes, G)
        @test sets == [[1, 3], [2], [4]]
        @test MEM._hdfe_dof([[1, 1, 2]], [2]).n_absorbed == 2
    end

    @testset "errors and edge cases" begin
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; model=:re, absorb=[:entity])
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; twoway=true, absorb=[:entity])
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; absorb=[:entity, :entity])
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; absorb=[:not_a_column])
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1]; absorb=[:cohort])
        @test_throws ArgumentError absorb_fe(y, hcat(x1), Vector{Int}[])
        @test_throws DimensionMismatch absorb_fe(y, hcat(x1), [ids[1:10]])
        @test_throws DimensionMismatch absorb_fe(y, hcat(x1)[1:10, :], [ids])
        @test_throws ArgumentError absorb_fe(y, hcat(x1), [collect(1:n)])   # one level per obs
        @test_throws ArgumentError absorb_fe(y, hcat(x1), [ids]; maxiter=0)
        @test_throws ArgumentError absorb_fe(y, hcat(x1), [ids]; tol=0.0)
        # NaN would otherwise become a silent extra level (isequal(NaN,NaN) is true)
        @test_throws ArgumentError absorb_fe(y, hcat(x1), [vcat(NaN, Float64.(ids[2:end]))])

        df_nan = copy(df)
        df_nan.bad = fill(1.0, n)
        df_nan.bad[3] = NaN
        @test_throws ArgumentError estimate_xtreg(xtset(df_nan, :id, :t), :y, [:x1];
                                                  absorb=[:entity, :bad])

        # Reserved index aliases all resolve.
        for alias in (:entity, :id, :unit, :group)
            @test MEM._hdfe_dimension(pd, alias) == pd.group_id
        end
        for alias in (:time, :period)
            @test MEM._hdfe_dimension(pd, alias) == pd.time_id
        end
    end

    @testset "display and downstream integration" begin
        m_hd = estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity, :time])
        buf = IOBuffer()
        show(buf, m_hd)
        out = String(take!(buf))
        @test occursin("HDFE", out)
        @test occursin("Absorbed FE", out)
        @test occursin("Mobility groups", out)
        @test occursin("FE parameters", out)

        # A non-HDFE fit's display is untouched (golden safety).
        buf2 = IOBuffer()
        show(buf2, estimate_xtreg(pd, :y, [:x1, :x2]))
        out2 = String(take!(buf2))
        @test !occursin("HDFE", out2)
        @test !occursin("Absorbed FE", out2)

        # Wild cluster bootstrap (T243) re-absorbs with the fit's own dimensions:
        # absorb=[:entity] must reproduce the plain-FE bootstrap exactly.
        wb_plain = wild_cluster_bootstrap(estimate_xtreg(pd, :y, [:x1, :x2]), :x1;
                                          n_boot=99, rng=Random.MersenneTwister(5))
        wb_abs = wild_cluster_bootstrap(estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity]),
                                        :x1; n_boot=99, rng=Random.MersenneTwister(5))
        @test wb_abs.t_stat ≈ wb_plain.t_stat atol = 1e-10
        @test wb_abs.p_value == wb_plain.p_value

        wb_hd = wild_cluster_bootstrap(m_hd, :x1; n_boot=99, rng=Random.MersenneTwister(5))
        @test isfinite(wb_hd.t_stat)
        @test 0 <= wb_hd.p_value <= 1

        # The absorption settings travel with the fit, so a deliberately
        # un-accelerated fit bootstraps against its own design, not a re-tuned one.
        m_noaccel = estimate_xtreg(pd, :y, [:x1, :x2]; absorb=[:entity, :time],
                                   hdfe_accel=false)
        @test m_noaccel.hdfe.accel == false
        @test coef(m_noaccel) ≈ coef(m_hd) atol = 1e-9
        wb_na = wild_cluster_bootstrap(m_noaccel, :x1; n_boot=99,
                                       rng=Random.MersenneTwister(5))
        @test wb_na.t_stat ≈ wb_hd.t_stat atol = 1e-6
    end
end

@testset "#407: panel report() renders cov_type through _label" begin
    # The panel show bodies used to print `string(m.cov_type)`, so the internal
    # symbol leaked into the table ("Cov. type  cluster"). Every panel covariance
    # type must now render the same human-readable label the cross-sectional
    # estimators use.
    rng = Random.MersenneTwister(407)
    # T >= N so the PCSE contemporaneous covariance is full rank (Beck & Katz 1995).
    N_g = 8; T_p = 14; n = N_g * T_p
    df = DataFrame(id=repeat(1:N_g, inner=T_p), t=repeat(1:T_p, N_g),
                   x=randn(rng, n), w=randn(rng, n), z=randn(rng, n))
    df.x .+= 0.8 .* df.z
    df.y = 0.7 .* df.x .- 0.3 .* df.w .+ randn(rng, n)
    pd = xtset(df, :id, :t)

    covline(m) = begin
        io = IOBuffer(); show(io, m); s = String(take!(io))
        strip(only(filter(l -> occursin("Cov. type", l), split(s, '\n'))))
    end

    expected = Dict(:ols => "OLS",
                    :cluster => "Cluster-robust",
                    :twoway => "Two-way cluster-robust",
                    :driscoll_kraay => "Driscoll–Kraay (HAC)",
                    :pcse => "Panel-corrected (Beck–Katz)")

    for (ct, label) in expected
        m = estimate_xtreg(pd, :y, [:x, :w]; model=:fe, cov_type=ct)
        line = covline(m)
        @test occursin(label, line)
        # the raw symbol must not survive anywhere on the row
        @test !occursin(Regex("\\b" * String(ct) * "\\b"), line)
    end

    # PanelIVModel is the second site the defect lived at.
    for ct in (:cluster, :twoway, :driscoll_kraay)
        m = estimate_xtiv(pd, :y, [:x], [:w]; instruments=[:z], model=:fe, cov_type=ct)
        line = covline(m)
        @test occursin(expected[ct], line)
        @test !occursin(Regex("\\b" * String(ct) * "\\b"), line)
    end

    # `_label` still falls back to title-case for symbols it does not know, so an
    # unregistered type degrades to "Some New Type" rather than dumping the symbol.
    @test MacroEconometricModels._label(:some_new_type) == "Some New Type"
end
