using Test, MacroEconometricModels, DataFrames, Distributions, Random, Statistics
using StatsAPI: coef, vcov, residuals, predict, nobs, stderror, confint, r2

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
