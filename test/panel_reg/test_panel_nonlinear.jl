using Test, MacroEconometricModels, DataFrames, Distributions, Random, Statistics
using StatsAPI: coef, vcov, predict, nobs, stderror, confint, loglikelihood, aic, bic, dof, islinear

@testset "estimate_xtlogit -- pooled" begin
    rng = Random.MersenneTwister(1234)
    N_g = 50; T_p = 10; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    x2 = randn(rng, n)

    # True beta = [0, 1.0, -0.8] (intercept, x1, x2)
    eta = 1.0 .* x1 .- 0.8 .* x2
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtlogit(pd, :y, [:x1, :x2])

    @test m isa PanelLogitModel{Float64}
    @test m.method == :pooled
    @test m.converged
    @test m.n_obs == n
    @test m.n_groups == N_g

    # Coefficient recovery (pooled, so intercept ~ 0, x1 ~ 1.0, x2 ~ -0.8)
    b = coef(m)
    @test length(b) == 3  # intercept + 2
    @test abs(b[2] - 1.0) < 0.4   # x1
    @test abs(b[3] + 0.8) < 0.4   # x2

    # Clustered SEs should be positive
    se = stderror(m)
    @test all(se .> 0)

    # Predictions in [0, 1]
    yhat = predict(m)
    @test all(0 .<= yhat .<= 1)

    # Log-likelihood
    @test loglikelihood(m) < 0
    @test aic(m) > 0
    @test bic(m) > 0
end

@testset "estimate_xtlogit -- FE (conditional)" begin
    rng = Random.MersenneTwister(5678)
    N_g = 100; T_p = 10; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    x2 = randn(rng, n)

    # Add entity effects to create variation + ensure some groups have mixed y
    alpha = repeat(randn(rng, N_g), inner=T_p)
    eta = alpha .+ 0.8 .* x1 .- 0.5 .* x2
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:fe)

    @test m isa PanelLogitModel{Float64}
    @test m.method == :fe
    @test m.converged

    # FE logit has no intercept
    b = coef(m)
    @test length(b) == 2

    # Coefficient recovery (conditional logit)
    @test abs(b[1] - 0.8) < 0.5   # x1
    @test abs(b[2] + 0.5) < 0.5   # x2

    # sigma_u and rho should be nothing for FE
    @test m.sigma_u === nothing
    @test m.rho === nothing

    # SEs positive
    se = stderror(m)
    @test all(se .> 0)

    # Groups with no variation should be dropped
    @test m.n_groups <= N_g
end

@testset "estimate_xtlogit -- RE" begin
    rng = Random.MersenneTwister(9012)
    N_g = 50; T_p = 10; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)

    # RE with sigma_u = 1.0
    alpha = repeat(1.0 .* randn(rng, N_g), inner=T_p)
    eta = alpha .+ 0.8 .* x1
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtlogit(pd, :y, [:x1]; model=:re, maxiter=300)

    @test m isa PanelLogitModel{Float64}
    @test m.method == :re

    # sigma_u should be estimated (> 0)
    @test m.sigma_u !== nothing
    @test m.sigma_u > 0

    # rho should be between 0 and 1
    @test m.rho !== nothing
    @test 0 < m.rho < 1

    # Coefficients: intercept + x1
    b = coef(m)
    @test length(b) == 2

    # x1 coefficient should be positive
    @test b[2] > 0

    # SEs positive
    se = stderror(m)
    @test all(se .> 0)

    @test m.n_obs == n
    @test m.n_groups == N_g
end

@testset "estimate_xtlogit -- CRE" begin
    rng = Random.MersenneTwister(3456)
    N_g = 50; T_p = 10; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)

    alpha = repeat(randn(rng, N_g), inner=T_p)
    eta = alpha .+ 1.0 .* x1
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtlogit(pd, :y, [:x1]; model=:cre, maxiter=300)

    @test m isa PanelLogitModel{Float64}
    @test m.method == :cre

    # CRE has original + mean variables
    b = coef(m)
    @test length(b) == 3  # intercept + x1 + x1_mean

    # Variable names should include mean
    @test any(contains("_mean"), m.varnames)

    # sigma_u estimated
    @test m.sigma_u !== nothing
    @test m.sigma_u >= 0
end

@testset "estimate_xtprobit -- pooled" begin
    rng = Random.MersenneTwister(7890)
    N_g = 50; T_p = 10; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    x2 = randn(rng, n)

    d = Normal()
    eta = 0.8 .* x1 .- 0.5 .* x2
    p = cdf.(d, eta)
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtprobit(pd, :y, [:x1, :x2])

    @test m isa PanelProbitModel{Float64}
    @test m.method == :pooled
    @test m.converged

    b = coef(m)
    @test length(b) == 3
    @test abs(b[2] - 0.8) < 0.4
    @test abs(b[3] + 0.5) < 0.4

    se = stderror(m)
    @test all(se .> 0)
end

@testset "estimate_xtprobit -- RE" begin
    rng = Random.MersenneTwister(1122)
    N_g = 50; T_p = 10; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)

    alpha = repeat(0.8 .* randn(rng, N_g), inner=T_p)
    d = Normal()
    eta = alpha .+ 0.6 .* x1
    p = cdf.(d, eta)
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtprobit(pd, :y, [:x1]; model=:re, maxiter=300)

    @test m isa PanelProbitModel{Float64}
    @test m.method == :re

    @test m.sigma_u !== nothing
    @test m.sigma_u > 0

    @test m.rho !== nothing
    @test 0 < m.rho < 1

    b = coef(m)
    @test length(b) == 2
    @test b[2] > 0

    se = stderror(m)
    @test all(se .> 0)
end

@testset "estimate_xtprobit -- FE throws" begin
    rng = Random.MersenneTwister(3344)
    N_g = 10; T_p = 5; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    y = Float64.(rand(rng, n) .< 0.5)
    df = DataFrame(id=ids, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)

    @test_throws ArgumentError estimate_xtprobit(pd, :y, [:x1]; model=:fe)
end

@testset "Panel nonlinear -- StatsAPI" begin
    rng = Random.MersenneTwister(5566)
    N_g = 30; T_p = 8; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    eta = 0.5 .* x1
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(rand(rng, n) .< p)

    df = DataFrame(id=ids, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)

    m = estimate_xtlogit(pd, :y, [:x1])

    # StatsAPI interface
    @test length(coef(m)) == 2
    @test size(vcov(m)) == (2, 2)
    @test nobs(m) == n
    @test dof(m) == 2
    @test islinear(m) == false
    @test loglikelihood(m) < 0
    @test aic(m) > 0
    @test bic(m) > 0

    se = stderror(m)
    @test length(se) == 2
    @test all(se .> 0)

    ci = confint(m)
    @test size(ci) == (2, 2)
    @test all(ci[:, 1] .< ci[:, 2])

    yhat = predict(m)
    @test length(yhat) == n
end

@testset "Panel nonlinear -- display" begin
    rng = Random.MersenneTwister(7788)
    N_g = 20; T_p = 5; n = N_g * T_p
    ids = repeat(1:N_g, inner=T_p)
    ts = repeat(1:T_p, N_g)
    x1 = randn(rng, n)
    y = Float64.(rand(rng, n) .< 0.5)
    df = DataFrame(id=ids, t=ts, x1=x1, y=y)
    pd = xtset(df, :id, :t)

    m_logit = estimate_xtlogit(pd, :y, [:x1])
    m_probit = estimate_xtprobit(pd, :y, [:x1])

    # show() should work without error
    io = IOBuffer()
    show(io, m_logit)
    s_logit = String(take!(io))
    @test contains(s_logit, "Logit")
    @test contains(s_logit, "Pooled")

    show(io, m_probit)
    s_probit = String(take!(io))
    @test contains(s_probit, "Probit")
    @test contains(s_probit, "Pooled")
end
