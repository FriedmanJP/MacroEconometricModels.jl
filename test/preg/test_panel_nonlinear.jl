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

@testset "FE clogit log-space DP stability (T088)" begin
    dp = MacroEconometricModels._clogit_dp_logsum
    # (a) OVERFLOW + closed form (T_g=2, s=1): raw exp overflows to Inf; log-space stays finite
    ld, p = dp(reshape([1000.0, 1001.0], 2, 1), [1.0], 1)
    @test isfinite(ld)
    @test isapprox(ld, 1001.3132616875182; atol=1e-9)     # logaddexp(1000, 1001)
    @test isapprox(p, [0.2689414213699951, 0.7310585786300049]; atol=1e-8)   # softmax
    @test all(isfinite, p) && isapprox(sum(p), 1.0)

    # (b) BRUTE-FORCE cross-check (T_g=5, s=2) against explicit subset enumeration
    rng = Random.MersenneTwister(88); eta = randn(rng, 5) .* 0.5
    ld5, p5 = dp(reshape(eta, 5, 1), [1.0], 2)
    subs = [(i, j) for i in 1:5 for j in (i+1):5]
    denom_bf = sum(exp(eta[i] + eta[j]) for (i, j) in subs)
    @test isapprox(ld5, log(denom_bf); atol=1e-10)
    for t in 1:5
        gt = sum(exp(eta[i] + eta[j]) for (i, j) in subs if t == i || t == j)
        @test isapprox(p5[t], gt / denom_bf; atol=1e-10)
    end

    # (c) SCALE-EQUIVARIANCE through _xtlogit_fe: β(50·x) = β(x)/50, equal maximized loglik.
    #     With large |Xβ| the old raw-exp DP overflowed (NaN); the log-space DP stays valid.
    rng2 = Random.MersenneTwister(880); N_g = 60; T_p = 8; nn = N_g * T_p
    ids = repeat(1:N_g, inner=T_p); ts = repeat(1:T_p, N_g)
    x1 = randn(rng2, nn)
    alpha = repeat(randn(rng2, N_g), inner=T_p)
    y = Float64.(rand(rng2, nn) .< 1.0 ./ (1.0 .+ exp.(-(alpha .+ 0.7 .* x1))))
    m1 = estimate_xtlogit(xtset(DataFrame(id=ids, t=ts, x1=x1, y=y), :id, :t), :y, [:x1]; model=:fe)
    m2 = estimate_xtlogit(xtset(DataFrame(id=ids, t=ts, x1=50.0 .* x1, y=y), :id, :t), :y, [:x1]; model=:fe)
    @test all(isfinite, coef(m2))
    @test isapprox(coef(m2)[1], coef(m1)[1] / 50.0; rtol=1e-4)
    @test isapprox(loglikelihood(m1), loglikelihood(m2); rtol=1e-5)
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

@testset "Panel marginal effects" begin
    @testset "Pooled logit" begin
        rng = Random.MersenneTwister(4001)
        N_g = 50; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        eta = 1.0 .* x1 .- 0.8 .* x2
        p = 1.0 ./ (1.0 .+ exp.(-eta))
        y = Float64.(rand(rng, n) .< p)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtlogit(pd, :y, [:x1, :x2])

        me = marginal_effects(m)
        @test me isa MarginalEffects{Float64}
        @test length(me.effects) == 3  # intercept + 2 vars
        @test all(me.se .> 0)
        # AME for x1 should be positive, x2 negative
        @test me.effects[2] > 0
        @test me.effects[3] < 0
        # AMEs should be smaller in magnitude than raw coefficients (attenuation by f(eta))
        @test abs(me.effects[2]) < abs(coef(m)[2])
        @test abs(me.effects[3]) < abs(coef(m)[3])
    end

    @testset "RE logit — attenuation" begin
        rng = Random.MersenneTwister(4002)
        N_g = 50; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        alpha = repeat(1.0 .* randn(rng, N_g), inner=T_p)
        eta = alpha .+ 0.8 .* x1
        p = 1.0 ./ (1.0 .+ exp.(-eta))
        y = Float64.(rand(rng, n) .< p)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtlogit(pd, :y, [:x1]; model=:re, maxiter=300)

        me = marginal_effects(m)
        @test me isa MarginalEffects{Float64}
        @test length(me.effects) == 2  # intercept + x1
        @test all(me.se .> 0)
        # AME for x1 should be positive
        @test me.effects[2] > 0
        # Effects should be smaller than coefficients (attenuation)
        @test abs(me.effects[2]) < abs(coef(m)[2])
    end

    @testset "FE logit" begin
        rng = Random.MersenneTwister(4003)
        N_g = 100; T_p = 10; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        eta = alpha .+ 0.8 .* x1 .- 0.5 .* x2
        p = 1.0 ./ (1.0 .+ exp.(-eta))
        y = Float64.(rand(rng, n) .< p)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)
        m = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:fe)

        me = marginal_effects(m)
        @test me isa MarginalEffects{Float64}
        @test length(me.effects) == 2  # no intercept in FE
        @test all(me.se .> 0)
        @test me.effects[1] > 0  # x1 positive
        @test me.effects[2] < 0  # x2 negative
    end

    @testset "Pooled probit" begin
        rng = Random.MersenneTwister(4004)
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

        me = marginal_effects(m)
        @test me isa MarginalEffects{Float64}
        @test length(me.effects) == 3  # intercept + 2
        @test all(me.se .> 0)
        @test me.effects[2] > 0
        @test me.effects[3] < 0
        @test abs(me.effects[2]) < abs(coef(m)[2])
    end

    @testset "CRE logit — only original vars" begin
        rng = Random.MersenneTwister(4005)
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

        me = marginal_effects(m)
        @test me isa MarginalEffects{Float64}
        # CRE should only report original vars (not _cons and not _mean)
        @test length(me.effects) == 1
        @test all(me.se .> 0)
        @test me.effects[1] > 0
        @test !any(endswith("_mean"), me.varnames)
    end
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
