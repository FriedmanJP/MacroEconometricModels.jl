using Test, MacroEconometricModels, DataFrames, Distributions, Random, LinearAlgebra, StatsAPI

@testset "Panel IV Estimation" begin

    # =================================================================
    # DGP helper: endogenous x correlated with error, z is instrument
    # =================================================================
    function make_iv_panel(; N=50, Ti=20, beta_x=1.5, beta_endog=2.0, seed=42)
        rng = Random.MersenneTwister(seed)
        n = N * Ti
        id = repeat(1:N, inner=Ti)
        t = repeat(1:Ti, N)

        alpha_i = repeat(randn(rng, N) .* 2.0, inner=Ti)
        x_exog = randn(rng, n)
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        u = randn(rng, n)

        # Endogenous: correlated with error u
        x_endog = 0.6 .* z1 .+ 0.4 .* z2 .+ 0.5 .* u .+ randn(rng, n) .* 0.3

        y = alpha_i .+ beta_x .* x_exog .+ beta_endog .* x_endog .+ u

        df = DataFrame(id=id, t=t, y=y, x=x_exog, x_endog=x_endog, z1=z1, z2=z2)
        pd = xtset(df, :id, :t)
        return pd, beta_x, beta_endog
    end

    # =================================================================
    @testset "estimate_xtiv - FE-IV" begin
        pd, beta_x, beta_endog = make_iv_panel(N=50, Ti=20, seed=123)

        m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2], model=:fe)

        @test m isa PanelIVModel{Float64}
        @test m.method == :fe_iv
        @test length(coef(m)) == 2
        @test m.n_obs == 1000
        @test m.n_groups == 50

        # Coefficient recovery (with tolerance for finite sample)
        @test abs(coef(m)[1] - beta_x) < 0.3      # x exog
        @test abs(coef(m)[2] - beta_endog) < 0.5   # x endog (IV has more variance)

        # First-stage F should indicate strong instruments
        @test m.first_stage_f > 10.0

        # Sargan test (overidentified: 2 instruments, 1 endogenous)
        @test m.sargan_stat !== nothing
        @test m.sargan_pval !== nothing
        @test m.sargan_pval > 0.0  # valid instruments => should not reject

        # Variance components
        @test m.sigma_e > 0.0
        @test m.sigma_u >= 0.0
        @test 0.0 <= m.rho <= 1.0
    end

    # =================================================================
    @testset "estimate_xtiv - FD-IV" begin
        pd, beta_x, beta_endog = make_iv_panel(N=50, Ti=20, seed=456)

        m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2], model=:fd)

        @test m isa PanelIVModel{Float64}
        @test m.method == :fd_iv
        @test length(coef(m)) == 2

        # Coefficient recovery
        @test abs(coef(m)[1] - beta_x) < 0.5
        @test abs(coef(m)[2] - beta_endog) < 0.8

        # First-stage F
        @test m.first_stage_f > 5.0

        # Should have fewer obs due to differencing
        @test m.n_obs < 1000
    end

    # =================================================================
    @testset "estimate_xtiv - RE-IV (EC2SLS)" begin
        rng = Random.MersenneTwister(789)
        N, Ti = 60, 15
        n = N * Ti
        id = repeat(1:N, inner=Ti)
        t = repeat(1:Ti, N)

        # RE DGP: alpha_i uncorrelated with x
        alpha_i = repeat(randn(rng, N), inner=Ti)
        x_exog = randn(rng, n)
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        u = randn(rng, n)

        x_endog = 0.5 .* z1 .+ 0.3 .* z2 .+ 0.4 .* u .+ randn(rng, n) .* 0.3
        y = 1.0 .+ alpha_i .+ 1.5 .* x_exog .+ 2.0 .* x_endog .+ u

        df = DataFrame(id=id, t=t, y=y, x=x_exog, x_endog=x_endog, z1=z1, z2=z2)
        pd = xtset(df, :id, :t)

        m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2], model=:re)

        @test m isa PanelIVModel{Float64}
        @test m.method == :re_iv
        @test length(coef(m)) == 2
        @test m.n_obs == n

        # Coefficient recovery
        @test abs(coef(m)[1] - 1.5) < 0.4
        @test abs(coef(m)[2] - 2.0) < 0.6

        # Variance components
        @test m.sigma_e > 0.0
        @test m.sigma_u >= 0.0
    end

    # =================================================================
    @testset "estimate_xtiv - Hausman-Taylor" begin
        rng = Random.MersenneTwister(101)
        N, Ti = 200, 20
        n = N * Ti
        id = repeat(1:N, inner=Ti)
        t = repeat(1:Ti, N)

        # Time-varying exogenous (multiple for stronger instruments)
        x_tv_exog1 = randn(rng, n)
        x_tv_exog2 = randn(rng, n)

        # Time-invariant exogenous (e.g., gender)
        w_ti_exog_base = randn(rng, N)
        w_ti_exog = repeat(w_ti_exog_base, inner=Ti)

        # Time-invariant endogenous (e.g., education, correlated with alpha_i)
        w_ti_endog_base = randn(rng, N)
        w_ti_endog = repeat(w_ti_endog_base, inner=Ti)

        # alpha_i correlated with w_ti_endog
        alpha_i_base = 0.5 .* w_ti_endog_base .+ randn(rng, N) .* 0.5
        alpha_i = repeat(alpha_i_base, inner=Ti)

        # Time-varying endogenous: correlated with alpha_i
        x_tv_endog = 0.3 .* alpha_i .+ randn(rng, n)

        u = randn(rng, n) .* 0.3
        y = alpha_i .+ 1.0 .* x_tv_exog1 .+ 0.5 .* x_tv_exog2 .+
            1.5 .* x_tv_endog .+ 0.8 .* w_ti_exog .+ 1.2 .* w_ti_endog .+ u

        df = DataFrame(id=id, t=t, y=y,
                        x_tv_exog1=x_tv_exog1, x_tv_exog2=x_tv_exog2,
                        x_tv_endog=x_tv_endog,
                        w_ti_exog=w_ti_exog, w_ti_endog=w_ti_endog)
        pd = xtset(df, :id, :t)

        m = estimate_xtiv(pd, :y, [:x_tv_exog1, :x_tv_exog2], [:x_tv_endog];
                          model=:hausman_taylor,
                          time_invariant_exog=[:w_ti_exog],
                          time_invariant_endog=[:w_ti_endog])

        @test m isa PanelIVModel{Float64}
        @test m.method == :hausman_taylor
        @test length(coef(m)) == 5
        @test m.n_obs == n
        @test m.n_groups == N

        # Time-varying coefficients should be well estimated from within
        @test abs(coef(m)[1] - 1.0) < 0.3   # tv_exog1
        @test abs(coef(m)[2] - 0.5) < 0.3   # tv_exog2
        @test abs(coef(m)[3] - 1.5) < 0.5   # tv_endog

        # Time-invariant coefficients have higher variance in HT
        @test abs(coef(m)[4] - 0.8) < 2.0   # ti_exog
        @test abs(coef(m)[5] - 1.2) < 3.0   # ti_endog

        # Variance components
        @test m.sigma_e > 0.0
        @test m.sigma_u >= 0.0
    end

    # =================================================================
    @testset "Panel IV - StatsAPI interface" begin
        pd, _, _ = make_iv_panel(N=30, Ti=10, seed=999)
        m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2], model=:fe)

        @test length(coef(m)) == 2
        @test size(vcov(m)) == (2, 2)
        @test nobs(m) == 300
        @test length(stderror(m)) == 2
        @test all(stderror(m) .> 0)
        @test length(residuals(m)) == 300
        @test length(predict(m)) == 300
        @test dof(m) == 2
        @test StatsAPI.dof_residual(m) == 300 - 2 - 30  # n - k - N for FE
        @test r2(m) isa Float64
        @test islinear(m) == true

        # Confidence intervals
        ci = confint(m)
        @test size(ci) == (2, 2)
        @test all(ci[:, 1] .< coef(m))
        @test all(ci[:, 2] .> coef(m))
    end

    # =================================================================
    @testset "Panel IV - display" begin
        pd, _, _ = make_iv_panel(N=30, Ti=10, seed=777)
        m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2], model=:fe)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Panel IV", output) || occursin("FE-IV", output)
        @test occursin("x_endog", output)
        @test occursin("z1", output)
        @test occursin("1st-stage F", output) || occursin("first", lowercase(output))
    end

    # =================================================================
    @testset "Panel IV - input validation" begin
        pd, _, _ = make_iv_panel(N=20, Ti=10, seed=111)

        # Bad model
        @test_throws ArgumentError estimate_xtiv(pd, :y, [:x], [:x_endog];
                                                  instruments=[:z1], model=:bad)

        # Missing instruments for non-HT model
        @test_throws ArgumentError estimate_xtiv(pd, :y, [:x], [:x_endog];
                                                  instruments=Symbol[], model=:fe)

        # Missing endogenous
        @test_throws ArgumentError estimate_xtiv(pd, :y, [:x], Symbol[];
                                                  instruments=[:z1], model=:fe)

        # Variable not in panel
        @test_throws ArgumentError estimate_xtiv(pd, :y, [:nonexistent], [:x_endog];
                                                  instruments=[:z1], model=:fe)
    end

    # =================================================================
    @testset "Panel IV - exactly identified" begin
        pd, _, _ = make_iv_panel(N=40, Ti=15, seed=222)

        # 1 instrument, 1 endogenous: exactly identified => no Sargan
        m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1], model=:fe)
        @test m.sargan_stat === nothing
        @test m.sargan_pval === nothing
        @test m.first_stage_f > 0.0
    end

    # =================================================================
    @testset "Panel IV - cov_type options" begin
        pd, _, _ = make_iv_panel(N=30, Ti=10, seed=333)

        m_ols = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2],
                               model=:fe, cov_type=:ols)
        m_cl = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z1, :z2],
                              model=:fe, cov_type=:cluster)

        @test m_ols isa PanelIVModel{Float64}
        @test m_cl isa PanelIVModel{Float64}
        # Cluster SE typically larger
        @test stderror(m_cl)[1] > 0.0
        @test stderror(m_ols)[1] > 0.0
    end

end
