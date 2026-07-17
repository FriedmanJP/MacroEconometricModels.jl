using Test, MacroEconometricModels, DataFrames, Random, Statistics, LinearAlgebra
using Logging: with_logger

@testset "Panel Specification Tests" begin

    # =========================================================================
    # 1. Hausman Test — X correlated with alpha_i triggers rejection
    # =========================================================================
    @testset "Hausman test" begin
        rng = Random.MersenneTwister(1234)
        N_g = 50; T_p = 20; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        # Entity effects correlated with X (endogeneity for RE)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        x1 = 0.5 .* alpha .+ randn(rng, n)   # correlated with alpha
        x2 = randn(rng, n)
        y = alpha .+ 2.0 .* x1 .- 1.0 .* x2 .+ 0.3 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)

        fe = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
        re = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)

        # On this finite sample V_FE − V_RE is INDEFINITE. The corrected test reports the
        # (genuinely negative) generalized-inverse statistic with df = rank(dV) < k and does
        # NOT spuriously reject — the old abs()+df=k reported this as a significant rejection.
        tl = Test.TestLogger(respect_maxlog=false)
        ht = with_logger(() -> hausman_test(fe, re), tl)
        @test ht isa PanelTestResult{Float64}
        @test ht.statistic < 0                    # negative form no longer masked by abs()
        @test ht.df == 1                          # rank(dV) < k = 2
        @test ht.pvalue == 1.0                    # never a spurious rejection when chi2 ≤ 0
        @test occursin("non-PSD", ht.description)
        @test any(r -> occursin("not positive semidefinite", r.message), tl.logs)  # warns once

        # Test show method
        io = IOBuffer()
        show(io, ht)
        @test length(String(take!(io))) > 0

        # Error: wrong model types
        @test_throws ArgumentError hausman_test(re, fe)
    end

    @testset "Hausman generalized-inverse numerics (T085)" begin
        hq = MacroEconometricModels._hausman_quadratic_form
        # (a) full-rank PSD: equals the classical inverse quadratic form, df = k
        chi2, df, nonpsd = hq([1.0, 2.0], [2.0 0.0; 0.0 3.0])
        @test chi2 ≈ (1.0^2 / 2 + 2.0^2 / 3)
        @test df == 2 && nonpsd == false
        A = [1.5 0.4; 0.4 2.2]; b = [0.7, -1.1]
        @test hq(b, A)[1] ≈ dot(b, inv(A) * b)           # generalized inverse == inverse when PD
        # (b) indefinite dV: negative form NOT masked; flagged; df drops to the positive rank
        chi2n, dfn, np = hq([1.0, 1.0], [1.0 0.0; 0.0 -0.5])
        @test chi2n < 0 && np == true && dfn == 1
        # (c) rank-deficient PSD: Moore-Penrose inverse, reduced df, not flagged non-PSD
        chi2r, dfr, npr = hq([1.0, 1.0], [1.0 0.0; 0.0 0.0])
        @test chi2r ≈ 1.0 && dfr == 1 && npr == false
    end

    # =========================================================================
    # 2. Breusch-Pagan LM Test — Large sigma_u triggers rejection
    # =========================================================================
    @testset "Breusch-Pagan LM test" begin
        rng = Random.MersenneTwister(2345)
        N_g = 50; T_p = 20; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        # Large entity effects -> sigma_u^2 >> 0
        alpha = repeat(3.0 .* randn(rng, N_g), inner=T_p)
        x1 = randn(rng, n)
        y = alpha .+ 1.5 .* x1 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        re = estimate_xtreg(pd, :y, [:x1]; model=:re)
        bp = breusch_pagan_test(re)

        @test bp isa PanelTestResult{Float64}
        @test bp.statistic > 0
        @test bp.df == 1
        @test bp.pvalue < 0.05  # should reject: RE needed

        # Error: wrong model type
        fe = estimate_xtreg(pd, :y, [:x1]; model=:fe)
        @test_throws ArgumentError breusch_pagan_test(fe)
    end

    # =========================================================================
    # 3. F-test for FE — Large entity effects triggers rejection
    # =========================================================================
    @testset "F-test for fixed effects" begin
        rng = Random.MersenneTwister(3456)
        N_g = 30; T_p = 15; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        # Large entity effects
        alpha = repeat(5.0 .* randn(rng, N_g), inner=T_p)
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        y = alpha .+ 1.0 .* x1 .- 0.5 .* x2 .+ 0.3 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
        pd = xtset(df, :id, :t)

        fe = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
        ft = f_test_fe(fe)

        @test ft isa PanelTestResult{Float64}
        @test ft.statistic > 0
        @test ft.df == (N_g - 1, n - N_g - 2)
        @test ft.pvalue < 0.05  # should reject: entity effects present

        # Error: wrong model type
        re = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)
        @test_throws ArgumentError f_test_fe(re)
    end

    # =========================================================================
    # 4. Pesaran CD Test — Common shock creates cross-sectional dependence
    # =========================================================================
    @testset "Pesaran CD test" begin
        rng = Random.MersenneTwister(4567)
        N_g = 30; T_p = 20; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        # Common time shock creates cross-sectional dependence
        common_shock = repeat(2.0 .* randn(rng, T_p), N_g)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        x1 = randn(rng, n)
        y = alpha .+ 1.0 .* x1 .+ common_shock .+ 0.3 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        fe = estimate_xtreg(pd, :y, [:x1]; model=:fe)
        cd = pesaran_cd_test(fe)

        @test cd isa PanelTestResult{Float64}
        @test cd.df == 1
        @test cd.pvalue < 0.05  # should reject: cross-sectional dependence

        # Also works with RE
        re = estimate_xtreg(pd, :y, [:x1]; model=:re)
        cd_re = pesaran_cd_test(re)
        @test cd_re isa PanelTestResult{Float64}
    end

    # =========================================================================
    # 5. Wooldridge AR Test — AR(1) errors trigger serial correlation
    # =========================================================================
    @testset "Wooldridge AR(1) test" begin
        rng = Random.MersenneTwister(5678)
        N_g = 50; T_p = 25; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        alpha = repeat(randn(rng, N_g), inner=T_p)
        x1 = randn(rng, n)

        # Generate AR(1) errors with rho=0.7
        e = zeros(n)
        ar_rho = 0.7
        for g in 1:N_g
            offset = (g - 1) * T_p
            e[offset + 1] = randn(rng)
            for t in 2:T_p
                e[offset + t] = ar_rho * e[offset + t - 1] + randn(rng)
            end
        end
        y = alpha .+ 1.5 .* x1 .+ e

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        fe = estimate_xtreg(pd, :y, [:x1]; model=:fe)
        wt = wooldridge_ar_test(fe)

        @test wt isa PanelTestResult{Float64}
        @test wt.statistic > 0
        @test wt.df == (1, N_g - 1)
        @test wt.pvalue < 0.05  # should reject: serial correlation present

        # Error: wrong model type
        re = estimate_xtreg(pd, :y, [:x1]; model=:re)
        @test_throws ArgumentError wooldridge_ar_test(re)
    end

    # =========================================================================
    # 6. Modified Wald Test — Heterogeneous sigma_i triggers rejection
    # =========================================================================
    @testset "Modified Wald test" begin
        rng = Random.MersenneTwister(6789)
        N_g = 40; T_p = 20; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        alpha = repeat(randn(rng, N_g), inner=T_p)
        x1 = randn(rng, n)

        # Heterogeneous error variance across groups
        sigma_vec = repeat(0.5 .+ 3.0 .* rand(rng, N_g), inner=T_p)
        e = sigma_vec .* randn(rng, n)
        y = alpha .+ 2.0 .* x1 .+ e

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        fe = estimate_xtreg(pd, :y, [:x1]; model=:fe)
        mw = modified_wald_test(fe)

        @test mw isa PanelTestResult{Float64}
        @test mw.statistic > 0
        @test mw.df == N_g
        @test mw.pvalue < 0.05  # should reject: groupwise heteroskedasticity

        # Error: wrong model type
        re = estimate_xtreg(pd, :y, [:x1]; model=:re)
        @test_throws ArgumentError modified_wald_test(re)
    end

    # =========================================================================
    # Non-rejection cases (sanity checks)
    # =========================================================================
    @testset "Non-rejection under null" begin
        rng = Random.MersenneTwister(9999)
        N_g = 30; T_p = 20; n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)

        # No entity effects, no serial correlation, homoskedastic, no CSD
        x1 = randn(rng, n)
        y = 1.0 .+ 2.0 .* x1 .+ 0.5 .* randn(rng, n)

        df = DataFrame(id=ids, t=ts, x1=x1, y=y)
        pd = xtset(df, :id, :t)

        fe = estimate_xtreg(pd, :y, [:x1]; model=:fe)
        re = estimate_xtreg(pd, :y, [:x1]; model=:re)

        # Hausman: should NOT reject (RE is consistent when no correlation)
        ht = hausman_test(fe, re)
        @test ht.pvalue > 0.01

        # F-test FE: no entity effects -> should generally NOT reject
        ft = f_test_fe(fe)
        @test ft.pvalue > 0.01

        # Pesaran CD: iid errors -> should NOT reject
        cd = pesaran_cd_test(fe)
        @test cd.pvalue > 0.01
    end
end
