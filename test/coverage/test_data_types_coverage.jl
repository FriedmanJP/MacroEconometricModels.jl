# Coverage tests targeting 9 source files with low coverage:
#   src/data/convert.jl, src/data/types.jl, src/data/filter.jl, src/data/transform.jl,
#   src/lp/types.jl, src/arima/types.jl, src/arch/types.jl,
#   src/did/event_study.jl, src/factor/generalized.jl

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const M = MacroEconometricModels
const _suppress_warnings = M._suppress_warnings

@testset "Data Types Coverage" begin

    # =========================================================================
    # 1. src/data/convert.jl — TimeSeriesData dispatch wrappers
    # =========================================================================
    @testset "convert.jl — to_matrix / to_vector" begin
        rng = Random.MersenneTwister(7001)
        mat = randn(rng, 100, 3)
        ts = TimeSeriesData(mat; varnames=["a", "b", "c"])

        # to_matrix
        @test M.to_matrix(ts) === ts.data

        # to_vector for single-variable TimeSeriesData
        ts1 = TimeSeriesData(randn(rng, 50))
        @test M.to_vector(ts1) == ts1.data[:, 1]

        # to_vector with multi-var throws
        @test_throws ArgumentError M.to_vector(ts)

        # to_vector by index
        @test M.to_vector(ts, 2) == ts.data[:, 2]
        @test_throws BoundsError M.to_vector(ts, 0)
        @test_throws BoundsError M.to_vector(ts, 4)

        # to_vector by name
        @test M.to_vector(ts, "b") == ts.data[:, 2]
        @test_throws ArgumentError M.to_vector(ts, "nonexistent")
    end

    @testset "convert.jl — multivariate dispatches" begin
        rng = Random.MersenneTwister(7002)
        Y = randn(rng, 120, 3)
        ts = TimeSeriesData(Y; varnames=["y1", "y2", "y3"])

        # estimate_var dispatch
        m = estimate_var(ts, 2)
        @test m isa M.VARModel
        @test size(m.B, 1) > 0

        # estimate_bvar dispatch
        post = estimate_bvar(ts, 2; n_draws=50)
        @test post isa M.BVARPosterior

        # estimate_factors dispatch
        fac = estimate_factors(ts, 2)
        @test fac isa M.FactorModel

        # estimate_lp dispatch
        lp = estimate_lp(ts, 1, 4)
        @test lp isa M.LPModel

        # johansen_test dispatch
        coint = make_cointegrated_data(; T_obs=200, n=3, rank=1, seed=7002)
        ts_coint = TimeSeriesData(coint)
        jt = johansen_test(ts_coint, 2)
        @test jt isa M.JohansenResult
    end

    @testset "convert.jl — univariate dispatches" begin
        rng = Random.MersenneTwister(7003)
        y = make_ar1_data(; n=300, seed=7003)
        ts1 = TimeSeriesData(y; varname="gdp")

        # ARIMA family
        ar = estimate_ar(ts1, 2)
        @test ar isa M.ARModel
        ma = estimate_ma(ts1, 1; method=:css)
        @test ma isa M.MAModel
        arma = estimate_arma(ts1, 1, 1; method=:css)
        @test arma isa M.ARMAModel
        arima = estimate_arima(ts1, 1, 0, 1; method=:css)
        @test arima isa M.ARIMAModel

        # Filters
        hp = hp_filter(ts1)
        @test hp isa M.HPFilterResult
        ham = hamilton_filter(ts1)
        @test ham isa M.HamiltonFilterResult
        bn = beveridge_nelson(ts1)
        @test bn isa M.BeveridgeNelsonResult
        bk = baxter_king(ts1)
        @test bk isa M.BaxterKingResult
        bhp = boosted_hp(ts1)
        @test bhp isa M.BoostedHPResult

        # Unit root tests
        adf = adf_test(ts1)
        @test adf isa M.ADFResult
        kp = kpss_test(ts1)
        @test kp isa M.KPSSResult
        pp = pp_test(ts1)
        @test pp isa M.PPResult
        za = za_test(ts1)
        @test za isa M.ZAResult
        ng = ngperron_test(ts1)
        @test ng isa M.NgPerronResult
    end

    @testset "convert.jl — volatility dispatches" begin
        rng = Random.MersenneTwister(7004)
        y = simulate_arch1(; n=500, seed=7004)
        ts1 = TimeSeriesData(y; varname="vol")

        arch = estimate_arch(ts1, 1)
        @test arch isa M.ARCHModel
        garch = estimate_garch(ts1)
        @test garch isa M.GARCHModel
    end

    @testset "convert.jl — VECM dispatch" begin
        rng = Random.MersenneTwister(7005)
        coint = make_cointegrated_data(; T_obs=200, n=3, rank=1, seed=7005)
        ts = TimeSeriesData(coint; varnames=["x1", "x2", "x3"])
        vecm = estimate_vecm(ts, 2; rank=1)
        @test vecm isa M.VECMModel
    end

    # =========================================================================
    # 2. src/data/types.jl — constructors, getindex, rename, setters
    # =========================================================================
    @testset "types.jl — non-float fallbacks" begin
        # Integer matrix -> Float64
        ts_int = TimeSeriesData(ones(Int, 10, 2))
        @test eltype(ts_int.data) == Float64

        # Integer vector -> Float64
        ts_ivec = TimeSeriesData(collect(1:20))
        @test eltype(ts_ivec.data) == Float64

        # CrossSectionData non-float
        cs_int = CrossSectionData(ones(Int, 5, 3))
        @test eltype(cs_int.data) == Float64
    end

    @testset "types.jl — DataFrame constructors" begin
        # TimeSeriesData from DataFrame
        df = DataFrame(a=randn(30), b=randn(30), c=["str" for _ in 1:30])
        ts_df = TimeSeriesData(df)
        @test nobs(ts_df) == 30
        @test nvars(ts_df) == 2  # only numeric columns
        @test ts_df.varnames == ["a", "b"]

        # DataFrame with missing values -> NaN
        df_miss = DataFrame(x=[1.0, missing, 3.0], y=[missing, 2.0, 4.0])
        ts_miss = TimeSeriesData(df_miss)
        @test isnan(ts_miss.data[1, 2])
        @test isnan(ts_miss.data[2, 1])

        # CrossSectionData from DataFrame
        cs_df = CrossSectionData(df)
        @test cs_df.N_obs == 30
        @test cs_df.n_vars == 2
    end

    @testset "types.jl — getindex" begin
        rng = Random.MersenneTwister(7010)
        ts = TimeSeriesData(randn(rng, 50, 3); varnames=["GDP", "CPI", "FFR"])

        # Column by string
        v = ts[:, "CPI"]
        @test v == ts.data[:, 2]

        # Column by string vector -> sub-TimeSeriesData
        sub = ts[:, ["GDP", "FFR"]]
        @test sub isa TimeSeriesData
        @test nvars(sub) == 2
        @test sub.varnames == ["GDP", "FFR"]

        # Column by integer
        v2 = ts[:, 1]
        @test v2 == ts.data[:, 1]
        @test_throws BoundsError ts[:, 0]
        @test_throws BoundsError ts[:, 4]

        # Not-found string
        @test_throws ArgumentError ts[:, "FAKE"]
        @test_throws ArgumentError ts[:, ["GDP", "FAKE"]]

        # CrossSectionData getindex by string
        cs = CrossSectionData(randn(rng, 10, 2); varnames=["a", "b"])
        @test cs[:, "a"] == cs.data[:, 1]
        @test_throws ArgumentError cs[:, "nope"]
    end

    @testset "types.jl — date indexing" begin
        rng = Random.MersenneTwister(7011)
        ts = TimeSeriesData(randn(rng, 4, 2); varnames=["x", "y"])

        # No dates set -> error
        @test_throws ArgumentError ts["2020Q1", :]

        # Set dates and index
        set_dates!(ts, ["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
        row = ts["2020Q2", :]
        @test row == ts.data[2, :]

        # Not-found date
        @test_throws ArgumentError ts["2021Q1", :]

        # Date range
        sub = ts[["2020Q1", "2020Q2"], :]
        @test sub isa TimeSeriesData
        @test nobs(sub) == 2

        # Range with bad date
        @test_throws ArgumentError ts[["2020Q1", "2025Q1"], :]

        # No dates set -> range error
        ts2 = TimeSeriesData(randn(rng, 4, 2))
        @test_throws ArgumentError ts2[["a"], :]
    end

    @testset "types.jl — rename_vars!" begin
        rng = Random.MersenneTwister(7012)

        # TimeSeriesData pair rename
        ts = TimeSeriesData(randn(rng, 20, 2); varnames=["old1", "old2"],
                            vardesc=Dict("old1" => "desc1"))
        rename_vars!(ts, "old1" => "new1")
        @test ts.varnames[1] == "new1"
        @test haskey(ts.vardesc, "new1")
        @test !haskey(ts.vardesc, "old1")

        # TimeSeriesData vector rename
        rename_vars!(ts, ["A", "B"])
        @test ts.varnames == ["A", "B"]

        # PanelData rename
        df_pd = DataFrame(id=repeat(1:3, inner=10), t=repeat(1:10, 3),
                          x=randn(rng, 30), y=randn(rng, 30))
        pd = xtset(df_pd, :id, :t)
        rename_vars!(pd, "x" => "X_new")
        @test pd.varnames[1] == "X_new"
        rename_vars!(pd, ["W", "Z"])
        @test pd.varnames == ["W", "Z"]

        # CrossSectionData rename
        cs = CrossSectionData(randn(rng, 5, 2); varnames=["c1", "c2"],
                              vardesc=Dict("c1" => "desc"))
        rename_vars!(cs, "c1" => "C1")
        @test cs.varnames[1] == "C1"
        @test haskey(cs.vardesc, "C1")
        rename_vars!(cs, ["P", "Q"])
        @test cs.varnames == ["P", "Q"]

        # Error: not found
        @test_throws ArgumentError rename_vars!(ts, "nonexist" => "x")
        @test_throws ArgumentError rename_vars!(pd, "nonexist" => "x")
        @test_throws ArgumentError rename_vars!(cs, "nonexist" => "x")

        # Error: wrong length
        @test_throws ArgumentError rename_vars!(ts, ["one"])
    end

    @testset "types.jl — setters" begin
        rng = Random.MersenneTwister(7013)
        ts = TimeSeriesData(randn(rng, 10, 2); varnames=["a", "b"])

        # set_time_index!
        set_time_index!(ts, collect(101:110))
        @test ts.time_index == collect(101:110)
        @test_throws ArgumentError set_time_index!(ts, [1, 2])

        # set_dates! — overwrite existing
        set_dates!(ts, ["d$i" for i in 1:10])
        @test dates(ts) == ["d$i" for i in 1:10]
        set_dates!(ts, ["e$i" for i in 1:10])
        @test dates(ts)[1] == "e1"
        @test_throws ArgumentError set_dates!(ts, ["a"])

        # set_desc!
        set_desc!(ts, "test description")
        @test desc(ts) == "test description"

        # set_vardesc! single
        set_vardesc!(ts, "a", "variable a")
        @test vardesc(ts, "a") == "variable a"
        @test_throws ArgumentError set_vardesc!(ts, "nonexist", "x")

        # set_vardesc! dict
        set_vardesc!(ts, Dict("a" => "A desc", "b" => "B desc"))
        @test vardesc(ts, "b") == "B desc"
        @test_throws ArgumentError set_vardesc!(ts, Dict("fake" => "x"))

        # vardesc(d, name) for missing key
        @test_throws ArgumentError vardesc(ts, "nonexistent_key")

        # CrossSectionData set_obs_id!
        cs = CrossSectionData(randn(rng, 5, 2))
        M.set_obs_id!(cs, [10, 20, 30, 40, 50])
        @test M.obs_id(cs) == [10, 20, 30, 40, 50]
        @test_throws ArgumentError M.set_obs_id!(cs, [1])
    end

    @testset "types.jl — accessors & StatsAPI" begin
        rng = Random.MersenneTwister(7014)
        ts = TimeSeriesData(randn(rng, 50, 3); varnames=["a", "b", "c"],
                            frequency=M.Quarterly, desc="test")

        @test nobs(ts) == 50
        @test nvars(ts) == 3
        @test varnames(ts) == ["a", "b", "c"]
        @test frequency(ts) == M.Quarterly
        @test M.time_index(ts) == collect(1:50)
        @test desc(ts) == "test"
        @test vardesc(ts) isa Dict

        # PanelData accessors
        df_pd = DataFrame(id=repeat(1:2, inner=10), t=repeat(1:10, 2),
                          x=randn(rng, 20), y=randn(rng, 20))
        pd = xtset(df_pd, :id, :t; frequency=M.Monthly)
        @test nobs(pd) == 20
        @test nvars(pd) == 2
        @test frequency(pd) == M.Monthly

        # CrossSectionData accessors
        cs = CrossSectionData(randn(rng, 8, 2))
        @test nobs(cs) == 8
        @test nvars(cs) == 2
    end

    @testset "types.jl — show methods" begin
        rng = Random.MersenneTwister(7015)

        # TimeSeriesData show
        ts = TimeSeriesData(randn(rng, 20, 2); varnames=["A", "B"],
                            frequency=M.Quarterly, desc="My data")
        io = IOBuffer()
        show(io, ts)
        s = String(take!(io))
        @test occursin("TimeSeriesData", s)
        @test occursin("Quarterly", s)
        @test occursin("My data", s)

        # MIME show
        io2 = IOBuffer()
        show(io2, MIME("text/plain"), ts)
        @test !isempty(String(take!(io2)))

        # PanelData show
        df_pd = DataFrame(id=repeat(1:3, inner=5), t=repeat(1:5, 3),
                          x=randn(rng, 15))
        pd = xtset(df_pd, :id, :t)
        io3 = IOBuffer()
        show(io3, pd)
        s_pd = String(take!(io3))
        @test occursin("PanelData", s_pd)
        @test occursin("balanced", s_pd)

        io4 = IOBuffer()
        show(io4, MIME("text/plain"), pd)
        @test !isempty(String(take!(io4)))

        # CrossSectionData show
        cs = CrossSectionData(randn(rng, 5, 2); varnames=["v1", "v2"], desc="CS data")
        io5 = IOBuffer()
        show(io5, cs)
        s_cs = String(take!(io5))
        @test occursin("CrossSectionData", s_cs)
        @test occursin("CS data", s_cs)

        io6 = IOBuffer()
        show(io6, MIME("text/plain"), cs)
        @test !isempty(String(take!(io6)))
    end

    @testset "types.jl — size / length / Matrix / Vector" begin
        rng = Random.MersenneTwister(7016)
        ts = TimeSeriesData(randn(rng, 10, 3))
        @test size(ts) == (10, 3)
        @test length(ts) == 30
        @test Matrix(ts) === ts.data

        ts1 = TimeSeriesData(randn(rng, 10))
        @test Vector(ts1) == ts1.data[:, 1]
        @test_throws ArgumentError Vector(ts)  # multi-var

        df_pd = DataFrame(id=repeat(1:2, inner=5), t=repeat(1:5, 2), x=randn(rng, 10))
        pd = xtset(df_pd, :id, :t)
        @test size(pd) == (10, 1)
        @test length(pd) == 10
        @test Matrix(pd) === pd.data

        cs = CrossSectionData(randn(rng, 4, 2))
        @test size(cs) == (4, 2)
        @test length(cs) == 8
        @test Matrix(cs) === cs.data

        # to_matrix for PanelData and CrossSectionData
        @test M.to_matrix(pd) === pd.data
        @test M.to_matrix(cs) === cs.data
    end

    # =========================================================================
    # 3. src/data/filter.jl — apply_filter dispatches
    # =========================================================================
    @testset "filter.jl — apply_filter TimeSeriesData single symbol" begin
        rng = Random.MersenneTwister(7020)
        y = cumsum(randn(rng, 200, 2), dims=1)
        ts = TimeSeriesData(y; varnames=["GDP", "CPI"])

        # Apply :hp to all variables
        ts_hp = apply_filter(ts, :hp)
        @test ts_hp isa TimeSeriesData
        @test nvars(ts_hp) == 2

        # Apply :hp with component=:trend
        ts_trend = apply_filter(ts, :hp; component=:trend)
        @test nobs(ts_trend) == nobs(ts)

        # Apply to selected variables only
        ts_sel = apply_filter(ts, :hp; vars=["GDP"])
        @test nvars(ts_sel) == 2
    end

    @testset "filter.jl — apply_filter TimeSeriesData per-variable specs" begin
        rng = Random.MersenneTwister(7021)
        y = cumsum(randn(rng, 200, 3), dims=1)
        ts = TimeSeriesData(y; varnames=["A", "B", "C"])

        # Per-variable: HP, Hamilton, nothing (pass-through)
        ts_mixed = apply_filter(ts, [:hp, :hamilton, nothing])
        @test ts_mixed isa TimeSeriesData
        @test nvars(ts_mixed) == 3

        # With pre-computed result
        hp_res = hp_filter(ts.data[:, 1])
        ts_pre = apply_filter(ts, [hp_res, :hp, nothing])
        @test ts_pre isa TimeSeriesData

        # Tuple specs with component override
        ts_tuple = apply_filter(ts, [(:hp, :trend), (:hp, :cycle), nothing])
        @test ts_tuple isa TimeSeriesData
    end

    @testset "filter.jl — apply_filter PanelData" begin
        rng = Random.MersenneTwister(7022)
        df = DataFrame(id=repeat(1:3, inner=100), t=repeat(1:100, 3),
                       x=cumsum(randn(rng, 300)), y=cumsum(randn(rng, 300)))
        pd = xtset(df, :id, :t)

        pd_hp = apply_filter(pd, :hp; component=:cycle)
        @test pd_hp isa M.PanelData
        @test pd_hp.n_groups == 3
    end

    @testset "filter.jl — _filter_valid_range dispatches" begin
        rng = Random.MersenneTwister(7023)
        y = cumsum(randn(rng, 200))

        hp_r = hp_filter(y)
        @test M._filter_valid_range(hp_r) == 1:hp_r.T_obs

        ham_r = hamilton_filter(y)
        @test M._filter_valid_range(ham_r) == ham_r.valid_range

        bn_r = beveridge_nelson(y)
        @test M._filter_valid_range(bn_r) == 1:bn_r.T_obs

        bk_r = baxter_king(y)
        @test M._filter_valid_range(bk_r) == bk_r.valid_range

        bhp_r = boosted_hp(y)
        @test M._filter_valid_range(bhp_r) == 1:bhp_r.T_obs
    end

    # =========================================================================
    # 4. src/data/transform.jl — apply_tcode, inverse_tcode
    # =========================================================================
    @testset "transform.jl — apply_tcode all 7 codes" begin
        y = [100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 115.0, 112.0, 120.0, 125.0]

        # tcode 1: levels
        @test apply_tcode(y, 1) == y
        # tcode 2: first difference
        d2 = apply_tcode(y, 2)
        @test length(d2) == 9
        @test d2[1] ≈ 2.0
        # tcode 3: second difference
        d3 = apply_tcode(y, 3)
        @test length(d3) == 8
        # tcode 4: log
        d4 = apply_tcode(y, 4)
        @test d4 ≈ log.(y)
        # tcode 5: diff of log
        d5 = apply_tcode(y, 5)
        @test length(d5) == 9
        @test d5[1] ≈ log(102.0) - log(100.0)
        # tcode 6: second diff of log
        d6 = apply_tcode(y, 6)
        @test length(d6) == 8
        # tcode 7: delta pct change
        d7 = apply_tcode(y, 7)
        @test length(d7) == 8

        # Error: invalid tcode
        @test_throws ArgumentError apply_tcode(y, 0)
        @test_throws ArgumentError apply_tcode(y, 8)

        # Error: non-positive for tcode >= 4
        @test_throws ArgumentError apply_tcode([-1.0, 2.0, 3.0], 4)
    end

    @testset "transform.jl — apply_tcode TimeSeriesData" begin
        rng = Random.MersenneTwister(7030)
        mat = abs.(randn(rng, 100, 3)) .+ 1.0
        ts = TimeSeriesData(mat; varnames=["a", "b", "c"])

        # Per-variable codes
        ts2 = apply_tcode(ts, [5, 5, 1])
        @test nobs(ts2) < nobs(ts)
        @test ts2.tcode == [5, 5, 1]

        # Single code for all
        ts3 = apply_tcode(ts, 2)
        @test nobs(ts3) == nobs(ts) - 1

        # Default: use stored tcode (all 1s = no-op)
        ts4 = apply_tcode(ts)
        @test nobs(ts4) == nobs(ts)
    end

    @testset "transform.jl — apply_tcode PanelData" begin
        rng = Random.MersenneTwister(7031)
        df = DataFrame(id=repeat(1:2, inner=50), t=repeat(1:50, 2),
                       x=abs.(randn(rng, 100)) .+ 1.0,
                       y=abs.(randn(rng, 100)) .+ 1.0)
        pd = xtset(df, :id, :t)

        pd2 = apply_tcode(pd, [5, 1])
        @test pd2 isa M.PanelData
        @test pd2.T_obs < pd.T_obs

        # Single code
        pd3 = apply_tcode(pd, 2)
        @test pd3 isa M.PanelData
    end

    @testset "transform.jl — inverse_tcode all 7 codes" begin
        y = [100.0, 102.0, 105.0, 103.0, 108.0]

        # tcode 1: no-op
        @test inverse_tcode(y, 1) == y

        # tcode 4: exp(log(x)) = x
        @test inverse_tcode(log.(y), 4) ≈ y

        # tcode 2: cumsum from x_prev
        yd = apply_tcode(y, 2)
        y_rec = inverse_tcode(yd, 2; x_prev=[100.0])
        @test y_rec ≈ y[2:end]

        # tcode 3: double cumsum from x_prev
        yd3 = apply_tcode(y, 3)
        y_rec3 = inverse_tcode(yd3, 3; x_prev=[100.0, 102.0])
        @test y_rec3 ≈ y[3:end]

        # tcode 5: exp of cumsum of log-diffs
        yd5 = apply_tcode(y, 5)
        y_rec5 = inverse_tcode(yd5, 5; x_prev=[100.0])
        @test y_rec5 ≈ y[2:end] atol=1e-10

        # tcode 6
        yd6 = apply_tcode(y, 6)
        y_rec6 = inverse_tcode(yd6, 6; x_prev=[100.0, 102.0])
        @test y_rec6 ≈ y[3:end] atol=1e-10

        # tcode 7
        yd7 = apply_tcode(y, 7)
        y_rec7 = inverse_tcode(yd7, 7; x_prev=[100.0, 102.0])
        @test y_rec7 ≈ y[3:end] atol=1e-10

        # Error cases
        @test_throws ArgumentError inverse_tcode(y, 0)
        @test_throws ArgumentError inverse_tcode(y, 2)  # no x_prev
        @test_throws ArgumentError inverse_tcode(y, 2; x_prev=Float64[])
        @test_throws ArgumentError inverse_tcode(y, 3; x_prev=[1.0])  # need 2
        @test_throws ArgumentError inverse_tcode(y, 5)  # no x_prev
        @test_throws ArgumentError inverse_tcode(y, 5; x_prev=Float64[])
        @test_throws ArgumentError inverse_tcode(y, 6; x_prev=[1.0])
        @test_throws ArgumentError inverse_tcode(y, 7; x_prev=[1.0])
    end

    # =========================================================================
    # 5. src/lp/types.jl — LP StatsAPI dispatches
    # =========================================================================
    @testset "lp/types.jl — LPModel StatsAPI" begin
        rng = Random.MersenneTwister(7040)
        Y = randn(rng, 200, 3)
        lp = _suppress_warnings() do
            estimate_lp(Y, 1, 8; lags=2)
        end

        @test coef(lp) isa Vector{<:Matrix}
        @test coef(lp, 0) isa Matrix
        @test coef(lp, 4) isa Matrix
        @test residuals(lp) isa Vector{<:Matrix}
        @test residuals(lp, 0) isa Matrix
        @test vcov(lp) isa Vector{<:Matrix}
        @test vcov(lp, 0) isa Matrix
        @test nobs(lp) == 200
        @test nobs(lp, 0) > 0
        @test dof(lp) > 0
        @test islinear(lp) == true
    end

    @testset "lp/types.jl — LPIVModel StatsAPI" begin
        rng = Random.MersenneTwister(7041)
        Y = randn(rng, 200, 3)
        Z = randn(rng, 200, 1)  # instrument
        lpiv = _suppress_warnings() do
            estimate_lp_iv(Y, 1, Z, 6; lags=2)
        end

        @test coef(lpiv) isa Vector{<:Matrix}
        @test coef(lpiv, 0) isa Matrix
        @test residuals(lpiv) isa Vector{<:Matrix}
        @test vcov(lpiv) isa Vector{<:Matrix}
        @test nobs(lpiv) == 200
        @test islinear(lpiv) == true
    end

    @testset "lp/types.jl — SmoothLPModel StatsAPI" begin
        rng = Random.MersenneTwister(7042)
        Y = randn(rng, 200, 3)
        slp = _suppress_warnings() do
            estimate_smooth_lp(Y, 1, 8; lags=2)
        end

        @test coef(slp) isa Matrix
        @test residuals(slp) isa Matrix
        @test vcov(slp) isa Matrix
        @test nobs(slp) == 200
        @test islinear(slp) == true
    end

    @testset "lp/types.jl — StateLPModel StatsAPI" begin
        rng = Random.MersenneTwister(7043)
        Y = randn(rng, 200, 3)
        sv = randn(rng, 200)  # state variable
        stlp = _suppress_warnings() do
            estimate_state_lp(Y, 1, sv, 6; lags=2)
        end

        @test residuals(stlp) isa Vector{<:Matrix}
        @test residuals(stlp, 0) isa Matrix
        @test nobs(stlp) == 200
        @test islinear(stlp) == true
    end

    if !FAST
    @testset "lp/types.jl — PropensityLPModel StatsAPI" begin
        rng = Random.MersenneTwister(7044)
        n = 200
        Y = randn(rng, n, 3)
        treatment = rand(rng, Bool, n)
        covariates = randn(rng, n, 2)
        plp = _suppress_warnings() do
            M.estimate_propensity_lp(Y, treatment, covariates, 4)
        end

        @test coef(plp) isa Vector{<:Matrix}
        @test residuals(plp) isa Vector{<:Matrix}
        @test vcov(plp) isa Vector{<:Matrix}
        @test nobs(plp) == n
        @test islinear(plp) == true
    end
    end

    # =========================================================================
    # 6. src/arima/types.jl — order accessors, StatsAPI, show
    # =========================================================================
    @testset "arima/types.jl — order accessors" begin
        rng = Random.MersenneTwister(7050)
        y = make_ar1_data(; n=300, seed=7050)

        ar = estimate_ar(y, 2)
        @test ar_order(ar) == 2
        @test ma_order(ar) == 0
        @test diff_order(ar) == 0

        ma = estimate_ma(y, 1; method=:css)
        @test ar_order(ma) == 0
        @test ma_order(ma) == 1
        @test diff_order(ma) == 0

        arma = estimate_arma(y, 1, 1; method=:css)
        @test ar_order(arma) == 1
        @test ma_order(arma) == 1
        @test diff_order(arma) == 0

        arima = estimate_arima(y, 1, 1, 1; method=:css)
        @test ar_order(arima) == 1
        @test ma_order(arima) == 1
        @test diff_order(arima) == 1
    end

    @testset "arima/types.jl — StatsAPI" begin
        rng = Random.MersenneTwister(7051)
        y = make_ar1_data(; n=300, seed=7051)

        ar = estimate_ar(y, 2)
        @test nobs(ar) == 300
        @test length(coef(ar)) == 3  # c + phi1 + phi2
        @test residuals(ar) isa Vector
        @test predict(ar) isa Vector  # fitted
        @test isfinite(loglikelihood(ar))
        @test isfinite(aic(ar))
        @test isfinite(bic(ar))
        @test dof(ar) == 2 + 0 + 2  # ar + ma + 2
        @test StatsAPI.dof_residual(ar) > 0
        @test islinear(ar) == true

        # r2
        r2_val = r2(ar)
        @test 0.0 <= r2_val <= 1.0

        # vcov
        V = vcov(ar)
        @test size(V, 1) == length(coef(ar))

        # confint
        ci = confint(ar)
        @test size(ci, 2) == 2
        @test all(ci[:, 1] .<= ci[:, 2])

        # MA model StatsAPI
        ma = estimate_ma(y, 1; method=:css)
        @test length(coef(ma)) == 2  # c + theta
        @test nobs(ma) == 300

        # ARMA model StatsAPI
        arma = estimate_arma(y, 1, 1; method=:css)
        @test length(coef(arma)) == 3  # c + phi + theta

        # ARIMA model StatsAPI
        arima = estimate_arima(y, 1, 1, 1; method=:css)
        @test length(coef(arima)) == 3  # c + phi + theta
    end

    @testset "arima/types.jl — show methods" begin
        rng = Random.MersenneTwister(7052)
        y = make_ar1_data(; n=300, seed=7052)

        ar = estimate_ar(y, 2)
        io = IOBuffer()
        show(io, ar)
        s = String(take!(io))
        @test occursin("AR(2)", s)

        ma = estimate_ma(y, 1; method=:css)
        io = IOBuffer()
        show(io, ma)
        @test occursin("MA(1)", String(take!(io)))

        arma = estimate_arma(y, 1, 1; method=:css)
        io = IOBuffer()
        show(io, arma)
        @test occursin("ARMA(1,1)", String(take!(io)))

        arima = estimate_arima(y, 1, 1, 1; method=:css)
        io = IOBuffer()
        show(io, arima)
        @test occursin("ARIMA(1,1,1)", String(take!(io)))
    end

    @testset "arima/types.jl — ARIMAOrderSelection show" begin
        rng = Random.MersenneTwister(7053)
        y = make_ar1_data(; n=200, seed=7053)
        sel = M.select_arima_order(y, 2, 2)
        @test sel isa M.ARIMAOrderSelection
        io = IOBuffer()
        show(io, sel)
        s = String(take!(io))
        @test occursin("ARIMA Order Selection", s)
        @test occursin("AIC", s)
        @test occursin("BIC", s)
    end

    @testset "arima/types.jl — ARIMAForecast show" begin
        rng = Random.MersenneTwister(7054)
        y = make_ar1_data(; n=200, seed=7054)
        ar = estimate_ar(y, 2)
        fc = forecast(ar, 12)
        io = IOBuffer()
        show(io, fc)
        s = String(take!(io))
        @test occursin("ARIMA Forecast", s)
        @test occursin("h=12", s)

        # Long forecast that triggers "..." truncation
        fc_long = forecast(ar, 15)
        io2 = IOBuffer()
        show(io2, fc_long)
        s2 = String(take!(io2))
        @test occursin("more", s2)
    end

    # =========================================================================
    # 7. src/arch/types.jl — volatility accessors, StatsAPI, show
    # =========================================================================
    @testset "arch/types.jl — ARCH accessors" begin
        y = simulate_arch1(; n=1000, seed=7060)
        m = estimate_arch(y, 1)

        @test arch_order(m) == 1
        @test persistence(m) ≈ sum(m.alpha)
        @test isfinite(halflife(m))
        @test isfinite(unconditional_variance(m))
    end

    @testset "arch/types.jl — ARCH StatsAPI" begin
        y = simulate_arch1(; n=1000, seed=7061)
        m = estimate_arch(y, 1)

        @test nobs(m) == 1000
        @test length(coef(m)) == 3  # mu + omega + alpha1
        @test residuals(m) isa Vector
        @test predict(m) isa Vector  # conditional variance
        @test isfinite(loglikelihood(m))
        @test isfinite(aic(m))
        @test isfinite(bic(m))
        @test dof(m) == 3  # 2 + q
        @test StatsAPI.dof_residual(m) > 0
        @test islinear(m) == false

        # vcov and confint (shared AbstractVolatilityModel dispatches)
        V = vcov(m)
        @test size(V, 1) == length(coef(m))
        ci = confint(m)
        @test size(ci, 2) == 2
    end

    @testset "arch/types.jl — ARCH show" begin
        y = simulate_arch1(; n=500, seed=7062)
        m = estimate_arch(y, 1)
        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test occursin("ARCH(1)", s)
        @test occursin("Persistence", s)
    end

    @testset "arch/types.jl — VolatilityForecast show" begin
        y = simulate_arch1(; n=500, seed=7063)
        m = estimate_arch(y, 1)
        fc = forecast(m, 10)
        io = IOBuffer()
        show(io, fc)
        s = String(take!(io))
        @test occursin("Volatility Forecast", s)
        @test occursin("arch", s)

        # Long forecast triggers truncation
        fc_long = forecast(m, 15)
        io2 = IOBuffer()
        show(io2, fc_long)
        s2 = String(take!(io2))
        @test occursin("more", s2)
    end

    @testset "arch/types.jl — GARCH/EGARCH/GJR-GARCH accessors" begin
        y = simulate_garch11(; n=1000, seed=7064)

        garch = estimate_garch(y)
        @test arch_order(garch) == 1
        @test M.garch_order(garch) == 1
        @test persistence(garch) ≈ sum(garch.alpha) + sum(garch.beta)
        @test isfinite(halflife(garch))
        @test isfinite(unconditional_variance(garch))

        egarch = estimate_egarch(y)
        @test arch_order(egarch) == 1
        @test M.garch_order(egarch) == 1
        @test persistence(egarch) ≈ sum(egarch.beta)
        @test isfinite(unconditional_variance(egarch))

        gjr = estimate_gjr_garch(y)
        @test arch_order(gjr) == 1
        @test M.garch_order(gjr) == 1
        @test persistence(gjr) ≈ sum(gjr.alpha) + sum(gjr.gamma) / 2 + sum(gjr.beta)
        @test isfinite(unconditional_variance(gjr))

        # GARCH/EGARCH/GJR show
        for (mod, label) in [(garch, "GARCH"), (egarch, "EGARCH"), (gjr, "GJR-GARCH")]
            io = IOBuffer()
            show(io, mod)
            s = String(take!(io))
            @test occursin(label, s)
        end

        # GARCH/EGARCH/GJR StatsAPI
        for mod in [garch, egarch, gjr]
            @test nobs(mod) == 1000
            @test length(coef(mod)) > 0
            @test residuals(mod) isa Vector
            @test predict(mod) isa Vector
            @test isfinite(loglikelihood(mod))
            @test isfinite(aic(mod))
            @test isfinite(bic(mod))
            @test dof(mod) > 0
            @test StatsAPI.dof_residual(mod) > 0
            @test islinear(mod) == false
        end
    end

    @testset "arch/types.jl — halflife edge cases" begin
        y = simulate_arch1(; n=500, seed=7065, omega=0.1, alpha1=0.99)
        m = estimate_arch(y, 1)
        # High persistence: halflife may be very large or Inf
        hl = halflife(m)
        @test hl >= 0 || hl == Inf
    end

    # =========================================================================
    # 8. src/did/event_study.jl — estimate_lp_did and estimate_event_study_lp
    # =========================================================================
    @testset "event_study.jl — estimate_event_study_lp with covariates" begin
        rng = Random.MersenneTwister(7070)
        n_units, n_periods = 30, 20
        treat_times = zeros(Int, n_units)
        # First 10 units treated at period 8
        for u in 1:10
            treat_times[u] = 8
        end
        # Units 11-30 are never-treated

        N_obs = n_units * n_periods
        data = Matrix{Float64}(undef, N_obs, 3)
        group_id = Vector{Int}(undef, N_obs)
        time_id = Vector{Int}(undef, N_obs)

        row = 1
        for i in 1:n_units
            alpha_i = randn(rng)
            for t in 1:n_periods
                te = (treat_times[i] > 0 && t >= treat_times[i]) ? 2.0 : 0.0
                y = alpha_i + 0.1 * t + te + 0.5 * randn(rng)
                data[row, 1] = y
                data[row, 2] = Float64(treat_times[i])
                data[row, 3] = randn(rng)  # covariate
                group_id[row] = i
                time_id[row] = t
                row += 1
            end
        end

        group_names = ["unit_$i" for i in 1:n_units]
        pd = PanelData{Float64}(data, ["outcome", "treat_time", "covariate"],
                                M.Quarterly, [1, 1, 1], group_id, time_id,
                                nothing, group_names, n_units, 3, N_obs, true,
                                ["test"], Dict{String,String}(), Symbol[])

        # Standard event study LP
        eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
                                        leads=2, lags=2, cluster=:unit)
        @test eslp isa M.EventStudyLP{Float64}
        @test eslp.clean_controls == false
        @test length(eslp.coefficients) == length(eslp.event_times)

        # With covariates
        eslp_cov = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
                                            leads=2, lags=2,
                                            covariates=["covariate"],
                                            cluster=:unit)
        @test eslp_cov isa M.EventStudyLP{Float64}
        @test length(eslp_cov.coefficients) > 0
    end

    @testset "event_study.jl — estimate_lp_did with clean controls" begin
        rng = Random.MersenneTwister(7071)
        n_units, n_periods = 30, 20
        treat_times = zeros(Int, n_units)
        for u in 1:10
            treat_times[u] = 8
        end

        N_obs = n_units * n_periods
        data = Matrix{Float64}(undef, N_obs, 3)
        group_id = Vector{Int}(undef, N_obs)
        time_id = Vector{Int}(undef, N_obs)

        row = 1
        for i in 1:n_units
            alpha_i = randn(rng)
            for t in 1:n_periods
                te = (treat_times[i] > 0 && t >= treat_times[i]) ? 2.0 : 0.0
                y = alpha_i + 0.1 * t + te + 0.5 * randn(rng)
                data[row, 1] = y
                data[row, 2] = Float64(treat_times[i])
                data[row, 3] = randn(rng)
                group_id[row] = i
                time_id[row] = t
                row += 1
            end
        end

        group_names = ["unit_$i" for i in 1:n_units]
        pd = PanelData{Float64}(data, ["outcome", "treat_time", "cov"],
                                M.Other, [1, 1, 1], group_id, time_id,
                                nothing, group_names, n_units, 3, N_obs, true,
                                [""], Dict{String,String}(), Symbol[])

        # LP-DiD with clean controls
        lpdid = estimate_lp_did(pd, "outcome", "treat_time", 5;
                                 pre_window=2, post_window=2, cluster=:unit)
        @test lpdid isa M.LPDiDResult{Float64}
        @test all(lpdid.se .>= 0)

        # LP-DiD with covariates
        lpdid_cov = estimate_lp_did(pd, "outcome", "treat_time", 5;
                                     pre_window=2, post_window=2,
                                     covariates=["cov"],
                                     cluster=:unit)
        @test lpdid_cov isa M.LPDiDResult{Float64}
    end

    # =========================================================================
    # 9. src/factor/generalized.jl — ic_criteria_gdfm, forecast CI, StatsAPI
    # =========================================================================
    @testset "generalized.jl — GDFM StatsAPI" begin
        rng = Random.MersenneTwister(7080)
        T_obs, N, q = 150, 15, 2
        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        @test predict(model) == model.common_component
        @test residuals(model) == model.idiosyncratic
        @test nobs(model) == T_obs
        @test dof(model) > 0

        r2_vals = r2(model)
        @test length(r2_vals) == N
        @test all(isfinite, r2_vals)

        # common_variance_share
        cvs = common_variance_share(model)
        @test length(cvs) == N
        @test all(0 .<= cvs .<= 1.0 + 0.01)

        # spectral_eigenvalue_plot_data
        spd = spectral_eigenvalue_plot_data(model)
        @test haskey(spd, :frequencies)
        @test haskey(spd, :eigenvalues)
    end

    @testset "generalized.jl — GDFM show" begin
        rng = Random.MersenneTwister(7081)
        X = randn(rng, 120, 10)
        model = estimate_gdfm(X, 2)
        io = IOBuffer()
        show(io, model)
        s = String(take!(io))
        @test occursin("Generalized Dynamic Factor Model", s)
        @test occursin("q=2", s)
    end

    @testset "generalized.jl — forecast ci_method=:none" begin
        rng = Random.MersenneTwister(7082)
        T_obs, N, q = 120, 10, 2
        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        fc_none = forecast(model, 5; ci_method=:none)
        @test fc_none isa M.FactorForecast
        @test fc_none.ci_method == :none
        @test size(fc_none.observables) == (5, N)
        @test all(isfinite, fc_none.observables)
    end

    @testset "generalized.jl — forecast ci_method=:theoretical" begin
        rng = Random.MersenneTwister(7083)
        T_obs, N, q = 120, 10, 2
        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        fc_th = forecast(model, 5; ci_method=:theoretical)
        @test fc_th isa M.FactorForecast
        @test fc_th.ci_method == :theoretical
        @test size(fc_th.observables_lower) == (5, N)
        @test size(fc_th.observables_upper) == (5, N)
        @test all(fc_th.observables_lower .<= fc_th.observables_upper .+ 1e-10)
    end

    @testset "generalized.jl — forecast ci_method=:bootstrap" begin
        rng = Random.MersenneTwister(7084)
        T_obs, N, q = 100, 8, 2
        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        fc_boot = forecast(model, 3; ci_method=:bootstrap, n_boot=100)
        @test fc_boot isa M.FactorForecast
        @test fc_boot.ci_method == :bootstrap
        @test size(fc_boot.factors) == (3, q)
        @test all(isfinite, fc_boot.observables)
    end

    @testset "generalized.jl — ic_criteria_gdfm" begin
        rng = Random.MersenneTwister(7085)
        T_obs, N = 150, 12
        max_q = 4
        X = randn(rng, T_obs, N)

        ic = ic_criteria_gdfm(X, max_q)
        @test length(ic.eigenvalue_ratios) >= 1
        @test length(ic.cumulative_variance) == max_q
        @test 1 <= ic.q_ratio <= max_q
        @test 1 <= ic.q_variance <= max_q
        @test issorted(ic.cumulative_variance)
    end

    @testset "generalized.jl — GDFM with non-float fallback" begin
        rng = Random.MersenneTwister(7086)
        X_int = rand(rng, 1:10, 100, 8)
        model = estimate_gdfm(X_int, 2)
        @test model isa M.GeneralizedDynamicFactorModel{Float64}
    end

    @testset "generalized.jl — forecast errors" begin
        rng = Random.MersenneTwister(7087)
        X = randn(rng, 100, 8)
        model = estimate_gdfm(X, 2)

        @test_throws ArgumentError forecast(model, 0)  # h < 1
        @test_throws ArgumentError forecast(model, 5; method=:invalid)
        @test_throws ArgumentError forecast(model, 5; ci_method=:invalid)
    end

end  # @testset "Data Types Coverage"
