# EV-25 (#433): Beck-Katz panel-corrected SE (PCSE) + Prais-Winsten AR(1) FGLS.
#
# Oracle discipline (Stata `xtpcse` / `plm` are NOT available in this environment):
#   Layer 1 — analytic identity / limiting case (PRIMARY):
#     * the time-by-time PCSE meat equals the block-Kronecker (Σ̂⊗I) sandwich,
#       recomputed independently inside the test on a small balanced panel;
#     * a diagonal, homoskedastic, balanced Σ̂ (no cross-section correlation) makes
#       PCSE SEs ≈ OLS SEs (loose tolerance).
#   Layer 2 — hand recomputation of the AR(1) formula and ρ̂ recovery on a seeded
#     AR(1) panel (ρ̂ ≈ true ρ), and the Prais-Winsten first-obs √(1-ρ̂²) weight is
#     PRESENT (omitting it silently degrades to Cochrane-Orcutt).
#   Layer 3 — self-consistency: :pcse point estimates == :ols point estimates.
# No published `xtpcse` reference numerics are hard-coded (Stata unavailable here);
# where such a value would go, the property/identity oracles above stand in.

using Test, MacroEconometricModels, DataFrames, Random, Statistics, LinearAlgebra
using StatsAPI: coef, vcov, stderror

const MEM = MacroEconometricModels

# Helper: build a balanced panel DataFrame + xtset.
function _balanced_panel(rng, N, T; rho=0.0, beta=(1.5, -0.8), sig=0.4)
    ids = repeat(1:N, inner=T)
    ts = repeat(1:T, N)
    n = N * T
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    alpha = repeat(randn(rng, N), inner=T)
    # AR(1) idiosyncratic error within each unit
    u = zeros(n)
    for i in 1:N
        e = randn(rng, T)
        v = zeros(T)
        v[1] = e[1] / sqrt(1 - rho^2 + 1e-12)
        for t in 2:T
            v[t] = rho * v[t-1] + e[t]
        end
        u[(i-1)*T+1:i*T] .= sig .* v
    end
    y = alpha .+ beta[1] .* x1 .+ beta[2] .* x2 .+ u
    df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=y)
    return xtset(df, :id, :t)
end

@testset "PCSE + Prais-Winsten (EV-25)" begin

    # -------------------------------------------------------------------------
    @testset "End-to-end: :pcse runs, point estimates unchanged vs :ols" begin
        rng = MersenneTwister(433)
        pd = _balanced_panel(rng, 10, 40)
        m_ols = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:ols)
        m_pcse = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:pcse)

        @test length(stderror(m_pcse)) == 2
        @test all(stderror(m_pcse) .> 0)
        # :pcse changes ONLY the vcov — point estimates identical to :ols.
        @test coef(m_pcse) ≈ coef(m_ols) atol=1e-10
        # vcov genuinely differs from OLS.
        @test !(vcov(m_pcse) ≈ vcov(m_ols))
        # header advertises Beck-Katz.
        s = sprint(show, m_pcse)
        @test occursin("Panel-corrected", s)
    end

    # -------------------------------------------------------------------------
    @testset "Analytic identity: time-by-time meat == block-Kronecker sandwich" begin
        # Small balanced panel; recompute the Beck-Katz sandwich two ways.
        rng = MersenneTwister(7)
        N, T = 4, 12
        pd = _balanced_panel(rng, N, T; rho=0.0)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:pcse)

        # Reconstruct within-demeaned design + residuals (FE, no intercept).
        y = pd.data[:, findfirst(==("y"), pd.varnames)]
        X = pd.data[:, [findfirst(==("x1"), pd.varnames), findfirst(==("x2"), pd.varnames)]]
        groups = pd.group_id; time_ids = pd.time_id
        ug = sort(unique(groups))
        ydm, _ = MEM._within_demean(y, groups, ug)
        Xdm, _ = MEM._within_demean_matrix(X, groups, ug)
        XtXinv = inv(Xdm' * Xdm)
        beta = XtXinv * (Xdm' * ydm)
        resid = ydm .- Xdm * beta

        V_fun = MEM._panel_pcse_vcov(Xdm, resid, XtXinv, groups, time_ids; unbalanced=:casewise)

        # Independent block-Kronecker recomputation (balanced ⇒ unit-major ordering).
        # E is N×T residual panel; Σ̂ = E E' / T. Ω = kron(Σ̂, I_T) in unit-major order.
        E = zeros(N, T)
        Xum = zeros(N * T, 2)
        rowmap = Dict((g, t) => r for (r, (g, t)) in enumerate(zip(groups, time_ids)))
        for (i, g) in enumerate(ug), (jt, t) in enumerate(sort(unique(time_ids)))
            r = rowmap[(g, t)]
            E[i, jt] = resid[r]
            Xum[(i-1)*T+jt, :] .= Xdm[r, :]
        end
        Sig = (E * E') ./ T
        Omega = kron(Sig, Matrix(1.0I, T, T))
        meat_ref = Xum' * Omega * Xum
        V_ref = XtXinv * meat_ref * XtXinv

        @test V_fun ≈ V_ref atol=1e-9 rtol=1e-8

        # vcov() from the model equals the hand sandwich too.
        @test vcov(m) ≈ V_ref atol=1e-9 rtol=1e-8
    end

    # -------------------------------------------------------------------------
    @testset "Limiting case: diagonal homoskedastic Σ̂ ⇒ PCSE SE ≈ OLS SE" begin
        # Cross-sectionally independent, homoskedastic errors + long T ⇒ Σ̂ → σ²I,
        # so PCSE V → σ²(X'X)⁻¹ ≈ OLS V. Loose tolerance (σ² uses /T vs /(n-k)).
        rng = MersenneTwister(2024)
        pd = _balanced_panel(rng, 6, 250; rho=0.0, sig=0.5)
        m_ols = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:ols)
        m_pcse = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:pcse)
        ratio = stderror(m_pcse) ./ stderror(m_ols)
        @test all(abs.(ratio .- 1) .< 0.15)
    end

    # -------------------------------------------------------------------------
    @testset "PCSE meat never inverts Σ̂ — hand check of accumulation" begin
        # Directly verify Σ_t X_t' Σ̂ X_t on a tiny hand example.
        # N=2, T=3 balanced. Build residuals/X and compare to explicit sum.
        groups = [1, 1, 1, 2, 2, 2]
        time_ids = [1, 2, 3, 1, 2, 3]
        resid = Float64[1.0, -0.5, 0.3, 0.2, 0.4, -0.1]
        X = Float64[1.0 0.0; 0.5 1.0; -1.0 0.5; 0.3 -0.2; 0.8 0.1; -0.4 0.7]
        XtXinv = inv(X' * X)
        V = MEM._panel_pcse_vcov(X, resid, XtXinv, groups, time_ids; unbalanced=:casewise)

        # Σ̂ over all 3 (fully-observed) periods.
        E = [resid[1] resid[2] resid[3]; resid[4] resid[5] resid[6]]
        Sig = (E * E') / 3
        meat = zeros(2, 2)
        for t in 1:3
            rows = [t, t + 3]           # unit1 at t, unit2 at t
            Xt = X[rows, :]
            meat .+= Xt' * Sig * Xt
        end
        @test V ≈ XtXinv * meat * XtXinv atol=1e-12
    end

    # -------------------------------------------------------------------------
    @testset "Prais-Winsten ρ̂ recovery ≈ true ρ (:common)" begin
        # Feed an AR(1) residual series with known ρ; the estimator should recover it.
        rng = MersenneTwister(99)
        N, T = 12, 300
        rho_true = 0.6
        groups = repeat(1:N, inner=T)
        time_ids = repeat(1:T, N)
        resid = zeros(N * T)
        for i in 1:N
            e = randn(rng, T)
            v = zeros(T); v[1] = e[1]
            for t in 2:T; v[t] = rho_true * v[t-1] + e[t]; end
            resid[(i-1)*T+1:i*T] .= v
        end
        ug = sort(unique(groups))
        rvec = MEM._prais_winsten_rho(resid, groups, time_ids, ug, :common)
        @test all(rvec .== rvec[1])            # common ⇒ constant
        @test isapprox(rvec[1], rho_true; atol=0.05)

        # panel_specific ⇒ per-unit, each near the true ρ.
        rvec_ps = MEM._prais_winsten_rho(resid, groups, time_ids, ug, :panel_specific)
        @test length(rvec_ps) == N
        @test isapprox(mean(rvec_ps), rho_true; atol=0.05)
    end

    # -------------------------------------------------------------------------
    @testset "PW first-obs √(1-ρ²) weight is PRESENT (not Cochrane-Orcutt)" begin
        # 2 units, 4 periods. Known residuals give a specific ρ̂; check first row.
        groups = [1, 1, 1, 1, 2, 2, 2, 2]
        time_ids = [1, 2, 3, 4, 1, 2, 3, 4]
        # simple AR(1)-ish resid so ρ̂ well-defined and nonzero
        resid = Float64[1.0, 0.6, 0.36, 0.2, 0.8, 0.5, 0.3, 0.15]
        y = collect(1.0:8.0)
        X = reshape(collect(11.0:18.0), 8, 1)

        ug = sort(unique(groups))
        rvec = MEM._prais_winsten_rho(resid, groups, time_ids, ug, :common)
        rho = rvec[1]
        @test rho > 0
        w = sqrt(1 - rho^2)

        ypw, Xpw, rho_out = MEM._prais_winsten_transform(y, X, groups, time_ids, resid; ar1=:common)
        @test rho_out ≈ rho

        # Unit 1 first obs (row 1): must be √(1-ρ²)·y₁, NOT dropped and NOT a difference.
        @test ypw[1] ≈ w * y[1] atol=1e-12
        @test Xpw[1, 1] ≈ w * X[1, 1] atol=1e-12
        # It differs from a plain first-difference (Cochrane-Orcutt would drop the first obs
        # entirely; a naive quasi-difference against a zero would give y₁).
        @test !isapprox(ypw[1], y[1]; atol=1e-6)
        # Subsequent obs: quasi-difference y_t - ρ y_{t-1}.
        @test ypw[2] ≈ y[2] - rho * y[1] atol=1e-12
        @test ypw[3] ≈ y[3] - rho * y[2] atol=1e-12
        # Unit 2 restarts its own first-obs weighting.
        @test ypw[5] ≈ w * y[5] atol=1e-12
    end

    # -------------------------------------------------------------------------
    @testset "estimate_xtreg ar1=:common / :panel_specific run end-to-end" begin
        rng = MersenneTwister(101)
        pd = _balanced_panel(rng, 10, 40; rho=0.5)

        m_c = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:pcse, ar1=:common)
        @test m_c.ar1_rho isa Real
        @test -1 < m_c.ar1_rho < 1
        @test all(stderror(m_c) .> 0)

        m_ps = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:pcse, ar1=:panel_specific)
        @test m_ps.ar1_rho isa AbstractVector
        @test length(m_ps.ar1_rho) == 10

        # AR(1) FGLS changes the point estimates (unlike :pcse alone).
        m_plain = estimate_xtreg(pd, :y, [:x1, :x2]; cov_type=:pcse, ar1=:none)
        @test m_plain.ar1_rho === nothing
        @test !(coef(m_c) ≈ coef(m_plain))

        # ρ̂ appears in report output.
        s = sprint(show, m_c)
        @test occursin("AR(1) rho", s)

        # ar1 rejected for unsupported models.
        @test_throws ArgumentError estimate_xtreg(pd, :y, [:x1, :x2]; model=:fd, ar1=:common)
    end

    # -------------------------------------------------------------------------
    @testset "Unbalanced: :casewise and :pairwise both run" begin
        rng = MersenneTwister(55)
        pd = _balanced_panel(rng, 8, 30)
        # Drop some rows to unbalance (keep every period fully observed for at least casewise).
        df = DataFrame(id=pd.group_id, t=pd.time_id,
                       x1=pd.data[:, findfirst(==("x1"), pd.varnames)],
                       x2=pd.data[:, findfirst(==("x2"), pd.varnames)],
                       y=pd.data[:, findfirst(==("y"), pd.varnames)])
        # remove a scattered handful of (id,t) but never wipe an entire period for casewise:
        drop = falses(nrow(df))
        drop[3] = true; drop[57] = true; drop[120] = true
        df2 = df[.!drop, :]
        pd2 = xtset(df2, :id, :t)

        m_cw = estimate_xtreg(pd2, :y, [:x1, :x2]; cov_type=:pcse, pcse_unbalanced=:casewise)
        @test all(stderror(m_cw) .> 0)
        m_pw = estimate_xtreg(pd2, :y, [:x1, :x2]; cov_type=:pcse, pcse_unbalanced=:pairwise)
        @test all(stderror(m_pw) .> 0)
    end

    # -------------------------------------------------------------------------
    @testset "T<N casewise ⇒ warn (no garbage inverse)" begin
        # N=6 units, only T=3 fully-observed periods ⇒ rank-deficient Σ̂.
        rng = MersenneTwister(3)
        pd = _balanced_panel(rng, 6, 3)
        y = pd.data[:, findfirst(==("y"), pd.varnames)]
        X = pd.data[:, [findfirst(==("x1"), pd.varnames), findfirst(==("x2"), pd.varnames)]]
        groups = pd.group_id; time_ids = pd.time_id
        ug = sort(unique(groups))
        ydm, _ = MEM._within_demean(y, groups, ug)
        Xdm, _ = MEM._within_demean_matrix(X, groups, ug)
        XtXinv = inv(Xdm' * Xdm)
        beta = XtXinv * (Xdm' * ydm)
        resid = ydm .- Xdm * beta
        V = @test_logs (:warn,) match_mode=:any MEM._panel_pcse_vcov(
            Xdm, resid, XtXinv, groups, time_ids; unbalanced=:casewise)
        @test all(isfinite, V)   # finite, not NaN — Σ̂ never inverted
    end

    # -------------------------------------------------------------------------
    @testset "refs() renders for PanelRegModel (includes PCSE/PW keys)" begin
        io = IOBuffer()
        refs(io, :PanelRegModel; format=:text)
        s = String(take!(io))
        @test occursin("Beck", s)
        @test occursin("Prais", s)
    end
end
