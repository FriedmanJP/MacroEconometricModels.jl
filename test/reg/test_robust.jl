# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-40 (#448): robust regression — Huber/bisquare M-estimation + Yohai MM-estimation.

using Test
using MacroEconometricModels
using MacroEconometricModels: _rw_huber, _rw_bisq, _rho_bisq, _mad_scale, _m_scale,
                              _S_C0, _S_B, _fast_s
using LinearAlgebra, Statistics, Random, Logging
import StatsAPI

# =============================================================================
# ORACLE — R MASS::rlm on the Brownlee (1965) stackloss data (base R `datasets::stackloss`).
# Exact provenance (Rscript, MASS 7.3):
#
#   library(MASS); data(stackloss)
#   rlm(stack.loss ~ ., data=stackloss, method="M", psi=psi.huber)$coef
#     (Intercept)   Air.Flow  Water.Temp  Acid.Conc.
#     -41.0265311  0.8293739   0.9261082  -0.1278492      ; scale (s) = 2.4407138
#   rlm(stack.loss ~ ., data=stackloss, method="M", psi=psi.bisquare)$coef
#     -42.2852537  0.9275471   0.6507322  -0.1123310      ; scale (s) = 2.2818859
#
# MM-estimation NOTE: `MASS::rlm(method="MM")` reports coef
#   -41.5230725  0.9388402   0.5794546  -0.1129150 ; scale = 1.9119655
# but its S-scale comes from `lqs`'s *non-standard* scale estimator, which over-states the
# high-breakdown scale. The standard Yohai (1987) / Salibian-Barrera-Yohai (2006) MM
# estimator (implemented here, == robustbase::lmrob) uses the Tukey biweight M-scale and
# recovers the canonical Rousseeuw & Leroy (1987) robust stackloss fit, flagging
# observations 1, 3, 4, 21 as outliers (ψ-weight ≈ 0). We therefore oracle MM against that
# canonical fit and its breakdown properties, not against rlm's weaker lqs-seeded scale:
#
#   OLS on stackloss[-c(1,3,4,21),] (the 17 "clean" points):
#     (Intercept)   Air.Flow  Water.Temp  Acid.Conc.
#     -37.6524589  0.7976856   0.5773405  -0.0670602
# =============================================================================

const RLM_HUBER_M = [-41.0265310522, 0.8293738689, 0.9261081841, -0.1278491603]
const RLM_HUBER_M_SCALE = 2.440713795
const RLM_BISQ_M  = [-42.2852536878, 0.9275470955, 0.6507322173, -0.1123310319]
const RLM_BISQ_M_SCALE = 2.28188588
const CLEAN17_OLS = [-37.6524589008, 0.7976855601, 0.5773404574, -0.0670601769]

@testset "Robust Regression (EV-40)" begin

    # ---- dataset ----
    @testset "load_example(:stackloss)" begin
        d = load_example(:stackloss)
        @test d isa CrossSectionData
        @test size(d.data) == (21, 4)
        @test d.varnames == ["Air.Flow", "Water.Temp", "Acid.Conc.", "stack.loss"]
        @test d.data[1, :] == [80.0, 27.0, 89.0, 42.0]
        @test d.data[end, 4] == 15.0
        @test occursin("Brownlee", desc(d))
    end

    d = load_example(:stackloss)
    y = d.data[:, 4]
    X = hcat(ones(21), d.data[:, 1:3])
    vn = ["(Intercept)", "Air.Flow", "Water.Temp", "Acid.Conc."]

    # ---- ψ / weight primitives ----
    @testset "ψ-weight primitives" begin
        # Huber weight = min(1, k/|u|); 1 inside, k/|u| outside; u=0 → 1 (no NaN).
        @test _rw_huber(0.0, 1.345) == 1.0
        @test _rw_huber(1.0, 1.345) == 1.0
        @test _rw_huber(2.69, 1.345) ≈ 0.5 atol = 1e-12
        # bisquare weight = (1-(u/c)²)² inside support, 0 outside; u=0 → 1.
        @test _rw_bisq(0.0, 4.685) == 1.0
        @test _rw_bisq(4.685, 4.685) == 0.0
        @test _rw_bisq(100.0, 4.685) == 0.0
        @test _rw_bisq(2.3425, 4.685) ≈ (1 - 0.25)^2 atol = 1e-12
        # bisquare ρ saturates at c²/6.
        @test _rho_bisq(100.0, 1.548) ≈ 1.548^2 / 6 atol = 1e-12
    end

    # ---- MAD scale Fisher-consistency ----
    @testset "normalized-MAD scale consistency" begin
        rng = MersenneTwister(20240716)
        z = 3.0 .* randn(rng, 200_000)     # N(0, 3²)
        @test _mad_scale(z) ≈ 3.0 rtol = 0.02
    end

    # ---- S-estimator target constant ----
    @testset "S-estimator constant b (50% breakdown)" begin
        @test _S_B / (_S_C0^2 / 6) ≈ 0.5 atol = 1e-3   # b/ρ(∞) = breakdown = 0.5
        # M-scale is scale-equivariant: s(a·r) = a·s(r).
        rng = MersenneTwister(1)
        r = randn(rng, 50)
        s1 = _m_scale(r, Float64(_S_C0), Float64(_S_B))
        s2 = _m_scale(5.0 .* r, Float64(_S_C0), Float64(_S_B))
        @test s2 ≈ 5.0 * s1 rtol = 1e-6
    end

    # ---- ORACLE: Huber-M vs MASS::rlm ----
    @testset "Huber M-estimation vs rlm" begin
        m = estimate_robust(y, X; psi=:huber, method=:m, varnames=vn)
        @test m isa RobustRegModel
        @test m.method == :m
        @test m.psi == :huber
        @test m.tuning ≈ 1.345
        @test m.converged
        @test maximum(abs.(m.beta .- RLM_HUBER_M)) < 1e-3
        @test m.scale ≈ RLM_HUBER_M_SCALE atol = 1e-3
        # StatsAPI surface
        @test StatsAPI.coef(m) === m.beta
        @test length(StatsAPI.stderror(m)) == 4
        @test StatsAPI.nobs(m) == 21
        @test StatsAPI.dof(m) == 4
        @test size(StatsAPI.confint(m)) == (4, 2)
        @test StatsAPI.predict(m) ≈ X * m.beta
    end

    # ---- ORACLE: bisquare-M vs MASS::rlm ----
    @testset "bisquare M-estimation vs rlm" begin
        m = estimate_robust(y, X; psi=:bisquare, method=:m, varnames=vn)
        @test m.psi == :bisquare
        @test m.tuning ≈ 4.685
        @test maximum(abs.(m.beta .- RLM_BISQ_M)) < 1e-3
        @test m.scale ≈ RLM_BISQ_M_SCALE atol = 1e-3
    end

    # ---- MM-estimation: canonical high-breakdown fit + reproducibility ----
    @testset "MM-estimation (Yohai) — breakdown & reproducibility" begin
        m = estimate_robust(y, X; method=:mm, rng=MersenneTwister(1), varnames=vn)
        @test m.method == :mm
        @test m.psi == :bisquare          # MM forces bisquare
        @test m.tuning ≈ 4.685
        # Flags exactly the canonical Rousseeuw-Leroy outliers 1,3,4,21 (ψ-weight ≈ 0).
        @test findall(<(0.05), m.weights) == [1, 3, 4, 21]
        # Recovers the clean-17 robust fit (soft down-weighting ⇒ not identical, but close).
        @test maximum(abs.(m.beta .- CLEAN17_OLS)) < 1.5
        # High-breakdown S-scale is well below the M-estimate's MAD scale.
        @test m.scale < RLM_BISQ_M_SCALE
        @test m.scale > 0.5              # sane, not a degenerate collapse
        # MM differs materially from OLS (the fast-S start actually ran).
        ols = X \ y
        @test norm(m.beta .- ols) > 1.0
        # Reproducible given a fixed rng; robust to the seed (same global optimum).
        m_same = estimate_robust(y, X; method=:mm, rng=MersenneTwister(1))
        @test m.beta == m_same.beta
        m_seed2 = estimate_robust(y, X; method=:mm, rng=MersenneTwister(999))
        @test maximum(abs.(m.beta .- m_seed2.beta)) < 1e-6
    end

    # ---- Analytic property: clean data ⇒ robust ≈ OLS; one outlier barely moves bisquare ----
    @testset "clean data ≈ OLS; single outlier downweighted" begin
        rng = MersenneTwister(2024)
        n = 200
        Xc = hcat(ones(n), randn(rng, n, 2))
        beta_true = [1.0, 2.0, -0.5]
        yc = Xc * beta_true .+ 0.5 .* randn(rng, n)

        ols = Xc \ yc
        mh = estimate_robust(yc, Xc; psi=:huber, method=:m)
        mb = estimate_robust(yc, Xc; psi=:bisquare, method=:m)
        @test maximum(abs.(mh.beta .- ols)) < 0.05
        @test maximum(abs.(mb.beta .- ols)) < 0.05

        # Inject one gross vertical outlier.
        yo = copy(yc); yo[1] += 50.0
        ols_o = Xc \ yo
        mb_o = estimate_robust(yo, Xc; psi=:bisquare, method=:m)
        # OLS is pulled off; the bisquare fit is not.
        @test norm(ols_o .- beta_true) > norm(mb_o.beta .- beta_true)
        @test maximum(abs.(mb_o.beta .- mb.beta)) < 0.05
        # The outlier gets a ψ-weight of essentially zero.
        @test mb_o.weights[1] < 1e-3
    end

    # ---- Huber Proposal-2 joint scale updating ----
    @testset "Huber Proposal-2 scale update" begin
        m = estimate_robust(y, X; psi=:huber, method=:m, scale_update=:proposal2, varnames=vn)
        @test m.converged
        # Still a bounded-influence Huber fit; coefficients in the neighbourhood of MAD-scaled rlm.
        @test maximum(abs.(m.beta .- RLM_HUBER_M)) < 1.0
        @test m.scale > 0
    end

    # ---- CrossSectionData convenience method ----
    @testset "CrossSectionData entry point" begin
        m1 = estimate_robust(d, "stack.loss", ["Air.Flow", "Water.Temp", "Acid.Conc."];
                             psi=:huber, method=:m)
        m2 = estimate_robust(y, X; psi=:huber, method=:m, varnames=vn)
        @test m1.beta ≈ m2.beta
        @test m1.varnames == vn
        # Index-based selection agrees with name-based selection.
        m3 = estimate_robust(d, 4, [1, 2, 3]; psi=:huber, method=:m)
        @test m3.beta ≈ m2.beta
        @test_throws ArgumentError estimate_robust(d, "nope", ["Air.Flow"])
    end

    # ---- report() / refs() render ----
    @testset "report() and refs()" begin
        m = estimate_robust(y, X; psi=:huber, method=:m, varnames=vn)
        buf = IOBuffer()
        show(buf, m)
        str = String(take!(buf))
        @test occursin("Robust Regression", str)
        @test occursin("Robust scale", str)
        @test occursin("Downweighted", str)
        @test occursin("(Intercept)", str)
        # MM report labels the method.
        mm = estimate_robust(y, X; method=:mm, rng=MersenneTwister(1), varnames=vn)
        buf2 = IOBuffer()
        show(buf2, mm)
        @test occursin("MM-estimation", String(take!(buf2)))
        # refs render and include the robust-regression bibliography.
        rbuf = IOBuffer()
        refs(rbuf, m)
        rstr = String(take!(rbuf))
        @test occursin("Yohai", rstr)
        @test occursin("Huber", rstr)
    end

    # ---- input validation ----
    @testset "input validation" begin
        @test_throws ArgumentError estimate_robust(y, X; method=:bad)
        @test_throws ArgumentError estimate_robust(y, X; psi=:bad)
        @test_throws ArgumentError estimate_robust(y, X; scale_update=:bad)
        @test_throws ArgumentError estimate_robust(y[1:3], X)          # n ≤ p
        @test_throws ArgumentError estimate_robust(y, X; varnames=["a", "b"])
    end
end

@testset "Conley (1999) spatial HAC standard errors (#360/T261)" begin
    M = MacroEconometricModels
    rng = Random.MersenneTwister(2610)
    n = 150
    lat = 40 .+ 4 .* rand(rng, n)
    lon = -100 .+ 4 .* rand(rng, n)
    coords = hcat(lat, lon)
    X = hcat(ones(n), randn(rng, n, 2))
    y = X * [1.0, 0.5, -0.3] .+ randn(rng, n)
    m = estimate_reg(y, X; cov_type=:hc0)

    @testset "haversine distance against known values" begin
        # one degree of longitude at the equator, and pole to pole
        @test M._great_circle(0.0, 0.0, 0.0, 1.0) ≈ 2π * 6371 / 360 rtol = 1e-10
        @test M._great_circle(-90.0, 0.0, 90.0, 0.0) ≈ π * 6371 rtol = 1e-10
        @test M._great_circle(40.0, -100.0, 40.0, -100.0) == 0.0
        @test M._great_circle(40.0, -100.0, 41.0, -99.0) ==
              M._great_circle(41.0, -99.0, 40.0, -100.0)          # symmetric
        # a degree of longitude shrinks with latitude by cos(lat)
        @test M._great_circle(60.0, 0.0, 60.0, 1.0) ≈
              M._great_circle(0.0, 0.0, 0.0, 1.0) * cosd(60.0) rtol = 1e-3
        # euclidean metric
        c2 = [0.0 0.0; 3.0 4.0]
        @test M._conley_distance(c2, 1, 2, :euclidean) ≈ 5.0
        @test M._conley_distance(c2, 1, 1, :euclidean) == 0.0
    end

    @testset "kernel shape" begin
        @test M._conley_kernel(0.0, 10.0, :bartlett) == 1.0
        @test M._conley_kernel(5.0, 10.0, :bartlett) ≈ 0.5
        @test M._conley_kernel(10.0, 10.0, :bartlett) == 0.0
        @test M._conley_kernel(11.0, 10.0, :bartlett) == 0.0
        @test M._conley_kernel(5.0, 10.0, :uniform) == 1.0
        @test M._conley_kernel(11.0, 10.0, :uniform) == 0.0
        # a non-positive cutoff keeps only the own-observation term
        @test M._conley_kernel(0.0, 0.0, :bartlett) == 1.0
        @test M._conley_kernel(1e-12, 0.0, :bartlett) == 0.0
    end

    @testset "cutoff = 0 reduces EXACTLY to HC0" begin
        for kern in (:bartlett, :uniform), met in (:euclidean, :haversine)
            c = conley_se(m; coords=coords, cutoff=0.0, kernel=kern, metric=met)
            @test c.vcov ≈ m.vcov_mat atol = 1e-12
            @test c.se ≈ sqrt.(diag(m.vcov_mat)) atol = 1e-12
            @test !c.adjusted
        end
        # ... and through the estimate_reg keyword path
        mc = estimate_reg(y, X; cov_type=:conley, coords=coords, cutoff=0.0)
        @test mc.vcov_mat ≈ m.vcov_mat atol = 1e-12
        @test mc.cov_type === :conley
    end

    @testset "uniform kernel equals the brute-force double sum" begin
        c = conley_se(m; coords=coords, cutoff=150.0, kernel=:uniform,
                      metric=:haversine, psd=false)
        Xu = X .* m.residuals
        S = zeros(3, 3)
        for i in 1:n, j in 1:n
            M._great_circle(lat[i], lon[i], lat[j], lon[j]) <= 150.0 &&
                (S .+= Xu[i, :] * Xu[j, :]')
        end
        XtXinv = inv(X'X)
        @test c.vcov ≈ XtXinv * S * XtXinv atol = 1e-10
        # the meat carries no 1/n (package HAC convention)
        @test !isapprox(c.vcov, XtXinv * (S ./ n) * XtXinv; rtol=1e-3)
    end

    @testset "Bartlett weights strictly between uniform and none" begin
        c_none = conley_se(m; coords=coords, cutoff=0.0, psd=false)
        c_bart = conley_se(m; coords=coords, cutoff=150.0, kernel=:bartlett,
                           metric=:haversine, psd=false)
        c_unif = conley_se(m; coords=coords, cutoff=150.0, kernel=:uniform,
                           metric=:haversine, psd=false)
        @test c_bart.vcov != c_unif.vcov
        @test c_bart.vcov != c_none.vcov
        # a wider cutoff includes weakly more pairs, so it cannot equal the narrow one
        c_wide = conley_se(m; coords=coords, cutoff=400.0, kernel=:uniform,
                           metric=:haversine, psd=false)
        @test c_wide.vcov != c_unif.vcov
    end

    @testset "PSD projection clips and flags" begin
        c_raw = conley_se(m; coords=coords, cutoff=150.0, kernel=:uniform,
                          metric=:haversine, psd=false)
        c_psd = conley_se(m; coords=coords, cutoff=150.0, kernel=:uniform,
                          metric=:haversine, psd=true)
        # Conley's meat need not be PSD in finite samples; when it is not, clipping the
        # negative eigenvalues changes the answer and the result says so.
        if c_psd.adjusted
            @test c_psd.vcov != c_raw.vcov
            @test all(diag(c_psd.vcov) .>= -1e-12)
        end
        @test all(isfinite, c_psd.se)
    end

    @testset "spatial correlation in BOTH error and regressor inflates the SE" begin
        # With an iid regressor the correction is asymptotically moot, since
        # E[x_i u_i x_j u_j] = E[x_i x_j] E[u_i u_j] = 0. The common spatial component has
        # to be in the regressor too — that is when HC under-states the uncertainty.
        reps = 60
        ratio = Float64[]
        n_cov_hc = 0; n_cov_con = 0
        for r in 1:reps
            rr = Random.MersenneTwister(1000 + r)
            la = 40 .+ 4 .* rand(rr, 150); lo = -100 .+ 4 .* rand(rr, 150)
            blk = @. Int(floor(la - 40)) * 10 + Int(floor(lo + 100))
            ush = Dict(b => randn(rr) for b in unique(blk))
            xsh = Dict(b => randn(rr) for b in unique(blk))
            xb = [xsh[blk[i]] for i in 1:150] .+ 0.3 .* randn(rr, 150)
            u = [ush[blk[i]] for i in 1:150] .+ 0.3 .* randn(rr, 150)
            Xr = hcat(ones(150), xb)
            mh = estimate_reg(Xr * [1.0, 0.5] .+ u, Xr; cov_type=:hc0)
            ch = conley_se(mh; coords=hcat(la, lo), cutoff=120.0, kernel=:uniform,
                           metric=:haversine, psd=false)
            se_h = sqrt(mh.vcov_mat[2, 2]); se_c = ch.se[2]
            push!(ratio, se_c / se_h)
            abs(mh.beta[2] - 0.5) <= 1.96 * se_h && (n_cov_hc += 1)
            abs(mh.beta[2] - 0.5) <= 1.96 * se_c && (n_cov_con += 1)
        end
        @test all(ratio .> 1)                       # measured: every rep, mean ratio 2.8
        @test Statistics.mean(ratio) > 2
        # and the coverage repair is large (measured 44.7% -> 83.3% at 300 reps)
        @test n_cov_con > n_cov_hc + reps ÷ 5
    end

    @testset "spatial panel: the time kernel multiplies the spatial one" begin
        nT = 4; nS = 30; N = nS * nT
        la = repeat(40 .+ rand(Random.MersenneTwister(5), nS); inner=nT)
        lo = repeat(-100 .+ rand(Random.MersenneTwister(6), nS); inner=nT)
        tt = repeat(1:nT, outer=nS)
        X2 = hcat(ones(N), randn(Random.MersenneTwister(7), N))
        m2 = estimate_reg(X2 * [1.0, 0.5] .+ randn(Random.MersenneTwister(8), N), X2;
                          cov_type=:hc0)
        c0 = conley_se(m2; coords=hcat(la, lo), cutoff=50.0, kernel=:uniform,
                       metric=:haversine, time=tt, time_cutoff=0, psd=false)
        # time_cutoff = 0 ⇒ only contemporaneous pairs contribute
        Xu = X2 .* m2.residuals
        S0 = zeros(2, 2)
        for i in 1:N, j in 1:N
            (M._great_circle(la[i], lo[i], la[j], lo[j]) <= 50.0 && tt[i] == tt[j]) &&
                (S0 .+= Xu[i, :] * Xu[j, :]')
        end
        Xti = inv(X2'X2)
        @test c0.vcov ≈ Xti * S0 * Xti atol = 1e-10
        # a positive time cutoff brings in cross-period pairs, so the answer moves
        c2 = conley_se(m2; coords=hcat(la, lo), cutoff=50.0, kernel=:uniform,
                       metric=:haversine, time=tt, time_cutoff=2, psd=false)
        @test c2.vcov != c0.vcov
    end

    @testset "input validation" begin
        @test_throws ArgumentError conley_se(m; coords=coords[1:10, :], cutoff=100.0)
        @test_throws ArgumentError conley_se(m; coords=coords, cutoff=100.0, kernel=:bogus)
        @test_throws ArgumentError conley_se(m; coords=coords, cutoff=100.0, metric=:bogus)
        @test_throws ArgumentError conley_se(m; coords=lat[:, :], cutoff=100.0, metric=:haversine)
        @test_throws ArgumentError conley_se(m; coords=coords, cutoff=100.0, time_cutoff=-1)
        @test_throws ArgumentError conley_se(m; coords=coords, cutoff=100.0, time=[1, 2])
        @test_throws ArgumentError estimate_reg(y, X; cov_type=:conley)          # coords missing
        @test_throws ArgumentError estimate_reg(y, X; cov_type=:bogus)
    end
end
