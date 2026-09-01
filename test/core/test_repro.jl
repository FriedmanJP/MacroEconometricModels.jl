# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Random
using LinearAlgebra

const _MEM = MacroEconometricModels

@testset "Reproducibility manifest (T246/#345)" begin

    @testset "capture_manifest populates the environment and never throws" begin
        m = capture_manifest()
        @test m isa ReproManifest
        @test m.seed === nothing
        @test m.n_threads == Threads.nthreads()
        @test m.julia_version == string(VERSION)
        @test !isempty(m.package_version)
        @test haskey(m.dependency_versions, "Distributions")
        @test haskey(m.dependency_versions, "StatsAPI")
        @test !isempty(m.timestamp)
        @test m.git_sha isa String              # a sha or "unknown", never an error
        @test m.git_dirty isa Bool
        @test m.os == string(Sys.KERNEL)

        m2 = capture_manifest(; seed=42, settings=Dict{String,Any}("reps" => 100))
        @test m2.seed == 42
        @test m2.settings["reps"] == 100
    end

    @testset "manifest ↔ dict round-trip (serialization bridge)" begin
        m = capture_manifest(; seed=7, settings=Dict{String,Any}("burnin" => 200, "thin" => 2))
        d = _MEM._manifest_to_dict(m)
        @test d["__manifest__"] == true
        m2 = _MEM._manifest_from_dict(d)
        @test m2.seed == m.seed
        @test m2.n_threads == m.n_threads
        @test m2.julia_version == m.julia_version
        @test m2.package_version == m.package_version
        @test m2.git_sha == m.git_sha
        @test m2.settings["burnin"] == 200
        @test m2.dependency_versions == m.dependency_versions
        @test _MEM._manifest_to_dict(nothing) === nothing
        @test _MEM._manifest_from_dict(nothing) === nothing
    end

    @testset "BVAR posterior carries a manifest and reproduces bit-for-bit" begin
        Y = randn(MersenneTwister(1), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=100, seed=20260717)
        @test post.manifest isa ReproManifest
        @test post.manifest.seed == 20260717

        rep = reproduce(post)
        @test rep isa ReproReport
        @test rep.matched === true
        @test occursin("matched", rep.note)

        # same seed ⇒ identical draws; different seed ⇒ different draws
        post_same = estimate_bvar(Y, 2; n_draws=100, seed=20260717)
        @test post.B_draws == post_same.B_draws
        @test post.Sigma_draws == post_same.Sigma_draws
        post_diff = estimate_bvar(Y, 2; n_draws=100, seed=999)
        @test post.B_draws != post_diff.B_draws
    end

    @testset "BVAR gibbs sampler reproduces (burnin/thin recorded)" begin
        Y = randn(MersenneTwister(5), 70, 2)
        post = estimate_bvar(Y, 2; n_draws=60, sampler=:gibbs, thin=2, seed=314)
        @test post.manifest.settings["thin"] == 2
        @test post.manifest.settings["burnin"] == 200   # gibbs default recorded
        @test reproduce(post).matched === true
    end

    @testset "BVAR without a seed: manifest present, reproduction declines" begin
        Y = randn(MersenneTwister(2), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=50)          # no seed
        @test post.manifest isa ReproManifest
        @test post.manifest.seed === nothing
        rep = reproduce(post)
        @test rep.matched === missing
        @test occursin("no recorded seed", rep.note)
    end

    @testset "bootstrap IRF carries a manifest and reproduces via reproduce(ir, model)" begin
        Y = randn(MersenneTwister(3), 100, 2)
        model = estimate_var(Y, 2)
        ir = irf(model, 10; ci_type=:bootstrap, reps=80, seed=123)
        @test ir.manifest isa ReproManifest
        @test ir.manifest.seed == 123
        @test ir.manifest.settings["reps"] == 80
        @test ir.manifest.settings["method"] == "cholesky"

        rep = reproduce(ir, model)
        @test rep.matched === true

        # single-arg form asks for the source model rather than throwing
        rep1 = reproduce(ir)
        @test rep1.matched === missing
        @test occursin("source model", rep1.note)

        # a deterministic (no-CI) IRF has no manifest
        ir_none = irf(model, 10)
        @test ir_none.manifest === nothing
    end

    @testset "thread-count caveat wording in the report note" begin
        # A fabricated manifest whose thread count differs from the current one
        # exercises both caveat branches deterministically.
        m = _MEM.ReproManifest(1, Threads.nthreads() + 1, "v", "p",
                               Dict{String,String}(), "os", "mach", "ts", "sha", false,
                               Dict{String,Any}())
        matched = _MEM._finalize_repro([_MEM.ReproFieldDiff("x", true, 0.0)], m)
        @test matched.matched === true
        @test occursin("thread-count-invariant", matched.note)

        mismatched = _MEM._finalize_repro([_MEM.ReproFieldDiff("x", false, 1.0)], m)
        @test mismatched.matched === false
        @test occursin("thread count changed", mismatched.note)
    end

    @testset "generic reproduce fallback for unsupported types" begin
        rep = reproduce(42)
        @test rep.matched === missing
        @test occursin("not implemented", rep.note)
    end

    @testset "reproducibility footer is opt-in (#521)" begin
        Y = randn(MersenneTwister(4), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=50, seed=5)
        model = estimate_var(Y, 2)
        boot = irf(model, 10; ci_type=:bootstrap, reps=20, seed=5)

        # Off by default, so no git revision is baked into built docs or any
        # other durable artifact that captures `show`/`report` output.
        for obj in (post, boot)
            @test !occursin("Reproducibility:", sprint(show, obj))
        end

        # Opt in via the IOContext key or the `report` keyword.
        s = sprint(io -> show(IOContext(io, :repro => true), post))
        @test occursin("Reproducibility:", s)
        @test occursin("seed=5", s)

        s_irf = sprint(io -> show(IOContext(io, :repro => true), boot))
        @test occursin("Reproducibility:", s_irf)
        @test occursin("seed=5", s_irf)

        # `report` plumbs the keyword into that same IOContext key.
        @test (redirect_stdout(devnull) do; report(post; repro=true) end; true)
        @test (redirect_stdout(devnull) do; report(boot; repro=true) end; true)
        @test (redirect_stdout(devnull) do; report(post) end; true)
        @test (redirect_stdout(devnull) do; report(boot) end; true)

        post_ns = estimate_bvar(Y, 2; n_draws=50)
        s_ns = sprint(io -> show(IOContext(io, :repro => true), post_ns))
        @test occursin("seed=unset", s_ns)

        # A manifest-free result never emits the footer, opt-in or not.
        plain = irf(model, 10)
        @test plain.manifest === nothing
        @test !occursin("Reproducibility:",
                        sprint(io -> show(IOContext(io, :repro => true), plain)))
    end

    @testset "ReproManifest display gates only the git line (#521)" begin
        Y = randn(MersenneTwister(4), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=50, seed=5)
        m = post.manifest

        # Shown by default — asking for the manifest is asking for provenance.
        full = sprint(show, m)
        @test occursin("ReproManifest", full)
        @test occursin("git", full)
        @test occursin(m.git_sha, full)

        # `:repro => false` drops it, leaving every stable field in place.
        masked = sprint(io -> show(IOContext(io, :repro => false), m))
        @test occursin("ReproManifest", masked)
        @test !occursin(m.git_sha, masked)
        @test !occursin("(dirty)", masked)
        for field in ("seed", "threads", "julia", "package", "os / machine", "captured")
            @test occursin(field, masked)
        end

        # The revision is suppressed from the display only, never from the data.
        @test m.git_sha isa AbstractString

        @test occursin("ReproReport", sprint(show, reproduce(post)))
    end
end

# =============================================================================
# RSER-13 / #786 — seed=/manifest/reproduce for rng-only estimators
# =============================================================================

function _rser13_did_panel(; n_units=12, n_periods=10, seed=42)
    rng = MersenneTwister(seed)
    n_cohorts = 2
    units_per = n_units ÷ (n_cohorts + 1)
    treat_times = zeros(Int, n_units)
    for c in 1:n_cohorts
        for u in ((c - 1) * units_per + 1):(c * units_per)
            treat_times[u] = 4 + 2 * (c - 1)
        end
    end
    N_obs = n_units * n_periods
    data = Matrix{Float64}(undef, N_obs, 2)
    group_id = Vector{Int}(undef, N_obs)
    time_id = Vector{Int}(undef, N_obs)
    row = 1
    for i in 1:n_units
        a = randn(rng)
        for t in 1:n_periods
            te = (treat_times[i] > 0 && t >= treat_times[i]) ? 1.5 : 0.0
            data[row, 1] = a + 0.1 * t + te + 0.4 * randn(rng)
            data[row, 2] = Float64(treat_times[i])
            group_id[row] = i
            time_id[row] = t
            row += 1
        end
    end
    PanelData{Float64}(data, ["outcome", "treat_time"], Quarterly, [1, 1],
                       group_id, time_id, nothing, ["u$i" for i in 1:n_units],
                       n_units, 2, N_obs, true, String[], Dict{String,String}(), Symbol[])
end

function _rser13_pvar(; N=8, T_total=14, m=2, seed=11)
    rng = MersenneTwister(seed)
    data_mat = zeros(N * T_total, m)
    for i in 1:N
        mu = randn(rng, m)
        off = (i - 1) * T_total
        data_mat[off + 1, :] = mu + 0.1 * randn(rng, m)
        for t in 2:T_total
            data_mat[off + t, :] = mu + 0.3 * data_mat[off + t - 1, :] + 0.1 * randn(rng, m)
        end
    end
    pd = PanelData{Float64}(data_mat, ["y1", "y2"], Quarterly, [1, 1],
                            repeat(1:N, inner=T_total), repeat(1:T_total, outer=N),
                            nothing, ["g$i" for i in 1:N],
                            N, 2, N * T_total, true, ["pvar test panel"], Dict{String,String}(), Symbol[])
    estimate_pvar_feols(pd, 1; dependent_vars=["y1", "y2"])
end

@testset "RSER-13 seed=/manifest/reproduce (#786)" begin

    @testset "listed result types carry a trailing manifest field" begin
        types = (SVModel{Float64}, TVPVARPosterior{Float64}, MFVARPosterior{Float64},
                 BayesianFAVAR{Float64}, StructuralDFM{Float64}, PVARModel{Float64},
                 SMMModel{Float64}, QuantileRegModel{Float64}, RobustRegModel{Float64},
                 ThresholdModel{Float64}, STARModel{Float64}, MSForecast{Float64},
                 STARForecast{Float64}, ThresholdForecast{Float64},
                 WildClusterBootstrap{Float64}, DIDResult{Float64}, StructuralLP{Float64},
                 ConditionalForecast{Float64}, BayesianHistoricalDecomposition{Float64},
                 OPPResult{Float64}, OPPSequence{Float64}, PolicyCausalEffects{Float64},
                 PolicyForecast{Float64}, SpanningDiagnostic{Float64},
                 ModelBankMember{Float64}, NARDLMultipliers{Float64},
                 HansenLinearityTest{Float64})
        for Tw in types
            @test :manifest in fieldnames(Tw)
            @test fieldnames(Tw)[end] === :manifest
        end
        # Identification helpers do not grow a manifest field (SID-19 coordination).
        @test !(:manifest in fieldnames(SignIdentifiedSet{Float64}))
        @test !(:manifest in fieldnames(UhligSVARResult{Float64}))
        @test !(:manifest in fieldnames(AriasSVARResult{Float64}))
        @test !(:manifest in fieldnames(ICASVARResult{Float64}))
    end

    @testset "seed= identity; rng= still works (identification helpers)" begin
        Q1 = generate_Q(3; seed=17)
        Q2 = generate_Q(3; seed=17)
        @test Q1 == Q2
        Q3 = generate_Q(3; seed=18)
        @test Q1 != Q3
        Qr = generate_Q(3; rng=MersenneTwister(17))
        @test Qr isa Matrix
        # seed wins when both are passed
        @test generate_Q(3; rng=MersenneTwister(99), seed=17) == Q1
    end

    @testset "estimate_sv: seed, manifest, reproduce, round-trip" begin
        y = randn(MersenneTwister(3), 40)
        m = estimate_sv(y; n_samples=12, burnin=6, seed=20260717)
        @test m.manifest isa ReproManifest
        @test m.manifest.seed == 20260717
        @test estimate_sv(y; n_samples=12, burnin=6, seed=20260717).h_draws == m.h_draws
        @test estimate_sv(y; n_samples=12, burnin=6, seed=9).h_draws != m.h_draws
        @test reproduce(m).matched === true
        m2 = _MEM._reconstruct_from_container(_MEM._build_container(m))
        @test m2.manifest.seed == 20260717
        @test reproduce(m2).matched === true
        m_ns = estimate_sv(y; n_samples=8, burnin=4)
        @test m_ns.manifest isa ReproManifest
        @test m_ns.manifest.seed === nothing
        @test reproduce(m_ns).matched === missing
        # rng= still works
        @test estimate_sv(y; n_samples=8, burnin=4, rng=MersenneTwister(1)) isa SVModel
    end

    @testset "estimate_tvpvar: seed, manifest, reproduce, round-trip" begin
        Y = randn(MersenneTwister(4), 40, 2)
        post = estimate_tvpvar(Y, 1; tvp=true, sv=true, n_draws=6, n_burn=6,
                               n_train=8, seed=314)
        @test post.manifest isa ReproManifest
        @test post.manifest.seed == 314
        @test reproduce(post).matched === true
        post2 = _MEM._reconstruct_from_container(_MEM._build_container(post))
        @test reproduce(post2).matched === true
        @test estimate_tvpvar(Y, 1; n_draws=6, n_burn=6, n_train=8,
                              rng=MersenneTwister(1)) isa TVPVARPosterior
    end

    @testset "pvar_bootstrap_irf: seed, manifest, reproduce, round-trip" begin
        model = _rser13_pvar()
        boot = pvar_bootstrap_irf(model, 3; n_draws=6, seed=42)
        @test boot.manifest isa ReproManifest
        @test boot.manifest.seed == 42
        @test boot.model isa PVARModel
        @test boot.model.boot_draws == boot.draws
        boot2 = pvar_bootstrap_irf(model, 3; n_draws=6, seed=42)
        @test boot.draws == boot2.draws
        @test reproduce(boot.model).matched === true
        loaded = _MEM._reconstruct_from_container(_MEM._build_container(boot.model))
        @test reproduce(loaded).matched === true
        @test pvar_bootstrap_irf(model, 3; n_draws=4,
                                 rng=MersenneTwister(1)).draws isa Array
    end

    @testset "PVARModel v1 payload without boot_* keys still loads" begin
        @test SERIALIZATION_FORMAT_VERSION == 1
        model = _rser13_pvar()
        c = _MEM._build_container(model)
        @test c["format_version"] == 1
        payload = c["payload"]
        for k in ("boot_irf", "boot_lower", "boot_upper", "boot_draws", "manifest")
            @test haskey(payload, k)
            delete!(payload, k)
        end
        loaded = _MEM._reconstruct_from_container(c)
        @test loaded isa PVARModel
        @test loaded.boot_irf === nothing
        @test loaded.boot_lower === nothing
        @test loaded.boot_upper === nothing
        @test loaded.boot_draws === nothing
        @test loaded.manifest === nothing
        @test loaded.Phi == model.Phi
    end

    @testset "estimate_structural_dfm: seed, manifest on both methods" begin
        X = randn(MersenneTwister(1), 80, 8)
        sdfm = estimate_structural_dfm(X, 2; r=2, p=1, H=4, seed=1)
        @test sdfm.manifest isa ReproManifest
        @test sdfm.manifest.seed == 1
        @test sdfm.manifest.settings["method"] == "fglr"
        sdfm_g = estimate_structural_dfm(X, 2; r=2, p=1, H=4, method=:gdfm_var, seed=7)
        @test sdfm_g.manifest isa ReproManifest
        @test sdfm_g.manifest.seed == 7
        @test sdfm_g.manifest.settings["method"] == "gdfm_var"
        gdfm = estimate_gdfm(X, 2)
        sdfm2 = estimate_structural_dfm(gdfm; r=2, p=1, H=4, seed=3)
        @test sdfm2.manifest.seed == 3
        sdfm_ns = estimate_structural_dfm(X, 2; r=2, p=1, H=4)
        @test sdfm_ns.manifest isa ReproManifest
        @test sdfm_ns.manifest.seed === nothing
    end

    @testset "estimate_did bootstrap SEs: seed, manifest, reproduce, round-trip" begin
        pd = _rser13_did_panel()
        r = estimate_did(pd, :outcome, :treat_time; method=:did_multiplegt,
                         leads=1, horizon=2, n_boot=8, seed=1234)
        @test r.manifest isa ReproManifest
        @test r.manifest.seed == 1234
        r_same = estimate_did(pd, :outcome, :treat_time; method=:did_multiplegt,
                              leads=1, horizon=2, n_boot=8, seed=1234)
        @test r.att == r_same.att
        @test r.se == r_same.se
        @test reproduce(r).matched === true
        r2 = _MEM._reconstruct_from_container(_MEM._build_container(r))
        @test reproduce(r2).matched === true
        @test estimate_did(pd, :outcome, :treat_time; method=:did_multiplegt,
                           leads=1, horizon=2, n_boot=6,
                           rng=MersenneTwister(1)) isa DIDResult
    end

    @testset "forecast(::MSRegModel): seed, manifest, reproduce, round-trip" begin
        rng = MersenneTwister(510)
        n = 120
        y = zeros(n)
        s = 1
        P = [0.9 0.1; 0.2 0.8]
        mu = (-0.5, 1.0)
        for t in 1:n
            s = rand(rng) < P[s, 1] ? 1 : 2
            y[t] = mu[s] + 0.4 * (t == 1 ? 0.0 : y[t-1]) + 0.3 * randn(rng)
        end
        ms = estimate_ms_ar(y, 1; k_regimes=2)
        fc = forecast(ms, 4; reps=30, seed=77)
        @test fc isa MSForecast
        @test fc.manifest isa ReproManifest
        @test fc.manifest.seed == 77
        @test forecast(ms, 4; reps=30, seed=77).forecast == fc.forecast
        @test reproduce(fc).matched === true
        fc2 = _MEM._reconstruct_from_container(_MEM._build_container(fc))
        @test reproduce(fc2).matched === true
        @test forecast(ms, 3; reps=10, rng=MersenneTwister(2)) isa MSForecast
    end

    @testset "estimate_opp: seed, manifest, reproduce, round-trip" begin
        H, n_s = 6, 2
        Tx0 = randn(MersenneTwister(8), H, n_s)
        v0 = randn(MersenneTwister(9), H)
        noises = 0.05 .* randn(MersenneTwister(2), 12)
        D = cat((Tx0 .* (1 + e) for e in noises)...; dims=3)
        ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx0], Theta_x_draws=[D])
        d = [hcat((v0 .* (1 + e) for e in noises)...)]
        fc = PolicyForecast{Float64}([:u], [v0], d, H, "t")
        loss = policy_loss([:u], H; lambda=[1.0])
        r = estimate_opp(fc, ce, loss; n_sim=24, seed=5)
        @test r.manifest isa ReproManifest
        @test r.manifest.seed == 5
        @test estimate_opp(fc, ce, loss; n_sim=24, seed=5).delta_draws == r.delta_draws
        @test reproduce(r).matched === true
        r2 = _MEM._reconstruct_from_container(_MEM._build_container(r))
        @test reproduce(r2).matched === true
        @test estimate_opp(fc, ce, loss; n_sim=16,
                           rng=MersenneTwister(3)) isa OPPResult
    end

    @testset "two-arg reproduce(ir, model) is unchanged" begin
        Y = randn(MersenneTwister(3), 80, 2)
        model = estimate_var(Y, 2)
        ir = irf(model, 8; ci_type=:bootstrap, reps=40, seed=123)
        @test reproduce(ir, model).matched === true
        @test reproduce(ir).matched === missing
    end
end

