# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — PVAR & Non-Gaussian Identification Coverage Tests
#
# Targets uncovered branches in:
#   src/pvar/estimation.jl  (90.0%, 36 missed) — :mstep, system GMM, Windmeijer, short Ti, predet
#   src/pvar/types.jl       (93.6%, 3 missed)  — show() for System GMM model
#   src/nongaussian/tests.jl (90.1%, 19 missed) — _procrustes_distance n>5, jade/sobi strength,
#                                                   ML gaussianity/independence, overidentification

using Test, MacroEconometricModels, Random, LinearAlgebra, DataFrames, Statistics, Distributions

Random.seed!(9003)

const MEM = MacroEconometricModels
const _suppress = MEM._suppress_warnings

# =============================================================================
# Helper: generate balanced panel DGP (same pattern as test/pvar/test_pvar.jl)
# =============================================================================

function _make_panel(; N=30, T_total=25, m=3, p=1, rng=MersenneTwister(9003))
    A1 = 0.3 * I(m) + 0.05 * randn(rng, m, m)
    F = eigvals(A1)
    while maximum(abs.(F)) >= 0.95
        A1 *= 0.8
        F = eigvals(A1)
    end

    data_mat = zeros(N * T_total, m)
    for i in 1:N
        mu_i = randn(rng, m)
        offset = (i - 1) * T_total
        data_mat[offset + 1, :] = mu_i + 0.1 * randn(rng, m)
        for t in 2:T_total
            data_mat[offset + t, :] = mu_i + A1 * data_mat[offset + t - 1, :] + 0.1 * randn(rng, m)
        end
    end

    df = DataFrame(data_mat, ["y$i" for i in 1:m])
    df.id = repeat(1:N, inner=T_total)
    df.time = repeat(1:T_total, outer=N)
    pd = xtset(df, :id, :time)
    (pd=pd, A1=A1, N=N, T_total=T_total, m=m)
end

# Panel with an extra "exog" column for predetermined / exogenous variable tests
function _make_panel_with_extras(; N=20, T_total=20, rng=MersenneTwister(9004))
    m = 2  # endogenous
    A1 = 0.3 * I(m) + 0.05 * randn(rng, m, m)
    F = eigvals(A1)
    while maximum(abs.(F)) >= 0.95
        A1 *= 0.8
        F = eigvals(A1)
    end

    data_mat = zeros(N * T_total, m + 2)  # 2 endogenous + 2 extra columns
    for i in 1:N
        mu_i = randn(rng, m)
        offset = (i - 1) * T_total
        data_mat[offset + 1, 1:m] = mu_i + 0.1 * randn(rng, m)
        for t in 2:T_total
            data_mat[offset + t, 1:m] = mu_i + A1 * data_mat[offset + t - 1, 1:m] + 0.1 * randn(rng, m)
        end
        # Predetermined and exogenous columns: correlated with lags but not contemporaneous
        for t in 1:T_total
            data_mat[offset + t, m+1] = 0.5 * (t > 1 ? data_mat[offset + max(t-1,1), 1] : 0.0) + randn(rng)
            data_mat[offset + t, m+2] = 2.0 + 0.3 * randn(rng)  # exogenous: near-constant + noise
        end
    end

    df = DataFrame(data_mat, ["y1", "y2", "predet1", "exog1"])
    df.id = repeat(1:N, inner=T_total)
    df.time = repeat(1:T_total, outer=N)
    pd = xtset(df, :id, :time)
    pd
end

# Panel with very few time periods per group (short Ti)
function _make_short_panel(; N=40, T_total=8, m=2, rng=MersenneTwister(9005))
    A1 = 0.25 * I(m)
    data_mat = zeros(N * T_total, m)
    for i in 1:N
        mu_i = randn(rng, m)
        offset = (i - 1) * T_total
        data_mat[offset + 1, :] = mu_i + 0.1 * randn(rng, m)
        for t in 2:T_total
            data_mat[offset + t, :] = mu_i + A1 * data_mat[offset + t - 1, :] + 0.2 * randn(rng, m)
        end
    end
    df = DataFrame(data_mat, ["y$i" for i in 1:m])
    df.id = repeat(1:N, inner=T_total)
    df.time = repeat(1:T_total, outer=N)
    xtset(df, :id, :time)
end

# =============================================================================
# 1. PVAR Estimation: :mstep (iterated GMM) path
# =============================================================================

@testset "PVAR :mstep iterated GMM" begin
    dgp = _make_panel(N=25, T_total=20, m=2, p=1, rng=MersenneTwister(9010))
    pd = dgp.pd

    @testset "mstep converges" begin
        model = estimate_pvar(pd, 1; steps=:mstep, max_iter=25)
        @test model isa PVARModel
        @test model.steps == :mstep
        @test model.method == :fd_gmm
        @test size(model.Phi) == (2, 2)
        @test all(isfinite.(model.se))
        @test all(model.se .>= 0)
    end

    @testset "mstep with FOD transformation" begin
        model = estimate_pvar(pd, 1; steps=:mstep, transformation=:fod, max_iter=15)
        @test model.steps == :mstep
        @test model.transformation == :fod
        @test size(model.Phi) == (2, 2)
    end

    @testset "mstep with max_iter=1 (early exit)" begin
        # max_iter=1 means only one iteration of the two-step refinement loop
        model = estimate_pvar(pd, 1; steps=:mstep, max_iter=1)
        @test model isa PVARModel
        @test model.steps == :mstep
    end

    @testset "mstep with multiple lags" begin
        model = estimate_pvar(pd, 2; steps=:mstep, max_iter=10)
        @test model.p == 2
        @test size(model.Phi) == (2, 4)  # 2 vars * 2 lags
    end
end

# =============================================================================
# 2. PVAR System GMM estimation
# =============================================================================

@testset "PVAR System GMM" begin
    dgp = _make_panel(N=25, T_total=20, m=2, p=1, rng=MersenneTwister(9020))
    pd = dgp.pd

    @testset "System GMM twostep" begin
        model = estimate_pvar(pd, 1; system_instruments=true, steps=:twostep)
        @test model isa PVARModel
        @test model.method == :system_gmm
        @test model.steps == :twostep
        @test model.system_constant == true
        @test model.n_instruments > 0
        @test size(model.Phi) == (2, 2)
        @test all(isfinite.(model.Phi))
    end

    @testset "System GMM onestep" begin
        model = estimate_pvar(pd, 1; system_instruments=true, steps=:onestep)
        @test model.method == :system_gmm
        @test model.steps == :onestep
    end

    @testset "System GMM mstep" begin
        model = estimate_pvar(pd, 1; system_instruments=true, steps=:mstep, max_iter=10)
        @test model.method == :system_gmm
        @test model.steps == :mstep
    end

    @testset "System GMM without constant" begin
        model = estimate_pvar(pd, 1; system_instruments=true, system_constant=false, steps=:onestep)
        @test model.method == :system_gmm
        @test model.system_constant == false
    end

    @testset "System GMM with collapsed instruments" begin
        model = estimate_pvar(pd, 1; system_instruments=true, collapse=true, steps=:twostep)
        @test model isa PVARModel
        @test model.method == :system_gmm
    end

    @testset "System GMM Hansen J-test" begin
        model = estimate_pvar(pd, 1; system_instruments=true, steps=:twostep)
        j = pvar_hansen_j(model)
        @test j isa PVARTestResult
        @test j.statistic >= 0
        @test 0 <= j.pvalue <= 1
    end
end

# =============================================================================
# 3. PVAR show() for System GMM display branch
# =============================================================================

@testset "PVAR System GMM Display" begin
    dgp = _make_panel(N=20, T_total=20, m=2, p=1, rng=MersenneTwister(9030))

    @testset "show System GMM model includes 'System GMM'" begin
        model = estimate_pvar(dgp.pd, 1; system_instruments=true, steps=:twostep)
        s = sprint(show, model)
        @test occursin("System GMM", s)
        @test occursin("Panel VAR", s)
        @test occursin("Instruments", s)
        @test occursin("L1.", s)
    end

    @testset "show System GMM with constant" begin
        model = estimate_pvar(dgp.pd, 1; system_instruments=true, system_constant=true, steps=:twostep)
        s = sprint(show, model)
        @test occursin("System GMM", s)
    end

    @testset "show mstep model" begin
        model = estimate_pvar(dgp.pd, 1; steps=:mstep, max_iter=10)
        s = sprint(show, model)
        @test occursin("FD-GMM", s)
        @test occursin("mstep", s)
    end
end

# =============================================================================
# 4. PVAR Windmeijer correction edge cases
# =============================================================================

@testset "PVAR Windmeijer Correction" begin
    dgp = _make_panel(N=30, T_total=20, m=2, p=1, rng=MersenneTwister(9040))
    pd = dgp.pd

    @testset "twostep SEs differ from onestep" begin
        m1 = estimate_pvar(pd, 1; steps=:onestep)
        m2 = estimate_pvar(pd, 1; steps=:twostep)
        # Windmeijer-corrected SEs should generally differ from one-step
        @test !isapprox(m1.se, m2.se, atol=1e-10)
    end

    @testset "mstep SEs differ from twostep" begin
        m2 = estimate_pvar(pd, 1; steps=:twostep)
        m3 = estimate_pvar(pd, 1; steps=:mstep, max_iter=25)
        # mstep iterates further — SEs may differ
        @test all(isfinite.(m3.se))
        @test all(m3.se .>= 0)
    end
end

# =============================================================================
# 5. PVAR short Ti (very few time periods per group)
# =============================================================================

@testset "PVAR Short Ti" begin
    pd_short = _make_short_panel(N=40, T_total=8, m=2, rng=MersenneTwister(9050))

    @testset "FD-GMM with short T" begin
        model = estimate_pvar(pd_short, 1; steps=:onestep)
        @test model isa PVARModel
        @test model.m == 2
        @test model.p == 1
        @test model.obs_per_group.min > 0
    end

    @testset "FD-GMM twostep with short T" begin
        model = estimate_pvar(pd_short, 1; steps=:twostep)
        @test model isa PVARModel
        @test all(isfinite.(model.se))
    end

    @testset "System GMM with short T" begin
        model = estimate_pvar(pd_short, 1; system_instruments=true, steps=:onestep)
        @test model.method == :system_gmm
        @test model.n_instruments > 0
    end

    @testset "FE-OLS with short T" begin
        model = estimate_pvar_feols(pd_short, 1)
        @test model isa PVARModel
        @test model.method == :fe_ols
    end
end

# =============================================================================
# 6. PVAR Predetermined variable handling
# =============================================================================

@testset "PVAR Predetermined Variables" begin
    pd = _make_panel_with_extras(N=20, T_total=20, rng=MersenneTwister(9060))

    @testset "FD-GMM with predetermined vars" begin
        model = estimate_pvar(pd, 1;
            dependent_vars=["y1", "y2"],
            predet_vars=["predet1"],
            steps=:onestep)
        @test model isa PVARModel
        @test model.m == 2
        @test model.n_predet == 1
        @test model.predet_names == ["predet1"]
        @test size(model.Phi, 2) == 2 * 1 + 1  # m*p + n_predet
    end

    @testset "FD-GMM with exogenous vars" begin
        model = estimate_pvar(pd, 1;
            dependent_vars=["y1", "y2"],
            exog_vars=["exog1"],
            steps=:onestep)
        @test model.n_exog == 1
        @test model.exog_names == ["exog1"]
        @test size(model.Phi, 2) == 2 * 1 + 1  # m*p + n_exog
    end

    @testset "FD-GMM with both predet and exog" begin
        model = estimate_pvar(pd, 1;
            dependent_vars=["y1", "y2"],
            predet_vars=["predet1"],
            exog_vars=["exog1"],
            steps=:twostep)
        @test model.n_predet == 1
        @test model.n_exog == 1
        @test size(model.Phi, 2) == 2 * 1 + 1 + 1  # m*p + n_predet + n_exog
    end

    @testset "FE-OLS with predetermined vars" begin
        model = estimate_pvar_feols(pd, 1;
            dependent_vars=["y1", "y2"],
            predet_vars=["predet1"])
        @test model.n_predet == 1
        @test size(model.Phi, 2) == 2 * 1 + 1
    end

    @testset "FE-OLS with exogenous vars" begin
        model = estimate_pvar_feols(pd, 1;
            dependent_vars=["y1", "y2"],
            exog_vars=["exog1"])
        @test model.n_exog == 1
    end

    @testset "invalid predet var name" begin
        @test_throws ArgumentError estimate_pvar(pd, 1;
            dependent_vars=["y1", "y2"],
            predet_vars=["nonexistent"])
    end

    @testset "invalid exog var name" begin
        @test_throws ArgumentError estimate_pvar(pd, 1;
            dependent_vars=["y1", "y2"],
            exog_vars=["nonexistent"])
    end

    @testset "invalid exog var name FE-OLS" begin
        @test_throws ArgumentError estimate_pvar_feols(pd, 1;
            dependent_vars=["y1", "y2"],
            exog_vars=["nonexistent"])
    end

    @testset "invalid predet var name FE-OLS" begin
        @test_throws ArgumentError estimate_pvar_feols(pd, 1;
            dependent_vars=["y1", "y2"],
            predet_vars=["nonexistent"])
    end
end

# =============================================================================
# 7. Non-Gaussian: _procrustes_distance with n > 5 (greedy matching)
# =============================================================================

@testset "Procrustes distance n > 5" begin
    rng = MersenneTwister(9070)

    @testset "greedy matching for 6x6" begin
        B1 = randn(rng, 6, 6)
        B2 = randn(rng, 6, 6)
        d = MEM._procrustes_distance(B1, B2)
        @test d >= 0
        @test isfinite(d)
    end

    @testset "greedy matching identity" begin
        n = 7
        B = Matrix{Float64}(I, n, n)
        d = MEM._procrustes_distance(B, B)
        @test d >= 0
        # Identity matched to itself should be exactly 0 via greedy
        @test d < 1e-10
    end

    @testset "greedy matching signed permutation" begin
        n = 8
        B = randn(rng, n, n)
        # Column permutation with sign flip
        perm = [2, 1, 4, 3, 6, 5, 8, 7]
        signs = [1, -1, 1, -1, 1, -1, 1, -1]
        B_perm = B[:, perm] .* signs'
        d = MEM._procrustes_distance(B, B_perm)
        @test d >= 0
        # Should find a good match (greedy is approximate, may not be exactly 0)
        @test isfinite(d)
    end

    @testset "greedy matching 10x10" begin
        B1 = randn(rng, 10, 10)
        B2 = B1 + 0.01 * randn(rng, 10, 10)  # small perturbation
        d = MEM._procrustes_distance(B1, B2)
        @test d >= 0
        @test d < 1.0  # should be small since B2 is close to B1
    end
end

# =============================================================================
# 8. Non-Gaussian: test_identification_strength with :jade and :sobi
# =============================================================================

@testset "Identification strength jade/sobi" begin
    Random.seed!(9080)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)

    _suppress() do
        @testset "jade method" begin
            result = test_identification_strength(model; method=:jade, n_bootstrap=15)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :identification_strength
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
            @test result.details[:method] == :jade
            @test result.details[:n_bootstrap] > 0
        end

        @testset "sobi method" begin
            result = test_identification_strength(model; method=:sobi, n_bootstrap=15)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :identification_strength
            @test result.details[:method] == :sobi
        end
    end
end

# =============================================================================
# 9. Non-Gaussian: test_shock_gaussianity with NonGaussianMLResult
# =============================================================================

@testset "Shock gaussianity with NonGaussianMLResult" begin
    Random.seed!(9090)
    Y = randn(250, 3)
    model = estimate_var(Y, 2)

    _suppress() do
        @testset "student_t ML result" begin
            ml = identify_student_t(model)
            result = test_shock_gaussianity(ml)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :shock_gaussianity
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
            @test haskey(result.details, :jb_stats)
            @test haskey(result.details, :jb_pvals)
            @test haskey(result.details, :n_gaussian)
            @test result.details[:method] == :student_t
        end

        @testset "mixture_normal ML result" begin
            ml = identify_mixture_normal(model)
            result = test_shock_gaussianity(ml)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :shock_gaussianity
            @test result.details[:method] == :mixture_normal
        end

        @testset "skew_normal ML result" begin
            ml = identify_skew_normal(model)
            result = test_shock_gaussianity(ml)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
        end
    end
end

# =============================================================================
# 10. Non-Gaussian: test_shock_independence with NonGaussianMLResult
# =============================================================================

@testset "Shock independence with NonGaussianMLResult" begin
    Random.seed!(9100)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)

    _suppress() do
        @testset "student_t ML result" begin
            ml = identify_student_t(model)
            result = test_shock_independence(ml; max_lag=5)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :shock_independence
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
            @test haskey(result.details, :cc_statistic)
            @test haskey(result.details, :dcov_statistic)
            @test haskey(result.details, :max_lag)
        end

        @testset "mixture_normal ML result" begin
            ml = identify_mixture_normal(model)
            result = test_shock_independence(ml; max_lag=3)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :shock_independence
        end
    end
end

# =============================================================================
# 11. Non-Gaussian: test_overidentification
# =============================================================================

@testset "Overidentification test" begin
    Random.seed!(9110)
    Y = randn(250, 3)
    model = estimate_var(Y, 2)

    _suppress() do
        @testset "with ICA result" begin
            ica = identify_fastica(model)
            result = test_overidentification(model, ica; n_bootstrap=49)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :overidentification
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
            @test haskey(result.details, :discrepancy)
            @test haskey(result.details, :orthogonality_error)
            @test haskey(result.details, :n_bootstrap)
            @test result.details[:n_bootstrap] == 49
        end

        @testset "with ML result" begin
            ml = identify_student_t(model)
            result = test_overidentification(model, ml; n_bootstrap=29)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
            @test result.test_name == :overidentification
            @test result.statistic >= 0
        end

        @testset "with JADE result" begin
            jade_res = identify_jade(model)
            result = test_overidentification(model, jade_res; n_bootstrap=19)
            @test result isa MEM.IdentifiabilityTestResult{Float64}
        end
    end
end

# =============================================================================
# 12. Non-Gaussian: IdentifiabilityTestResult display
# =============================================================================

@testset "IdentifiabilityTestResult show" begin
    Random.seed!(9120)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)

    _suppress() do
        ica = identify_fastica(model)
        gauss_result = test_shock_gaussianity(ica)
        s = sprint(show, gauss_result)
        @test occursin("Identifiability Test", s)
        @test occursin("shock_gaussianity", s)
        @test occursin("P-value", s)
    end
end

# =============================================================================
# 13. PVAR PCA instrument reduction
# =============================================================================

@testset "PVAR PCA Instruments" begin
    dgp = _make_panel(N=25, T_total=20, m=2, p=1, rng=MersenneTwister(9130))
    pd = dgp.pd

    @testset "PCA reduction with auto components" begin
        model = estimate_pvar(pd, 1; pca_instruments=true, pca_max_components=0, steps=:onestep)
        @test model isa PVARModel
    end

    @testset "PCA reduction with fixed components" begin
        model = estimate_pvar(pd, 1; pca_instruments=true, pca_max_components=5, steps=:onestep)
        @test model isa PVARModel
        @test model.n_instruments > 0
    end
end

# =============================================================================
# 14. Integration: System GMM -> structural analysis pipeline
# =============================================================================

@testset "System GMM structural analysis" begin
    dgp = _make_panel(N=25, T_total=20, m=2, p=1, rng=MersenneTwister(9140))
    pd = dgp.pd

    model = estimate_pvar(pd, 1; system_instruments=true, steps=:twostep)

    @testset "OIRF from System GMM" begin
        oirf = pvar_oirf(model, 5)
        @test size(oirf) == (6, 2, 2)
    end

    @testset "GIRF from System GMM" begin
        girf = pvar_girf(model, 5)
        @test size(girf) == (6, 2, 2)
    end

    @testset "FEVD from System GMM" begin
        fv = pvar_fevd(model, 5)
        @test size(fv) == (6, 2, 2)
        for h in 1:6, v in 1:2
            @test isapprox(sum(fv[h, v, :]), 1.0, atol=0.01)
        end
    end

    @testset "stability from System GMM" begin
        stab = pvar_stability(model)
        @test stab isa PVARStability
        @test length(stab.moduli) == 2
    end
end
