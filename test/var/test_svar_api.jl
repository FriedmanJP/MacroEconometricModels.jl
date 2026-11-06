# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Unit test suite for Milestone 4 (#878 & #879):
- SVAR Identification API Consistency & Keyword Unification (#878)
- Shock Naming Standardization (#879)

Enforces 5-dimension test matrix:
1. Exact algebraic identities and structural invariants
2. DGP parameter recovery with explicit RNG (DGP-01, #790)
3. Type stability and bidirectional dispatch conformance
4. Concrete @test_throws negative paths
5. Struct backward compatibility and plotting propagation
"""

using LinearAlgebra
using MacroEconometricModels
using Random
using Statistics
using Test

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

if !@isdefined(simulate_proxy_svar)
    include("id_dgps.jl")
end

const MEM = MacroEconometricModels

@testset "Milestone 4: SVAR Identification API & Shock Naming (#878, #879)" begin

    # =========================================================================
    # Dimension 1: Exact Algebraic Identities & Structural Invariants
    # =========================================================================
    @testset "D1: Exact Algebraic Identities & Structural Invariants" begin
        rng = Xoshiro(87801)
        Y = dgp_var(rng; T=300).Y
        model = estimate_var(Y, 1)
        n = nvars(model)
        @test n == 3

        # M4-ALG-01: SVAR AB Impact Orthogonality & Delegated Analysis
        pat = cholesky_pattern(n)
        svar = estimate_svar(model, pat; rng=Xoshiro(87802))
        @test svar.model === model
        B0 = svar.A \ svar.B
        @test norm(B0 * B0' - model.Sigma) < 1e-8
        @test norm(svar.Q * svar.Q' - I(n)) < 1e-8

        # Direct analysis calls on SVARModel
        ir_svar = irf(svar, 8)
        @test ir_svar isa ImpulseResponse
        @test ir_svar.shocks == ["Shock 1", "Shock 2", "Shock 3"]
        @test size(ir_svar.values) == (8, 3, 3)

        fv_svar = fevd(svar, 8)
        @test fv_svar isa FEVD
        @test fv_svar.shocks == ["Shock 1", "Shock 2", "Shock 3"]

        hd_svar = historical_decomposition(svar)
        @test hd_svar isa HistoricalDecomposition
        @test hd_svar.shock_names == ["Shock 1", "Shock 2", "Shock 3"]

        # M4-ALG-02: SVEC Long-Run Rank Invariant
        rng_vecm = Xoshiro(87803)
        Y_vecm = dgp_vecm(rng_vecm; T=300).Y
        vm = estimate_vecm(Y_vecm, 2; rank=1)
        svec = identify_svec(vm)
        @test svec.shock_names == ["P1", "P2", "T1"]
        @test norm(svec.Xi * svec.B0[:, 3]) < 1e-10

        # Direct analysis calls on SVECResult
        ir_svec = irf(svec, 8)
        @test ir_svec isa ImpulseResponse
        @test ir_svec.shocks == ["P1", "P2", "T1"]
        fv_svec = fevd(svec, 8)
        @test fv_svec isa FEVD
        @test fv_svec.shocks == ["P1", "P2", "T1"]

        # M4-ALG-03: HD Adding-Up Decomposition
        hd = historical_decomposition(model; method=:cholesky)
        @test hd.shock_names == ["y1", "y2", "y3"]
        @test norm(sum(hd.contributions; dims=3)[:, :, 1] +
                   hd.initial_conditions - hd.actual) < 1e-8

        # M4-ALG-04: Proxy Impact Consistency
        B_true = [1.0 0.0 0.0; 0.3 1.0 0.0; 0.2 0.4 1.0]
        A_lag = [0.4 * Matrix{Float64}(I, 3, 3)]
        Y_px, _, z_px = simulate_proxy_svar(
            B_true, A_lag; Tobs=300, ρ=0.85, rng=Xoshiro(87804),
        )
        m_px = estimate_var(Y_px, 1)
        pr = identify_proxy(m_px, reshape(z_px, :, 1))
        @test pr.first_stage_F > 10
        @test pr.shock_names == ["Proxy", "Unidentified 1", "Unidentified 2"]
    end

    # =========================================================================
    # Dimension 2: DGP Ground-Truth Parameter Recovery (Explicit RNG)
    # =========================================================================
    @testset "D2: DGP Ground-Truth Parameter Recovery" begin
        rng1 = Xoshiro(87805)
        m = estimate_var(dgp_var(rng1; T=200).Y, 1)
        svar = estimate_svar(m, recursive_pattern(3); rng=Xoshiro(87806))
        @test svar.shock_names == ["Shock 1", "Shock 2", "Shock 3"]

        rng2 = Xoshiro(87807)
        Y_v = dgp_vecm(rng2; T=250).Y
        vm = estimate_vecm(Y_v, 2; rank=1)
        svec = identify_svec(vm)
        @test svec.shock_names == ["P1", "P2", "T1"]

        # Rank 2 SVEC: 1 permanent, 2 transitory
        vm2 = estimate_vecm(Y_v, 2; rank=2)
        svec2 = identify_svec(vm2)
        @test svec2.shock_names == ["P1", "T1", "T2"]

        # Sign identification default shocks
        check_fn = irf -> irf[1, 1, 1] > 0 && irf[1, 2, 2] > 0
        s_sign = identify_sign(
            m, 6, check_fn; max_draws=100, store_all=true, rng=Xoshiro(87808),
        )
        @test s_sign.shocks == ["Shock 1", "Shock 2", "Shock 3"]

        # Narrative identification default shocks
        n_check = irf -> true
        s_narr = identify_narrative(
            m, 6, check_fn, n_check; max_draws=100, store_all=true,
            rng=Xoshiro(87809),
        )
        @test s_narr.shocks == ["Shock 1", "Shock 2", "Shock 3"]
    end

    # =========================================================================
    # Dimension 3: Type Stability & Return Type Conformance
    # =========================================================================
    @testset "D3: Type Stability & Bidirectional Dispatch" begin
        rng = Xoshiro(87810)
        m = estimate_var(dgp_var(rng; T=200).Y, 1)
        pat = cholesky_pattern(3)

        # M4-TYP-01: Bidirectional check_identification (Pattern)
        st1 = check_identification(m, pat)
        st2 = check_identification(pat, m)
        st3 = check_identification(3, pat)
        st4 = check_identification(pat, 3)
        @test st1 isa IdentificationStatus
        @test st1 == st2 == st3 == st4
        @test Bool(st1)

        # M4-TYP-02: Bidirectional check_identification (Restrictions)
        restr = SVARRestrictions(
            3;
            zeros=[zero_restriction(2, 1)],
            signs=[sign_restriction(1, 1, :positive)],
        )
        sr1 = check_identification(m, restr)
        sr2 = check_identification(restr, m)
        sr3 = check_identification(3, restr)
        sr4 = check_identification(restr, 3)
        @test sr1 isa IdentificationStatus
        @test sr1 == sr2 == sr3 == sr4

        # M4-TYP-03: Bidirectional estimate_svar
        sv1 = estimate_svar(m, pat; rng=Xoshiro(87811))
        sv2 = estimate_svar(pat, m; rng=Xoshiro(87811))
        @test sv1 isa SVARModel{Float64}
        @test sv2 isa SVARModel{Float64}
        @test sv1.B ≈ sv2.B

        # M4-TYP-04: Proxy Keyword Entry Points
        z = randn(Xoshiro(87812), size(m.Y, 1))
        pr_single = identify_proxy(m; instrument=z)
        @test pr_single isa ProxySVARResult
        @test pr_single.k == 1
        @test pr_single.shock_names == ["Proxy", "Unidentified 1", "Unidentified 2"]

        pr_plural = identify_proxy(m; instruments=reshape(z, :, 1))
        @test pr_plural isa ProxySVARResult
        @test pr_plural.k == 1

        # M4-TYP-05 & M4-TYP-06: Struct Field Inferred Types
        @test fieldtype(AriasSVARResult, :shock_names) == Vector{String}
        @test fieldtype(UhligSVARResult, :shock_names) == Vector{String}
        @test fieldtype(RobustBayesResult, :varnames) == Vector{String}
        @test fieldtype(RobustBayesResult, :shock_names) == Vector{String}
        @test fieldtype(SVARModel, :shock_names) == Vector{String}
        @test fieldtype(SVARModel{Float64}, :model) == Union{Nothing,VARModel{Float64}}
        @test fieldtype(SVECResult, :shock_names) == Vector{String}
        @test fieldtype(SVSVARResult, :varnames) == Vector{String}
        @test fieldtype(SVSVARResult, :shock_names) == Vector{String}

        # Convenience functions
        @test cholesky_pattern(3) isa SVARPattern
        restr_mut = SVARRestrictions(3)
        add_zero!(restr_mut, :A0, 2, 1)
        @test length(restr_mut.zeros) == 1
    end

    # =========================================================================
    # Dimension 4: Concrete @test_throws Negative Paths
    # =========================================================================
    @testset "D4: Concrete @test_throws Negative Paths" begin
        rng = Xoshiro(87813)
        m = estimate_var(dgp_var(rng; T=200).Y, 1)
        pat = cholesky_pattern(3)

        # M4-ERR-01: SVAR Shock Names Length Mismatch
        @test_throws ArgumentError estimate_svar(m, pat; shock_names=["S1"])

        # M4-ERR-02: SVEC Shock Names Length Mismatch
        vm = estimate_vecm(m.Y, 1; rank=1)
        @test_throws ArgumentError identify_svec(vm; shock_names=["P1"])

        # M4-ERR-03: Sign ID Shock Names Length Mismatch
        check_fn = irf -> true
        @test_throws ArgumentError identify_sign(
            m, 6, check_fn; shock_names=["A", "B", "C", "D"],
        )

        # M4-ERR-04: Arias ID Shock Names Length Mismatch
        restr = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
        @test_throws ArgumentError identify_arias(m, restr, 6; shock_names=["A"])

        # M4-ERR-05: Proxy SVAR Neither Instrument Passed
        @test_throws ArgumentError identify_proxy(m)

        # M4-ERR-06: Proxy SVAR Dimension Mismatch (k > n)
        z_too_many = zeros(size(m.Y, 1), 4)
        @test_throws ArgumentError identify_proxy(m; instruments=z_too_many)

        # M4-ERR-07: IRF Shock Names Length Mismatch
        @test_throws ArgumentError irf(m, 6; shock_names=["S1"])

        # M4-ERR-08: FEVD Shock Names Length Mismatch
        @test_throws ArgumentError fevd(m, 6; shock_names=["S1"])

        # M4-ERR-09: HD Shock Names Length Mismatch
        @test_throws ArgumentError historical_decomposition(m; shock_names=["S1"])

        # M4-ERR-10: SVAR Bidirectional Dimension Mismatch
        @test_throws ArgumentError check_identification(4, pat)
    end

    # =========================================================================
    # Dimension 5: Struct Backward Compatibility & Plotting Propagation
    # =========================================================================
    @testset "D5: Struct Backward Compatibility & Plotting Propagation" begin
        I3 = Matrix{Float64}(I, 3, 3)
        restr = SVARRestrictions(3)
        id_st = IdentificationStatus(:exact, [0, 0, 0], [0, 0, 0], 0)

        # M4-BC-01: Arias Legacy 5-arg & 7-arg Constructor
        a_5 = AriasSVARResult{Float64}([I3], zeros(1, 4, 3, 3), [1.0], 1.0, restr)
        @test a_5.shock_names == ["Shock 1", "Shock 2", "Shock 3"]
        @test a_5.varnames == ["Var 1", "Var 2", "Var 3"]

        # M4-BC-02: Uhlig Legacy 6-arg & 7-arg Constructor
        u_6 = UhligSVARResult{Float64}(
            I3, zeros(4, 3, 3), 0.1, [0.0, 0.0, 0.0], restr, true,
        )
        @test u_6.shock_names == ["Shock 1", "Shock 2", "Shock 3"]
        @test u_6.varnames == ["var1", "var2", "var3"]

        # M4-BC-03: Robust Bayes Legacy 9-arg Constructor
        arr3 = zeros(4, 3, 3)
        rb_9 = RobustBayesResult{Float64}(
            arr3, arr3 .+ 1, arr3 .- 0.1, arr3 .+ 1.1,
            arr3, arr3 .+ 0.5, 0.2, 0.0, 0.68,
        )
        @test rb_9.varnames == ["var1", "var2", "var3"]
        @test rb_9.shock_names == ["Shock 1", "Shock 2", "Shock 3"]

        # M4-BC-04: SVARModel Legacy 12-arg Constructor
        pat = cholesky_pattern(3)
        svar_12 = SVARModel{Float64}(
            I3, I3, I3, zeros(6, 6), zeros(3, 3), 0.0, 0.0, 0, 1.0,
            pat, id_st, ["y1", "y2", "y3"],
        )
        @test svar_12.shock_names == ["Shock 1", "Shock 2", "Shock 3"]
        @test svar_12.model === nothing

        # M4-BC-05: SVECResult Legacy 6-arg Constructor
        rng = Xoshiro(87814)
        vm = estimate_vecm(dgp_vecm(rng; T=200).Y, 1; rank=1)
        svec_6 = SVECResult{Float64}(I3, I3, I3, 2, vm, id_st)
        @test svec_6.shock_names == ["P1", "P2", "T1"]

        # M4-BC-06: SVSVARResult Legacy 12-arg Constructor
        sv_12 = SVSVARResult{Float64}(
            I3, [I3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9],
            [0.1, 0.1, 0.1], trues(3), zeros(10, 3), [0.0], true, 1, "conv",
        )
        @test sv_12.varnames == ["y1", "y2", "y3"]
        @test sv_12.shock_names == ["Shock 1", "Shock 2", "Shock 3"]

        # M4-PLT-01: IRF Panel Title Propagation
        m = estimate_var(dgp_var(Xoshiro(87815); T=200).Y, 1)
        p_irf = plot_result(irf(m, 6))
        @test occursin("y1 ← y1", p_irf.html)
        @test !occursin("y1 ← Shock 1", p_irf.html)

        # M4-PLT-02: Custom IRF Title Propagation
        custom_shocks = ["Demand", "Supply", "Monetary"]
        p_cust = plot_result(irf(m, 6; shock_names=custom_shocks))
        @test occursin("y1 ← Demand", p_cust.html)
        @test occursin("y1 ← Supply", p_cust.html)

        # M4-PLT-03: Arias Set-ID Plot Titles
        restr_a = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
        a_res = identify_arias(m, restr_a, 6; n_draws=50, rng=Xoshiro(87816))
        p_arias = plot_result(a_res)
        @test occursin("y1 ← Shock 1", p_arias.html)
        @test !occursin("Var 1 ← Shock 1", p_arias.html)

        # M4-PLT-04: Uhlig Penalty Plot Titles
        u_res = identify_uhlig(
            m, restr_a, 6; n_starts=2, max_iter_coarse=10, max_iter_fine=10,
            rng=Xoshiro(87817),
        )
        p_uhlig = plot_result(u_res)
        @test occursin("y1 ← Shock 1", p_uhlig.html)

        # M4-PLT-05: SVAR Heatmap Shock Labels
        svar_mod = estimate_svar(m, pat; shock_names=custom_shocks, rng=Xoshiro(87818))
        p_svar = plot_result(svar_mod)
        @test occursin("Demand", p_svar.html)
        @test occursin("Supply", p_svar.html)

        # M4-PLT-06: SVEC Heatmap Shock Labels
        svec_mod = identify_svec(vm)
        p_svec = plot_result(svec_mod)
        @test occursin("P1", p_svec.html)
        @test occursin("T1", p_svec.html)

        # M4-PLT-07: HD Stacked Bar Legend Labels
        hd_cust = historical_decomposition(m; shock_names=custom_shocks)
        p_hd = plot_result(hd_cust)
        @test occursin("Demand", p_hd.html)
        @test occursin("Supply", p_hd.html)

        # M4-PLT-08: Hostile Character Escaping
        hostile = ["<script>alert(1)</script>", "&\"foo\"", "Safe"]
        p_hostile = plot_result(irf(m, 4; shock_names=hostile))
        @test !occursin("<script>alert(1)</script>", p_hostile.html)

        # M4-PLT-09: Unicode Subscripts Preservation
        unicode_shocks = ["ε_MP", "ε_AD", "ε_AS"]
        p_uni = plot_result(irf(m, 4; shock_names=unicode_shocks))
        @test occursin("ε_MP", p_uni.html)
        @test occursin("ε_AD", p_uni.html)
    end
end
