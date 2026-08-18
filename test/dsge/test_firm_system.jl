# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels

const _MFFS = MacroEconometricModels

@testset "G-18: FirmSystem Khan-Thomas 2008 (#652)" begin

@testset "kind, to_spec, household methods do not fire" begin
    @test FirmSystem <: AbstractAgentSystem
    @test !hasfield(FirmSystem, :individual)
    fs = khan_thomas_example(; n_k=12, n_eps=3)
    @test fs isa FirmSystem{Float64}
    @test length(fs.k_grid) == 12
    @test length(fs.productivity.states) == 3
    @test all(>(0), fs.productivity.states)
    @test _MFFS.aggregation(fs) == [:K, :N, :Y]
    @test _MFFS.idiosyncratic(fs) === fs.productivity
    @test issorted(fs.k_grid)

    spec = to_spec(fs)
    @test spec isa ModelSpec
    @test _MFFS.has_kind(spec, FirmSystem)
    @test !_MFFS.has_kind(spec, HouseholdSystem)
    @test keys(spec.agents) == (:firms,)
    @test only(values(spec.agents)) === fs
    @test_throws ArgumentError _MFFS._hh(spec)

    spec_named = to_spec(fs; agent_name=:plants)
    @test keys(spec_named.agents) == (:plants,)
    @test _MFFS.has_kind(spec_named, FirmSystem)
    @test !_MFFS.has_kind(spec_named, HouseholdSystem)

    # solve / compute_steady_state must name FirmSystem, not route through HA.
    err = try
        solve(spec)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("FirmSystem", msg)
    @test !occursin("HouseholdSystem", msg)
    @test !occursin("exactly one HouseholdSystem", msg)

    err2 = try
        compute_steady_state(spec)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("FirmSystem", sprint(showerror, err2))

    @test_throws ArgumentError khan_thomas_steady_state(spec; hh_solver=:egm)
    @test occursin("FirmSystem", sprint(show, fs))
end

@testset "constructor validation" begin
    fs = khan_thomas_example(; n_k=8, n_eps=2)
    @test_throws ArgumentError FirmSystem{Float64}(fs.k_grid, fs.productivity,
        0.5, 0.6, 0.07, 0.97, 1.0, 0.01, 0.01, 2.4, 0.8, 0.01, 1.0)
    @test_throws ArgumentError FirmSystem{Float64}(fs.k_grid, fs.productivity,
        0.25, 0.64, 0.07, 1.5, 1.0, 0.01, 0.01, 2.4, 0.8, 0.01, 1.0)
    @test_throws ArgumentError FirmSystem{Float64}([0.1, 0.2], fs.productivity,
        0.25, 0.64, 0.07, 0.97, 1.0, 0.01, 0.01, 2.4, 0.8, 0.01, 1.0)
end

@testset "static labor FOC" begin
    z, e, k, α, ν, w = 1.0, 1.0, 2.0, 0.256, 0.640, 1.0
    n, y, π = _MFFS._kt_static(z, e, k, α, ν, w)
    @test n ≈ (ν * z * e * k^α / w)^(1 / (1 - ν))
    @test y ≈ z * e * k^α * n^ν
    @test π ≈ (1 - ν) * y atol=1e-12
    @test _MFFS._kt_static(z, e, 0.0, α, ν, w) == (0.0, 0.0, 0.0)
end

@testset "coarse-grid SS vs Khan-Thomas Table 1 / Table 2" begin
    # Khan & Thomas (2008, Econometrica 76(2)) Table 1: γ=1.016, δ=0.069 so
    # BGP I/K = γ − 1 + δ = 0.085. Text (p. 12) targets 10% I/K (US FAT
    # 1954–2002). Table 2 row (2) lumpy + plant TFP: inaction 0.073 (LRD 0.081).
    # Coarse n_k=16, n_ε=3 is not the paper's 15-state chain; I/K is an
    # accounting identity on a stationary distribution, inaction is allowed a
    # wide band around the Table 2 moment.
    fs = khan_thomas_example(; n_k=16, n_eps=3)
    spec = to_spec(fs)
    @test _MFFS.has_kind(spec, FirmSystem)
    ss = khan_thomas_steady_state(spec; max_iter=20, tol=2e-4,
                                  vfi_tol=1e-5, vfi_max_iter=80)
    @test ss isa KhanThomasSteadyState
    @test ss.method === :mit
    @test ss.converged
    @test ss.K > 0 && ss.Y > 0 && ss.N > 0
    @test isfinite(ss.I) && isfinite(ss.C)
    ik = ss.I / ss.K
    @test 0.06 < ik < 0.14
    # (S,s) inaction = share with |i/k| ≤ max(0.01, b). Table 2 row (2) is
    # 0.073 (LRD 0.081); a 16×3 grid inflates the band. Require an interior
    # rate, not the frictionless ~0 or the traditional-lumpy 0.79-at-b=0.
    @test 0.05 < ss.inaction < 0.95
    # Nested GE (KT Appendix B): w clears the frictionless household FOC;
    # lumpy C stays close (paper: aggregates nearly identical).
    @test abs(ss.w - fs.phi * ss.C) / ss.w < 0.10
    txt = sprint(report, ss)
    @test occursin("Inaction", txt)
    @test occursin("I/K", txt)
    @test occursin("method", lowercase(txt)) || occursin(":mit", txt)
    @test occursin("KhanThomasSteadyState", sprint(show, ss))
end

@testset "MIT IRF of Y to TFP (method=:mit)" begin
    fs = khan_thomas_example(; n_k=12, n_eps=3)
    ss = khan_thomas_steady_state(fs; max_iter=20, tol=1e-5,
                                  vfi_tol=1e-5, vfi_max_iter=60)
    @test ss.converged
    Z = [ss.firm.Z * (1 + 0.02 * ss.firm.rho_z^(t - 1)) for t in 1:8]
    tr = khan_thomas_mit(ss, Z)
    @test tr isa KhanThomasTransition
    @test tr.method === :mit
    @test tr.converged
    @test tr.K[1] ≈ ss.K rtol=1e-8          # K predetermined
    @test all(isfinite, tr.Y) && all(isfinite, tr.I)
    @test tr.Y[1] != ss.Y                   # TFP moves output on impact

    resp = irf(ss, 8; shock_size=0.02)
    @test resp isa ImpulseResponse
    @test resp.variables == ["Y", "I", "K", "N", "C", "Z"]
    @test resp.shocks == ["Z"]
    @test all(isfinite, resp.values)
    @test resp.values[1, 1, 1] > 0          # Y IRF finite and positive on impact
    @test resp.values[1, 6, 1] ≈ 0.02 * ss.firm.Z atol=1e-12
    @test resp.values[1, 3, 1] ≈ 0 atol=1e-8  # K predetermined
    @test occursin(":mit", sprint(show, tr))
end

end
