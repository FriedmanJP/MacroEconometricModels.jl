# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random
using StatsAPI
using ForwardDiff

const MEM = MacroEconometricModels
const _suppress_warnings = MEM._suppress_warnings

# Helper: spectral density f_u(ω) = (1/2π) Θ(e^{-iω}) C C' Θ(e^{iω})'
function _test_ma_spectrum(Theta::Array{T,3}, C::Matrix{T}, omega::Real) where {T<:AbstractFloat}
    n, _, q = size(Theta)
    z = exp(-im * Float64(omega))
    Theta_z = Matrix{ComplexF64}(I, n, n)
    for j in 1:q
        Theta_z .+= ComplexF64.(@view(Theta[:, :, j])) .* (z^j)
    end
    A_z = Theta_z * ComplexF64.(C)
    return (A_z * A_z') ./ (2 * pi)
end

# Helper: exact MA autocovariance Γ_u(k) = Σ_{j=0}^{q-k} Θ_{j+k} Σ Θ_j'
function _test_ma_autocov(Theta::Array{T,3}, C::Matrix{T}, k::Int) where {T<:AbstractFloat}
    n, _, q = size(Theta)
    Sigma = Float64.(C) * Float64.(C)'
    G = zeros(Float64, n, n)
    k > q && return G
    for j in 0:(q - k)
        M1 = j + k == 0 ? Matrix{Float64}(I, n, n) : Float64.(Theta[:, :, j + k])
        M2 = j == 0 ? Matrix{Float64}(I, n, n) : Float64.(Theta[:, :, j])
        G .+= M1 * Sigma * M2'
    end
    return G
end

# Helper: simulate deterministic unit shock path through SVARMA(p, q)
function _test_impulse_sim(Phi::Array{T,3}, Theta::Array{T,3}, C::Matrix{T},
                           shock_idx::Int, H::Int) where {T<:AbstractFloat}
    n = size(C, 1)
    p = size(Phi, 3)
    q = size(Theta, 3)
    y_hist = zeros(Float64, H, n)
    eps_vec = Float64.(C[:, shock_idx])
    for h in 1:H
        val = zeros(Float64, n)
        if h == 1
            val .+= eps_vec
        elseif h - 1 <= q
            val .+= Float64.(Theta[:, :, h - 1]) * eps_vec
        end
        for k in 1:min(h - 1, p)
            val .+= Float64.(Phi[:, :, k]) * y_hist[h - k, :]
        end
        y_hist[h, :] .= val
    end
    return y_hist
end

@testset "NGID-05: Non-Fundamental SSVARMA(p, q) Core (#905)" begin

    @testset "1. Export Hygiene" begin
        for sym in (:SVARMAModel, :ssvarma_filter_shocks, :fundamentalness_regime,
                    :blaschke_representations, :dgp_ssvarma)
            @test isdefined(MacroEconometricModels, sym)
        end
    end

    @testset "2. SVARMAModel Type, Properties, StatsAPI, RSER-01 Compatibility & Display" begin
        sim = dgp_ssvarma(; T=120, seed=9051)
        Phi_vec = [sim.Phi[:, :, 1]]
        Theta_vec = [sim.Theta[:, :, 1]]

        m_3d = SVARMAModel(sim.Y, sim.Phi, sim.Theta, sim.C;
                           mu=sim.mu, distributions=[:student_t, :student_t],
                           dist_params=[[5.0], [5.0]], loglik=-320.5,
                           aic=665.0, bic=698.0, varnames=["y1", "y2"],
                           shock_names=["s1", "s2"], method=:test)
        m_vec = SVARMAModel(sim.Y, Phi_vec, Theta_vec, sim.C; mu=sim.mu)
        m_mat = SVARMAModel(sim.Y, sim.Phi[:, :, 1], sim.Theta[:, :, 1], sim.C; mu=sim.mu)
        m_pure_ma = SVARMAModel(sim.Y, nothing, Theta_vec, sim.C)
        m_pure_ar = SVARMAModel(sim.Y, Phi_vec, Matrix{Float64}[], sim.C)

        @test m_3d isa AbstractVARModel
        @test m_vec.Phi ≈ m_3d.Phi
        @test m_vec.Theta ≈ m_3d.Theta
        @test m_mat.Phi ≈ m_3d.Phi
        @test m_mat.Theta ≈ m_3d.Theta
        @test m_pure_ma.p == 0 && size(m_pure_ma.Phi) == (2, 2, 0)
        @test m_pure_ar.q == 0 && size(m_pure_ar.Theta) == (2, 2, 0)

        # Convenient virtual properties via getproperty / propertynames
        @test :ma_eigenvalues in propertynames(m_3d)
        @test :nonfundamental_order in propertynames(m_3d)
        @test :Sigma in propertynames(m_3d)
        @test :residuals in propertynames(m_3d)
        @test m_3d.ma_eigenvalues ≈ Complex.(m_3d.ma_eig_re, m_3d.ma_eig_im)
        @test m_3d.nonfundamental_order == 1
        @test m_3d.regime === :mixed_nonfundamental
        @test m_3d.gmr_region === :II
        @test m_3d.Sigma ≈ m_3d.C * m_3d.C'
        @test m_3d.residuals == m_3d.shocks
        @test m_3d.distribution === :student_t
        @test length(m_3d.Phi_list) == 1 && m_3d.Phi_list[1] ≈ m_3d.Phi[:, :, 1]
        @test length(m_3d.Theta_list) == 1 && m_3d.Theta_list[1] ≈ m_3d.Theta[:, :, 1]

        # Accessors and StatsAPI interface
        @test nvars(m_3d) == 2
        @test nlags(m_3d) == 1
        @test effective_nobs(m_3d) == 119
        @test varnames(m_3d) == ["y1", "y2"]
        @test StatsAPI.nobs(m_3d) == 120
        @test length(StatsAPI.coef(m_3d)) == 2 + 4 + 4 + 4
        @test StatsAPI.residuals(m_3d) === m_3d.shocks
        @test StatsAPI.loglikelihood(m_3d) == -320.5
        @test StatsAPI.aic(m_3d) == 665.0
        @test StatsAPI.bic(m_3d) == 698.0
        @test StatsAPI.dof(m_3d) == 2 + 4 + 4 + 4 + 2
        @test StatsAPI.dof_residual(m_3d) == 119 * 2 - StatsAPI.dof(m_3d)
        @test !StatsAPI.islinear(m_3d)

        # RSER-01 compatibility: serialization-safe field types & positional reconstruction
        for fname in fieldnames(SVARMAModel)
            val = getfield(m_3d, fname)
            @test !(val isa AbstractArray{<:Complex})
            @test !(val isa Vector{<:AbstractMatrix})
            @test !(val isa AbstractDict)
        end
        raw_fields = Any[getfield(m_3d, f) for f in fieldnames(SVARMAModel)]
        m_rebuilt = SVARMAModel{Float64}(raw_fields...)
        @test m_rebuilt.Y == m_3d.Y
        @test m_rebuilt.Theta == m_3d.Theta
        @test m_rebuilt.ma_eigenvalues == m_3d.ma_eigenvalues
        @test m_rebuilt.regime == m_3d.regime
        @test m_rebuilt.gmr_region == m_3d.gmr_region

        # Display & report under text and LaTeX backends
        txt_show = sprint(show, m_3d)
        txt_rep = sprint(report, m_3d)
        @test occursin("SVARMA(1, 1)", txt_show)
        @test occursin("mixed_nonfundamental", txt_show)
        @test txt_show == txt_rep
        ltx = with_display_backend(:latex) do
            sprint(report, m_3d)
        end
        @test occursin("SVARMA(1, 1)", ltx)
    end

    @testset "3. Schur Forward/Backward Shock Filter — Interior Recovery (< 1e-10)" begin
        C_true = [1.2 -0.35; 0.45 0.9]
        Phi_true = [0.35 0.1; -0.08 0.3]

        # (a) Fundamental (|λ| < 1)
        Theta_fund = [-0.45 0.1; -0.15 0.35]
        sim_f = dgp_ssvarma(; T=400, Phi=Phi_true, Theta=Theta_fund, C=C_true,
                            dist=:student_t, nu=5.0, seed=9052)
        res_f = ssvarma_filter_shocks(sim_f.Y, sim_f.mu, sim_f.Phi, sim_f.Theta, sim_f.C)
        @test res_f.n_fundamental == 2 && res_f.n_nonfundamental == 0
        @test maximum(abs, res_f.shocks[40:end, :] .- sim_f.eta[41:end, :]) < 1e-10

        # (b) Mixed non-fundamental GMR eq. 4.27 (Θ = [-0.5 0; 1 -2], λ = 0.5, 2.0, m = 1)
        Theta_mix = [-0.5 0.0; 1.0 -2.0]
        sim_m = dgp_ssvarma(; T=400, Phi=Phi_true, Theta=Theta_mix, C=C_true,
                            dist=:laplace, seed=9053)
        res_m = ssvarma_filter_shocks(sim_m.Y, sim_m.mu, sim_m.Phi, sim_m.Theta, sim_m.C)
        @test res_m.n_fundamental == 1 && res_m.n_nonfundamental == 1
        # At boundary gap ≥ 40, |2.0|^{-40} < 1e-12; also check stronger roots for 20:380
        @test maximum(abs, res_m.shocks[40:360, :] .- sim_m.eta[41:361, :]) < 1e-10

        Theta_mix_strong = [-0.25 0.0; 1.0 -4.0]
        sim_ms = dgp_ssvarma(; T=400, Phi=nothing, Theta=Theta_mix_strong, C=C_true,
                             dist=:student_t, nu=5.0, seed=9054)
        res_ms = ssvarma_filter_shocks(sim_ms.Y, sim_ms.Theta, sim_ms.C)
        @test maximum(abs, res_ms.shocks[20:380, :] .- sim_ms.eta[20:380, :]) < 1e-10

        # (c) Purely non-fundamental (both |λ| > 1)
        Theta_pure = [-2.0 0.2; -0.4 -2.5]
        sim_p = dgp_ssvarma(; T=400, Phi=Phi_true, Theta=Theta_pure, C=C_true,
                            dist=:mixture_normal_asym, seed=9055)
        res_p = ssvarma_filter_shocks(sim_p.Y, sim_p.mu, sim_p.Phi, sim_p.Theta, sim_p.C)
        @test res_p.n_fundamental == 0 && res_p.n_nonfundamental == 2
        @test maximum(abs, res_p.shocks[40:360, :] .- sim_p.eta[41:361, :]) < 1e-10

        # (d) Complex-conjugate non-fundamental (λ = 1.6 ± 1.2i, |λ| = 2.0)
        # -Θ₁ = [1.6 1.2; -1.2 1.6] => Θ₁ = [-1.6 -1.2; 1.2 -1.6]
        Theta_cplx = [-1.6 -1.2; 1.2 -1.6]
        sim_c = dgp_ssvarma(; T=400, Phi=Phi_true, Theta=Theta_cplx, C=C_true,
                            dist=:skew, seed=9056)
        res_c = ssvarma_filter_shocks(sim_c.Y, sim_c.mu, sim_c.Phi, sim_c.Theta, sim_c.C)
        @test res_c.n_fundamental == 0 && res_c.n_nonfundamental == 2
        @test maximum(abs, res_c.shocks[40:360, :] .- sim_c.eta[41:361, :]) < 1e-10

        # (e) Higher-order q = 2 (mixed real + complex roots in 4×4 companion)
        R1 = [0.4 0.1; -0.1 0.3]
        R2 = [0.5 0.0; 0.3 2.0]
        Theta_q2 = [-(R1 + R2), R1 * R2]
        sim_q2 = dgp_ssvarma(; T=400, Phi=Phi_true, Theta=Theta_q2, C=C_true,
                             dist=:student_t, nu=5.0, seed=9057)
        res_q2 = ssvarma_filter_shocks(sim_q2.Y, sim_q2.mu, sim_q2.Phi, sim_q2.Theta, sim_q2.C)
        @test res_q2.n_fundamental == 3 && res_q2.n_nonfundamental == 1
        @test maximum(abs, res_q2.shocks[40:360, :] .- sim_q2.eta[41:361, :]) < 1e-10

        # (f) Pure SVAR (q = 0)
        sim_q0 = dgp_ssvarma(; T=100, Phi=Phi_true, Theta=nothing, C=C_true, seed=9058)
        res_q0 = ssvarma_filter_shocks(sim_q0.Y, sim_q0.mu, sim_q0.Phi, sim_q0.Theta, sim_q0.C)
        @test maximum(abs, res_q0.shocks .- sim_q0.eta[2:end, :]) < 1e-12

        # (g) Unit-root and singular-C guards
        rng = Xoshiro(90581)
        U_bad1 = randn(rng, 30, 2)
        U_bad2 = randn(rng, 30, 2)
        @test_throws IdentificationError ssvarma_filter_shocks(U_bad1, [-1.0 0.0; 0.0 -0.5], C_true)
        @test_throws SingularSystemError ssvarma_filter_shocks(U_bad2, Theta_mix, [1.0 2.0; 2.0 4.0])
    end

    @testset "4. Exact Finite-Sample Log-Jacobian vs ForwardDiff (< 1e-12)" begin
        rng = Xoshiro(9059)
        T_small = 8
        n = 2
        C_mat = [1.15 -0.3; 0.4 0.85]
        U0 = randn(rng, T_small, n)

        # (a) Fundamental q = 1 and q = 2: direct ForwardDiff on U -> shocks
        for Th in (reshape([-0.4 0.1; -0.1 0.3], 2, 2, 1),
                   cat([-0.6 0.0; 0.1 -0.5], [0.08 0.0; -0.02 0.06]; dims=3))
            res = ssvarma_filter_shocks(U0, Th, C_mat)
            J_fd = ForwardDiff.jacobian(u -> vec(ssvarma_filter_shocks(reshape(u, T_small, n), Th, C_mat).shocks),
                                        vec(U0))
            logjac_fd = -first(logabsdet(J_fd)) / T_small
            @test isapprox(res.log_jacobian, logjac_fd; atol=1e-12)
        end

        # (b) Mixed, purely non-fundamental, complex-conjugate, and q = 2 non-fundamental:
        #     exact augmented finite-sample bijection (z_init, U, z_term) -> (eps_pre, shocks)
        cases_nf = [
            ("mixed q=1", reshape([-0.5 0.0; 1.0 -2.0], 2, 2, 1)),
            ("pure nonfund q=1", reshape([-2.0 0.2; -0.4 -2.5], 2, 2, 1)),
            ("complex nonfund q=1", reshape([-1.6 -1.2; 1.2 -1.6], 2, 2, 1)),
            ("mixed q=2", cat(-([0.4 0.1; -0.1 0.3] + [0.5 0.0; 0.3 2.0]),
                              [0.4 0.1; -0.1 0.3] * [0.5 0.0; 0.3 2.0]; dims=3)),
        ]
        for (label, Th) in cases_nf
            res = ssvarma_filter_shocks(U0, Th, C_mat)
            m1 = res.n_fundamental
            m2 = res.n_nonfundamental
            x0 = vcat(zeros(m1), vec(U0), zeros(m2))
            J_aug = ForwardDiff.jacobian(x -> begin
                zi = @view x[1:m1]
                Um = reshape(@view(x[(m1 + 1):(m1 + T_small * n)]), T_small, n)
                zt = @view x[(m1 + T_small * n + 1):end]
                out = MEM._svarma_filter_augmented(Um, Th, C_mat, zi, zt)
                vcat(vec(out.eps_pre), vec(out.shocks))
            end, x0)
            logjac_fd = -first(logabsdet(J_aug)) / T_small
            @test isapprox(res.log_jacobian, logjac_fd; atol=1e-12)
            # Also check decomposition log_jacobian = log|det C| + Σ_{|λ|>1} log|λ|
            sum_unstable = sum(log, abs.(filter(z -> abs(z) > 1.0, res.eigenvalues)); init=0.0)
            @test isapprox(res.log_jacobian, log(abs(det(C_mat))) + sum_unstable; atol=1e-12)
        end
    end

    @testset "5. Fundamentalness Regime & GMR (2020, Fig. 6) Region I–V Classifier" begin
        # Companion matrix A = -Θ₁ has tr(A) = τ, det(A) = δ via A = [τ -δ; 1 0] -> Θ₁ = [-τ δ; -1 0]
        from_tr_det(tau, delta) = [-tau delta; -1.0 0.0]

        # Region I: fundamental (τ = 0.5, δ = 0.2 -> 1 - τ + δ = 0.7 > 0, 1 + τ + δ = 1.7 > 0, δ < 1)
        r1 = fundamentalness_regime(from_tr_det(0.5, 0.2))
        @test r1.regime === :fundamental && r1.gmr_region === :I
        @test r1.n_fundamental == 2 && r1.n_nonfundamental == 0

        # Region II: mixed right wing (1 - τ + δ < 0, 1 + τ + δ > 0), e.g. GMR eq. 4.27 (τ = 2.5, δ = 1.0)
        r2a = fundamentalness_regime([-0.5 0.0; 1.0 -2.0])
        @test r2a.regime === :mixed_nonfundamental && r2a.gmr_region === :II
        @test r2a.n_fundamental == 1 && r2a.n_nonfundamental == 1
        # Check δ = 1 with |τ| > 2 stays inside Region II (not unit root!)
        r2b = fundamentalness_regime(from_tr_det(2.5, 1.0))
        @test r2b.regime === :mixed_nonfundamental && r2b.gmr_region === :II

        # Region III: purely non-fundamental top wedge (δ > 1, 1 ± τ + δ > 0), complex & real
        r3_cplx = fundamentalness_regime(from_tr_det(1.0, 2.0))
        @test r3_cplx.regime === :purely_nonfundamental && r3_cplx.gmr_region === :III
        r3_real = fundamentalness_regime(from_tr_det(3.5, 3.0))
        @test r3_real.regime === :purely_nonfundamental && r3_real.gmr_region === :III

        # Region IV: mixed left wing (1 + τ + δ < 0, 1 - τ + δ > 0)
        r4 = fundamentalness_regime(from_tr_det(-2.5, 1.0))
        @test r4.regime === :mixed_nonfundamental && r4.gmr_region === :IV

        # Region V: purely non-fundamental bottom wedge (1 - τ + δ < 0, 1 + τ + δ < 0)
        r5 = fundamentalness_regime(from_tr_det(0.0, -2.0))
        @test r5.regime === :purely_nonfundamental && r5.gmr_region === :V

        # Exact GMR Fig. 6 boundary perturbations (1 ∓ τ + δ = 0, and δ = 1 for |τ| < 2)
        eps_b = 1e-4
        @test fundamentalness_regime(from_tr_det(0.0, 1.0 - eps_b)).gmr_region === :I
        @test fundamentalness_regime(from_tr_det(0.0, 1.0 + eps_b)).gmr_region === :III
        @test fundamentalness_regime(from_tr_det(0.0, 1.0)).regime === :unit_root
        @test fundamentalness_regime(from_tr_det(0.0, 1.0)).gmr_region === :unit_root

        # Boundary 1 - τ + δ = 0 (at δ = 0.2, τ = 1.2)
        @test fundamentalness_regime(from_tr_det(1.2 - eps_b, 0.2)).gmr_region === :I
        @test fundamentalness_regime(from_tr_det(1.2 + eps_b, 0.2)).gmr_region === :II
        @test fundamentalness_regime(from_tr_det(1.2, 0.2)).regime === :unit_root

        # Boundary 1 + τ + δ = 0 (at δ = 0.2, τ = -1.2)
        @test fundamentalness_regime(from_tr_det(-1.2 + eps_b, 0.2)).gmr_region === :I
        @test fundamentalness_regime(from_tr_det(-1.2 - eps_b, 0.2)).gmr_region === :IV
        @test fundamentalness_regime(from_tr_det(-1.2, 0.2)).regime === :unit_root

        # Boundary between II and V (1 + τ + δ = 0 at δ = -2.0, τ = 1.0)
        @test fundamentalness_regime(from_tr_det(1.0 + eps_b, -2.0)).gmr_region === :II
        @test fundamentalness_regime(from_tr_det(1.0 - eps_b, -2.0)).gmr_region === :V

        # Boundary between IV and V (1 - τ + δ = 0 at δ = -2.0, τ = -1.0)
        @test fundamentalness_regime(from_tr_det(-1.0 - eps_b, -2.0)).gmr_region === :IV
        @test fundamentalness_regime(from_tr_det(-1.0 + eps_b, -2.0)).gmr_region === :V

        # Non-(2,1) systems have gmr_region === :none
        r_3d = fundamentalness_regime(diagm([-0.4, -1.5, -2.0]))
        @test r_3d.regime === :mixed_nonfundamental && r_3d.gmr_region === :none
        @test r_3d.n_fundamental == 1 && r_3d.n_nonfundamental == 2
        @test fundamentalness_regime(nothing, Matrix{Float64}(I, 2, 2)).gmr_region === :none
    end

    @testset "6. Blaschke Second-Moment Equivalence (< 1e-11)" begin
        C_mat = [1.1 0.35; -0.4 0.95]
        freqs = range(0.0, pi; length=64)

        function check_blaschke_case(Theta_in, C_in, expected_count::Int)
            reps = blaschke_representations(Theta_in, C_in)
            @test length(reps) == expected_count
            Theta0 = reps[1].Theta
            C0 = reps[1].C
            q = size(Theta0, 3)

            n_fund = count(r -> r.regime === :fundamental, reps)
            @test n_fund == 1

            for rep in reps
                @test eltype(rep.Theta) == eltype(C0)
                @test eltype(rep.C) == eltype(C0)
                @test all(isfinite, rep.Theta) && all(isfinite, rep.C)
                # Spectral density equivalence across ω ∈ [0, π]
                max_spec_err = maximum(w -> norm(_test_ma_spectrum(rep.Theta, rep.C, w) -
                                                 _test_ma_spectrum(Theta0, C0, w)), freqs)
                @test max_spec_err < 1e-11
                # Exact time-domain MA autocovariance equivalence for k = 0..q
                for k in 0:q
                    @test norm(_test_ma_autocov(rep.Theta, rep.C, k) -
                               _test_ma_autocov(Theta0, C0, k)) < 1e-11
                end
            end
            return reps
        end

        # (a) Mixed real n = 2, q = 1 (2 real roots -> 4 representations)
        reps_mix = check_blaschke_case([-0.5 0.0; 1.0 -2.0], C_mat, 4)
        @test reps_mix[1].regime === :mixed_nonfundamental
        fund_rep = only(filter(r -> r.regime === :fundamental, reps_mix))
        @test all(abs.(fund_rep.eigenvalues) .< 1.0)

        # (b) Complex-conjugate non-fundamental n = 2, q = 1 (1 pair -> 2 representations)
        reps_cplx = check_blaschke_case([-2.4 1.0; -2.0 0.0], C_mat, 2)
        @test reps_cplx[1].regime === :purely_nonfundamental
        @test reps_cplx[2].regime === :fundamental

        # (c) Univariate q = 2 with complex conjugate pair (1 pair -> 2 representations)
        reps_1d = check_blaschke_case(cat([-2.4;;], [2.0;;]; dims=3), [1.5;;], 2)
        @test reps_1d[2].regime === :fundamental

        # (d) Bivariate q = 2 with 2 real + 1 complex-conjugate pair (r = 3 -> 8 representations)
        R1u = [2.4 -1.0; 2.0 0.0]
        R2 = [0.5 0.0; 0.3 1.8]
        check_blaschke_case([-(R1u + R2), R1u * R2], C_mat, 8)

        # (e) Repeated root and Jordan block
        reps_rep = check_blaschke_case([-2.0 0.0; 0.0 -2.0], C_mat, 4)
        @test length(unique(r -> round.(r.Theta; digits=6), reps_rep)) == 4
        check_blaschke_case([-2.0 1.0; 0.0 -2.0], C_mat, 4)

        # (f) Singular Θ₁ (one zero eigenvalue -> only 1 flippable root -> 2 representations)
        check_blaschke_case([-2.0 0.0; 1.0 0.0], C_mat, 2)

        # (g) Unit-root throws IdentificationError
        @test_throws IdentificationError blaschke_representations([-1.0 0.0; 0.0 -0.5], C_mat)
    end

    @testset "7. Causal IRF, Hump-Shaped Non-Fundamental Dynamics, and FEVD" begin
        H = 20
        Phi1 = [0.5 0.1; -0.05 0.4]
        Theta1 = [-0.5 0.0; 1.0 -2.0]
        C_mat = [1.0 0.3; -0.4 0.9]
        sim = dgp_ssvarma(; T=80, Phi=Phi1, Theta=Theta1, C=C_mat, seed=9060)
        m = SVARMAModel(sim.Y, sim.Phi, sim.Theta, sim.C; mu=sim.mu)

        ir = irf(m, H)
        @test ir isa ImpulseResponse{Float64}
        @test size(ir.values) == (H, 2, 2)
        @test ir.horizon == H
        @test ir.ci_type === :none
        @test ir.values[1, :, :] ≈ C_mat
        for j in 1:2
            @test ir.values[:, :, j] ≈ _test_impulse_sim(m.Phi, m.Theta, m.C, j, H) atol=1e-12
        end

        # GMR (2020, fn. 6): non-fundamental representation exhibits hump-shaped IRF peaking after h = 0
        # whereas its Blaschke-equivalent fundamental representation is monotone!
        C1d = [1.0;;]
        Phi1d = reshape([0.5;;], 1, 1, 1)
        Th_nf = reshape([2.0;;], 1, 1, 1)    # u_t = ε_t + 2 ε_{t-1}, root z = -0.5 inside unit circle
        reps_1d = blaschke_representations(Th_nf, C1d)
        rep_f = only(filter(r -> r.regime === :fundamental, reps_1d))
        Y_dummy = zeros(10, 1)
        m_nf = SVARMAModel(Y_dummy, Phi1d, Th_nf, C1d; shocks=zeros(9, 1))
        m_f = SVARMAModel(Y_dummy, Phi1d, rep_f.Theta, rep_f.C; shocks=zeros(9, 1))
        ir_nf = irf(m_nf, 10).values[:, 1, 1]
        ir_f = irf(m_f, 10).values[:, 1, 1]
        @test abs(ir_nf[2]) > abs(ir_nf[1])   # hump-shaped peak at h = 1
        @test abs(ir_f[1]) >= abs(ir_f[2])    # fundamental impact dominates
        @test abs(ir_nf[1] - ir_f[1]) > 0.5   # distinct IRFs despite identical spectrum

        # Structural FEVD sums to 1 across shocks at every variable and horizon
        fv = fevd(m, H)
        @test fv isa FEVD{Float64}
        @test size(fv.decomposition) == (2, 2, H)
        @test size(fv.proportions) == (2, 2, H)
        @test all(x -> 0.0 <= x <= 1.0 + 1e-12, fv.proportions)
        for h in 1:H, i in 1:2
            @test isapprox(sum(fv.proportions[i, :, h]), 1.0; atol=1e-12)
        end
        for h in 2:H, i in 1:2, j in 1:2
            @test fv.decomposition[i, j, h] >= fv.decomposition[i, j, h - 1] - 1e-14
        end
    end

    @testset "8. Float32 Support & dgp_ssvarma Distribution Coverage" begin
        Phi32 = Float32[0.4 0.1; -0.05 0.35]
        Theta32 = Float32[-0.5 0.0; 1.0 -2.0]
        C32 = Float32[1.0 0.3; -0.4 0.9]
        sim32 = dgp_ssvarma(; T=160, Phi=Phi32, Theta=Theta32, C=C32, seed=9061)
        @test eltype(sim32.Y) == Float32
        @test eltype(sim32.eta) == Float32
        @test eltype(sim32.Phi) == Float32
        @test eltype(sim32.Theta) == Float32
        @test eltype(sim32.C) == Float32

        # Bit-exact reproducibility via seed
        sim32_b = dgp_ssvarma(; T=160, Phi=Phi32, Theta=Theta32, C=C32, seed=9061)
        @test sim32.Y == sim32_b.Y && sim32.eta == sim32_b.eta

        res32 = ssvarma_filter_shocks(sim32.Y, sim32.mu, sim32.Phi, sim32.Theta, sim32.C)
        @test eltype(res32.shocks) == Float32
        @test res32.log_jacobian isa Float32
        @test maximum(abs, res32.shocks[30:130, :] .- sim32.eta[31:131, :]) < 1f-4

        reg32 = fundamentalness_regime(Theta32, C32)
        @test reg32.regime === :mixed_nonfundamental
        @test reg32.gmr_region === :II
        @test eltype(reg32.eigenvalues) == ComplexF32

        reps32 = blaschke_representations(Theta32, C32)
        @test length(reps32) == 4
        @test all(r -> eltype(r.Theta) == Float32 && eltype(r.C) == Float32, reps32)

        m32 = SVARMAModel(sim32.Y, sim32.Phi, sim32.Theta, sim32.C; mu=sim32.mu)
        @test m32 isa SVARMAModel{Float32}
        @test irf(m32, 8) isa ImpulseResponse{Float32}
        @test fevd(m32, 8) isa FEVD{Float32}

        # All supported non-Gaussian shock families in dgp_ssvarma
        for dist in (:student_t, :laplace, :mixture, :mixture_normal_asym,
                     :skew, :skew_normal, :ged, :gauss)
            d = dgp_ssvarma(; T=100, dist=dist, skew=2.0, seed=9062)
            @test size(d.Y) == (100, 2) && size(d.eta) == (100, 2)
            @test all(isfinite, d.Y) && all(isfinite, d.eta)
        end
    end

end
