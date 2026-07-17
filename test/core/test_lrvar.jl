# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Statistics
using LinearAlgebra
using Random

const MEM = MacroEconometricModels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Discrete Lyapunov Γ₀ = A Γ₀ A' + Σ via vec: vec(Γ₀) = (I − A⊗A)⁻¹ vec(Σ).
function _lyap_gamma0(A::AbstractMatrix, Σ::AbstractMatrix)
    k = size(A, 1)
    G = (I(k * k) - kron(A, A)) \ vec(Σ)
    return reshape(G, k, k)
end

# Simulate a stationary VAR(1): U_t = A U_{t-1} + e_t, e_t ~ N(0, Σ_e).
function _sim_var1(A::AbstractMatrix, Σ::AbstractMatrix, T::Int; rng, burn::Int=500)
    k = size(A, 1)
    L = cholesky(Symmetric(Σ)).L
    U = zeros(T, k)
    x = zeros(k)
    for t in 1:(T + burn)
        x = A * x + L * randn(rng, k)
        t > burn && (U[t - burn, :] .= x)
    end
    return U
end

@testset "Long-run variance toolkit (EV-12)" begin

    # =======================================================================
    # 1. End-to-end: vector + matrix, every kernel, every bandwidth selector
    # =======================================================================
    @testset "end-to-end runs" begin
        rng = MersenneTwister(42)
        U = randn(rng, 300, 3)
        u = randn(rng, 300)

        for kernel in (:bartlett, :parzen, :qs, :quadratic_spectral, :tukey_hanning)
            for bw in (:andrews, :nw94, 6)
                Ω = lrcov(U; kernel=kernel, bandwidth=bw)
                @test size(Ω) == (3, 3)
                @test issymmetric(Ω) || maximum(abs.(Ω - Ω')) < 1e-10
                @test isposdef(Symmetric(Ω) + 1e-8I)   # PSD-projected
                Λ = lrcov_oneside(U; kernel=kernel, bandwidth=bw)
                @test size(Λ) == (3, 3)
                s = lrvar(u; kernel=kernel, bandwidth=bw)
                @test s isa Real && s ≥ 0
            end
        end

        # prewhiten on/off
        @test size(lrcov(U; prewhiten=true)) == (3, 3)
        @test lrvar(u; prewhiten=true) ≥ 0

        # VARHAC vector + matrix, both criteria
        @test size(varhac(U; ic=:aic)) == (3, 3)
        @test size(varhac(U; ic=:bic)) == (3, 3)
        @test varhac(u) ≥ 0
        @test size(varhac(U; max_lag=4)) == (3, 3)
        @test size(varhac(U; max_lag=0)) == (3, 3)   # white-noise fallback → Γ₀
    end

    # =======================================================================
    # 2. Analytic identity Ω = Λ + Λ' − Γ₀  (machine tolerance)
    # =======================================================================
    @testset "one/two-sided consistency Ω = Λ + Λ' − Γ₀" begin
        rng = MersenneTwister(7)
        U = randn(rng, 250, 2)
        n = size(U, 1)
        Ud = U .- mean(U, dims=1)
        Γ0 = (Ud' * Ud) / n

        for kernel in (:bartlett, :parzen, :qs, :tukey_hanning)
            for bw in (5, 12)
                # Use the RAW two-sided accumulator (skip PSD projection) so the algebraic
                # identity holds to machine precision (projection would perturb Ω).
                _, Λ, Γ0acc = MEM._lrv_accumulate(Ud, bw, MEM._lrv_kernel(kernel))
                Ω, _, _ = MEM._lrv_accumulate(Ud, bw, MEM._lrv_kernel(kernel))
                @test maximum(abs.(Ω - (Λ + Λ' - Γ0acc))) < 1e-12
                @test maximum(abs.(Γ0acc - Γ0)) < 1e-12
            end
        end
    end

    # =======================================================================
    # 3. Internal cross-check: lrcov/lrvar reproduce long_run_* exactly
    #    (they share the _hac_meat kernel and the T⁻¹ normalization)
    # =======================================================================
    @testset "matches long_run_covariance / long_run_variance" begin
        rng = MersenneTwister(99)
        U = randn(rng, 200, 3)
        u = U[:, 1]
        for kernel in (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning)
            for bw in (4, 9)
                @test maximum(abs.(lrcov(U; kernel=kernel, bandwidth=bw) -
                                   long_run_covariance(U; bandwidth=bw, kernel=kernel))) < 1e-12
                @test abs(lrvar(u; kernel=kernel, bandwidth=bw) -
                          long_run_variance(u; bandwidth=bw, kernel=kernel)) < 1e-12
            end
        end
    end

    # =======================================================================
    # 4. One-sided Λ equals the DIRECT Σ_{j≥0} k(j/b) Γ_j
    # =======================================================================
    @testset "one-sided Λ = direct one-sided kernel sum" begin
        rng = MersenneTwister(123)
        A = [0.5 0.1; -0.2 0.4]
        Σe = [1.0 0.3; 0.3 0.8]
        U = _sim_var1(A, Σe, 1000; rng=rng)
        n = size(U, 1)
        Ud = U .- mean(U, dims=1)
        bw = 8
        kernel = :bartlett

        Γ0 = (Ud' * Ud) / n
        Λ_direct = Matrix(Γ0)
        for j in 1:bw
            w = MEM.kernel_weight(j, bw, kernel, Float64)
            Γj = (Ud[(j+1):n, :]' * Ud[1:(n-j), :]) / n
            Λ_direct .+= w * Γj
        end
        @test maximum(abs.(lrcov_oneside(U; kernel=kernel, bandwidth=bw) - Λ_direct)) < 1e-10

        # Λ is genuinely one-sided (asymmetric here) — NOT the symmetric Ω.
        @test maximum(abs.(Λ_direct - Λ_direct')) > 1e-3
    end

    # =======================================================================
    # 5. Analytic VAR(1) long-run objects
    #    True two-sided Ω = (I−A)⁻¹ Σ_e (I−A)⁻ᵀ ; one-sided Λ = (I−A)⁻¹ Γ₀.
    # =======================================================================
    @testset "VAR(1) analytic Ω and Λ (large sample)" begin
        rng = MersenneTwister(2024)
        A = [0.6 0.0; 0.2 0.3]
        Σe = [1.0 0.2; 0.2 1.5]
        U = _sim_var1(A, Σe, 8000; rng=rng)
        k = 2
        D = inv(I(k) - A)
        Ω_true = D * Σe * D'
        Γ0_true = _lyap_gamma0(A, Σe)
        Λ_true = D * Γ0_true

        # Kernel HAC with a generous bandwidth should approach the analytic Ω.
        Ω_hat = lrcov(U; kernel=:qs, bandwidth=:andrews)
        @test norm(Ω_hat - Ω_true) / norm(Ω_true) < 0.20

        Λ_hat = lrcov_oneside(U; kernel=:qs, bandwidth=:andrews)
        @test norm(Λ_hat - Λ_true) / norm(Λ_true) < 0.25
    end

    # =======================================================================
    # 6. VARHAC recovers B(1)⁻¹ Σ B(1)⁻ᵀ of a fixed-seed VAR(1)
    # =======================================================================
    @testset "VARHAC recovers zero-frequency spectral density" begin
        rng = MersenneTwister(555)
        A = [0.5 0.1; 0.0 0.4]
        Σe = [1.0 0.25; 0.25 0.9]
        U = _sim_var1(A, Σe, 6000; rng=rng)
        k = 2
        D = inv(I(k) - A)
        Ω_true = D * Σe * D'

        Ω_vh = varhac(U; ic=:aic)
        @test norm(Ω_vh - Ω_true) / norm(Ω_true) < 0.15

        # Scalar AR(1): analytic LRV = σ²/(1−ρ)².
        ρ = 0.7
        x = zeros(6000)
        for t in 2:6000
            x[t] = ρ * x[t-1] + randn(rng)
        end
        lrv_true = 1.0 / (1 - ρ)^2
        @test abs(varhac(x) - lrv_true) / lrv_true < 0.20
    end

    # =======================================================================
    # 7. Refactor safety: newey_west unchanged after rerouting through _hac_meat
    #    Independent re-implementation of the ORIGINAL inline meat formula.
    # =======================================================================
    @testset "newey_west regression guard (bit-for-bit meat)" begin
        rng = MersenneTwister(2718)
        n, k = 180, 3
        X = hcat(ones(n), randn(rng, n, k - 1))
        u = randn(rng, n)
        XtX_inv = inv(X' * X)

        for kernel in (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning)
            bw = 5
            # ORIGINAL formula, recomputed independently.
            M = X .* u
            m = size(M, 1)
            S = M' * M
            jmax = kernel == :quadratic_spectral ? (m - 1) : bw
            for j in 1:jmax
                w = MEM.kernel_weight(j, bw, kernel, Float64)
                w == 0 && continue
                Γj = M[(j+1):m, :]' * M[1:(m-j), :]
                S .+= w * (Γj + Γj')
            end
            V_expected = XtX_inv * S * XtX_inv
            V_expected = (V_expected + V_expected') / 2

            V_actual = newey_west(X, u; bandwidth=bw, kernel=kernel)
            @test V_actual == V_expected           # bit-for-bit
        end
        # The shared _hac_meat kernel IS lrcov's un-normalized two-sided accumulator
        # (before PSD projection): T · Ω = _hac_meat, verified on a PSD-safe Bartlett meat.
        M = X .* u
        Ωraw, _, _ = MEM._lrv_accumulate(M .- mean(M, dims=1), 5, :bartlett)
        @test MEM._hac_meat(M .- mean(M, dims=1), 5, :bartlett) ≈ n * Ωraw atol=1e-8
    end

    # =======================================================================
    # 8. Newey–West (1994) automatic bandwidth
    # =======================================================================
    @testset "optimal_bandwidth_nw94" begin
        rng = MersenneTwister(11)
        # Persistent AR(1) → longer bandwidth than iid.
        ρ = 0.8
        x = zeros(500)
        for t in 2:500
            x[t] = ρ * x[t-1] + randn(rng)
        end
        bw_persist = optimal_bandwidth_nw94(x; kernel=:bartlett)
        bw_iid = optimal_bandwidth_nw94(randn(rng, 500); kernel=:bartlett)
        @test bw_persist isa Int && bw_persist ≥ 0
        @test bw_iid isa Int && bw_iid ≥ 0
        @test bw_persist > bw_iid

        # Matrix input aggregates to a scalar and returns a valid lag.
        @test optimal_bandwidth_nw94(randn(rng, 400, 2); kernel=:parzen) ≥ 0
        @test optimal_bandwidth_nw94(randn(rng, 400, 2); kernel=:quadratic_spectral) ≥ 0
        # prewhite lowers the pilot constant (3 vs 4).
        @test optimal_bandwidth_nw94(x; kernel=:bartlett, prewhiten=true) ≥ 0
    end

    # =======================================================================
    # 9. Prewhitening
    # =======================================================================
    @testset "Andrews–Monahan prewhitening" begin
        rng = MersenneTwister(321)
        A = [0.6 0.1; 0.1 0.5]
        Σe = [1.0 0.2; 0.2 1.0]
        U = _sim_var1(A, Σe, 4000; rng=rng)
        k = 2
        D = inv(I(k) - A)
        Ω_true = D * Σe * D'

        Ω_pw = lrcov(U; kernel=:bartlett, bandwidth=:andrews, prewhiten=true)
        @test isposdef(Symmetric(Ω_pw) + 1e-8I)
        # Prewhitened Bartlett recovers the analytic VAR(1) LRV to a loose tolerance.
        @test norm(Ω_pw - Ω_true) / norm(Ω_true) < 0.35

        # One-sided prewhitened recolor reduces to (I−A)⁻¹Γ₀ on VAR(1) data.
        Γ0_true = _lyap_gamma0(A, Σe)
        Λ_true = D * Γ0_true
        Λ_pw = lrcov_oneside(U; kernel=:bartlett, bandwidth=:andrews, prewhiten=true)
        @test norm(Λ_pw - Λ_true) / norm(Λ_true) < 0.25

        # Near-unit-root moments: prewhitening falls back gracefully (no throw, PSD result).
        rng2 = MersenneTwister(4)
        rw = cumsum(randn(rng2, 300, 2), dims=1)
        Ω_rw = @test_logs (:warn,) match_mode=:any lrcov(rw; prewhiten=true)
        @test size(Ω_rw) == (2, 2)
    end

    # =======================================================================
    # 10. Options + error handling
    # =======================================================================
    @testset "options and errors" begin
        rng = MersenneTwister(1)
        U = randn(rng, 100, 2)

        # demean=false differs from demean=true on a non-zero-mean series.
        Um = U .+ [5.0 -3.0]
        @test lrcov(Um; demean=true, bandwidth=4) ≈ lrcov(U; demean=true, bandwidth=4) atol=1e-8
        @test !(lrcov(Um; demean=false, bandwidth=4) ≈ lrcov(Um; demean=true, bandwidth=4))

        # :qs alias == :quadratic_spectral
        @test lrcov(U; kernel=:qs, bandwidth=6) == lrcov(U; kernel=:quadratic_spectral, bandwidth=6)

        @test_throws ArgumentError lrcov(U; kernel=:not_a_kernel)
        @test_throws ArgumentError lrcov(U; bandwidth=:not_a_selector)
        @test_throws ArgumentError lrcov(U; bandwidth=-3)
        @test_throws ArgumentError varhac(U; ic=:xic)

        # Float32 input stays Float32.
        U32 = Float32.(U)
        @test eltype(lrcov(U32; bandwidth=4)) == Float32
        @test lrvar(Float32.(U[:, 1])) isa Float32
    end

    # =======================================================================
    # 11. refs() renders for the toolkit
    # =======================================================================
    @testset "refs(:lrvar)" begin
        io = IOBuffer()
        refs(io, :lrvar)
        s = String(take!(io))
        @test occursin("Andrews", s)
        @test occursin("Newey", s)
        @test occursin("Den Haan", s) || occursin("Haan", s)
    end
end
