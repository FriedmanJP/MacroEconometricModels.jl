# SDFM-17 recovery / oracle + SDFM-18 proxy identification.
# Included from test_structural_dfm.jl. Run standalone:
#   MACRO_FAST_TESTS=1 julia --project=. -e 'include("test/fixtures.jl"); include("test/factor/test_sdfm_recovery.jl")'

using Test
using LinearAlgebra
using Statistics
using Random
using DelimitedFiles
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _SDFM_REF = joinpath(@__DIR__, "..", "oracle", "sdfm_ref")

function _align_irf_columns(est::Array{T,3}, truth::Array{T,3}) where {T}
    _, _, q = size(est)
    used = falses(q)
    out = similar(est)
    for j in 1:q
        best = 0
        bests = zero(T)
        for k in 1:q
            used[k] && continue
            d = dot(@view(est[1, :, k]), @view(truth[1, :, j]))
            if abs(d) >= abs(bests)
                bests = d
                best = k
            end
        end
        used[best] = true
        sg = bests < 0 ? -one(T) : one(T)
        out[:, :, j] = sg .* est[:, :, best]
    end
    out
end

function _rel_rmse(est, truth)
    num = sqrt(mean(abs2, est .- truth))
    den = sqrt(mean(abs2, truth))
    den <= 0 ? num : num / den
end

"""True if `A` matches `B` after independent ±1 row and column signs (PCA/eigen LAPACK)."""
function _same_up_to_row_col_signs(A::AbstractMatrix, B::AbstractMatrix; atol=1e-6, rtol=1e-6)
    size(A) == size(B) || return false
    r, c = size(A)
    (r > 8 || c > 8) && throw(ArgumentError("_same_up_to_row_col_signs: r,c ≤ 8"))
    sr = Vector{eltype(A)}(undef, r)
    sc = Vector{eltype(A)}(undef, c)
    for maskr in 0:(2^r - 1), maskc in 0:(2^c - 1)
        @inbounds for i in 1:r
            sr[i] = ((maskr >> (i - 1)) & 1) == 1 ? -one(eltype(A)) : one(eltype(A))
        end
        @inbounds for j in 1:c
            sc[j] = ((maskc >> (j - 1)) & 1) == 1 ? -one(eltype(A)) : one(eltype(A))
        end
        ok = true
        @inbounds for j in 1:c, i in 1:r
            if !isapprox(sr[i] * A[i, j] * sc[j], B[i, j]; atol=atol, rtol=rtol)
                ok = false
                break
            end
        end
        ok && return true
    end
    false
end

@testset "SDFM recovery, fixture, and proxy" begin

    @testset "committed FGLR Cholesky fixture" begin
        X = readdlm(joinpath(_SDFM_REF, "X.csv"), ',', Float64)
        Kref = readdlm(joinpath(_SDFM_REF, "K.csv"), ',', Float64)
        B0ref = readdlm(joinpath(_SDFM_REF, "B0.csv"), ',', Float64)
        irfref = readdlm(joinpath(_SDFM_REF, "irf.csv"), ',', Float64)
        sdfm = estimate_structural_dfm(X, 2; r=2, p=1, H=12, identification=:cholesky,
            order=[1, 2], standardize=true, method=:fglr)
        # Factor-row and shock-column signs of K and B0 follow LAPACK eigen/PCA
        # (Windows CI flipped two signs vs the committed Linux fixture).
        @test _same_up_to_row_col_signs(sdfm.K, Kref; atol=1e-6, rtol=1e-6)
        @test _same_up_to_row_col_signs(sdfm.B0, B0ref; atol=1e-6, rtol=1e-6)
        ir = irf(sdfm, 12).values
        H, N, q = size(ir)
        irref3 = reshape(irfref, H, N, q)
        aligned = _align_irf_columns(ir, irref3)
        @test aligned ≈ irref3 atol=1e-6 rtol=1e-6
        m2 = MacroEconometricModels._reconstruct_from_container(
            MacroEconometricModels._build_container(sdfm))
        @test m2.identification === sdfm.identification
        @test m2.K ≈ sdfm.K atol=1e-10
    end

    if !FAST
        @testset "FGLR recovers r>q DGP; gdfm_var does not" begin
            rng = Random.MersenneTwister(72602)
            T_obs, N, q, rstat = 500, 60, 2, 4
            Φ = [0.5 0.0; 0.1 0.4]
            Λ = 0.4 .* randn(rng, N, rstat)
            Λ[1:2, 1:2] .= [1.0 0.0; 0.6 1.0]
            Λ[1:2, 3:4] .= [0.3 0.1; 0.2 0.4]
            f = zeros(T_obs + 1, q)
            εtrue = zeros(T_obs, q)
            X = zeros(T_obs, N)
            true_irf = zeros(12, N, q)
            for h in 1:12
                Ah = h == 1 ? Matrix{Float64}(I, q, q) : Φ^(h - 1)
                Alag = h == 1 ? zeros(q, q) : Φ^(h - 2)
                true_irf[h, :, :] = Λ * vcat(Ah, Alag)
            end
            f[1, :] = randn(rng, q)
            for t in 1:T_obs
                εtrue[t, :] = randn(rng, q)
                f[t + 1, :] = Φ * f[t, :] + εtrue[t, :]
                Fstat = vcat(f[t + 1, :], f[t, :])
                X[t, :] = Λ * Fstat + 0.15 .* randn(rng, N)
            end
            sdfm = estimate_structural_dfm(X, q; r=rstat, method=:fglr,
                identification=:cholesky, order=[1, 2], p=1, H=12, standardize=false)
            est = _align_irf_columns(irf(sdfm, 12).values, true_irf)
            @test _rel_rmse(est, true_irf) < 0.15
            εhat = structural_shocks(sdfm)
            Tε = min(size(εhat, 1), size(εtrue, 1))
            εe = εhat[end - Tε + 1:end, :]
            εt = εtrue[end - Tε + 1:end, :]
            for j in 1:q
                c = [cor(εe[:, k], εt[:, j]) for k in 1:q]
                k = argmax(abs.(c))
                @test abs(c[k]) > 0.9
            end
            sdfm_g = estimate_structural_dfm(X, q; method=:gdfm_var,
                identification=:cholesky, p=1, H=12, standardize=false)
            est_g = _align_irf_columns(irf(sdfm_g, 12).values, true_irf)
            @test _rel_rmse(est_g, true_irf) >= 0.15
        end

        @testset "proxy identification recovers the instrumented shock" begin
            rng = Random.MersenneTwister(72701)
            T_obs, N, q = 600, 20, 2
            F = zeros(T_obs, q)
            εtrue = zeros(T_obs, q)
            F[1, :] = randn(rng, q)
            for t in 2:T_obs
                εtrue[t, :] = randn(rng, q)
                F[t, :] = 0.4 .* F[t - 1, :] .+ εtrue[t, :]
            end
            εtrue[1, :] = F[1, :]
            Λ = randn(rng, N, q)
            Λ[1, :] .= [1.2, 0.1]
            X = F * Λ' .+ 0.2 .* randn(rng, T_obs, N)
            names = ["y$i" for i in 1:N]
            z = εtrue[:, 1] .+ 0.5 .* randn(rng, T_obs)
            sdfm = estimate_structural_dfm(X, q; r=2, p=1, H=8, standardize=false,
                identification=:proxy, instrument=z, normalize=("y1", 1.0),
                varnames=names)
            @test sdfm.identification === :proxy
            @test sdfm.first_stage_F > 10
            ir = irf(sdfm, 1).values
            true_imp = Λ[:, 1]
            true_imp = true_imp ./ true_imp[1]
            est_imp = ir[1, :, 1]
            rel = norm(est_imp - true_imp) / norm(true_imp)
            @test rel < 0.15
            z_noise = randn(rng, T_obs)
            @test_logs (:warn, r"(?i)weak") estimate_structural_dfm(X, q; r=2, p=1, H=6,
                standardize=false, identification=:proxy, instrument=z_noise,
                normalize=("y1", 1.0), varnames=names)
        end
    else
        @testset "proxy identification (FAST)" begin
            rng = Random.MersenneTwister(72711)
            T_obs, N, q = 120, 10, 2
            F = zeros(T_obs, q)
            ε1 = randn(rng, T_obs)
            F[1, :] = randn(rng, q)
            for t in 2:T_obs
                F[t, :] = 0.3 .* F[t - 1, :] .+ [ε1[t], randn(rng)]
            end
            Λ = randn(rng, N, q)
            Λ[1, 1] = 1.0
            X = F * Λ' .+ 0.2 .* randn(rng, T_obs, N)
            z = ε1 .+ 0.3 .* randn(rng, T_obs)
            sdfm = estimate_structural_dfm(X, q; r=2, p=1, H=6, standardize=false,
                identification=:proxy, instrument=z, normalize=(1, 1.0))
            @test isfinite(sdfm.first_stage_F)
            @test all(isfinite, sdfm.B0)
        end
    end
end
