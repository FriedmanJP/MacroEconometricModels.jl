# CF-06 (#386): stacked IRF-matching target + CTW covariance transform.
using Test
using LinearAlgebra
using Random
using Statistics
using MacroEconometricModels

const MEM = MacroEconometricModels

# Toy container with recognizable entries: value = 100*var + 10*shock + h,
# var 1 = :pi (outcome), var 2 = :r (instrument).
function _cf06_container(; H=3, n_s=2, nd=40, rng=MersenneTwister(66))
    point(v) = [100.0 * v + 10.0 * k + h for h in 1:H, k in 1:n_s]
    noise(v) = begin
        D = Array{Float64,3}(undef, H, n_s, nd)
        for d in 1:nd
            D[:, :, d] = point(v) + 0.1 * randn(rng, H, n_s)
        end
        D
    end
    PolicyCausalEffects(outcomes=[:pi], instruments=[:r],
                        Theta_x=[point(1)], Theta_z=[point(2)],
                        Theta_x_draws=[noise(1)], Theta_z_draws=[noise(2)],
                        source=:manual)
end

@testset "IRF-matching target + CTW (CF-06)" begin
    H, n_s, nd = 3, 2, 40
    ce = _cf06_container(; H=H, n_s=n_s, nd=nd)

    @testset "stacking order + index" begin
        t = stacked_irf_target(ce)
        m = 2 * n_s * H
        @test length(t.theta_hat) == m
        @test length(t.index) == m
        # shock-major CMW order: [pi^s1; r^s1; pi^s2; r^s2], horizons inside
        expected_first = [(var=:pi, shock=1, h=1), (var=:pi, shock=1, h=2),
                          (var=:pi, shock=1, h=3), (var=:r, shock=1, h=1)]
        @test t.index[1:4] == expected_first
        @test t.index[7] == (var=:pi, shock=2, h=1)   # second shock block starts
        @test t.index[m] == (var=:r, shock=2, h=3)
        # every element equals its recognizable value
        for (i, e) in enumerate(t.index)
            v = e.var == :pi ? 1 : 2
            @test t.theta_hat[i] == 100.0 * v + 10.0 * e.shock + e.h
        end
        # variable-major variant
        tv = stacked_irf_target(ce; order=:variable_major)
        @test tv.index[1:3] == [(var=:pi, shock=1, h=1), (var=:pi, shock=1, h=2),
                                (var=:pi, shock=1, h=3)]
        @test tv.index[4] == (var=:pi, shock=2, h=1)
    end

    @testset "V_bar equals hand-stacked draw covariance" begin
        t = stacked_irf_target(ce)
        Dmat = Matrix{Float64}(undef, length(t.index), nd)
        for (i, e) in enumerate(t.index)
            src = e.var == :pi ? ce.Theta_x_draws[1] : ce.Theta_z_draws[1]
            Dmat[i, :] = src[e.h, e.shock, :]
        end
        @test t.V_bar ≈ cov(Dmat, dims=2) atol = 1e-12
    end

    @testset "scale, drop, inflate" begin
        t = @test_logs (:info, r"scale factors") match_mode = :any begin
            stacked_irf_target(ce; scale=Dict(:pi => 0.25))
        end
        @test t.theta_hat[1] == 0.25 * 111.0
        t0 = stacked_irf_target(ce)
        i_pi = findall(e -> e.var == :pi, t0.index)
        @test t.V_bar[i_pi[1], i_pi[1]] ≈ 0.0625 * t0.V_bar[i_pi[1], i_pi[1]] atol = 1e-12

        td = stacked_irf_target(ce; drop=[(:pi, 1, 2)])
        @test length(td.theta_hat) == 2 * n_s * H - 1
        @test all(e -> (e.var, e.shock, e.h) != (:pi, 1, 2), td.index)
        @test_throws ArgumentError stacked_irf_target(ce; drop=[(:pi, 9, 1)])

        ti = stacked_irf_target(ce; inflate=[(:r, 2, 1)])
        j = findfirst(e -> (e.var, e.shock, e.h) == (:r, 2, 1), ti.index)
        @test ti.V_bar[j, j] == 1e6
        @test all(ti.V_bar[j, setdiff(1:end, j)] .== 0.0)
        @test_throws ArgumentError stacked_irf_target(ce; inflate=[(:r, 2, 99)])

        @test_throws ArgumentError stacked_irf_target(ce; scale=Dict(:gdp => 4.0))
        # draws required
        ce0 = PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(3, 2)])
        @test_throws ArgumentError stacked_irf_target(ce0)
    end

    @testset "CTW kernel" begin
        m, bl = 6, 3
        # diagonally dominant so damping keeps PSD-ness: exact entries checkable
        V = 0.2 * ones(m, m) + 0.8 * Matrix{Float64}(I, m, m)
        r = ctw_covariance(V, bl; bandwidth=2, eta=1.0)
        @test r.repair == 0.0
        # diagonal preserved exactly
        @test all(diag(r.V) .≈ 1.0)
        # within-block horizon distance 1: kernel (1 - 1/2) = 0.5
        @test r.V[1, 2] ≈ 0.2 * 0.5 atol = 1e-12
        # d = bandwidth: zeroed
        @test r.V[1, 3] ≈ 0.0 atol = 1e-12
        # cross-block, same horizon (d = 0): untouched
        @test r.V[1, 4] ≈ 0.2 atol = 1e-12
        # cross-block horizon distance 1: damped identically
        @test r.V[1, 5] ≈ 0.2 * 0.5 atol = 1e-12

        # eta exponent honored
        r2 = ctw_covariance(V, bl; bandwidth=2, eta=2.0)
        @test r2.V[1, 2] ≈ 0.2 * 0.25 atol = 1e-12

        @test_throws ArgumentError ctw_covariance(ones(5, 5), 3)
        @test_throws ArgumentError ctw_covariance(ones(4, 5), 4)
    end

    @testset "PSD repair" begin
        # indefinite after damping: [1 2; 2 1] -> [1 1.75; 1.75 1], eigmin < 0
        V = [1.0 2.0; 2.0 1.0]
        r = ctw_covariance(V, 2; bandwidth=8, eta=1.0)
        @test r.repair > 0.5
        @test eigmin(Symmetric(r.V)) >= 0.0
        # PSD input needs no repair
        rr = ctw_covariance(Matrix{Float64}(I, 4, 4), 2)
        @test rr.repair == 0.0
    end

    @testset "precision_of" begin
        rng = MersenneTwister(7)
        A = randn(rng, 5, 5)
        V = A * A' + 5.0 * I
        p = MEM.precision_of(V)
        @test p.precision * V ≈ Matrix{Float64}(I, 5, 5) atol = 1e-8
        @test p.logdet ≈ log(det(V)) atol = 1e-8
    end
end
