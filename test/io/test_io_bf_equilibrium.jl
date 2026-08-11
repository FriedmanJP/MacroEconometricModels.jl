using Test, MacroEconometricModels, LinearAlgebra, SparseArrays

# ── helpers (hard-coded economies only; no RNG) ──────────────────────────────

"""One-sector roundabout economy: ω intermediate share, labour share 1−ω."""
function _roundabout_io(ω::Float64=0.4)
    # x=1, Z=ω, Y=1−ω, va=1−ω  (GDP = 1−ω, Domar λ = 1/(1−ω))
    Z = reshape([ω], 1, 1)
    Y = reshape([1 - ω], 1, 1)
    va = reshape([1 - ω], 1, 1)
    return IOData(Z, Y, va; sectors=["X"])
end

"""Closed-form dlogY for the one-sector CES roundabout economy (E=1, no dlogL).

Hand derivation
---------------
Cost: p = A⁻¹ [ω p^{1−θ} + (1−ω) w^{1−θ}]^{1/(1−θ)}  (θ≠1)
      log p = −log A + ω log p + (1−ω) log w          (θ=1)
Single factor, E=1 ⇒ w L = 1 with L = Λ̃ = 1 ⇒ w = 1.
From the CES fixed point:
  p^{1−θ} = (1−ω) / (A^{1−θ} − ω)
  p = [(1−ω)/(A^{1−θ} − ω)]^{1/(1−θ)}
  dlogY = −log p = −1/(1−θ) · log( (1−ω)/(A^{1−θ} − ω) )
Cobb-Douglas limit θ→1: dlogY = log A / (1−ω) = λ · dlogA.
"""
function _roundabout_closed(dlogA::Float64, θ::Float64, ω::Float64=0.4)
    A = exp(dlogA)
    if abs(θ - 1.0) < 1e-12
        return dlogA / (1 - ω)
    else
        return -(1 / (1 - θ)) * log((1 - ω) / (A^(1 - θ) - ω))
    end
end

@testset "bf_equilibrium oracles" begin
    io = load_example(:wiot)

    # ── Oracle 1: Cobb-Douglas exactness for large shocks ────────────────────
    @testset "Oracle 1: Cobb-Douglas exactness" begin
        for nests in (:single, :two), factors in (:single, :va_cats)
            net = production_network(io; nests=nests, factors=factors,
                                     theta=1.0, sigma=1.0, epsilon=1.0, eta=1.0)
            λ_out = [net.lambda[g] for g in net.outer_nodes]
            Λ = net.lambda[net.M+2:end]
            for dA in ([0.5, -0.5], [0.5, 0.5], [-0.5, -0.3], [0.0, 0.0])
                eq = bf_equilibrium(net; dlogA=dA)
                @test eq.converged
                hult = dot(λ_out, dA)
                @test eq.dlogY ≈ hult atol=1e-10
                @test eq.hulten ≈ hult atol=1e-14
            end
            # factor-supply shock only
            dL = fill(0.2, net.F)
            eqL = bf_equilibrium(net; dlogL=dL)
            @test eqL.converged
            @test eqL.dlogY ≈ dot(Λ, dL) atol=1e-10
        end
    end

    # ── Oracle 2: Hulten gradient via central FD ─────────────────────────────
    @testset "Oracle 2: Hulten gradient (central FD)" begin
        net = production_network(io; theta=0.5, sigma=0.9, factors=:va_cats)
        λ_out = [net.lambda[g] for g in net.outer_nodes]
        ε = 1e-5
        g = zeros(net.n)
        for i in 1:net.n
            dAp = zeros(net.n); dAp[i] = ε
            dAm = zeros(net.n); dAm[i] = -ε
            yp = bf_equilibrium(net; dlogA=dAp).dlogY
            ym = bf_equilibrium(net; dlogA=dAm).dlogY
            g[i] = (yp - ym) / (2ε)
        end
        @test g ≈ λ_out atol=1e-6

        # multi-factor labour gradient
        Λ = net.lambda[net.M+2:end]
        gL = zeros(net.F)
        for f in 1:net.F
            dLp = zeros(net.F); dLp[f] = ε
            dLm = zeros(net.F); dLm[f] = -ε
            yp = bf_equilibrium(net; dlogL=dLp).dlogY
            ym = bf_equilibrium(net; dlogL=dLm).dlogY
            gL[f] = (yp - ym) / (2ε)
        end
        @test gL ≈ Λ atol=1e-6
    end

    # ── Oracle 4: closed-form roundabout economy ──────────────────────────────
    @testset "Oracle 4: closed-form roundabout" begin
        ω = 0.4
        io1 = _roundabout_io(ω)
        for θ in (0.5, 1.0, 2.0)
            net = production_network(io1; theta=θ, sigma=1.0)
            @test net.n == 1 && net.F == 1
            @test net.lambda[net.outer_nodes[1]] ≈ 1 / (1 - ω) atol=1e-12
            for dA in (-0.3, -0.1, 0.0, 0.1, 0.3, 0.5)
                eq = bf_equilibrium(net; dlogA=[dA])
                @test eq.converged
                closed = _roundabout_closed(dA, θ, ω)
                @test eq.dlogY ≈ closed atol=1e-10
            end
        end
        # θ-continuity at 1 ± 1e-6
        net_lo = production_network(io1; theta=1 - 1e-6)
        net_hi = production_network(io1; theta=1 + 1e-6)
        net_cd = production_network(io1; theta=1.0)
        dA = [0.2]
        y_lo = bf_equilibrium(net_lo; dlogA=dA).dlogY
        y_hi = bf_equilibrium(net_hi; dlogA=dA).dlogY
        y_cd = bf_equilibrium(net_cd; dlogA=dA).dlogY
        @test y_lo ≈ y_cd atol=1e-5
        @test y_hi ≈ y_cd atol=1e-5
    end

    # ── Structural invariants along the solve ────────────────────────────────
    @testset "structural invariants" begin
        net = production_network(io; theta=0.3, sigma=0.8, factors=:va_cats,
                                 nests=:two)
        eq = bf_equilibrium(net; dlogA=[0.1, -0.05], dlogL=[0.0, 0.05])
        @test eq.converged
        Ω = Matrix(eq.Omega)
        M, F = net.M, net.F
        N = 1 + M + F
        @test all(abs.(sum(Ω[1:1+M, :]; dims=2) .- 1) .< 1e-8)
        @test eq.lambda[1] ≈ 1 atol=1e-8
        @test sum(eq.Lambda) ≈ 1 atol=1e-8
        # prices positive
        @test all(eq.p .> 0) && all(eq.w .> 0)
    end

    # ── method=:fixedpoint fallback path ─────────────────────────────────────
    @testset "fixedpoint method" begin
        net = production_network(io; theta=0.5)
        eq = bf_equilibrium(net; dlogA=[0.1, 0.0], method=:fixedpoint)
        @test eq.converged
        eq_n = bf_equilibrium(net; dlogA=[0.1, 0.0], method=:newton)
        @test eq.dlogY ≈ eq_n.dlogY atol=1e-8
    end

    # ── zero shocks → zero change ────────────────────────────────────────────
    @testset "zero shock identity" begin
        net = production_network(io; theta=2.0, sigma=0.5)
        eq = bf_equilibrium(net)
        @test eq.converged
        @test abs(eq.dlogY) < 1e-12
        @test all(abs.(eq.dlog_p) .< 1e-10)
        @test all(abs.(log.(eq.p)) .< 1e-10)
    end

    # ── show / report / plot smoke ───────────────────────────────────────────
    @testset "show / report / plot" begin
        net = production_network(io)
        eq = bf_equilibrium(net; dlogA=[0.05, 0.0])
        s = sprint(show, eq)
        @test occursin("BFEquilibrium", s)
        mktemp() do _, ioh
            redirect_stdout(ioh) do
                report(eq)
            end
        end
        p = plot_result(eq)
        @test p isa PlotOutput
        p2 = plot_result(net)
        @test p2 isa PlotOutput
    end
end

# ── Backward-compat: legacy baqaee_farhi still works (untouched path) ────────
@testset "legacy baqaee_farhi regression pin" begin
    io = load_example(:wiot)
    bf = baqaee_farhi(io; theta=2.0, sigma=0.9)
    # Frozen numbers from the pre-IO2 scalar Hessian (absolute levels).
    @test bf.first_order ≈ [0.4878048780487805, 0.975609756097561] atol=1e-12
    @test bf.second_order ≈ bf.second_order' atol=1e-12   # symmetry
    # Non-zero under θ≠1; sign of diagonal ≥ 0 (variance weights)
    @test all(diag(bf.second_order) .>= -1e-10)
    # Cobb-Douglas still exact zero
    bf_cd = baqaee_farhi(io)
    @test all(abs.(bf_cd.second_order) .<= 1e-10)
end
