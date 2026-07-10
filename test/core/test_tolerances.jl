# T148 / #247 — central tolerance constants + perfect_foresight abstol exposure
@testset "T148: central tolerance constants (#247)" begin
    # --- Constants: Float64 pinned to the historical 1e-8; others eps-derived ---
    @test default_abstol(Float64) == 1e-8
    @test default_abstol() == 1e-8
    @test default_abstol(1.0) == 1e-8                       # value dispatch
    @test default_abstol(Float32) == sqrt(eps(Float32))
    @test default_abstol(1.0f0) == sqrt(eps(Float32))
    @test default_abstol(Float16) == sqrt(eps(Float16))     # eps-genericity, no eps(Float64) hardcode
    @test default_abstol(Float32) > default_abstol(Float64) # lower precision loosens

    @test default_reltol(Float64) == sqrt(eps(Float64))
    @test default_reltol() == sqrt(eps(Float64))
    @test default_reltol(Float32) == sqrt(eps(Float32))
    @test default_reltol(Float32) > default_reltol(Float64)

    # --- perfect_foresight exposes an abstol gate wired to the solver ---
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 50
    shocks = zeros(T_periods, 1); shocks[1, 1] = 1.0

    pf_default  = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks)
    pf_explicit = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks, abstol=1e-8)
    pf_loose    = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks, abstol=1e-1)
    @test pf_default.converged
    @test pf_loose.converged
    # Wiring the default through default_abstol(Float64) is byte-identical to the literal 1e-8.
    @test pf_default.path == pf_explicit.path

    # A sub-roundoff abstol the linear solve can never satisfy proves the kwarg genuinely
    # reaches the solver's convergence test (it would converge if abstol were ignored).
    pf_tight = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks,
                     abstol=1e-30, max_iter=3)
    @test !pf_tight.converged
end
