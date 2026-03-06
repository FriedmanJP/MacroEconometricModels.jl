# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Coverage tests for DSGE module
# Targets: display.jl, occbin.jl, pruning.jl, blanchard_kahn.jl, pfi.jl,
#          constraints.jl, gensys.jl, perturbation.jl, projection.jl

Random.seed!(9005)

const _suppress = MacroEconometricModels._suppress_warnings

@testset "DSGE Coverage" begin

# =========================================================================
# 1. display.jl -- LaTeX and HTML backends for DSGESpec
# =========================================================================
@testset "display.jl: LaTeX/HTML backends for DSGESpec" begin
    spec = @dsge begin
        parameters: alpha = 0.33, beta = 0.99, rho = 0.9
        endogenous: y, k
        exogenous: e
        y[t] = alpha * k[t-1] + e[t]
        k[t] = beta * y[t] + rho * k[t-1]
    end
    spec = compute_steady_state(spec)

    # -- Text backend (baseline) --
    MacroEconometricModels.set_display_backend(:text)
    s_text = sprint(show, spec)
    @test occursin("DSGE Model Specification", s_text)
    @test occursin("Endogenous", s_text)
    @test occursin("Steady State", s_text)

    # -- LaTeX backend --
    MacroEconometricModels.set_display_backend(:latex)
    s_latex = sprint(show, spec)
    @test occursin("\\begin{align}", s_latex)
    @test occursin("\\end{align}", s_latex)
    @test occursin("\\begin{tabular}", s_latex)
    @test occursin("\\hline", s_latex)
    # Check that Greek letters map correctly
    @test occursin("\\alpha", s_latex) || occursin("alpha", s_latex)
    @test occursin("\\beta", s_latex) || occursin("beta", s_latex)
    # Steady state in LaTeX
    @test occursin("\\bar{", s_latex)

    # -- HTML backend --
    MacroEconometricModels.set_display_backend(:html)
    s_html = sprint(show, spec)
    @test occursin("<div class=\"dsge-spec\">", s_html)
    @test occursin("<h3>", s_html)
    @test occursin("<table>", s_html)
    @test occursin("</table>", s_html)
    @test occursin("MathJax", s_html) || occursin("\\begin{align}", s_html)
    @test occursin("Steady State", s_html)
    @test occursin("</div>", s_html)

    # Restore text backend
    MacroEconometricModels.set_display_backend(:text)
end

@testset "display.jl: LaTeX/HTML for DSGESpec without steady state" begin
    spec_no_ss = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    # No steady state computed -- should not show steady state section

    MacroEconometricModels.set_display_backend(:latex)
    s = sprint(show, spec_no_ss)
    @test occursin("\\begin{align}", s)
    @test !occursin("\\bar{", s)  # no steady state

    MacroEconometricModels.set_display_backend(:html)
    s2 = sprint(show, spec_no_ss)
    @test occursin("<div class=\"dsge-spec\">", s2)
    @test !occursin("Steady State", s2)

    MacroEconometricModels.set_display_backend(:text)
end

@testset "display.jl: LaTeX/HTML for forward-looking model" begin
    spec_fwd = @dsge begin
        parameters: beta = 0.99, rho = 0.9
        endogenous: c, k
        exogenous: e
        c[t] = beta * E[t](c[t+1]) + e[t]
        k[t] = rho * k[t-1] + c[t]
    end
    spec_fwd = compute_steady_state(spec_fwd)

    # LaTeX with forward-looking: should have \mathbb{E}
    MacroEconometricModels.set_display_backend(:latex)
    s = sprint(show, spec_fwd)
    @test occursin("\\mathbb{E}", s) || occursin("E_t", s)

    # HTML with forward-looking
    MacroEconometricModels.set_display_backend(:html)
    s2 = sprint(show, spec_fwd)
    @test occursin("<div", s2)

    MacroEconometricModels.set_display_backend(:text)
end

@testset "display.jl: expression converters edge cases" begin
    # Test _expr_to_latex with division, power, unary minus
    endog = [:y, :k]
    exog = [:e]
    params = [:alpha]

    # Division -> \frac
    div_expr = :(y[t] / k[t-1])
    s = MacroEconometricModels._expr_to_latex(div_expr, endog, exog, params)
    @test occursin("\\frac", s)

    # Power -> ^{}
    pow_expr = :(k[t-1] ^ alpha)
    s2 = MacroEconometricModels._expr_to_latex(pow_expr, endog, exog, params)
    @test occursin("^{", s2)

    # Generic function: log()
    log_expr = :(log(y[t]))
    s3 = MacroEconometricModels._expr_to_latex(log_expr, endog, exog, params)
    @test occursin("\\log", s3)

    # Text division
    s4 = MacroEconometricModels._expr_to_text(div_expr, endog, exog, params)
    @test occursin("/", s4)

    # Unary minus
    neg_expr = :(-(y[t]))
    s5 = MacroEconometricModels._expr_to_text(neg_expr, endog, exog, params)
    @test startswith(s5, "-")

    s6 = MacroEconometricModels._expr_to_latex(neg_expr, endog, exog, params)
    @test startswith(s6, "-")
end

# =========================================================================
# 2. constraints.jl -- VariableBound and NonlinearConstraint
# =========================================================================
@testset "constraints.jl: VariableBound construction and edge cases" begin
    # Lower bound only
    vb1 = variable_bound(:i, lower=0.0)
    @test vb1.var_name == :i
    @test vb1.lower == 0.0
    @test vb1.upper === nothing

    # Upper bound only
    vb2 = variable_bound(:h, upper=1.0)
    @test vb2.var_name == :h
    @test vb2.lower === nothing
    @test vb2.upper == 1.0

    # Both bounds
    vb3 = variable_bound(:h, lower=0.0, upper=1.0)
    @test vb3.lower == 0.0
    @test vb3.upper == 1.0

    # Error: no bounds
    @test_throws ArgumentError variable_bound(:x)

    # Error: lower > upper
    @test_throws ArgumentError variable_bound(:x, lower=2.0, upper=1.0)

    # sprint should work (even if no custom show, uses default)
    s = sprint(show, vb1)
    @test !isempty(s)
end

@testset "constraints.jl: NonlinearConstraint" begin
    nc = nonlinear_constraint((y, yl, yle, e, theta) -> y[1] - 0.8 * y[2]; label="collateral")
    @test nc.label == "collateral"
    @test nc.fn([1.0, 1.0], [0.0], [0.0], [0.0], Dict()) ≈ 0.2

    # Default label
    nc2 = nonlinear_constraint((y, yl, yle, e, theta) -> y[1])
    @test nc2.label == "constraint"

    # sprint should work
    s = sprint(show, nc)
    @test !isempty(s)
end

@testset "constraints.jl: _validate_constraints" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, k
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        k[t] = y[t]
    end

    # Valid variable bound
    vb = variable_bound(:y, lower=0.0)
    MacroEconometricModels._validate_constraints(spec, [vb])

    # Invalid variable name
    vb_bad = variable_bound(:z, lower=0.0)
    @test_throws ArgumentError MacroEconometricModels._validate_constraints(spec, [vb_bad])

    # Valid nonlinear constraint
    nc = nonlinear_constraint((y, yl, yle, e, theta) -> y[1] - 0.5; label="test")
    MacroEconometricModels._validate_constraints(spec, [nc])
end

@testset "constraints.jl: _select_solver" begin
    # With override
    @test MacroEconometricModels._select_solver([], :ipopt) == :ipopt
    @test MacroEconometricModels._select_solver([], :path) == :path

    # Without override, no NonlinearConstraint
    vb = variable_bound(:y, lower=0.0)
    # Will be :path if PATHSolver loaded, :ipopt otherwise
    result = MacroEconometricModels._select_solver([vb], nothing)
    @test result in (:path, :ipopt)

    # With NonlinearConstraint -> always :ipopt
    nc = nonlinear_constraint((y, yl, yle, e, theta) -> y[1]; label="test")
    @test MacroEconometricModels._select_solver([nc], nothing) == :ipopt
end

@testset "constraints.jl: _extract_bounds" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, k
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        k[t] = y[t]
    end

    vb1 = variable_bound(:y, lower=0.0)
    vb2 = variable_bound(:k, lower=-1.0, upper=2.0)
    lower, upper = MacroEconometricModels._extract_bounds(spec, [vb1, vb2])
    @test lower[1] == 0.0
    @test upper[1] == Inf
    @test lower[2] == -1.0
    @test upper[2] == 2.0
end

# =========================================================================
# 3. gensys.jl -- sunspot / non-existence paths
# =========================================================================
@testset "gensys.jl: no forward-looking variables" begin
    # AR(1): all stable eigenvalues, no Pi columns -> eu=[1,1]
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test sol.eu == [1, 1]
end

@testset "gensys.jl: indeterminate equilibrium (sunspot)" begin
    # Model with more unstable eigenvalues than forward-looking variables
    # can produce eu[2] != 1
    # Use a model: x_t = 0.5 * E_t[x_{t+1}] + e_t
    # This has 1 forward-looking, 1 eigenvalue with |lambda| > 1 => unique
    # But with 2 forward-looking vars and 1 unstable eigenvalue => indeterminate
    _suppress() do
        # Manually construct a gensys call with mismatched dimensions
        n = 2
        Gamma0 = Matrix{Float64}(I, n, n)
        Gamma1 = 0.5 * Matrix{Float64}(I, n, n)
        C_vec = zeros(n)
        Psi = Matrix{Float64}(I, n, 1)
        Pi = Matrix{Float64}(I, n, 2)  # 2 forward-looking but only 0 unstable eigenvalues
        result = gensys(Gamma0, Gamma1, C_vec, Psi, Pi)
        # With all stable eigenvalues but 2 expectation errors: should be [1,1]
        # (over-determined forward block)
        @test length(result.eu) == 2
        @test length(result.eigenvalues) == n
    end
end

@testset "gensys.jl: explosive model (no stable solution)" begin
    _suppress() do
        # All eigenvalues unstable, with forward-looking variables
        n = 2
        # Gamma0 * y_t = Gamma1 * y_{t-1} + ...
        # eigenvalues = |T/S| -> if Gamma1 has large entries relative to Gamma0
        Gamma0 = [1.0 0.0; 0.0 1.0]
        Gamma1 = [3.0 0.0; 0.0 3.0]  # eigenvalues |lambda| = 3 > 1
        C_vec = zeros(2)
        Psi = [1.0; 0.0][:, :]
        Pi = zeros(2, 0)  # no forward-looking
        result = gensys(Gamma0, Gamma1, C_vec, Psi, Pi)
        # All unstable, no forward-looking: may still produce a solution
        @test length(result.eigenvalues) == 2
        # nstab should be 0
        @test result.eu[1] == 1 || result.eu[1] == 0  # depends on rank condition
    end
end

# =========================================================================
# 4. blanchard_kahn.jl -- error paths
# =========================================================================
@testset "blanchard_kahn.jl: determined equilibrium" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:blanchard_kahn)
    @test sol.eu == [1, 1]
    @test sol.method == :blanchard_kahn
end

@testset "blanchard_kahn.jl: indeterminate (more unstable than forward)" begin
    _suppress() do
        # Construct a case where n_unstable > n_forward => eu = [0,0]
        spec = DSGESpec{Float64}(
            [:y1, :y2], [:e], [:rho],
            Dict(:rho => 0.5),
            [:(y1[t]), :(y2[t])],
            [(y, yl, yle, eps, th) -> y[1] - 2.0 * yl[1] - eps[1],
             (y, yl, yle, eps, th) -> y[2] - 3.0 * yl[2]],
            0, Int[], Float64[]
        )
        spec = compute_steady_state(spec)
        ld = linearize(spec)
        # Manually fix linearized system to have specific eigenvalue structure
        # ld has Gamma0=I, Gamma1=diag(2,3), Pi=empty
        # eigenvalues |lambda| = 2, 3 => both unstable
        # n_forward = size(Pi,2) = 0 => n_unstable(2) > n_fwd(0) => eu=[0,0]
        sol = blanchard_kahn(ld, spec)
        @test sol.eu[1] == 0  # explosive -- no stable solution
    end
end

@testset "blanchard_kahn.jl: indeterminate (fewer unstable than forward)" begin
    _suppress() do
        # Use a model with forward-looking terms so Pi has columns
        spec_fwd = @dsge begin
            parameters: rho = 0.5
            endogenous: y1, y2
            exogenous: e
            y1[t] = rho * y1[t-1] + 0.01 * E[t](y1[t+1]) + 0.01 * E[t](y2[t+1]) + e[t]
            y2[t] = rho * y2[t-1] + 0.01 * E[t](y1[t+1]) + 0.01 * E[t](y2[t+1])
        end
        spec_fwd = compute_steady_state(spec_fwd)
        ld = linearize(spec_fwd)
        # eigenvalues are close to 0.5 => both stable => 0 unstable
        # n_forward = 2 (both variables appear with E[t]) => n_unstable(0) < n_fwd(2) => eu=[1,0]
        sol = blanchard_kahn(ld, spec_fwd)
        @test sol.eu == [1, 0]
    end
end

# =========================================================================
# 5. perturbation.jl -- order=1 path
# =========================================================================
@testset "perturbation.jl: order=1" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=1)
    @test sol isa PerturbationSolution{Float64}
    @test sol.order == 1
    @test sol.gxx === nothing
    @test sol.hxx === nothing
    @test sol.gσσ === nothing
    @test sol.hσσ === nothing

    # First-order IRF should work
    irf_result = irf(sol, 10)
    @test size(irf_result.values) == (10, 1, 1)
end

@testset "perturbation.jl: order=2" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=2)
    @test sol.order == 2
    @test sol.gxx !== nothing || sol.hxx !== nothing
end

@testset "perturbation.jl: order=3 works" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=3)
    @test sol.order == 3
end

@testset "perturbation.jl: multi-variable 2nd order" begin
    spec = @dsge begin
        parameters: rho = 0.9, beta = 0.99
        endogenous: c, k
        exogenous: e
        c[t] = beta * E[t](c[t+1]) + e[t]
        k[t] = rho * k[t-1] + c[t]
    end
    spec = compute_steady_state(spec)
    _suppress() do
        sol = perturbation_solver(spec; order=2)
        @test sol.order == 2
        @test size(sol.hx, 1) > 0  # has state rows
    end
end

# =========================================================================
# 6. pruning.jl -- simulate with GIRF and analytical_moments
# =========================================================================
@testset "pruning.jl: simulate order=1 perturbation" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=1)

    sim = simulate(sol, 100; rng=Random.MersenneTwister(42))
    @test size(sim) == (100, 1)
    @test all(isfinite, sim)
end

@testset "pruning.jl: simulate order=2 pruned" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=2)

    sim = simulate(sol, 100; rng=Random.MersenneTwister(42))
    @test size(sim) == (100, 1)
    @test all(isfinite, sim)

    # With custom shock draws
    shocks = randn(50, 1)
    sim2 = simulate(sol, 50; shock_draws=shocks)
    @test size(sim2) == (50, 1)

    # Antithetic variates
    sim3 = simulate(sol, 100; antithetic=true, rng=Random.MersenneTwister(42))
    @test size(sim3) == (100, 1)
end

@testset "pruning.jl: GIRF path" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=2)

    # GIRF via irf function
    irf_girf = irf(sol, 10; irf_type=:girf, n_draws=20)
    @test size(irf_girf.values) == (10, 1, 1)
    @test all(isfinite, irf_girf.values)
end

@testset "pruning.jl: analytical_moments covariance format" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    # Order 1: Lyapunov approach
    sol1 = perturbation_solver(spec; order=1)
    m1 = analytical_moments(sol1; lags=2)
    @test length(m1) > 0
    @test all(isfinite, m1)

    # Order 2: simulation-based moments
    sol2 = perturbation_solver(spec; order=2)
    m2 = analytical_moments(sol2; lags=1)
    @test length(m2) > 0
    @test all(isfinite, m2)
end

@testset "pruning.jl: analytical_moments GMM format" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    # Order 1: GMM format (means are zero)
    sol1 = perturbation_solver(spec; order=1)
    m1 = analytical_moments(sol1; format=:gmm, lags=3)
    @test length(m1) > 0
    @test all(isfinite, m1)

    # Order 2: GMM format with closed-form augmented Lyapunov
    sol2 = perturbation_solver(spec; order=2)
    m2 = analytical_moments(sol2; format=:gmm, lags=3)
    @test length(m2) > 0
    @test all(isfinite, m2)
end

@testset "pruning.jl: FEVD for PerturbationSolution" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=2)

    f = fevd(sol, 10)
    @test size(f.proportions, 1) == 1  # 1 variable
    @test size(f.proportions, 2) == 1  # 1 shock
    @test size(f.proportions, 3) == 10 # 10 horizons
end

@testset "pruning.jl: invalid irf_type" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=2)
    @test_throws ArgumentError irf(sol, 10; irf_type=:invalid)
end

@testset "pruning.jl: invalid format for analytical_moments" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = perturbation_solver(spec; order=1)
    @test_throws ArgumentError analytical_moments(sol; format=:invalid)
end

# =========================================================================
# 7. projection.jl -- grid types and quadrature
# =========================================================================
@testset "projection.jl: tensor grid collocation" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.collocation_solver(spec; degree=3, grid=:tensor,
                                  quadrature=:gauss_hermite, n_quad=3,
                                  max_iter=5, tol=1e-4)
        @test sol isa ProjectionSolution{Float64}
        @test sol.grid_type == :tensor
        @test sol.quadrature == :gauss_hermite
        @test sol.method == :projection
    end
end

@testset "projection.jl: monomial quadrature" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.collocation_solver(spec; degree=3, grid=:tensor,
                                  quadrature=:monomial,
                                  max_iter=5, tol=1e-4)
        @test sol.quadrature == :monomial
    end
end

@testset "projection.jl: evaluate_policy" begin
    local sol_eval  # declared outside _suppress so it's accessible later

    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol_eval = MacroEconometricModels.collocation_solver(spec; degree=3, max_iter=5, tol=1e-4)

        # Single point evaluation
        y_val = evaluate_policy(sol_eval, [0.0])
        @test length(y_val) == 1
        @test all(isfinite, y_val)

        # Multi-point evaluation
        X = reshape([0.0, 0.1, -0.1], 3, 1)
        Y = evaluate_policy(sol_eval, X)
        @test size(Y) == (3, 1)
    end

    # Out-of-bounds extrapolation (test outside _suppress so @test_warn can see the warning)
    big_state = [sol_eval.state_bounds[1, 2] * 2.0]
    @test_warn r"extrapolating" evaluate_policy(sol_eval, big_state)
end

@testset "projection.jl: max_euler_error" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.collocation_solver(spec; degree=3, max_iter=5, tol=1e-4)
        err = max_euler_error(sol; n_test=20, rng=Random.MersenneTwister(42))
        @test isfinite(err)
        @test err >= 0
    end
end

@testset "projection.jl: invalid grid/quadrature" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    @test_throws ArgumentError MacroEconometricModels.collocation_solver(spec; grid=:bad_grid)
    @test_throws ArgumentError MacroEconometricModels.collocation_solver(spec; quadrature=:bad_quad)
end

# =========================================================================
# 8. pfi.jl -- policy function iteration
# =========================================================================
@testset "pfi.jl: basic PFI" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.pfi_solver(spec; degree=3, max_iter=5, tol=1e-4)
        @test sol isa ProjectionSolution{Float64}
        @test sol.method == :pfi
    end
end

@testset "pfi.jl: PFI with damping" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.pfi_solver(spec; degree=3, max_iter=5, tol=1e-4, damping=0.5)
        @test sol isa ProjectionSolution{Float64}
        @test sol.method == :pfi
    end
end

@testset "pfi.jl: PFI with different quadrature" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.pfi_solver(spec; degree=3, max_iter=5, tol=1e-4,
                          quadrature=:monomial)
        @test sol.quadrature == :monomial
    end
end

@testset "pfi.jl: PFI max_iter convergence (non-converged)" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)

        # Very few iterations + tight tolerance -> won't converge
        sol = MacroEconometricModels.pfi_solver(spec; degree=3, max_iter=1, tol=1e-20)
        # May or may not converge in 1 iteration depending on initial guess quality
        @test sol isa ProjectionSolution{Float64}
    end
end

# =========================================================================
# 9. occbin.jl -- OccBin IRF and two-constraint paths
# =========================================================================
@testset "occbin.jl: occbin_irf one-constraint" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9, phi = 1.5
            endogenous: y, i
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = phi * y[t]
        end
        spec = compute_steady_state(spec)
        constraint = parse_constraint(:(i[t] >= 0), spec)

        oirf = occbin_irf(spec, constraint, 1, 30; magnitude=-2.0)
        @test oirf isa OccBinIRF{Float64}
        @test size(oirf.linear) == (30, 2)
        @test size(oirf.piecewise) == (30, 2)
        @test oirf.shock_name == "e"

        # Show method
        s = sprint(show, oirf)
        @test occursin("OccBin IRF", s)
    end
end

@testset "occbin.jl: occbin_irf invalid shock_idx" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y, i
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = y[t]
        end
        spec = compute_steady_state(spec)
        constraint = parse_constraint(:(i[t] >= 0), spec)
        @test_throws ArgumentError occbin_irf(spec, constraint, 5, 20)
    end
end

@testset "occbin.jl: two-constraint solver" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9, phi = 1.5
            endogenous: y, i, cap
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = phi * y[t]
            cap[t] = y[t]
        end
        spec = compute_steady_state(spec)

        c1 = parse_constraint(:(i[t] >= 0), spec)
        c2 = parse_constraint(:(cap[t] <= 0.5), spec)

        shock_path = zeros(30, 1)
        shock_path[1, 1] = -2.0  # negative shock to trigger ZLB

        sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=30)
        @test sol isa OccBinSolution{Float64}
        @test size(sol.regime_history, 2) == 2
        @test size(sol.piecewise_path, 1) == 30
    end
end

@testset "occbin.jl: two-constraint IRF" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9, phi = 1.5
            endogenous: y, i, cap
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = phi * y[t]
            cap[t] = y[t]
        end
        spec = compute_steady_state(spec)

        c1 = parse_constraint(:(i[t] >= 0), spec)
        c2 = parse_constraint(:(cap[t] <= 0.5), spec)

        oirf = occbin_irf(spec, c1, c2, 1, 20; magnitude=-2.0)
        @test oirf isa OccBinIRF{Float64}
        @test size(oirf.regime_history, 2) == 2
    end
end

@testset "occbin.jl: constraint parsing edge cases" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = y[t]
    end

    # LEQ constraint
    c = parse_constraint(:(y[t] <= 1.0), spec)
    @test c.direction == :leq
    @test c.bound == 1.0

    # Invalid operator
    @test_throws ArgumentError parse_constraint(:(y[t] == 0), spec)

    # Invalid LHS (not var[t])
    @test_throws ArgumentError parse_constraint(:(y >= 0), spec)

    # Invalid bound (symbol not number)
    @test_throws ArgumentError parse_constraint(:(y[t] >= x), spec)
end

@testset "occbin.jl: two-constraint with curb_retrench" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9, phi = 1.5
            endogenous: y, i, cap
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = phi * y[t]
            cap[t] = y[t]
        end
        spec = compute_steady_state(spec)

        c1 = parse_constraint(:(i[t] >= 0), spec)
        c2 = parse_constraint(:(cap[t] <= 0.5), spec)

        shock_path = zeros(30, 1)
        shock_path[1, 1] = -2.0

        sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=30,
                           curb_retrench=true)
        @test sol isa OccBinSolution{Float64}
    end
end

# =========================================================================
# 10. Display backends for solution types
# =========================================================================
@testset "display: DSGESolution, PerturbationSolution, ProjectionSolution in LaTeX/HTML" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    # DSGESolution
    sol_gensys = solve(spec; method=:gensys)
    MacroEconometricModels.set_display_backend(:text)
    s1 = sprint(show, sol_gensys)
    @test occursin("DSGE Solution", s1)

    # PerturbationSolution
    sol_pert = perturbation_solver(spec; order=2)
    s2 = sprint(show, sol_pert)
    @test occursin("Perturbation Solution", s2)

    # ProjectionSolution
    _suppress() do
        sol_proj = MacroEconometricModels.collocation_solver(spec; degree=3, max_iter=5, tol=1e-4)
        s3 = sprint(show, sol_proj)
        @test occursin("Projection Solution", s3)
    end
end

# =========================================================================
# 11. solve() dispatcher method coverage
# =========================================================================
@testset "solve() dispatcher: various methods" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    # :klein method
    sol_k = solve(spec; method=:klein)
    @test sol_k isa DSGESolution{Float64}
    @test sol_k.method == :klein

    # :perturbation method via solve()
    sol_p = solve(spec; method=:perturbation, order=1)
    @test sol_p isa PerturbationSolution{Float64}

    # :projection method via solve()
    _suppress() do
        sol_c = solve(spec; method=:projection, degree=3, max_iter=5, tol=1e-4)
        @test sol_c isa ProjectionSolution{Float64}
    end

    # :pfi method via solve()
    _suppress() do
        sol_pfi = solve(spec; method=:pfi, degree=3, max_iter=5, tol=1e-4)
        @test sol_pfi isa ProjectionSolution{Float64}
    end

    # Invalid method
    @test_throws ArgumentError solve(spec; method=:invalid)
end

@testset "solve() dispatcher: auto steady state" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    # Do NOT compute steady state -- solve() should do it automatically
    sol = solve(spec; method=:gensys)
    @test sol isa DSGESolution{Float64}
end

# =========================================================================
# 12. Smolyak grid path
# =========================================================================
@testset "projection.jl: smolyak grid" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9, beta = 0.5
            endogenous: c, k
            exogenous: e
            c[t] = beta * E[t](c[t+1]) + e[t]
            k[t] = rho * k[t-1] + c[t]
        end
        spec = compute_steady_state(spec)

        # Force smolyak grid
        sol = MacroEconometricModels.collocation_solver(spec; degree=3, grid=:smolyak, smolyak_mu=2,
                                  max_iter=5, tol=1e-3)
        @test sol.grid_type == :smolyak
    end
end

@testset "pfi.jl: smolyak grid" begin
    _suppress() do
        spec = @dsge begin
            parameters: rho = 0.9, beta = 0.5
            endogenous: c, k
            exogenous: e
            c[t] = beta * E[t](c[t+1]) + e[t]
            k[t] = rho * k[t-1] + c[t]
        end
        spec = compute_steady_state(spec)

        sol = MacroEconometricModels.pfi_solver(spec; degree=3, grid=:smolyak, smolyak_mu=2,
                          max_iter=5, tol=1e-3)
        @test sol.grid_type == :smolyak
        @test sol.method == :pfi
    end
end

# =========================================================================
# 13. Multi-variable pruned simulation
# =========================================================================
@testset "pruning.jl: multi-variable 2nd order simulation" begin
    spec = @dsge begin
        parameters: rho = 0.9, beta = 0.5
        endogenous: c, k
        exogenous: e
        c[t] = beta * E[t](c[t+1]) + e[t]
        k[t] = rho * k[t-1] + c[t]
    end
    spec = compute_steady_state(spec)

    _suppress() do
        sol = perturbation_solver(spec; order=2)
        sim = simulate(sol, 25; rng=Random.MersenneTwister(42))
        @test size(sim, 1) == 25
        @test size(sim, 2) >= 2  # at least 2 variables (may be augmented)
        @test all(isfinite, sim)
    end
end

@testset "pruning.jl: multi-variable GIRF" begin
    spec = @dsge begin
        parameters: rho = 0.9, beta = 0.5
        endogenous: c, k
        exogenous: e
        c[t] = beta * E[t](c[t+1]) + e[t]
        k[t] = rho * k[t-1] + c[t]
    end
    spec = compute_steady_state(spec)

    _suppress() do
        sol = perturbation_solver(spec; order=2)
        irf_result = irf(sol, 10; irf_type=:girf, n_draws=10)
        @test size(irf_result.values, 1) == 10
        @test all(isfinite, irf_result.values)
    end
end

# =========================================================================
# 14. Additional edge cases
# =========================================================================
@testset "display.jl: _format_num_display" begin
    @test MacroEconometricModels._format_num_display(3) == "3"
    @test MacroEconometricModels._format_num_display(3.0) == "3"
    @test MacroEconometricModels._format_num_display(3.14) == "3.14"
    @test MacroEconometricModels._format_num_display(Inf) == "Inf"
end

@testset "display.jl: _sym_to_latex" begin
    @test MacroEconometricModels._sym_to_latex(:alpha) == "alpha"
    @test MacroEconometricModels._sym_to_latex(:α) == "\\alpha"
    @test MacroEconometricModels._sym_to_latex(:φ_π) == "\\phi_{\\pi}"
end

@testset "display.jl: _time_offset" begin
    @test MacroEconometricModels._time_offset(:t) == 0
    @test MacroEconometricModels._time_offset(:(t + 1)) == 1
    @test MacroEconometricModels._time_offset(:(t - 1)) == -1
    @test MacroEconometricModels._time_offset(:(t - 2)) == -2
    @test MacroEconometricModels._time_offset(:x) == 0  # non-time symbol
end

@testset "display.jl: _equation_to_display" begin
    endog = [:y]
    exog = [:e]
    params = [:rho]

    # Binary subtraction -> LHS = RHS
    eq = :(y[t] - rho * y[t-1])
    s = MacroEconometricModels._equation_to_display(eq, endog, exog, params; mode=:text)
    @test occursin("=", s)

    # LaTeX mode
    s2 = MacroEconometricModels._equation_to_display(eq, endog, exog, params; mode=:latex)
    @test occursin("=", s2)

    # Non-subtraction -> expr = 0
    eq2 = :(y[t])
    s3 = MacroEconometricModels._equation_to_display(eq2, endog, exog, params; mode=:text)
    @test occursin("= 0", s3)
end

@testset "display.jl: _steady_state_latex" begin
    s = MacroEconometricModels._steady_state_latex([:y, :k], [1.0, 2.0])
    @test occursin("\\bar{y}", s)
    @test occursin("\\bar{k}", s)
    @test occursin("1", s)
    @test occursin("2", s)
end

@testset "constraints.jl: _check_jump_loaded" begin
    # Behavior depends on whether JuMP extension is loaded
    if hasmethod(MacroEconometricModels._jump_compute_steady_state,
                 Tuple{MacroEconometricModels.DSGESpec, Vector})
        # JuMP extension loaded (e.g., Pkg.test() environment) — should not throw
        @test MacroEconometricModels._check_jump_loaded() === nothing
    else
        # No JuMP extension — should throw ArgumentError
        @test_throws ArgumentError MacroEconometricModels._check_jump_loaded()
    end
end

@testset "constraints.jl: _path_available" begin
    # Should return a Bool (likely false without PATHSolver)
    @test MacroEconometricModels._path_available() isa Bool
end

@testset "pruning.jl: _dlyap_doubling" begin
    A = [0.5 0.0; 0.0 0.3]
    B = [1.0 0.0; 0.0 1.0]
    Sigma = MacroEconometricModels._dlyap_doubling(A, B)
    @test size(Sigma) == (2, 2)
    # Check Sigma = A*Sigma*A' + B
    @test norm(Sigma - (A * Sigma * A' + B)) < 1e-8
    @test issymmetric(Sigma) || norm(Sigma - Sigma') < 1e-10
end

@testset "occbin.jl: _find_last_binding_two" begin
    vm = BitMatrix([false false; false false; true false; false true; false false])
    @test MacroEconometricModels._find_last_binding_two(vm) == 4

    vm_none = falses(5, 2)
    @test MacroEconometricModels._find_last_binding_two(vm_none) == 0
end

end  # @testset "DSGE Coverage"
