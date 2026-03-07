#=
Kink Benchmark: Standard Chebyshev vs ReLU-Enriched Chebyshev
==============================================================
Consumption-savings model with borrowing constraint (1 state, 1 control).
Unique solution with a sharp kink at the constraint boundary.

Model:
  c + a' = R·a + w,  a' ≥ 0  (borrowing constraint)
  Euler: u'(c) = β·R·E[u'(c(a'))]  when a' > 0
         a' = 0                      when constrained

The policy function a'(a) has a kink at a* where the constraint switches
from binding to non-binding. Standard Chebyshev creates Gibbs oscillations;
ReLU enrichment captures the kink analytically.

Solves via endogenous grid method (EGM) for ground truth, then fits
Chebyshev and ReLU-enriched Chebyshev approximations.

Run: julia --project=. scripts/benchmark_zlb_relu.jl
=#

using LinearAlgebra, Printf, Statistics

# ============================================================================
# Section 1: Model + Utilities
# ============================================================================

struct SavingsModel{T}
    β::T; R::T; w::T; σ_c::T; ρ::T; σ_ε::T; a_min::T; a_max::T
end

function SavingsModel(; β=0.96, R=1.03, w=1.0, σ_c=2.0, ρ=0.0, σ_ε=0.2, a_min=0.0, a_max=6.0)
    SavingsModel(β, R, w, σ_c, ρ, σ_ε, a_min, a_max)
end

u_prime(c, σ) = c^(-σ)
u_prime_inv(x, σ) = x^(-1.0/σ)

function gauss_hermite(n::Int)
    J = SymTridiagonal(zeros(n), [sqrt(k / 2.0) for k in 1:n-1])
    F = eigen(J)
    return sqrt(2.0) .* F.values, sqrt(π) .* F.vectors[1, :] .^ 2 ./ sqrt(π)
end

to_cheb(a, lo, hi) = 2.0 * (a - lo) / (hi - lo) - 1.0
from_cheb(z, lo, hi) = lo + (z + 1.0) / 2.0 * (hi - lo)
cheb_nodes(d::Int) = [cos(π * k / d) for k in 0:d]

function cheb_basis(z, d::Int)
    T = Vector{Float64}(undef, d + 1)
    T[1] = 1.0
    d >= 1 && (T[2] = z)
    for j in 3:d+1; T[j] = 2z * T[j-1] - T[j-2]; end
    return T
end

function softplus(x, ε)
    t = x / ε
    t > 30.0 ? x : t < -30.0 ? 0.0 : ε * log(1.0 + exp(t))
end

# ============================================================================
# Section 2: EGM Ground Truth (Fine Grid)
# ============================================================================

"""
Solve consumption-savings by endogenous grid method on a fine grid.
With ρ=0 (iid income), the state is just assets a.
Returns (a_grid, c_policy, a_prime_policy, kink_location).
"""
function solve_egm(m::SavingsModel; N=5000, tol=1e-12, maxiter=5000)
    gh_nodes, gh_weights = gauss_hermite(7)
    n_q = length(gh_nodes)

    # Exogenous grid for a' (savings)
    a_prime_grid = collect(range(m.a_min, m.a_max, length=N))

    # Initialize consumption policy on an a-grid
    a_grid_uniform = collect(range(m.a_min, m.a_max, length=N))
    c_old = m.R .* a_grid_uniform .+ m.w .- a_grid_uniform .* 0.5  # consume half

    # EGM iteration
    for iter in 1:maxiter
        # For each a' on the exogenous grid, compute implied a using Euler equation
        a_endog = zeros(N)
        c_endog = zeros(N)

        for i in 1:N
            a_p = a_prime_grid[i]

            # E[u'(c(a'))] via quadrature
            # With iid income shocks: a' is next period's state
            # Income next period: w + ε, where ε ~ N(0, σ_ε²)
            Eu_prime = 0.0
            for q in 1:n_q
                income_next = m.w + m.σ_ε * gh_nodes[q]
                cash_next = m.R * a_p + income_next
                # Next period consumption (from current policy, interpolated)
                c_next = max(cash_next - linear_interp_egm(a_grid_uniform, c_old, a_p, cash_next), 1e-10)
                # Actually, for EGM with iid: c(a') is the consumption when state is a'
                # Total cash: R*a' + w + ε
                # c = cash - a'' where a'' = policy(a')
                # Use current consumption policy
                c_next_val = linear_interp_c(a_grid_uniform, c_old, a_p)
                if c_next_val <= 0.0
                    c_next_val = m.R * a_p + m.w + m.σ_ε * gh_nodes[q]  # consume everything
                end
                Eu_prime += gh_weights[q] * u_prime(c_next_val, m.σ_c)
            end

            # Euler equation: u'(c) = β R E[u'(c')]
            c_i = u_prime_inv(m.β * m.R * Eu_prime, m.σ_c)
            c_endog[i] = c_i

            # Endogenous grid: a such that c + a' = R*a + w → a = (c + a' - w) / R
            a_endog[i] = (c_i + a_p - m.w) / m.R
        end

        # Augment with constraint: at a = a_min, c = R*a_min + w (consume everything)
        # The endogenous grid might not start at a_min
        # Prepend the constrained region
        a_constrained = m.a_min
        c_constrained = m.R * m.a_min + m.w  # consume everything, a' = a_min

        # Build new policy on uniform grid by interpolation
        c_new = similar(c_old)
        for i in 1:N
            a = a_grid_uniform[i]
            if a <= a_endog[1]
                # Constrained: consume everything above a_min
                c_new[i] = m.R * a + m.w - m.a_min  # a' = a_min
            else
                c_new[i] = linear_interp_c(a_endog, c_endog, a)
            end
        end

        err = maximum(abs, c_new .- c_old)
        c_old .= c_new

        if err < tol
            # Find kink location
            a_star = a_endog[1]  # first point where Euler holds with equality
            a_prime_policy = m.R .* a_grid_uniform .+ m.w .- c_old
            return a_grid_uniform, c_old, a_prime_policy, a_star, iter
        end
    end

    a_prime_policy = m.R .* a_grid_uniform .+ m.w .- c_old
    a_star = a_grid_uniform[findfirst(a_prime_policy .> m.a_min + 1e-8)]
    return a_grid_uniform, c_old, a_prime_policy, a_star, maxiter
end

function linear_interp_c(grid, vals, x)
    x = clamp(x, first(grid), last(grid))
    i = searchsortedlast(grid, x)
    i = clamp(i, 1, length(grid) - 1)
    t = (x - grid[i]) / (grid[i+1] - grid[i])
    return vals[i] + t * (vals[i+1] - vals[i])
end

function linear_interp_egm(grid, vals, a_p, cash)
    # Not used in simplified version
    return linear_interp_c(grid, vals, a_p)
end

# ============================================================================
# Section 3: Chebyshev Fitting (Least Squares on EGM Solution)
# ============================================================================

"""
Fit Chebyshev polynomial to the EGM solution via least squares.
This measures APPROXIMATION quality, not solver quality.
"""
function fit_chebyshev(a_grid, policy, a_lo, a_hi; degree=15)
    n_basis = degree + 1

    # Basis matrix at grid points
    N = length(a_grid)
    B = zeros(N, n_basis)
    for i in 1:N
        z = to_cheb(a_grid[i], a_lo, a_hi)
        B[i, :] = cheb_basis(z, degree)
    end

    # Least squares fit
    coeffs = B \ policy
    return coeffs, degree, n_basis
end

"""
Fit ReLU-enriched Chebyshev to the EGM solution.
Basis: {T₀,...,T_d, ψ(a)} where ψ(a) = softplus(a* - a, ε).
"""
function fit_relu_chebyshev(a_grid, policy, a_lo, a_hi, a_star;
                             degree=10, ε_basis=0.005)
    n_cheb = degree + 1
    n_basis = n_cheb + 1

    N = length(a_grid)
    B = zeros(N, n_basis)
    for i in 1:N
        z = to_cheb(a_grid[i], a_lo, a_hi)
        B[i, 1:n_cheb] = cheb_basis(z, degree)
        B[i, n_basis] = softplus(a_star - a_grid[i], ε_basis)
    end

    coeffs = B \ policy
    return coeffs, degree, n_basis, ε_basis
end

# ============================================================================
# Section 4: Error Metrics
# ============================================================================

function approximation_errors(a_test, true_policy, approx_func)
    N = length(a_test)
    errs = [abs(true_policy[i] - approx_func(a_test[i])) for i in 1:N]
    return maximum(errs), mean(errs)
end

function near_kink_error(a_test, true_policy, approx_func, a_star; width=0.3)
    near = [i for i in eachindex(a_test) if abs(a_test[i] - a_star) < width]
    isempty(near) && return NaN
    errs = [abs(true_policy[i] - approx_func(a_test[i])) for i in near]
    return maximum(errs)
end

# ============================================================================
# Section 5: Policy Function Constructors
# ============================================================================

function make_cheb_func(coeffs, degree, a_lo, a_hi)
    a -> dot(cheb_basis(to_cheb(a, a_lo, a_hi), degree), coeffs)
end

function make_relu_func(coeffs, degree, n_basis, a_star, ε_basis, a_lo, a_hi)
    function f(a)
        z = to_cheb(a, a_lo, a_hi)
        b = vcat(cheb_basis(z, degree), softplus(a_star - a, ε_basis))
        dot(b, coeffs)
    end
end

# ============================================================================
# Section 6: Main Benchmark
# ============================================================================

function main()
    m = SavingsModel()

    println("Kink Benchmark: Consumption-Savings with Borrowing Constraint")
    println("=" ^ 90)
    @printf("Parameters: β=%.2f, R=%.2f, w=%.1f, σ=%.1f, σ_ε=%.2f, a ∈ [%.0f, %.0f]\n\n",
            m.β, m.R, m.w, m.σ_c, m.σ_ε, m.a_min, m.a_max)

    # Solve for ground truth
    print("Solving EGM (N=5000)...")
    a_grid, c_pol, ap_pol, a_star, egm_iters = solve_egm(m)
    @printf(" done (%d iters, a* = %.4f)\n\n", egm_iters, a_star)

    # Dense test grid
    a_test = collect(range(m.a_min, m.a_max, length=10000))
    ap_true = [linear_interp_c(a_grid, ap_pol, a) for a in a_test]

    # Fit approximations to savings policy a'(a)
    println("Fitting approximations to savings policy a'(a)...")

    # Reference: Chebyshev d=40
    coeffs_ref, deg_ref, nb_ref = fit_chebyshev(a_grid, ap_pol, m.a_min, m.a_max; degree=40)
    f_ref = make_cheb_func(coeffs_ref, deg_ref, m.a_min, m.a_max)

    # Standard Chebyshev d=15
    coeffs_ch, deg_ch, nb_ch = fit_chebyshev(a_grid, ap_pol, m.a_min, m.a_max; degree=15)
    f_ch = make_cheb_func(coeffs_ch, deg_ch, m.a_min, m.a_max)

    # ReLU-enriched d=10
    coeffs_rl, deg_rl, nb_rl, eps_rl = fit_relu_chebyshev(
        a_grid, ap_pol, m.a_min, m.a_max, a_star; degree=10, ε_basis=0.01)
    f_rl = make_relu_func(coeffs_rl, deg_rl, nb_rl, a_star, eps_rl, m.a_min, m.a_max)

    # Also try matched basis count: ReLU d=14 (15+1=16 basis = same as Cheb d=15)
    coeffs_rl2, deg_rl2, nb_rl2, eps_rl2 = fit_relu_chebyshev(
        a_grid, ap_pol, m.a_min, m.a_max, a_star; degree=14, ε_basis=0.01)
    f_rl2 = make_relu_func(coeffs_rl2, deg_rl2, nb_rl2, a_star, eps_rl2, m.a_min, m.a_max)

    # Errors
    max_ref, mean_ref = approximation_errors(a_test, ap_true, f_ref)
    max_ch, mean_ch = approximation_errors(a_test, ap_true, f_ch)
    max_rl, mean_rl = approximation_errors(a_test, ap_true, f_rl)
    max_rl2, mean_rl2 = approximation_errors(a_test, ap_true, f_rl2)

    kink_ref = near_kink_error(a_test, ap_true, f_ref, a_star)
    kink_ch = near_kink_error(a_test, ap_true, f_ch, a_star)
    kink_rl = near_kink_error(a_test, ap_true, f_rl, a_star)
    kink_rl2 = near_kink_error(a_test, ap_true, f_rl2, a_star)

    # Results table
    println()
    println("─" ^ 90)
    @printf("%-25s │ %5s │ %10s │ %10s │ %10s\n",
            "Method", "Basis", "Max Error", "Mean Error", "Near-Kink")
    println("─" ^ 90)
    @printf("%-25s │ %5d │ %10.2e │ %10.2e │ %10.2e\n",
            "Reference (d=40)", nb_ref, max_ref, mean_ref, kink_ref)
    @printf("%-25s │ %5d │ %10.2e │ %10.2e │ %10.2e\n",
            "Chebyshev (d=15)", nb_ch, max_ch, mean_ch, kink_ch)
    @printf("%-25s │ %5d │ %10.2e │ %10.2e │ %10.2e\n",
            "ReLU+Cheb (d=10)", nb_rl, max_rl, mean_rl, kink_rl)
    @printf("%-25s │ %5d │ %10.2e │ %10.2e │ %10.2e\n",
            "ReLU+Cheb (d=14, matched)", nb_rl2, max_rl2, mean_rl2, kink_rl2)
    println("─" ^ 90)
    println()

    # Analysis
    @printf("ReLU+Cheb d=10 (12 basis) vs Chebyshev d=15 (16 basis):\n")
    @printf("  Max error:    %.2e vs %.2e (%.1fx improvement)\n",
            max_rl, max_ch, max_ch / max_rl)
    @printf("  Near-kink:    %.2e vs %.2e (%.1fx improvement)\n",
            kink_rl, kink_ch, kink_ch / kink_rl)
    @printf("  Mean error:   %.2e vs %.2e (%.1fx improvement)\n\n",
            mean_rl, mean_ch, mean_ch / mean_rl)

    @printf("ReLU+Cheb d=14 (16 basis, matched) vs Chebyshev d=15 (16 basis):\n")
    @printf("  Max error:    %.2e vs %.2e (%.1fx improvement)\n",
            max_rl2, max_ch, max_ch / max_rl2)
    @printf("  Near-kink:    %.2e vs %.2e (%.1fx improvement)\n",
            kink_rl2, kink_ch, kink_ch / kink_rl2)
    @printf("  Mean error:   %.2e vs %.2e (%.1fx improvement)\n",
            mean_rl2, mean_ch, mean_ch / mean_rl2)
end

main()
