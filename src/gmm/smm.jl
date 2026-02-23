# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Simulated Method of Moments (SMM) estimation.

References:
- Ruge-Murcia, F. (2012). "Estimating Nonlinear DSGE Models by the Simulated Method
  of Moments." Journal of Economic Dynamics and Control, 36(6), 914-938.
- Lee, B.-S. & Ingram, B. F. (1991). "Simulation Estimation of Time-Series Models."
  Journal of Econometrics, 47(2-3), 197-205.
"""

using LinearAlgebra, Statistics, Distributions, Random

# =============================================================================
# SMMModel Type
# =============================================================================

"""
    SMMModel{T} <: AbstractGMMModel

Simulated Method of Moments estimator.

Shares the `AbstractGMMModel` interface with `GMMModel` --- `coef`, `vcov`, `nobs`,
`stderror`, `show`, `refs`, `report`, `j_test` all work.

# Fields
- `theta::Vector{T}` --- estimated parameters
- `vcov::Matrix{T}` --- asymptotic covariance matrix
- `n_moments::Int` --- number of moment conditions
- `n_params::Int` --- number of parameters
- `n_obs::Int` --- number of data observations
- `weighting::GMMWeighting{T}` --- weighting specification
- `W::Matrix{T}` --- final weighting matrix
- `g_bar::Vector{T}` --- moment discrepancy at solution
- `J_stat::T` --- Hansen J-test statistic
- `J_pvalue::T` --- J-test p-value
- `converged::Bool` --- convergence flag
- `iterations::Int` --- optimizer iterations
- `sim_ratio::Int` --- tau = simulation periods / data periods
"""
struct SMMModel{T<:AbstractFloat} <: AbstractGMMModel
    theta::Vector{T}
    vcov::Matrix{T}
    n_moments::Int
    n_params::Int
    n_obs::Int
    weighting::GMMWeighting{T}
    W::Matrix{T}
    g_bar::Vector{T}
    J_stat::T
    J_pvalue::T
    converged::Bool
    iterations::Int
    sim_ratio::Int

    function SMMModel{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W,
                          g_bar, J_stat, J_pvalue, converged, iterations,
                          sim_ratio) where {T<:AbstractFloat}
        @assert length(theta) == n_params
        @assert size(vcov) == (n_params, n_params)
        @assert size(W) == (n_moments, n_moments)
        @assert length(g_bar) == n_moments
        @assert n_moments >= n_params "SMM requires at least as many moments as parameters"
        @assert sim_ratio >= 1
        new{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W,
               g_bar, J_stat, J_pvalue, converged, iterations, sim_ratio)
    end
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(m::SMMModel) = m.theta
StatsAPI.vcov(m::SMMModel) = m.vcov
StatsAPI.nobs(m::SMMModel) = m.n_obs
StatsAPI.dof(m::SMMModel) = m.n_params
StatsAPI.islinear(::SMMModel) = false
StatsAPI.stderror(m::SMMModel) = sqrt.(max.(diag(m.vcov), zero(eltype(m.theta))))

is_overidentified(m::SMMModel) = m.n_moments > m.n_params
overid_df(m::SMMModel) = m.n_moments - m.n_params

function StatsAPI.confint(m::SMMModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    z = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.theta .- z .* se, m.theta .+ z .* se)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::SMMModel{T}) where {T}
    spec = Any[
        "Parameters"    m.n_params;
        "Moments"       m.n_moments;
        "Observations"  m.n_obs;
        "Sim ratio (τ)" m.sim_ratio;
        "Weighting"     string(m.weighting.method);
        "Converged"     m.converged ? "Yes" : "No";
        "Iterations"    m.iterations
    ]
    _pretty_table(io, spec;
        title = "SMM Estimation Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    se = stderror(m)
    param_names = ["θ[$i]" for i in 1:m.n_params]
    _coef_table(io, "Coefficients", param_names, m.theta, se; dist=:z)
    if is_overidentified(m)
        j_data = Any[
            "J-statistic" _fmt(m.J_stat);
            "P-value"     _format_pvalue(m.J_pvalue);
            "DF"          overid_df(m)
        ]
        _pretty_table(io, j_data;
            title = "Hansen J-test",
            column_labels = ["", ""],
            alignment = [:l, :r],
        )
    end
end

# =============================================================================
# J-Test
# =============================================================================

"""
    j_test(m::SMMModel{T}) -> NamedTuple

Hansen's J-test for overidentifying restrictions on an SMM model.

H0: All moment conditions are valid (E[g(theta_0)] = 0)
H1: Some moment conditions are violated

Returns:
- J_stat: Test statistic
- p_value: p-value from chi-squared distribution
- df: Degrees of freedom (n_moments - n_params)
- reject_05: Whether to reject at 5% level
"""
function j_test(m::SMMModel{T}) where {T}
    df = overid_df(m)
    if df <= 0
        return (J_stat=zero(T), p_value=one(T), df=0, reject_05=false,
                message="Model is just-identified, J-test not applicable")
    end
    (J_stat=m.J_stat, p_value=m.J_pvalue, df=df, reject_05=m.J_pvalue < T(0.05))
end

# =============================================================================
# Moment Functions
# =============================================================================

"""
    autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) -> Vector{T}

Compute standard DSGE moment vector from data matrix.

Returns: `[upper-triangle variance-covariance elements; diagonal autocovariances at each lag]`

For k variables, lags L: `k*(k+1)/2 + k*L` moments total.

# Arguments
- `data` --- T_obs x k data matrix
- `lags` --- number of autocovariance lags (default: 1)
"""
function autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) where {T<:AbstractFloat}
    n, k = size(data)
    means = vec(mean(data, dims=1))
    data_c = data .- means'

    moments = T[]

    # Upper triangle of variance-covariance matrix (1/n divisor)
    for i in 1:k
        for j in i:k
            push!(moments, dot(data_c[:, i], data_c[:, j]) / n)
        end
    end

    # Diagonal autocovariances at each lag (1/n divisor)
    for lag in 1:lags
        for i in 1:k
            acov = dot(data_c[(lag+1):n, i], data_c[1:(n-lag), i]) / n
            push!(moments, acov)
        end
    end

    moments
end

autocovariance_moments(data::AbstractMatrix{<:Real}; kwargs...) =
    autocovariance_moments(Float64.(data); kwargs...)

# =============================================================================
# Weighting Matrix
# =============================================================================

"""
    _moment_covariance(data, moments_fn; hac, bandwidth) -> Matrix{T}

Internal: Compute long-run covariance of per-observation moment contributions.
Shared by `smm_weighting_matrix` (inverted) and `smm_data_covariance` (raw).
"""
function _moment_covariance(data::AbstractMatrix{T}, moments_fn::Function;
                             hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    n = size(data, 1)
    m_full = moments_fn(data)
    q = length(m_full)

    G = Matrix{T}(undef, n, q)
    for t in 1:n
        G[t, :] = moments_fn(data[t:t, :])
    end
    G_demean = G .- mean(G, dims=1)

    if hac
        long_run_covariance(G_demean; bandwidth=bandwidth, kernel=:bartlett)
    else
        (G_demean' * G_demean) / n
    end
end

"""
    smm_weighting_matrix(data::AbstractMatrix{T}, moments_fn::Function;
                          hac::Bool=true, bandwidth::Int=0) -> Matrix{T}

Compute optimal SMM weighting matrix from data moment contributions.
Centers the per-observation moment contributions and applies HAC with Bartlett kernel.

# Arguments
- `data` — T_obs × k data matrix
- `moments_fn` — function computing moment vector from data
- `hac` — use HAC correction (default: true)
- `bandwidth` — HAC bandwidth, 0 = automatic: `floor(4*(n/100)^(2/9))`
"""
function smm_weighting_matrix(data::AbstractMatrix{T}, moments_fn::Function;
                               hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    Omega = _moment_covariance(data, moments_fn; hac=hac, bandwidth=bandwidth)
    Omega_sym = Hermitian((Omega + Omega') / 2)
    eigvals_O = eigvals(Omega_sym)
    if minimum(eigvals_O) < eps(T)
        Omega_reg = Omega_sym + T(1e-8) * I
        return Matrix{T}(inv(Omega_reg))
    end
    robust_inv(Matrix(Omega_sym))
end

"""
    smm_data_covariance(data::AbstractMatrix{T}, moments_fn::Function;
                          hac::Bool=true, bandwidth::Int=0) -> Matrix{T}

Compute long-run covariance Ω of data moment contributions for sandwich SE formula.
"""
function smm_data_covariance(data::AbstractMatrix{T}, moments_fn::Function;
                              hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    _moment_covariance(data, moments_fn; hac=hac, bandwidth=bandwidth)
end

# =============================================================================
# SMM Estimation
# =============================================================================

"""
    estimate_smm(simulator_fn, moments_fn, theta0, data;
                 sim_ratio=5, burn=100, weighting=:two_step,
                 bounds=nothing, hac=true, bandwidth=0,
                 max_iter=1000, tol=1e-8,
                 rng=Random.default_rng()) -> SMMModel{T}

Estimate parameters via Simulated Method of Moments.

Minimizes `Q(theta) = (m_data - m_sim(theta))' W (m_data - m_sim(theta))` where
`m_data` are data moments and `m_sim(theta)` are simulated moments.

# Arguments
- `simulator_fn(theta, T_periods, burn; rng)` --- simulates T_periods obs after discarding burn
- `moments_fn(data) -> Vector{T}` --- computes moment vector from any T x k data matrix
- `theta0` --- initial parameter guess
- `data` --- observed data matrix (T x k)

# Keywords
- `sim_ratio::Int=5` --- tau = simulation periods / data periods
- `burn::Int=100` --- burn-in periods for simulator
- `weighting::Symbol=:two_step` --- `:identity` or `:two_step`
- `bounds::Union{Nothing,ParameterTransform}=nothing` --- parameter bounds
- `hac::Bool=true` --- HAC for weighting matrix
- `bandwidth::Int=0` --- HAC bandwidth (0 = automatic)
- `max_iter::Int=1000` --- max optimizer iterations
- `tol=1e-8` --- convergence tolerance
- `rng` --- random number generator (copied inside objective for deterministic simulation)

# References
- Ruge-Murcia (2012), Lee & Ingram (1991)
"""
function estimate_smm(simulator_fn::Function, moments_fn::Function,
                      theta0::AbstractVector, data::AbstractMatrix;
                      sim_ratio::Int=5, burn::Int=100,
                      weighting::Symbol=:two_step,
                      bounds::Union{Nothing,ParameterTransform}=nothing,
                      hac::Bool=true, bandwidth::Int=0,
                      max_iter::Int=1000, tol::Real=1e-8,
                      rng=Random.default_rng())
    T_type = eltype(data) <: AbstractFloat ? eltype(data) : Float64
    data_T = Matrix{T_type}(data)
    theta0_T = T_type.(theta0)
    tol_T = T_type(tol)

    n_obs = size(data_T, 1)
    n_params = length(theta0_T)
    T_sim = sim_ratio * n_obs

    # Default bandwidth: Andrews (1991) plug-in rule when bandwidth=0
    # Per-observation moment contributions from autocovariance-type functions are
    # often degenerate (a single row has zero variance), which causes the automatic
    # bandwidth estimator in long_run_covariance to produce NaN. Compute a safe
    # default from the data dimensions.
    bw = bandwidth > 0 ? bandwidth : max(1, floor(Int, 4 * (n_obs / 100)^(2/9)))

    # Compute data moments
    m_data = moments_fn(data_T)
    n_moments = length(m_data)

    @assert n_moments >= n_params "SMM requires at least as many moments ($n_moments) as parameters ($n_params)"

    # Set up parameter transform
    has_bounds = bounds !== nothing
    if has_bounds
        phi0 = to_unconstrained(bounds, theta0_T)
    else
        phi0 = copy(theta0_T)
    end

    # SMM objective: Q(theta) = g(theta)' W g(theta) where g(theta) = m_data - m_sim(theta)
    # CRITICAL: use copy(rng) so the same random draws are used every call,
    # making m_sim(theta) a deterministic function of theta for smooth optimization.
    function smm_moment_discrepancy(theta_or_phi)
        theta = has_bounds ? to_constrained(bounds, theta_or_phi) : theta_or_phi
        sim_data = simulator_fn(theta, T_sim, burn; rng=copy(rng))
        m_sim = moments_fn(Matrix{T_type}(sim_data))
        m_data .- m_sim
    end

    function smm_objective(phi, W)
        g = smm_moment_discrepancy(phi)
        dot(g, W * g)
    end

    # ------------------------------------------------------------------
    # Step 1: Identity weighting
    # ------------------------------------------------------------------
    W1 = Matrix{T_type}(I, n_moments, n_moments)
    obj1(phi) = smm_objective(phi, W1)

    # Primary: NelderMead (derivative-free, robust for simulation-based objectives)
    result1 = Optim.optimize(obj1, phi0, Optim.NelderMead(),
                              Optim.Options(iterations=max_iter, f_reltol=tol_T))
    if !Optim.converged(result1)
        # Fallback: LBFGS
        result1_lbfgs = Optim.optimize(obj1, phi0, Optim.LBFGS(),
                                        Optim.Options(iterations=max_iter, f_reltol=tol_T))
        if Optim.minimum(result1_lbfgs) < Optim.minimum(result1)
            result1 = result1_lbfgs
        end
    end

    phi_hat = Optim.minimizer(result1)
    converged = Optim.converged(result1)
    iterations = Optim.iterations(result1)

    # ------------------------------------------------------------------
    # Step 2: Optimal weighting (if two_step)
    # ------------------------------------------------------------------
    if weighting == :two_step
        W2 = smm_weighting_matrix(data_T, moments_fn; hac=hac, bandwidth=bw)

        obj2(phi) = smm_objective(phi, W2)
        result2 = Optim.optimize(obj2, phi_hat, Optim.NelderMead(),
                                  Optim.Options(iterations=max_iter, f_reltol=tol_T))
        if !Optim.converged(result2)
            result2_lbfgs = Optim.optimize(obj2, phi_hat, Optim.LBFGS(),
                                            Optim.Options(iterations=max_iter, f_reltol=tol_T))
            if Optim.minimum(result2_lbfgs) < Optim.minimum(result2)
                result2 = result2_lbfgs
            end
        end

        phi_hat = Optim.minimizer(result2)
        W_final = W2
        converged = Optim.converged(result2)
        iterations += Optim.iterations(result2)
    else
        W_final = W1
    end

    # Map back to constrained space
    theta_hat = has_bounds ? to_constrained(bounds, phi_hat) : phi_hat

    # Final moment discrepancy
    g_bar = smm_moment_discrepancy(phi_hat)

    # ------------------------------------------------------------------
    # Numerical Jacobian of simulated moments w.r.t. theta
    # ------------------------------------------------------------------
    function sim_moments(theta)
        sim_data = simulator_fn(theta, T_sim, burn; rng=copy(rng))
        m = moments_fn(Matrix{T_type}(sim_data))
        # Replace NaN/Inf with large values to keep Jacobian finite
        replace!(x -> isfinite(x) ? x : T_type(1e10), m)
        m
    end
    D = numerical_gradient(sim_moments, theta_hat)

    # If Jacobian still contains non-finite values, fall back to zero (will produce
    # large SE but won't crash)
    if any(!isfinite, D)
        D = zeros(T_type, n_moments, n_params)
    end

    # ------------------------------------------------------------------
    # Variance-covariance computation
    # ------------------------------------------------------------------
    # Simulation correction factor: (1 + 1/tau) accounts for simulation noise
    sim_correction = one(T_type) + one(T_type) / T_type(sim_ratio)

    bread = D' * W_final * D
    # Regularize if needed to avoid singular matrix
    if any(!isfinite, bread) || det(bread) == zero(T_type)
        bread = bread + T_type(1e-8) * I
        # Replace any remaining NaN/Inf
        for i in eachindex(bread)
            if !isfinite(bread[i])
                bread[i] = zero(T_type)
            end
        end
        bread = bread + T_type(1e-8) * I
    end
    bread_inv = robust_inv(bread)

    if weighting == :two_step
        # Efficient GMM (W approx Omega^{-1}): V = (1 + 1/tau) * (D'WD)^{-1} / n
        vcov = sim_correction * bread_inv / T_type(n_obs)
    else
        # Sandwich: V = (1 + 1/tau) * (D'WD)^{-1} D'W Omega W D (D'WD)^{-1} / n
        Omega = smm_data_covariance(data_T, moments_fn; hac=hac, bandwidth=bw)
        meat = D' * W_final * Omega * W_final * D
        vcov = sim_correction * (bread_inv * meat * bread_inv) / T_type(n_obs)
    end

    # Delta method SE correction for transforms
    if has_bounds
        J_transform = transform_jacobian(bounds, phi_hat)
        vcov = J_transform * vcov * J_transform'
    end

    # Ensure symmetric
    vcov = (vcov + vcov') / 2

    # ------------------------------------------------------------------
    # J-statistic (Hansen overidentification test)
    # ------------------------------------------------------------------
    J_stat, J_pvalue = if n_moments > n_params
        J = T_type(n_obs) * dot(g_bar, W_final * g_bar)
        J = max(J, zero(T_type))
        df = n_moments - n_params
        (J, one(T_type) - cdf(Chisq(df), J))
    else
        (zero(T_type), one(T_type))
    end

    weighting_spec = GMMWeighting{T_type}(weighting, max_iter, tol_T)

    SMMModel{T_type}(
        theta_hat, vcov, n_moments, n_params, n_obs, weighting_spec,
        W_final, g_bar, J_stat, J_pvalue, converged, iterations, sim_ratio
    )
end
