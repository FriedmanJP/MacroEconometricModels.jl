# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Reiter (2009) method for linearizing heterogeneous agent models.

Treats the distribution histogram as part of the state vector and linearizes
the entire system around the stationary equilibrium.  Because the full
distribution lives in an N_a × N_e dimensional space, SVD dimensionality
reduction projects the distribution dynamics onto the dominant singular
vectors, yielding a tractable linear system.

# References
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and
  perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.
"""

using SparseArrays, LinearAlgebra, Random

# =============================================================================
# Shared GE-closure helpers for _reiter_linearize
# =============================================================================

"""
    _price_sensitivity_reduced(ss, ip, grid, income, U_k, dist_ss, Lambda_ss;
                               dr_step=1e-5, dw_step=1e-5) → (g_r_red, g_w_red)

Reduced-space distribution response to a unit change in the interest rate `r` and
the wage `w`: `g_r = ∂Λ/∂r · d_ss` and `g_w = ∂Λ/∂w · d_ss`, each projected onto the
reduction basis `U_k`. Computed by finite-difference re-solves of the EGM policy
(perturb one price, re-solve, difference the transition matrix).

This is the price→distribution kernel shared by the Huggett and Aiyagari closures of
`_reiter_linearize`. The *closures* differ (Huggett clears the bond market ∫a'=0;
Aiyagari uses the firm FOC with predetermined `K`), but the way household policies —
hence the distribution — respond to prices is common.
"""
function _price_sensitivity_reduced(ss::HASteadyState{T}, ip::IndividualProblem{T},
                                     grid::HAGrid{T}, income::IncomeProcess{T},
                                     U_k::AbstractMatrix{T}, dist_ss::AbstractVector{T},
                                     Lambda_ss::AbstractMatrix{T};
                                     dr_step::T=T(1e-5),
                                     dw_step::T=T(1e-5)) where {T<:AbstractFloat}
    prices_r = copy(ss.prices); prices_r[:r] = ss.prices[:r] + dr_step
    _, a_pol_r = _egm_solve(ip, grid, income, prices_r; max_iter=1000, tol=T(1e-10))
    Lambda_r = _build_transition_matrix(a_pol_r, grid, income)

    prices_w = copy(ss.prices); prices_w[:w] = ss.prices[:w] + dw_step
    _, a_pol_w = _egm_solve(ip, grid, income, prices_w; max_iter=1000, tol=T(1e-10))
    Lambda_w = _build_transition_matrix(a_pol_w, grid, income)

    g_r = (Lambda_r * dist_ss .- Lambda_ss * dist_ss) ./ dr_step
    g_w = (Lambda_w * dist_ss .- Lambda_ss * dist_ss) ./ dw_step
    return U_k' * g_r, U_k' * g_w
end

"""
    _aiyagari_foc_derivatives(r_ss, w_ss, K_ss, alpha, delta, Z_val)
        → (dr_dK, dw_dK, dr_dZ, dw_dZ)

Firm-FOC price sensitivities for the Aiyagari GE closure with predetermined capital
`K`: `r = α Z (K/L)^(α-1) − δ`, `w = (1−α) Z (K/L)^α`. In steady-state form,

    ∂r/∂K = (α−1)(r+δ)/K < 0,   ∂w/∂K = α w/K > 0,
    ∂r/∂Z = (r+δ)/Z,            ∂w/∂Z = w/Z.

A higher predetermined capital lowers the rate and raises the wage — the GE feedback
the reduced Reiter system must carry through the `K` state column.
"""
function _aiyagari_foc_derivatives(r_ss::T, w_ss::T, K_ss::T, alpha::T, delta::T,
                                   Z_val::T) where {T<:AbstractFloat}
    dr_dK = (alpha - one(T)) * (r_ss + delta) / K_ss
    dw_dK = alpha * w_ss / K_ss
    dr_dZ = (r_ss + delta) / Z_val
    dw_dZ = w_ss / Z_val
    return dr_dK, dw_dK, dr_dZ, dw_dZ
end

# Diagnose reduced-transition stability WITHOUT mutating G1 (#234). The previous
# code silently shrank every eigenvalue by `0.999/ρ` whenever the spectral radius
# ρ exceeded 1, uniformly distorting all dynamics to mask a missing-GE-block /
# wrong-Jacobian bug and reporting determinacy on a genuinely explosive system.
# #229/#230 restore genuine stability (ρ ≈ 0.9 Huggett, 0.9997 Aiyagari/KS), so
# this diagnostic should stay silent for the shipped examples; if it fires, the
# reduced HA system really is indeterminate/explosive and must be investigated.
function _reiter_warn_unstable(G1::AbstractMatrix{T}, label::AbstractString) where {T<:AbstractFloat}
    rho = maximum(abs, eigvals(G1))
    if rho >= one(T) + T(1e-8)
        @warn "Reiter ($label): reduced HA transition spectral radius ρ = " *
              "$(round(rho; digits=8)) ≥ 1 — the reduced system is indeterminate " *
              "or explosive (likely an incomplete GE block or a mis-scaled " *
              "Jacobian). No silent eigenvalue rescaling is applied (#234)."
    end
    return rho
end

# =============================================================================
# _reiter_linearize — SVD-reduced linearization of the HA model
# =============================================================================

"""
    _reiter_linearize(ss, ip, grid, income; n_reduced=50, dx=1e-6, n_sim=200, rng=nothing)
        → (G1, impact, n_reduced_actual, explained_variance)

Linearize a heterogeneous agent model around its stationary distribution using
the Reiter (2009) method, then reduce dimensionality via SVD.

The distribution histogram is part of the state vector.  The full transition
matrix Λ maps the N-dimensional distribution one period forward.  Because N
can be large, we identify the reachable subspace by simulating random
perturbations of the distribution through Λ, then retain only the top
singular vectors of the resulting deviation matrix.

The reduced state is `[d̃_t; K_t; Z_t]` where `d̃ = U_k' (d − d_ss)` are
the SVD-compressed distribution deviations, `K` is aggregate capital, and `Z`
is a TFP shock following an AR(1) with persistence `ρ_z` read from the spec
parameters (`het_params[:rho_z]`; #236).

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium
- `ip::IndividualProblem{T}` — household problem
- `grid::HAGrid{T}` — one-asset grid
- `income::IncomeProcess{T}` — income process
- `n_reduced::Int` — maximum number of retained singular vectors (default 50)
- `dx::Real` — perturbation scale for distribution probing (default 1e-6)
- `n_sim::Int` — number of random distribution perturbations (default 200)
- `rng` — random number generator (default `MersenneTwister(1234)`)

# Returns
- `G1::Matrix{T}` — `(n_red + n_agg) × (n_red + n_agg)` transition matrix
- `impact::Matrix{T}` — `(n_red + n_agg) × 1` shock impact vector
- `n_reduced_actual::Int` — actual number of retained singular vectors
- `explained_variance::T` — fraction of variance captured by retained vectors
- `U_k::Matrix{T}` — the `N × n_red` reduction basis (`N = n_a·n_e`), so callers can
  project reduced-state deviations back to the full distribution (`d_dev = U_k·d̃`)

# References
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and
  perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.
"""
function _reiter_linearize(ss::HASteadyState{T}, ip::IndividualProblem{T},
                            grid::HAGrid{T}, income::IncomeProcess{T};
                            n_reduced::Int=50, dx::Real=T(1e-6),
                            n_sim::Int=200,
                            model::Symbol=:aiyagari,
                            het_params::Dict{Symbol,T}=Dict{Symbol,T}(),
                            rng::Union{Nothing,AbstractRNG}=nothing) where {T<:AbstractFloat}
    @assert grid.n_dims == 1 "Reiter linearization requires a one-asset grid"
    @assert ip.n_asset_dims == 1 "Reiter linearization requires a one-asset individual problem"

    rng_actual = isnothing(rng) ? Random.MersenneTwister(1234) : rng
    dx_T = T(dx)

    n_a = grid.n_points[1]
    n_e = grid.n_income
    N = n_a * n_e
    a_grid = grid.grids[1]

    # ── Step 1: Extract steady-state objects ──────────────────────────────────

    a_pol_ss = ss.policies[:savings]
    dist_ss = vec(ss.distribution)        # N-vector (column-major: income slowest)
    K_ss = ss.aggregates[:K]

    # Build steady-state transition matrix
    Lambda_ss = _build_transition_matrix(a_pol_ss, grid, income)

    # ── Step 2: Build aggregate-relevant reduction basis via SVD ────────────
    # The distribution evolves as δ_{t+1} = Λ δ_t.  We care about directions
    # that matter for aggregate capital K = a_vec' d.  Build a matrix whose
    # columns span the reachable subspace weighted by both the transition
    # dynamics and the capital aggregator.
    #
    # Strategy: construct a "controllability-like" matrix by iterating Λ on
    # random perturbations *and* an "observability-like" matrix by iterating
    # Λ' on the capital aggregation vector.  The SVD of their combination
    # captures the directions most relevant for aggregate dynamics.

    # Λ_ss is an N×N sparse transition (N = n_a·n_e). Only sparse mat-vec/mat-mat
    # and a tall-thin SVD of the (dense, N×n_obs) observability matrix are needed,
    # so we never densify Λ_ss (#242). The economy `svd(O_mat)` below is kept
    # deterministic (NOT swapped for a randomized SVD).

    # Build the capital aggregation vector
    a_vec_pre = zeros(T, N)
    @inbounds for j in 1:n_e
        offset = (j - 1) * n_a
        for i in 1:n_a
            a_vec_pre[offset + i] = a_grid[i]
        end
    end

    # Observability matrix: rows are a_vec' Λ^k for k = 0, 1, ..., n_obs-1
    # This identifies which distribution directions affect K at various horizons.
    n_obs = min(N - 1, max(n_reduced * 3, 50))
    O_mat = zeros(T, N, n_obs)
    v_obs = copy(a_vec_pre)
    for k in 1:n_obs
        O_mat[:, k] .= v_obs
        v_obs = Lambda_ss' * v_obs          # sparse transpose mat-vec
    end

    # SVD of O_mat to get the dominant observable directions
    F_obs = svd(O_mat)

    n_available = min(length(F_obs.S), N - 1)
    n_red = min(n_reduced, n_available)

    # Threshold: retain singular values above a relative threshold
    s_thresh = F_obs.S[1] * T(1e-10)
    n_above = count(s -> s > s_thresh, F_obs.S[1:n_red])
    n_red = max(n_above, 1)

    # Reduction basis: left singular vectors of the observability matrix
    U_k = F_obs.U[:, 1:n_red]   # N × n_red

    # ── Step 3: Build reduced distribution transition ─────────────────────────
    # G1_dist = U_k' Λ_ss U_k  (project transition into reduced coordinates)

    G1_dist = U_k' * (Lambda_ss * U_k)    # n_red × n_red (sparse×dense, then project)

    # ── Step 4: Capital loading ───────────────────────────────────────────────
    # K is a linear function of the distribution: K = a_grid' * d
    # In reduced coordinates: δK = a_loading' * d̃  where a_loading = U_k' * a_vec
    # and a_vec[i + (j-1)*n_a] = a_grid[i]

    a_vec = zeros(T, N)
    @inbounds for j in 1:n_e
        offset = (j - 1) * n_a
        for i in 1:n_a
            a_vec[offset + i] = a_grid[i]
        end
    end
    K_loading = U_k' * a_vec   # n_red vector

    # ── Step 4b: Explained variance ───────────────────────────────────────────
    # Measure how well the reduced basis captures the distribution dynamics
    # that matter for aggregate capital.  For random perturbations δ, compare
    # the full aggregate capital response ΔK_full = a_vec' Λ δ with the
    # reduced reconstruction ΔK_red = K_loading' G1_dist (U_k' δ).

    n_test = min(n_sim, 100)
    var_full = zero(T)
    var_resid = zero(T)

    for k_test in 1:n_test
        noise = randn(rng_actual, T, N)
        noise .-= mean(noise)
        delta_test = dx_T .* noise

        # Full response
        dK_full = dot(a_vec, Lambda_ss * delta_test)   # sparse mat-vec

        # Reduced response
        d_tilde = U_k' * delta_test
        dK_red = dot(K_loading, G1_dist * d_tilde)

        var_full += dK_full^2
        var_resid += (dK_full - dK_red)^2
    end

    explained = var_full > zero(T) ? one(T) - var_resid / var_full : one(T)
    explained = clamp(explained, zero(T), one(T))

    # ── Huggett (zero net supply): aggregate block with r pinned by clearing ──
    # State [d̃; w], where w is the aggregate endowment level (AR(1) shock). The bond
    # clears every period (∫a' = 0), which pins the rate statically:
    #   dr_t = -(1/A_r) (A_w·dw_t + sav_load'·d̃_t),
    # with A_r = ∂(∫a')/∂r, A_w = ∂(∫a')/∂w, sav_load = U_k'·a'_ss. Substituting into
    # the distribution transition d̃_{t+1} = G1_dist·d̃_t + g_r·dr_t + g_w·dw_t gives the
    # closed reduced system. (No hard-coded Cobb-Douglas channel — unlike the Aiyagari
    # path below — since Huggett has no production.)
    if model === :huggett
        rho = T(get(het_params, :rho_e, 0.9))
        a_sav_ss = vec(a_pol_ss)
        dr_step = T(1e-5)
        dw_step = T(1e-5)

        prices_r = copy(ss.prices); prices_r[:r] = ss.prices[:r] + dr_step
        _, a_pol_r = _egm_solve(ip, grid, income, prices_r; max_iter=1000, tol=T(1e-10))
        Lambda_r = _build_transition_matrix(a_pol_r, grid, income)

        prices_w = copy(ss.prices); prices_w[:w] = ss.prices[:w] + dw_step
        _, a_pol_w = _egm_solve(ip, grid, income, prices_w; max_iter=1000, tol=T(1e-10))
        Lambda_w = _build_transition_matrix(a_pol_w, grid, income)

        # Distribution responses (projected) and aggregate-savings sensitivities.
        g_r = (Lambda_r * dist_ss .- Lambda_ss * dist_ss) ./ dr_step
        g_w = (Lambda_w * dist_ss .- Lambda_ss * dist_ss) ./ dw_step
        gr_red = U_k' * g_r
        gw_red = U_k' * g_w
        A_r = sum((vec(a_pol_r) .- a_sav_ss) .* dist_ss) / dr_step
        A_w = sum((vec(a_pol_w) .- a_sav_ss) .* dist_ss) / dw_step
        sav_load = U_k' * a_sav_ss

        @assert abs(A_r) > eps(T) "Huggett Reiter: ∂(asset demand)/∂r ≈ 0 (cannot pin rate)"

        channel_w = gw_red .- gr_red .* (A_w / A_r)        # response of d̃ to a w shock

        n_total = n_red + 1
        G1 = zeros(T, n_total, n_total)
        G1[1:n_red, 1:n_red] .= G1_dist .- (gr_red ./ A_r) * sav_load'
        G1[1:n_red, n_red + 1] .= channel_w
        G1[n_red + 1, n_red + 1] = rho                      # w_{t+1} = ρ·w_t + ε

        impact_vec = zeros(T, n_total, 1)
        impact_vec[n_red + 1, 1] = one(T)
        impact_vec[1:n_red, 1] .= channel_w

        _reiter_warn_unstable(G1, "Huggett")

        return G1, impact_vec, n_red, explained, U_k
    end

    # ── Step 6: Aiyagari general-equilibrium block (#230) ─────────────────────
    # State [d̃_t (n_red); K_t (1); Z_t (1)]. Prices come from the firm FOC with a
    # PREDETERMINED capital K, so the interest rate responds to capital:
    #   r = α Z (K/L)^(α-1) − δ,  w = (1−α) Z (K/L)^α,
    #   dr = (∂r/∂K) dK + (∂r/∂Z) dZ,   dw = (∂w/∂K) dK + (∂w/∂Z) dZ,
    #   ∂r/∂K < 0,  ∂w/∂K > 0  (see _aiyagari_foc_derivatives).
    # The distribution responds to prices via the shared price-sensitivity kernel:
    #   d̃_{t+1} = G1_dist·d̃_t + g_r·dr_t + g_w·dw_t
    #           = G1_dist·d̃_t + (g_r ∂r/∂K + g_w ∂w/∂K)·dK_t
    #                         + (g_r ∂r/∂Z + g_w ∂w/∂Z)·dZ_t.
    # Populating the K column (g_r ∂r/∂K + g_w ∂w/∂K) is exactly the GE feedback the
    # old code omitted — capital fed back into nothing and r never responded.
    n_agg = 2  # K and Z
    n_total = n_red + n_agg

    # Read alpha/delta/rho_z from the spec (no magic-number literals). solve(:reiter)
    # merges the aggregate-spec parameters into het_params, so examples (which store
    # these in agg_spec) and @dsge models (which store them in het_params) both work
    # (#236). A genuinely missing key errors informatively rather than defaulting.
    for k in (:alpha, :delta, :rho_z)
        haskey(het_params, k) ||
            error("Reiter (Aiyagari) linearization requires parameter :$k in spec params")
    end
    alpha_val = T(het_params[:alpha])
    delta_val = T(het_params[:delta])
    Z_val     = T(get(het_params, :Z, one(T)))   # SS TFP level, defaults to 1
    rho_z     = T(het_params[:rho_z])

    r_ss = ss.prices[:r]
    w_ss = ss.prices[:w]

    # Firm-FOC price sensitivities (predetermined K).
    dr_dK, dw_dK, dr_dZ, dw_dZ =
        _aiyagari_foc_derivatives(r_ss, w_ss, K_ss, alpha_val, delta_val, Z_val)

    # Reduced distribution response to r and w (shared kernel, as in Huggett).
    g_r_red, g_w_red = _price_sensitivity_reduced(ss, ip, grid, income, U_k,
                                                  dist_ss, Lambda_ss)

    # Reduced-distribution columns for the K and Z aggregate states.
    K_column = g_r_red .* dr_dK .+ g_w_red .* dw_dK   # ∂d̃_{t+1}/∂K_t (GE feedback)
    Z_column = g_r_red .* dr_dZ .+ g_w_red .* dw_dZ   # ∂d̃_{t+1}/∂Z_t

    G1 = zeros(T, n_total, n_total)
    G1[1:n_red, 1:n_red] .= G1_dist
    G1[1:n_red, n_red + 1] .= K_column          # K feeds back via the price channel
    G1[1:n_red, n_red + 2] .= Z_column

    # Capital row: K_{t+1} = K_loading' · d̃_{t+1}.
    G1[n_red + 1, 1:n_red] .= vec(K_loading' * G1_dist)
    G1[n_red + 1, n_red + 1] = dot(K_loading, K_column)
    G1[n_red + 1, n_red + 2] = dot(K_loading, Z_column)

    # TFP AR(1): Z_{t+1} = ρ_z · Z_t + ε.
    G1[n_red + 2, n_red + 2] = rho_z

    # ── Step 7: Impact vector (shock ε enters through Z) ──────────────────────
    impact_vec = zeros(T, n_total, 1)
    impact_vec[n_red + 2, 1] = one(T)
    impact_vec[1:n_red, 1] .= Z_column
    impact_vec[n_red + 1, 1] = dot(K_loading, Z_column)

    # ── Step 8: Diagnose stability (no silent rescale; #234) ──────────────────
    _reiter_warn_unstable(G1, "Aiyagari")

    return G1, impact_vec, n_red, explained, U_k
end
