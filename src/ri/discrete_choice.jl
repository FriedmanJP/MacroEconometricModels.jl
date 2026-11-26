# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Discrete-choice Rational Inattention (RI) solver, consideration-set KKT
certificate, IIA breakdown diagnostic, and econometric RI logit estimator.

Implements:
- `solve_ri_discrete`: Safeguarded SQUAREM Blahut–Arimoto + active-set Newton
  with KKT revival (Matějka & McKay 2015; Caplin, Dean & Leahy 2019).
- `ri_iia_diagnostic`: Independence of Irrelevant Alternatives (IIA) breakdown
  diagnostic under menu expansion (Red-Bus / Blue-Bus benchmark).
- `estimate_ri_logit`: Maximum-likelihood econometric estimator for state-dependent
  and unconditional RI discrete choice across multiple environments.

# References
- Matějka, F., & McKay, A. (2015). "Rational Inattention to Discrete Choices:
  A New Foundation for the Multinomial Logit Model." *American Economic Review*,
  105(1), 272–298.
- Caplin, A., Dean, M., & Leahy, J. (2019). "Rational Inattention, Optimal
  Consideration Sets, and Stochastic Choice." *Review of Economic Studies*,
  86(3), 1061–1094.
"""

# =============================================================================
# Types
# =============================================================================

"""
    RIDiscreteChoiceSolution{T<:AbstractFloat}

Solution to the Matějka–McKay (2015) / Caplin–Dean–Leahy (2019) discrete-choice
Rational Inattention problem for a `J × S` payoff matrix `U` and prior `μ ∈ Δ^{S-1}`.

# Fields
- `payoffs::Matrix{T}` — `J × S` payoff matrix `U_{j,s} = u(a_j, v_s)`
- `prior::Vector{T}` — prior state distribution `μ ∈ Δ^{S-1}`
- `cond_probs::Matrix{T}` — `J × S` conditional choice probabilities `P(j | s)`
- `uncond_probs::Vector{T}` — length-`J` unconditional probabilities `P(j) = Σ_s μ_s P(j | s)`
- `consideration_set::Vector{Int}` — indices `findall(>(tol), uncond_probs)`
- `kkt_multipliers::Vector{T}` — CDL KKT multipliers `ζ_j(P)` for `j = 1..J`
- `omega::T` — marginal information cost `ω` per nat
- `mutual_info_bits::T` — mutual information `I(a; v) / ln(2)` in bits
- `mutual_info_nats::T` — mutual information `I(a; v)` in nats
- `expected_utility::T` — `Σ_{j,s} μ_s P(j | s) U_{j,s}`
- `net_welfare::T` — `expected_utility - omega * mutual_info_nats`
- `converged::Bool` — whether the global KKT optimality condition converged
- `iterations::Int` — total solver iterations performed
"""
struct RIDiscreteChoiceSolution{T<:AbstractFloat}
    payoffs::Matrix{T}
    prior::Vector{T}
    cond_probs::Matrix{T}
    uncond_probs::Vector{T}
    consideration_set::Vector{Int}
    kkt_multipliers::Vector{T}
    omega::T
    mutual_info_bits::T
    mutual_info_nats::T
    expected_utility::T
    net_welfare::T
    converged::Bool
    iterations::Int
end

"""
    RIIIADiagnostic{T<:AbstractFloat}

Diagnostic comparing base-menu and expanded-menu RI discrete-choice solutions
to quantify violations of Independence of Irrelevant Alternatives (IIA).

# Fields
- `base_solution::RIDiscreteChoiceSolution{T}` — solution on base `J × S` payoffs
- `expanded_solution::RIDiscreteChoiceSolution{T}` — solution on `(J+1) × S` payoffs
- `uncond_odds_ratio_shift::Matrix{T}` — `J × J` matrix of `|(P_exp(j)/P_exp(k)) / (P_base(j)/P_base(k)) - 1|`
- `cond_odds_ratio_max_diff::T` — maximum relative conditional odds-ratio shift across `(j, k, s)`
- `max_iia_violation::T` — overall maximum IIA violation across unconditional and conditional odds
"""
struct RIIIADiagnostic{T<:AbstractFloat}
    base_solution::RIDiscreteChoiceSolution{T}
    expanded_solution::RIDiscreteChoiceSolution{T}
    uncond_odds_ratio_shift::Matrix{T}
    cond_odds_ratio_max_diff::T
    max_iia_violation::T
end

"""
    RILogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel

Econometric Rational Inattention discrete-choice model estimated by maximum
likelihood across `M` choice environments.

# Fields
- `choices::Vector{Int}` — observed discrete choices in `1:J` (length `N`)
- `menu_attributes::Array{T,4}` — `J × S × K × M` attribute array
- `base_payoff::Union{Nothing,Array{T,3}}` — optional `J × S × M` known baseline payoff
- `normalize::Symbol` — scale normalization (`:none`, `:omega_one`, or `:first_coef`)
- `env_id::Vector{Int}` — environment indices in `1:M` (length `N`)
- `state_id::Union{Nothing,Vector{Int}}` — optional realized state indices in `1:S` (length `N`)
- `priors::Matrix{T}` — `S × M` matrix of state priors per environment
- `beta::Vector{T}` — length-`K` preference parameter vector
- `omega::T` — marginal information cost `ω > 0`
- `vcov_mat::Matrix{T}` — `(K + 1) × (K + 1)` covariance matrix for `[β; ω]`
- `solutions::Vector{RIDiscreteChoiceSolution{T}}` — length-`M` equilibrium RI solutions
- `loglik::T` — maximized log-likelihood
- `loglik_mnl::T` — McFadden fixed-intercept conditional logit benchmark log-likelihood
- `lr_vs_mnl::T` — likelihood ratio comparison `2 * (loglik - loglik_mnl)`
- `lr_pval::T` — chi-squared p-value comparing RI vs MNL benchmark
- `aic::T` — Akaike information criterion
- `bic::T` — Bayesian information criterion
- `pseudo_r2::T` — McFadden pseudo-R² relative to empirical-share null
- `cov_type::Symbol` — covariance estimator (`:hessian`, `:sandwich`, or `:robust`)
- `varnames::Vector{String}` — attribute names (length `K`)
- `converged::Bool` — whether MLE optimization converged
- `iterations::Int` — optimizer iteration count
- `n_obs::Int` — number of choice observations `N`
"""
struct RILogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    choices::Vector{Int}
    menu_attributes::Array{T,4}
    base_payoff::Union{Nothing,Array{T,3}}
    normalize::Symbol
    env_id::Vector{Int}
    state_id::Union{Nothing,Vector{Int}}
    priors::Matrix{T}
    beta::Vector{T}
    omega::T
    vcov_mat::Matrix{T}
    solutions::Vector{RIDiscreteChoiceSolution{T}}
    loglik::T
    loglik_mnl::T
    lr_vs_mnl::T
    lr_pval::T
    aic::T
    bic::T
    pseudo_r2::T
    cov_type::Symbol
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    n_obs::Int
end

# =============================================================================
# Internal Solver Helpers (Discrete Choice RI)
# =============================================================================

"""Euclidean projection onto the probability simplex Δ^{n-1} (permutation-equivariant)."""
function _ri_project_simplex(v::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(v)
    u = sort(v; rev=true)
    cssv = zero(T)
    theta = zero(T)
    @inbounds for j in 1:n
        cssv += u[j]
        t = (cssv - one(T)) / T(j)
        if u[j] - t > zero(T)
            theta = t
        end
    end
    out = Vector{T}(undef, n)
    @inbounds for j in 1:n
        out[j] = max(v[j] - theta, zero(T))
    end
    s = sum(out)
    if s > zero(T)
        out ./= s
    else
        fill!(out, one(T) / T(n))
    end
    out
end

"""Evaluate convex potential Φ(P) only, with log-sum-exp fallback when D_s == 0."""
function _ri_eval_phi(P::AbstractVector{T}, W::Matrix{T}, logW::Matrix{T},
                      mu_norm::Vector{T}) where {T<:AbstractFloat}
    J, S = size(W)
    phi = zero(T)
    @inbounds for s in 1:S
        mu_s = mu_norm[s]
        mu_s > zero(T) || continue
        d_s = zero(T)
        for j in 1:J
            d_s += P[j] * W[j, s]
        end
        if d_s > zero(T)
            phi -= mu_s * log(d_s)
        else
            lmax = T(-Inf)
            for j in 1:J
                P[j] > zero(T) && (lmax = max(lmax, log(P[j]) + logW[j, s]))
            end
            isfinite(lmax) || return T(Inf)
            sumexp = zero(T)
            for j in 1:J
                P[j] > zero(T) && (sumexp += exp(log(P[j]) + logW[j, s] - lmax))
            end
            phi -= mu_s * (lmax + log(sumexp))
        end
    end
    phi
end

"""Evaluate Φ(P), denominator vector D, and KKT multipliers ζ(P)."""
function _ri_eval_phi_zeta!(zeta::Vector{T}, D::Vector{T}, P::AbstractVector{T},
                            W::Matrix{T}, logW::Matrix{T},
                            mu_norm::Vector{T}) where {T<:AbstractFloat}
    J, S = size(W)
    fill!(zeta, zero(T))
    phi = zero(T)
    @inbounds for s in 1:S
        mu_s = mu_norm[s]
        if mu_s == zero(T)
            D[s] = zero(T)
            continue
        end
        d_s = zero(T)
        for j in 1:J
            d_s += P[j] * W[j, s]
        end
        if d_s > zero(T)
            D[s] = d_s
            phi -= mu_s * log(d_s)
            w_s = mu_s / d_s
            for j in 1:J
                zeta[j] += w_s * W[j, s]
            end
        else
            lmax = T(-Inf)
            for j in 1:J
                P[j] > zero(T) && (lmax = max(lmax, log(P[j]) + logW[j, s]))
            end
            if !isfinite(lmax)
                D[s] = zero(T)
                return T(Inf)
            end
            sumexp = zero(T)
            for j in 1:J
                P[j] > zero(T) && (sumexp += exp(log(P[j]) + logW[j, s] - lmax))
            end
            logD_s = lmax + log(sumexp)
            D[s] = zero(T)
            phi -= mu_s * logD_s
            for j in 1:J
                zeta[j] += mu_s * exp(min(logW[j, s] - logD_s, T(700)))
            end
        end
    end
    phi
end

"""Solve the discrete-choice RI problem for a fixed marginal cost ω > 0,
collapsing exact duplicate payoff rows so symmetry is preserved bit-for-bit."""
function _solve_ri_omega(U_mat::Matrix{T}, mu_norm::Vector{T}, ω::T,
                         tol::T, max_iter::Int) where {T<:AbstractFloat}
    J, S = size(U_mat)

    if J == 1
        cond_probs = ones(T, 1, S)
        uncond_probs = T[one(T)]
        eu = dot(@view(U_mat[1, :]), mu_norm)
        return RIDiscreteChoiceSolution{T}(
            copy(U_mat), copy(mu_norm), cond_probs, uncond_probs,
            Int[1], T[one(T)], ω, zero(T), zero(T), eu, eu, true, 0
        )
    end

    # Group exact duplicate rows to guarantee permutation symmetry
    group_id = zeros(Int, J)
    groups = Vector{Vector{Int}}()
    rep_rows = Int[]
    @inbounds for j in 1:J
        found = 0
        row_j = @view U_mat[j, :]
        for g in 1:length(rep_rows)
            if row_j == @view(U_mat[rep_rows[g], :])
                found = g
                break
            end
        end
        if found == 0
            push!(rep_rows, j)
            push!(groups, Int[j])
            group_id[j] = length(rep_rows)
        else
            push!(groups[found], j)
            group_id[j] = found
        end
    end

    G = length(rep_rows)
    if G < J
        U_uniq = U_mat[rep_rows, :]
        sol_u = _solve_ri_omega_unique(U_uniq, mu_norm, ω, tol, max_iter)
        cond_probs = Matrix{T}(undef, J, S)
        uncond_probs = Vector{T}(undef, J)
        kkt_multipliers = Vector{T}(undef, J)
        @inbounds for g in 1:G
            grp = groups[g]
            mg = T(length(grp))
            p_g = sol_u.uncond_probs[g] / mg
            z_g = sol_u.kkt_multipliers[g]
            for j in grp
                uncond_probs[j] = p_g
                kkt_multipliers[j] = z_g
                for s in 1:S
                    cond_probs[j, s] = sol_u.cond_probs[g, s] / mg
                end
            end
        end
        consideration_set = findall(>(tol), uncond_probs)
        return RIDiscreteChoiceSolution{T}(
            copy(U_mat), copy(mu_norm), cond_probs, uncond_probs,
            consideration_set, kkt_multipliers, ω,
            sol_u.mutual_info_bits, sol_u.mutual_info_nats,
            sol_u.expected_utility, sol_u.net_welfare,
            sol_u.converged, sol_u.iterations
        )
    end

    _solve_ri_omega_unique(U_mat, mu_norm, ω, tol, max_iter)
end

function _solve_ri_omega_unique(U_mat::Matrix{T}, mu_norm::Vector{T}, ω::T,
                                tol::T, max_iter::Int) where {T<:AbstractFloat}
    J, S = size(U_mat)

    if J == 1
        cond_probs = ones(T, 1, S)
        uncond_probs = T[one(T)]
        eu = dot(@view(U_mat[1, :]), mu_norm)
        return RIDiscreteChoiceSolution{T}(
            copy(U_mat), copy(mu_norm), cond_probs, uncond_probs,
            Int[1], T[one(T)], ω, zero(T), zero(T), eu, eu, true, 0
        )
    end

    # Log-sum-exp shifted weights W_{j,s} = exp((U_{j,s} - m_s) / ω) ∈ [0, 1]
    logW = Matrix{T}(undef, J, S)
    W = Matrix{T}(undef, J, S)
    @inbounds for s in 1:S
        m_s = maximum(@view U_mat[:, s])
        for j in 1:J
            lw = (U_mat[j, s] - m_s) / ω
            logW[j, s] = lw
            W[j, s] = exp(lw)
        end
    end

    tol_eff = max(tol, T(64) * eps(T))
    kkt_tol = min(tol_eff, max(T(1e-12), T(64) * eps(T)))
    kkt_gate = max(tol_eff, T(1e-11))
    prune_thresh = max(T(1e-10), T(100) * eps(T))

    # Initialize at uniform distribution to preserve duplicate-row symmetry
    P = fill(one(T) / T(J), J)
    zeta0 = Vector{T}(undef, J)
    zeta1 = Vector{T}(undef, J)
    zeta_try = Vector{T}(undef, J)
    D = Vector{T}(undef, S)
    P1 = Vector{T}(undef, J)
    P2 = Vector{T}(undef, J)
    P_cand = Vector{T}(undef, J)
    P_next = Vector{T}(undef, J)
    r_vec = Vector{T}(undef, J)
    v_vec = Vector{T}(undef, J)

    total_iters = 0

    # ---- Phase 1: Safeguarded SQUAREM Blahut–Arimoto ----
    ba_max = min(max_iter, 400)
    prev_support = trues(J)
    stable_count = 0

    for it in 1:ba_max
        total_iters += 1
        _ri_eval_phi_zeta!(zeta0, D, P, W, logW, mu_norm)
        @inbounds for j in 1:J
            P1[j] = P[j] * zeta0[j]
        end
        s1 = sum(P1)
        s1 > zero(T) && (P1 ./= s1)

        _ri_eval_phi_zeta!(zeta1, D, P1, W, logW, mu_norm)
        @inbounds for j in 1:J
            P2[j] = P1[j] * zeta1[j]
        end
        s2 = sum(P2)
        s2 > zero(T) && (P2 ./= s2)

        ba_res = zero(T)
        @inbounds for j in 1:J
            ba_res = max(ba_res, abs(P2[j] - P1[j]))
        end

        cur_support = P2 .> prune_thresh
        if cur_support == prev_support
            stable_count += 1
        else
            stable_count = 0
            prev_support = cur_support
        end

        if ba_res < T(1e-8) || (it >= 20 && stable_count >= 15 && ba_res < T(1e-3))
            P .= P2
            break
        end

        @inbounds for j in 1:J
            r_vec[j] = P1[j] - P[j]
            v_vec[j] = (P2[j] - P1[j]) - r_vec[j]
        end
        r2 = dot(r_vec, r_vec)
        v2 = dot(v_vec, v_vec)

        if v2 > T(1e-30) && r2 > T(1e-30)
            alpha = -max(one(T), min(sqrt(r2 / v2), T(64)))
            phi2 = _ri_eval_phi(P2, W, logW, mu_norm)
            accepted = false
            for _ in 1:6
                @inbounds for j in 1:J
                    P_cand[j] = P[j] - (T(2) * alpha) * r_vec[j] + (alpha * alpha) * v_vec[j]
                end
                P_proj = _ri_project_simplex(P_cand)
                phi_cand = _ri_eval_phi(P_proj, W, logW, mu_norm)
                if isfinite(phi_cand) && phi_cand <= phi2
                    _ri_eval_phi_zeta!(zeta_try, D, P_proj, W, logW, mu_norm)
                    @inbounds for j in 1:J
                        P_next[j] = P_proj[j] * zeta_try[j]
                    end
                    sn = sum(P_next)
                    if sn > zero(T)
                        P_next ./= sn
                        P .= P_next
                        accepted = true
                        break
                    end
                end
                alpha = (alpha - one(T)) / T(2)
            end
            accepted || (P .= P2)
        else
            P .= P2
        end
    end

    # ---- Phase 2: Active-Set Symmetric Simplex Newton + KKT Revival ----
    converged = false
    P_try = similar(P)
    P_rev = similar(P)

    for _ in 1:30
        C = findall(>(prune_thresh), P)
        isempty(C) && (C = Int[argmax(P)])
        @inbounds for j in 1:J
            (j in C) || (P[j] = zero(T))
        end
        sC = sum(@view P[C])
        if sC > zero(T)
            P[C] ./= sC
        else
            P[C] .= one(T) / T(length(C))
        end

        # Damped symmetric KKT Newton on active set C
        for _ in 1:30
            length(C) <= 1 && break
            total_iters >= max_iter && break
            total_iters += 1

            phi_cur = _ri_eval_phi_zeta!(zeta0, D, P, W, logW, mu_norm)
            m = length(C)
            kkt_act = maximum(i -> abs(zeta0[C[i]] - one(T)), 1:m)
            kkt_act < kkt_tol && break

            # Fallback to BA if any active state experienced W underflow (D[s] == 0)
            has_zero_D = any(s -> mu_norm[s] > zero(T) && D[s] <= zero(T), 1:S)
            if has_zero_D
                P[C] .*= @view zeta0[C]
                P[C] ./= sum(@view P[C])
                continue
            end

            W_C = @view W[C, :]
            weights_s = Vector{T}(undef, S)
            @inbounds for s in 1:S
                weights_s[s] = mu_norm[s] > zero(T) ? mu_norm[s] / (D[s] * D[s]) : zero(T)
            end
            H_C = W_C * Diagonal(weights_s) * W_C'
            ridge = max(T(1e-11), T(100) * eps(T)) * max(one(T), tr(H_C) / T(m))
            H_reg = Symmetric(H_C + ridge * I)
            L_chol = safe_cholesky(H_reg; silent=true)
            F_chol = Cholesky(L_chol, 'L', 0)

            zeta_C = zeta0[C]
            u_vec = F_chol \ zeta_C
            ones_m = ones(T, m)
            v_one = F_chol \ ones_m
            denom_v = sum(v_one)
            if !isfinite(denom_v) || abs(denom_v) <= T(1e-30)
                P[C] .*= zeta_C
                P[C] ./= sum(@view P[C])
                continue
            end
            nu = sum(u_vec) / denom_v
            d = u_vec .- nu .* v_one
            d .-= mean(d)

            if maximum(abs, d) < max(T(1e-14), T(10) * eps(T))
                break
            end

            dir_deriv = -dot(zeta_C, d)
            if dir_deriv >= zero(T)
                P[C] .*= zeta_C
                P[C] ./= sum(@view P[C])
                continue
            end

            alpha_max = one(T)
            @inbounds for i in 1:m
                if d[i] < zero(T)
                    alpha_max = min(alpha_max, -P[C[i]] / d[i])
                end
            end

            alpha = alpha_max
            step_accepted = false
            for _ in 1:35
                fill!(P_try, zero(T))
                @inbounds for i in 1:m
                    P_try[C[i]] = max(P[C[i]] + alpha * d[i], zero(T))
                end
                s_try = sum(@view P_try[C])
                if s_try > zero(T)
                    P_try[C] ./= s_try
                    phi_try = _ri_eval_phi(P_try, W, logW, mu_norm)
                    if isfinite(phi_try) && phi_try <= phi_cur + T(1e-4) * alpha * dir_deriv
                        P .= P_try
                        step_accepted = true
                        break
                    end
                end
                alpha *= T(0.5)
            end

            if !step_accepted
                P[C] .*= zeta_C
                P[C] ./= sum(@view P[C])
            else
                _ri_eval_phi_zeta!(zeta1, D, P, W, logW, mu_norm)
                drop_thresh = max(T(1e-12), T(100) * eps(T))
                keep = Int[C[i] for i in 1:m if !(P[C[i]] <= drop_thresh && zeta1[C[i]] < one(T) - T(1e-8))]
                if length(keep) < m && !isempty(keep)
                    for j in C
                        (j in keep) || (P[j] = zero(T))
                    end
                    C = keep
                    P[C] ./= sum(@view P[C])
                end
            end
        end

        # Up to 20 BA polishing steps on C
        for _ in 1:20
            _ri_eval_phi_zeta!(zeta0, D, P, W, logW, mu_norm)
            max_act_err = maximum(j -> abs(zeta0[j] - one(T)), C)
            max_act_err < kkt_tol && break
            total_iters >= max_iter && break
            total_iters += 1
            P[C] .*= @view zeta0[C]
            sC2 = sum(@view P[C])
            sC2 > zero(T) && (P[C] ./= sC2)
        end

        _ri_eval_phi_zeta!(zeta0, D, P, W, logW, mu_norm)
        drop_thresh = max(T(1e-12), T(100) * eps(T))
        C_pruned = Int[j for j in C if !(P[j] <= drop_thresh && zeta0[j] < one(T) - T(1e-8))]
        if length(C_pruned) < length(C) && !isempty(C_pruned)
            for j in C
                (j in C_pruned) || (P[j] = zero(T))
            end
            C = C_pruned
            P[C] ./= sum(@view P[C])
            _ri_eval_phi_zeta!(zeta0, D, P, W, logW, mu_norm)
        end

        # Global KKT check over all j = 1..J
        max_act_err = maximum(j -> abs(zeta0[j] - one(T)), C)
        inactive = Int[j for j in 1:J if !(j in C)]
        if isempty(inactive)
            if max_act_err <= kkt_gate
                converged = true
                break
            end
        else
            z_inact_max = maximum(j -> zeta0[j], inactive)
            if z_inact_max <= one(T) + kkt_gate && max_act_err <= kkt_gate
                converged = true
                break
            end
            if z_inact_max > one(T) + kkt_gate
                # Re-inject all inactive coordinates tied at z_inact_max to keep symmetry
                revive = Int[j for j in inactive if zeta0[j] >= z_inact_max - T(1e-13)]
                phi_before = _ri_eval_phi(P, W, logW, mu_norm)
                delta = T(1e-4)
                for _ in 1:25
                    P_rev .= (one(T) - delta) .* P
                    w_rev = delta / T(length(revive))
                    for j in revive
                        P_rev[j] = w_rev
                    end
                    phi_rev = _ri_eval_phi(P_rev, W, logW, mu_norm)
                    if isfinite(phi_rev) && phi_rev < phi_before
                        break
                    end
                    delta *= T(0.5)
                end
                P .= P_rev
            end
        end
        total_iters >= max_iter && break
    end

    # Compute final conditional probabilities P(j | s)
    cond_probs = zeros(T, J, S)
    @inbounds for s in 1:S
        d_s = zero(T)
        for j in 1:J
            P[j] > zero(T) && (d_s += P[j] * W[j, s])
        end
        if d_s > zero(T)
            for j in 1:J
                P[j] > zero(T) && (cond_probs[j, s] = P[j] * W[j, s] / d_s)
            end
        else
            lmax = T(-Inf)
            for j in 1:J
                P[j] > zero(T) && (lmax = max(lmax, log(P[j]) + logW[j, s]))
            end
            sumexp = zero(T)
            for j in 1:J
                P[j] > zero(T) && (sumexp += exp(log(P[j]) + logW[j, s] - lmax))
            end
            logD_s = lmax + log(sumexp)
            for j in 1:J
                P[j] > zero(T) && (cond_probs[j, s] = exp(log(P[j]) + logW[j, s] - logD_s))
            end
        end
        col_sum = sum(@view cond_probs[:, s])
        col_sum > zero(T) && (cond_probs[:, s] ./= col_sum)
    end

    uncond_probs = cond_probs * mu_norm
    @inbounds for j in 1:J
        P[j] == zero(T) && (uncond_probs[j] = zero(T))
    end
    u_sum = sum(uncond_probs)
    u_sum > zero(T) && (uncond_probs ./= u_sum)

    consideration_set = findall(>(tol), uncond_probs)
    _ri_eval_phi_zeta!(zeta0, D, uncond_probs, W, logW, mu_norm)

    if !converged
        act = isempty(consideration_set) ? findall(>(zero(T)), uncond_probs) : consideration_set
        max_act = isempty(act) ? zero(T) : maximum(j -> abs(zeta0[j] - one(T)), act)
        max_all = maximum(zeta0)
        converged = (max_act <= kkt_gate) && (max_all <= one(T) + kkt_gate)
    end

    eu = zero(T)
    mi_nats = zero(T)
    @inbounds for s in 1:S
        mu_s = mu_norm[s]
        mu_s > zero(T) || continue
        for j in 1:J
            p_js = cond_probs[j, s]
            p_js > zero(T) || continue
            eu += mu_s * p_js * U_mat[j, s]
            p_j = uncond_probs[j]
            if p_j > zero(T)
                mi_nats += mu_s * p_js * (log(p_js) - log(p_j))
            end
        end
    end
    mi_nats = max(mi_nats, zero(T))
    mi_bits = mi_nats / log(T(2))
    net_welfare = eu - ω * mi_nats

    RIDiscreteChoiceSolution{T}(
        copy(U_mat), copy(mu_norm), cond_probs, uncond_probs,
        consideration_set, copy(zeta0), ω, mi_bits, mi_nats,
        eu, net_welfare, converged, total_iters
    )
end

"""Brent's 1D root-finder on bracket [a, b] with f(a) * f(b) <= 0."""
function _ri_dc_brent_root(f, a0::T, b0::T, fa0::T, fb0::T;
                        xtol::T=T(1e-12), ftol::T=T(1e-11),
                        maxiter::Int=80) where {T<:AbstractFloat}
    a, b = a0, b0
    fa, fb = fa0, fb0
    c, fc = b, fb
    d = b - a
    e = d
    for _ in 1:maxiter
        if (fb > zero(T) && fc > zero(T)) || (fb < zero(T) && fc < zero(T))
            c = a; fc = fa
            d = b - a; e = d
        end
        if abs(fc) < abs(fb)
            a = b; b = c; c = a
            fa = fb; fb = fc; fc = fa
        end
        tol1 = T(2) * eps(T) * abs(b) + T(0.5) * xtol
        xm = T(0.5) * (c - b)
        if abs(xm) <= tol1 || abs(fb) <= ftol
            return b
        end
        if abs(e) >= tol1 && abs(fa) > abs(fb)
            s = fb / fa
            if a == c
                p = T(2) * xm * s
                q = one(T) - s
            else
                q = fa / fc
                r = fb / fc
                p = s * (T(2) * xm * q * (q - r) - (b - a) * (r - one(T)))
                q = (q - one(T)) * (r - one(T)) * (s - one(T))
            end
            p > zero(T) ? (q = -q) : (p = -p)
            if T(2) * p < min(T(3) * xm * q - abs(tol1 * q), abs(e * q))
                e = d
                d = p / q
            else
                d = xm; e = d
            end
        else
            d = xm; e = d
        end
        a = b; fa = fb
        b += abs(d) > tol1 ? d : (xm >= zero(T) ? tol1 : -tol1)
        fb = f(b)
    end
    b
end

"""
    solve_ri_discrete(U::AbstractMatrix, mu::AbstractVector;
                      omega=nothing, capacity=nothing, units::Symbol=:bits,
                      tol::Real=1e-12, max_iter::Int=5000) -> RIDiscreteChoiceSolution{T}

Solve the discrete-choice Rational Inattention problem (Matějka & McKay 2015;
Caplin, Dean & Leahy 2019) given `J × S` payoff matrix `U` and prior `mu`.

Provide either marginal information cost `omega > 0` (per nat) OR mutual
information constraint `capacity > 0` (in `:bits` or `:nats`, controlled by `units`).

# Algorithm
Combines safeguarded SQUAREM Blahut–Arimoto iterations with an active-set
symmetric KKT Newton solver, Armijo backtracking, and global KKT revival across
all alternatives so that duplicate payoff rows remain symmetric and optimal
consideration sets satisfy `ζ_j = 1` for `j ∈ C` and `ζ_j <= 1` for `j ∉ C`.

# References
- Matějka, F., & McKay, A. (2015). *American Economic Review* 105(1), 272–298.
- Caplin, A., Dean, M., & Leahy, J. (2019). *Review of Economic Studies* 86(3), 1061–1094.
"""
function solve_ri_discrete(U::AbstractMatrix, mu::AbstractVector;
                           omega=nothing, capacity=nothing,
                           units::Symbol=:bits, tol::Real=1e-12,
                           max_iter::Int=5000)
    J, S = size(U)
    (J >= 1 && S >= 1) || throw(ArgumentError("Payoff matrix U must be non-empty, got size ($J, $S)"))
    length(mu) == S || throw(ArgumentError("Prior mu must have length $S (got $(length(mu)))"))
    (omega === nothing) == (capacity === nothing) &&
        throw(ArgumentError("Provide exactly one of `omega` or `capacity`"))
    units in (:bits, :nats) ||
        throw(ArgumentError("units must be :bits or :nats, got :$units"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))

    T = promote_type(float(eltype(U)), float(eltype(mu)))
    T <: AbstractFloat || (T = Float64)
    U_mat = Matrix{T}(U)
    mu_vec = Vector{T}(mu)
    _validate_data(U_mat, "U")
    _validate_data(mu_vec, "mu")

    all(>=(zero(T)), mu_vec) || throw(ArgumentError("Prior mu must be non-negative"))
    mu_sum = sum(mu_vec)
    mu_sum > zero(T) || throw(ArgumentError("Prior mu must have positive sum"))
    mu_norm = mu_vec ./ mu_sum
    tol_T = T(tol)

    if omega !== nothing
        ω_val = T(omega)
        (isfinite(ω_val) && ω_val > zero(T)) ||
            throw(ArgumentError("omega must be finite and positive, got $omega"))
        return _solve_ri_omega(U_mat, mu_norm, ω_val, tol_T, max_iter)
    end

    # Capacity-constrained mode: Brent-solve I_nats(ω) = κ_nats
    cap_val = T(capacity)
    (isfinite(cap_val) && cap_val > zero(T)) ||
        throw(ArgumentError("capacity must be finite and positive, got $capacity"))
    κ_nats = units === :bits ? cap_val * log(T(2)) : cap_val

    J == 1 && throw(ArgumentError("Cannot achieve positive capacity κ_nats=$κ_nats with J=1 alternative (max entropy = 0)"))

    H_mu = zero(T)
    best_opts = Int[]
    @inbounds for s in 1:S
        mu_s = mu_norm[s]
        if mu_s > zero(T)
            H_mu -= mu_s * log(mu_s)
            push!(best_opts, argmax(@view U_mat[:, s]))
        end
    end
    n_distinct = length(unique!(best_opts))
    H_distinct = log(T(n_distinct))
    max_H = min(H_mu, H_distinct)
    κ_nats < max_H || throw(ArgumentError(
        "Requested capacity κ_nats=$(κ_nats) must be strictly less than maximum achievable entropy $(max_H)"
    ))

    payoff_span = maximum(U_mat) - minimum(U_mat)
    payoff_span > zero(T) || throw(ArgumentError(
        "All payoffs are identical; maximum achievable mutual information is 0"
    ))
    scale = max(one(T), payoff_span)

    solve_tol = min(tol_T, max(T(1e-12), T(64) * eps(T)))
    f_cap(ω_try) = _solve_ri_omega(U_mat, mu_norm, ω_try, solve_tol, max_iter).mutual_info_nats - κ_nats

    ω_lo = T(0.05) * scale
    f_lo = f_cap(ω_lo)
    for _ in 1:40
        f_lo > zero(T) && break
        ω_lo *= T(0.25)
        ω_lo < T(1e-12) * scale && break
        f_lo = f_cap(ω_lo)
    end
    f_lo > zero(T) || throw(ArgumentError(
        "Requested capacity κ_nats=$(κ_nats) exceeds maximum achievable mutual information for U and mu"
    ))

    ω_hi = max(scale, ω_lo * T(2))
    f_hi = f_cap(ω_hi)
    for _ in 1:40
        f_hi < zero(T) && break
        ω_hi *= T(2)
        f_hi = f_cap(ω_hi)
    end
    f_hi < zero(T) || throw(ArgumentError(
        "Failed to bracket upper shadow cost ω for requested capacity κ_nats=$(κ_nats)"
    ))

    ω_star = _ri_dc_brent_root(f_cap, ω_lo, ω_hi, f_lo, f_hi;
                            xtol=max(T(1e-12), T(16) * eps(T)) * max(one(T), ω_hi),
                            ftol=max(T(1e-11), T(64) * eps(T)))
    _solve_ri_omega(U_mat, mu_norm, ω_star, tol_T, max_iter)
end

# =============================================================================
# IIA Breakdown Diagnostic
# =============================================================================

"""
    ri_iia_diagnostic(U::AbstractMatrix, mu::AbstractVector,
                      new_option_payoffs::AbstractVector;
                      omega=nothing, capacity=nothing, units::Symbol=:bits,
                      tol::Real=1e-12) -> RIIIADiagnostic{T}

Diagnose violations of Independence of Irrelevant Alternatives (IIA) in
Rational Inattention discrete choice when adding a new alternative with
state-contingent payoffs `new_option_payoffs` (Matějka & McKay 2015, §II).

# References
- Matějka, F., & McKay, A. (2015). *American Economic Review* 105(1), 272–298.
"""
function ri_iia_diagnostic(U::AbstractMatrix, mu::AbstractVector,
                           new_option_payoffs::AbstractVector;
                           omega=nothing, capacity=nothing,
                           units::Symbol=:bits, tol::Real=1e-12)
    J, S = size(U)
    length(new_option_payoffs) == S || throw(ArgumentError(
        "new_option_payoffs must have length $S (got $(length(new_option_payoffs)))"
    ))

    base_sol = solve_ri_discrete(U, mu; omega=omega, capacity=capacity, units=units, tol=tol)
    T = eltype(base_sol.omega)
    new_row = Vector{T}(new_option_payoffs)
    _validate_data(new_row, "new_option_payoffs")

    U_exp = vcat(base_sol.payoffs, reshape(new_row, 1, S))
    exp_sol = solve_ri_discrete(U_exp, base_sol.prior; omega=omega, capacity=capacity, units=units, tol=tol)

    tol_T = T(tol)
    Pb = base_sol.uncond_probs
    Pe = exp_sol.uncond_probs
    Cb = base_sol.cond_probs
    Ce = exp_sol.cond_probs

    uncond_shift = zeros(T, J, J)
    cond_max_diff = zero(T)

    @inbounds for j in 1:J, k in 1:J
        j == k && continue
        if Pb[j] > tol_T && Pb[k] > tol_T
            if Pe[j] > tol_T && Pe[k] > tol_T
                r_base = Pb[j] / Pb[k]
                r_exp = Pe[j] / Pe[k]
                uncond_shift[j, k] = abs(r_exp / r_base - one(T))
            elseif Pe[j] <= tol_T && Pe[k] > tol_T
                uncond_shift[j, k] = one(T)
            elseif Pe[j] > tol_T && Pe[k] <= tol_T
                uncond_shift[j, k] = T(Inf)
            end
        end

        for s in 1:S
            base_sol.prior[s] > zero(T) || continue
            if Cb[j, s] > tol_T && Cb[k, s] > tol_T
                if Ce[j, s] > tol_T && Ce[k, s] > tol_T
                    cr_base = Cb[j, s] / Cb[k, s]
                    cr_exp = Ce[j, s] / Ce[k, s]
                    cond_max_diff = max(cond_max_diff, abs(cr_exp / cr_base - one(T)))
                elseif Ce[j, s] <= tol_T && Ce[k, s] > tol_T
                    cond_max_diff = max(cond_max_diff, one(T))
                elseif Ce[j, s] > tol_T && Ce[k] <= tol_T
                    cond_max_diff = max(cond_max_diff, T(Inf))
                end
            end
        end
    end

    max_uncond = isempty(uncond_shift) ? zero(T) : maximum(uncond_shift)
    max_violation = max(max_uncond, cond_max_diff)

    RIIIADiagnostic{T}(base_sol, exp_sol, uncond_shift, cond_max_diff, max_violation)
end

# =============================================================================
# Differentiable Implicit Solver for ForwardDiff (1st & 2nd Order Duals)
# =============================================================================

_ri_primal(x::Real) = x
_ri_primal(x::ForwardDiff.Dual) = _ri_primal(ForwardDiff.value(x))

"""
Differentiable per-market RI conditional & unconditional probabilities.
For `ForwardDiff.Dual` inputs (1st or 2nd order), solves the exact primal fixed
point in `Float64`, lifts the active support `C` to `Td`, and runs 6 symmetric
KKT Newton steps on `Φ(P_C; U_m, ω)` in `Td` to propagate exact implicit
derivatives cleanly through both gradients and Hessians.
"""
function _ri_market_probs(U_m::AbstractMatrix{T}, mu_m::AbstractVector{Tm},
                          ω::T) where {T<:AbstractFloat,Tm<:AbstractFloat}
    sol = solve_ri_discrete(U_m, mu_m; omega=ω, tol=min(T(1e-12), T(100) * eps(T) + T(1e-13)))
    (sol.cond_probs, sol.uncond_probs)
end

function _ri_market_probs(U_m::AbstractMatrix{Td}, mu_m::AbstractVector{Tm},
                          ω::Td) where {Td<:ForwardDiff.Dual,Tm<:AbstractFloat}
    J, S = size(U_m)
    U_val = Matrix{Float64}(undef, J, S)
    @inbounds for j in 1:J, s in 1:S
        U_val[j, s] = Float64(_ri_primal(U_m[j, s]))
    end
    ω_val = Float64(_ri_primal(ω))
    mu_val = Float64.(mu_m)

    sol_val = solve_ri_discrete(U_val, mu_val; omega=ω_val, tol=1e-13)
    C = sol_val.consideration_set
    isempty(C) && (C = Int[argmax(sol_val.uncond_probs)])
    m_act = length(C)

    if m_act == 1
        P_full = zeros(Td, J)
        P_full[C[1]] = one(Td)
        cond_probs = zeros(Td, J, S)
        @inbounds for s in 1:S
            cond_probs[C[1], s] = one(Td)
        end
        return (cond_probs, P_full)
    end

    P_C = Td.(sol_val.uncond_probs[C])
    P_C ./= sum(P_C)

    W_C = Matrix{Td}(undef, m_act, S)
    @inbounds for s in 1:S
        m_s = Td(maximum(@view U_val[:, s]))
        for i in 1:m_act
            W_C[i, s] = exp((U_m[C[i], s] - m_s) / ω)
        end
    end
    mu_td = Td.(mu_m)

    # 6 symmetric KKT Newton steps on Φ(P_C; U_m, ω) in Dual arithmetic
    for _ in 1:6
        D = vec(P_C' * W_C)
        any(s -> mu_val[s] > 0.0 && _ri_primal(D[s]) <= 1e-300, 1:S) && break
        inv_D = Vector{Td}(undef, S)
        inv_D2 = Vector{Td}(undef, S)
        @inbounds for s in 1:S
            if mu_val[s] > 0.0
                inv_D[s] = mu_td[s] / D[s]
                inv_D2[s] = mu_td[s] / (D[s] * D[s])
            else
                inv_D[s] = zero(Td)
                inv_D2[s] = zero(Td)
            end
        end
        zeta_C = W_C * inv_D
        H_C = W_C * Diagonal(inv_D2) * W_C'
        tr_val = Float64(_ri_primal(tr(H_C)))
        ridge = Td(1e-13 * max(1.0, tr_val / m_act))
        H_reg = H_C + ridge * I
        ones_m = ones(Td, m_act)
        K_mat = vcat(hcat(H_reg, ones_m), hcat(ones_m', zero(Td)))
        rhs = vcat(zeta_C, zero(Td))
        sol_kkt = K_mat \ rhs
        P_C = P_C .+ @view(sol_kkt[1:m_act])
        P_C ./= sum(P_C)
    end

    P_full = zeros(Td, J)
    P_full[C] .= P_C

    D_fin = vec(P_C' * W_C)
    cond_probs = zeros(Td, J, S)
    @inbounds for s in 1:S
        if _ri_primal(D_fin[s]) > 1e-300
            for i in 1:m_act
                cond_probs[C[i], s] = P_C[i] * W_C[i, s] / D_fin[s]
            end
        else
            for i in 1:m_act
                cond_probs[C[i], s] = P_C[i]
            end
        end
    end
    (cond_probs, P_full)
end

# =============================================================================
# McFadden Fixed-Intercept Conditional Logit Benchmark & Empirical Null
# =============================================================================

function _ri_null_loglik(choices::Vector{Int}, J::Int, ::Type{T}) where {T<:AbstractFloat}
    n = length(choices)
    ll = zero(T)
    for j in 1:J
        n_j = count(==(j), choices)
        n_j > 0 && (ll += T(n_j) * log(T(n_j) / T(n)))
    end
    ll
end

"""Fit McFadden fixed-intercept conditional logit benchmark on choice data."""
function _fit_ri_mnl_benchmark(choices::Vector{Int}, menu_attrs::Array{T,4},
                               env_vec::Vector{Int}, state_vec::Union{Nothing,Vector{Int}},
                               priors_mat::Matrix{T},
                               base_payoff::Union{Nothing,Array{T,3}}) where {T<:AbstractFloat}
    J, S, K, M = size(menu_attrs)
    N = length(choices)
    has_base = base_payoff !== nothing
    Jm1 = J - 1
    P_mnl = Jm1 + K + (has_base ? 1 : 0)

    # Precompute observation-specific J × K_eff attribute design matrices
    K_eff = K + (has_base ? 1 : 0)
    X_obs = Array{T,3}(undef, N, J, K_eff)
    @inbounds for i in 1:N
        m = env_vec[i]
        if state_vec !== nothing
            s = state_vec[i]
            for j in 1:J
                for k in 1:K
                    X_obs[i, j, k] = menu_attrs[j, s, k, m]
                end
                if has_base
                    X_obs[i, j, K + 1] = base_payoff[j, s, m]
                end
            end
        else
            for j in 1:J
                for k in 1:K
                    acc = zero(T)
                    for s in 1:S
                        acc += priors_mat[s, m] * menu_attrs[j, s, k, m]
                    end
                    X_obs[i, j, k] = acc
                end
                if has_base
                    acc_b = zero(T)
                    for s in 1:S
                        acc_b += priors_mat[s, m] * base_payoff[j, s, m]
                    end
                    X_obs[i, j, K + 1] = acc_b
                end
            end
        end
    end

    function _mnl_negll(psi::AbstractVector{Tp}) where {Tp}
        alpha = @view psi[1:Jm1]
        gamma = @view psi[J:P_mnl]
        ll = zero(Tp)
        v = Vector{Tp}(undef, J)
        @inbounds for i in 1:N
            vmax = Tp(-Inf)
            for j in 1:J
                val = j == 1 ? zero(Tp) : alpha[j - 1]
                for k in 1:K_eff
                    val += X_obs[i, j, k] * gamma[k]
                end
                v[j] = val
                val > vmax && (vmax = val)
            end
            denom = zero(Tp)
            for j in 1:J
                denom += exp(v[j] - vmax)
            end
            ll += v[choices[i]] - vmax - log(denom)
        end
        -ll
    end

    psi0 = zeros(T, P_mnl)
    g! = (G, p) -> ForwardDiff.gradient!(G, _mnl_negll, p)
    res = Optim.optimize(_mnl_negll, g!, psi0, Optim.LBFGS(),
                         Optim.Options(iterations=500, g_tol=T(1e-9)))
    psi_hat = Optim.minimizer(res)
    loglik_mnl = -T(Optim.minimum(res))
    gamma_hat = K > 0 ? Vector{T}(psi_hat[J:(J + K - 1)]) : zeros(T, 0)
    eta_hat = has_base ? T(psi_hat[P_mnl]) : one(T)
    (loglik_mnl=loglik_mnl, gamma_mnl=gamma_hat, eta_mnl=eta_hat, n_params=P_mnl)
end

# =============================================================================
# Core Optimization Driver for `estimate_ri_logit`
# =============================================================================

struct _RIContribs{T<:AbstractFloat,ST,BT}
    choices::Vector{Int}
    menu_flats::Vector{Matrix{T}}
    base_payoff::BT
    obs_by_env::Vector{Vector{Int}}
    state_vec::ST
    priors_mat::Matrix{T}
    J::Int
    S::Int
    K::Int
    M::Int
    mode::Symbol
end

function (c::_RIContribs{T})(theta::AbstractVector{Td}) where {T,Td}
    N = length(c.choices)
    contribs = Vector{Td}(undef, N)
    p_floor = Td(1e-15)
    J, S, K, M = c.J, c.S, c.K, c.M
    if c.mode === :case_a
        gamma = @view theta[1:K]
        inv_omega = exp(clamp(theta[K + 1], Td(-20), Td(20)))
        B = c.base_payoff
        @inbounds for m in 1:M
            U_eff = K > 0 ? reshape(c.menu_flats[m] * gamma, J, S) : zeros(Td, J, S)
            for j in 1:J, s in 1:S
                U_eff[j, s] += inv_omega * B[j, s, m]
            end
            cond_m, uncond_m = _ri_market_probs(U_eff, @view(c.priors_mat[:, m]), one(Td))
            for i in c.obs_by_env[m]
                p_i = c.state_vec === nothing ? uncond_m[c.choices[i]] : cond_m[c.choices[i], c.state_vec[i]]
                contribs[i] = log(max(p_i, p_floor))
            end
        end
    elseif c.mode === :omega_one
        beta = @view theta[1:K]
        @inbounds for m in 1:M
            U_m = reshape(c.menu_flats[m] * beta, J, S)
            cond_m, uncond_m = _ri_market_probs(U_m, @view(c.priors_mat[:, m]), one(Td))
            for i in c.obs_by_env[m]
                p_i = c.state_vec === nothing ? uncond_m[c.choices[i]] : cond_m[c.choices[i], c.state_vec[i]]
                contribs[i] = log(max(p_i, p_floor))
            end
        end
    else
        beta = @view theta[1:K]
        omega = exp(clamp(theta[K + 1], Td(-20), Td(20)))
        @inbounds for m in 1:M
            U_m = reshape(c.menu_flats[m] * beta, J, S)
            cond_m, uncond_m = _ri_market_probs(U_m, @view(c.priors_mat[:, m]), omega)
            for i in c.obs_by_env[m]
                p_i = c.state_vec === nothing ? uncond_m[c.choices[i]] : cond_m[c.choices[i], c.state_vec[i]]
                contribs[i] = log(max(p_i, p_floor))
            end
        end
    end
    contribs
end

struct _RINegLogLik{C}
    contribs::C
end
(f::_RINegLogLik)(theta) = -sum(f.contribs(theta))

struct _RINegLogLikGrad!{F}
    negll::F
end
(g::_RINegLogLikGrad!)(G, theta) = ForwardDiff.gradient!(G, g.negll, theta)

"""Run multi-start L-BFGS optimization on `negll` and compute vcov."""
function _ri_optimize_and_vcov(contribs_fn::_RIContribs{T}, starts::Vector{Vector{T}},
                               cov_type::Symbol, tol::T, max_iter::Int) where {T<:AbstractFloat}
    negll = _RINegLogLik(contribs_fn)
    g! = _RINegLogLikGrad!(negll)
    g_tol_eff = max(tol, T(1e-7))
    opts = Optim.Options(iterations=max_iter, g_tol=g_tol_eff)

    best_theta = nothing
    best_nll = T(Inf)
    best_conv = false
    best_iters = 0

    for th0 in starts
        val0 = try
            negll(th0)
        catch
            T(Inf)
        end
        isfinite(val0) || continue

        res = Optim.optimize(negll, g!, th0, Optim.LBFGS(), opts)
        cur_th = Vector{T}(Optim.minimizer(res))
        cur_nll = T(Optim.minimum(res))
        cur_conv = Optim.converged(res)
        cur_iters = Optim.iterations(res)

        if !cur_conv && isfinite(cur_nll)
            res_bfgs = Optim.optimize(negll, g!, cur_th, Optim.BFGS(), opts)
            cur_iters += Optim.iterations(res_bfgs)
            if Optim.minimum(res_bfgs) <= cur_nll
                cur_th = Vector{T}(Optim.minimizer(res_bfgs))
                cur_nll = T(Optim.minimum(res_bfgs))
                cur_conv = Optim.converged(res_bfgs)
            end
        end

        # Derivative-free Nelder-Mead check/polish when L-BFGS/BFGS halts at a
        # consideration-set boundary kink (piecewise-smooth V-cusp where g jumps sign)
        if !cur_conv && isfinite(cur_nll)
            res_nm = Optim.optimize(negll, cur_th, Optim.NelderMead(),
                                    Optim.Options(iterations=300,
                                                  f_abstol=max(tol, T(1e-9)),
                                                  x_abstol=max(tol, T(1e-7))))
            cur_iters += Optim.iterations(res_nm)
            if Optim.minimum(res_nm) <= cur_nll + T(1e-12)
                nm_th = Vector{T}(Optim.minimizer(res_nm))
                nm_nll = T(Optim.minimum(res_nm))
                if nm_nll < cur_nll - T(1e-6)
                    # Moved off kink into smoother region; do quick L-BFGS polish
                    res2 = Optim.optimize(negll, g!, nm_th, Optim.LBFGS(), opts)
                    cur_iters += Optim.iterations(res2)
                    if Optim.minimum(res2) <= nm_nll
                        cur_th = Vector{T}(Optim.minimizer(res2))
                        cur_nll = T(Optim.minimum(res2))
                        cur_conv = Optim.converged(res2) || Optim.converged(res_nm)
                    else
                        cur_th = nm_th
                        cur_nll = nm_nll
                        cur_conv = Optim.converged(res_nm)
                    end
                else
                    cur_th = nm_th
                    cur_nll = nm_nll
                    cur_conv = Optim.converged(res_nm)
                end
            end
        end

        if isfinite(cur_nll) && (
            best_theta === nothing ||
            (cur_nll < best_nll - T(1e-8)) ||
            (abs(cur_nll - best_nll) <= T(1e-8) && cur_conv && !best_conv)
        )
            best_theta = cur_th
            best_nll = cur_nll
            best_conv = cur_conv
            best_iters = cur_iters
        end
    end

    best_theta === nothing && error("RI logit optimization failed from all starting points")
    theta_hat = best_theta

    H_raw = ForwardDiff.hessian(negll, theta_hat)
    H_sym = Matrix{T}(T(0.5) * (H_raw + H_raw'))
    H_inv = Matrix{T}(robust_inv(Symmetric(H_sym); silent=true))

    V_theta = if cov_type === :hessian
        Matrix{T}(T(0.5) * (H_inv + H_inv'))
    else
        S_mat = ForwardDiff.jacobian(contribs_fn, theta_hat)
        J_OPG = S_mat' * S_mat
        V_sw = H_inv * J_OPG * H_inv
        Matrix{T}(T(0.5) * (V_sw + V_sw'))
    end

    (theta_hat=theta_hat, loglik=-best_nll, V_theta=V_theta,
     converged=best_conv, iterations=best_iters)
end

function _ri_build_contribs_fn(choices::Vector{Int},
                               menu_flats::Vector{Matrix{T}},
                               base_payoff::Union{Nothing,Array{T,3}},
                               obs_by_env::Vector{Vector{Int}},
                               state_vec::Union{Nothing,Vector{Int}},
                               priors_mat::Matrix{T},
                               J::Int, S::Int, K::Int, M::Int,
                               mode::Symbol) where {T<:AbstractFloat}
    _RIContribs{T,typeof(state_vec),typeof(base_payoff)}(
        choices, menu_flats, base_payoff, obs_by_env, state_vec, priors_mat,
        J, S, K, M, mode
    )
end

"""
    estimate_ri_logit(choices, menu_attributes, env_id;
                      state_id=nothing, priors=nothing,
                      base_payoff=nothing,
                      normalize=(base_payoff === nothing ? :omega_one : :none),
                      cov_type=:hessian, varnames=nothing,
                      tol=1e-10, max_iter=400) -> RILogitModel{T}

Estimate a Rational Inattention (RI) discrete-choice model (Matějka & McKay 2015)
across `M` environments by maximum likelihood using `Optim.LBFGS` with
`ForwardDiff` implicit differentiation through `solve_ri_discrete`.

# Scale Identification (`base_payoff` and `normalize`)
Without a known payoff numeraire, `P^{(m)}(j | s)` depends only on `U / ω = X (β / ω)`,
so multiplying `(β, ω)` by any scalar `α > 0` leaves the likelihood invariant.
Four modes are supported:
- **Case A (`base_payoff !== nothing`, `normalize = :none`, default when `base_payoff` given)**:
  Payoffs are `U^{(m)}_{j,s} = base_payoff[j,s,m] + Σ_k menu_attributes[j,s,k,m] β_k`.
  Separately identifies both `β` (`K` parameters) and `ω` (`dof = K + 1`) via `(γ = β/ω, log(1/ω))`
  and delta-method covariance transformation.
- **Case B (`base_payoff === nothing`, `normalize = :first_coef`)**:
  Fixes `β[1] = 1` (`K >= 1`), estimating `β[2:K]` and `ω` (`dof = K`), with zero row/col 1 in `vcov_mat`.
- **Case C (`base_payoff === nothing`, `normalize = :omega_one`, default when `base_payoff === nothing`)**:
  Fixes `ω = 1.0`, estimating `β = β_true / ω_true` (`dof = K`), with zero row/col `K + 1` in `vcov_mat`.
- **Unidentified fallback (`base_payoff === nothing`, `normalize = :none`)**:
  Optimizes `[β; log(ω)]` (`dof = K + 1`) with `robust_inv` (`pinv`) covariance; only `β / ω`
  is point-identified because `(α β, α ω)` yields identical choice probabilities.

# 3D `base_payoff`-only convenience methods (`K = 0`)
- `estimate_ri_logit(choices, base_payoff::AbstractArray{<:Real,3}, env_id; kwargs...)`
- `estimate_ri_logit(choices, env_id, base_payoff::AbstractArray{<:Real,3}; kwargs...)`
- `estimate_ri_logit(choices, base_payoff::AbstractArray{<:Real,3}; kwargs...)` (when `M = 1`)

# References
- Matějka, F., & McKay, A. (2015). *American Economic Review* 105(1), 272–298.
- Caplin, A., Dean, M., & Leahy, J. (2019). *Review of Economic Studies* 86(3), 1061–1094.
"""
function estimate_ri_logit(choices::AbstractVector{<:Integer},
                           menu_attributes::AbstractArray{<:Real,4},
                           env_id::AbstractVector{<:Integer};
                           state_id::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                           priors::Union{Nothing,Symbol,AbstractMatrix}=nothing,
                           base_payoff::Union{Nothing,AbstractArray{<:Real,3}}=nothing,
                           normalize::Symbol=(base_payoff === nothing ? :omega_one : :none),
                           cov_type::Symbol=:hessian,
                           varnames::Union{Nothing,Vector{String}}=nothing,
                           tol::Real=1e-10,
                           max_iter::Int=400)
    J, S, K, M = size(menu_attributes)
    J >= 2 || throw(ArgumentError("Need at least 2 alternatives (got J=$J)"))
    (S >= 1 && K >= 0 && M >= 1) ||
        throw(ArgumentError("Invalid dimensions (J=$J, S=$S, K=$K, M=$M)"))

    N = length(choices)
    N >= 1 || throw(ArgumentError("choices must be non-empty"))
    length(env_id) == N || throw(ArgumentError("env_id must have length $N (got $(length(env_id)))"))

    choices_vec = Vector{Int}(choices)
    env_vec = Vector{Int}(env_id)
    all(c -> 1 <= c <= J, choices_vec) || throw(ArgumentError("choices must be in 1:$J"))
    all(m -> 1 <= m <= M, env_vec) || throw(ArgumentError("env_id must be in 1:$M"))

    state_vec = state_id === nothing ? nothing : Vector{Int}(state_id)
    if state_vec !== nothing
        length(state_vec) == N || throw(ArgumentError("state_id must have length $N (got $(length(state_vec)))"))
        all(s -> 1 <= s <= S, state_vec) || throw(ArgumentError("state_id must be in 1:$S"))
    end

    normalize in (:none, :omega_one, :first_coef) ||
        throw(ArgumentError("normalize must be :none, :omega_one, or :first_coef; got :$normalize"))
    cov_type in (:hessian, :sandwich, :robust) ||
        throw(ArgumentError("cov_type must be :hessian, :sandwich, or :robust; got :$cov_type"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))

    if base_payoff !== nothing
        size(base_payoff) == (J, S, M) || throw(ArgumentError(
            "base_payoff must have size ($J, $S, $M), got $(size(base_payoff))"
        ))
        normalize === :none || throw(ArgumentError(
            "normalize must be :none when base_payoff is provided (got :$normalize)"
        ))
    else
        K >= 1 || throw(ArgumentError("Need K >= 1 menu attributes when base_payoff is nothing (got K=$K)"))
    end

    vn = varnames === nothing ? ["x$k" for k in 1:K] : copy(varnames)
    length(vn) == K || throw(ArgumentError("varnames must have length $K (got $(length(vn)))"))

    # Float type promotion
    T = float(eltype(menu_attributes))
    base_payoff !== nothing && (T = promote_type(T, float(eltype(base_payoff))))
    priors isa AbstractMatrix && (T = promote_type(T, float(eltype(priors))))
    T <: AbstractFloat || (T = Float64)

    menu_attrs = Array{T,4}(menu_attributes)
    K > 0 && _validate_data(reshape(menu_attrs, J * S * M, K), "menu_attributes")
    base_arr = base_payoff === nothing ? nothing : Array{T,3}(base_payoff)
    if base_arr !== nothing
        _validate_data(reshape(base_arr, J * S, M), "base_payoff")
    end
    tol_T = T(tol)

    # Build S × M priors matrix
    priors_mat = Matrix{T}(undef, S, M)
    if priors === nothing
        fill!(priors_mat, one(T) / T(S))
    elseif priors isa Symbol
        priors === :empirical || throw(ArgumentError("Symbol priors must be :empirical, got :$priors"))
        state_vec === nothing && throw(ArgumentError("priors=:empirical requires state_id !== nothing"))
        for m in 1:M
            idx = findall(==(m), env_vec)
            n_m = length(idx)
            for s in 1:S
                c_s = count(i -> state_vec[i] == s, idx)
                priors_mat[s, m] = (T(c_s) + one(T) / T(S)) / (T(n_m) + one(T))
            end
            priors_mat[:, m] ./= sum(@view priors_mat[:, m])
        end
    else
        size(priors) == (S, M) || throw(ArgumentError("priors matrix must have size ($S, $M), got $(size(priors))"))
        pm = Matrix{T}(priors)
        _validate_data(pm, "priors")
        all(>=(zero(T)), pm) || throw(ArgumentError("priors entries must be non-negative"))
        for m in 1:M
            csum = sum(@view pm[:, m])
            csum > zero(T) || throw(ArgumentError("Each column of priors must have positive sum"))
            pm[:, m] ./= csum
        end
        priors_mat .= pm
    end

    # ---- Case B delegation: normalize == :first_coef (fixes β[1] = 1) ----
    if base_arr === nothing && normalize === :first_coef
        base_sub = menu_attrs[:, :, 1, :]
        menu_sub = menu_attrs[:, :, 2:K, :]
        vn_sub = K > 1 ? vn[2:K] : String[]
        m_sub = estimate_ri_logit(
            choices_vec, menu_sub, env_vec;
            state_id=state_vec, priors=priors_mat,
            base_payoff=base_sub, normalize=:none,
            cov_type=cov_type, varnames=vn_sub,
            tol=tol, max_iter=max_iter
        )
        beta_full = vcat(one(T), m_sub.beta)
        vcov_full = zeros(T, K + 1, K + 1)
        vcov_full[2:(K + 1), 2:(K + 1)] .= m_sub.vcov_mat

        # Recompute McFadden benchmark and information criteria on full K with dof = K
        mnl = _fit_ri_mnl_benchmark(choices_vec, menu_attrs, env_vec, state_vec, priors_mat, nothing)
        ll_null = _ri_null_loglik(choices_vec, J, T)
        pseudo_r2 = ll_null == zero(T) ? zero(T) : one(T) - m_sub.loglik / ll_null
        npar = K
        aic_val = -T(2) * m_sub.loglik + T(2) * T(npar)
        bic_val = -T(2) * m_sub.loglik + log(T(N)) * T(npar)
        lr_stat = T(2) * (m_sub.loglik - mnl.loglik_mnl)
        df_lr = max(1, abs(mnl.n_params - npar))
        lr_pval = lr_stat > zero(T) ? T(ccdf(Chisq(df_lr), lr_stat)) : one(T)

        return RILogitModel{T}(
            choices_vec, menu_attrs, nothing, :first_coef,
            env_vec, state_vec, priors_mat,
            beta_full, m_sub.omega, vcov_full, m_sub.solutions,
            m_sub.loglik, mnl.loglik_mnl, lr_stat, lr_pval,
            aic_val, bic_val, pseudo_r2, cov_type, vn,
            m_sub.converged, m_sub.iterations, N
        )
    end

    # Pre-split observations and flattened attributes by environment
    obs_by_env = [findall(==(m), env_vec) for m in 1:M]
    menu_flats = [reshape(menu_attrs[:, :, :, m], J * S, K) for m in 1:M]

    # McFadden fixed-intercept MNL benchmark + warm start
    mnl = _fit_ri_mnl_benchmark(choices_vec, menu_attrs, env_vec, state_vec, priors_mat, base_arr)

    if base_arr !== nothing
        # ---- Case A: base_payoff given, normalize == :none ----
        contribs_a = _ri_build_contribs_fn(choices_vec, menu_flats, base_arr, obs_by_env,
                                           state_vec, priors_mat, J, S, K, M, :case_a)
        eta0 = clamp(mnl.eta_mnl > T(1e-3) ? mnl.eta_mnl : one(T), T(0.05), T(20))
        starts_a = Vector{T}[
            vcat(mnl.gamma_mnl, log(eta0)),
            vcat(mnl.gamma_mnl, zero(T)),
            vcat(T(0.5) .* mnl.gamma_mnl, log(T(2))),
            vcat(zeros(T, K), log(T(0.5))),
        ]
        res_opt = _ri_optimize_and_vcov(contribs_a, starts_a, cov_type, tol_T, max_iter)
        th = res_opt.theta_hat
        gamma_hat = K > 0 ? Vector{T}(th[1:K]) : zeros(T, 0)
        inv_omega_hat = exp(clamp(th[K + 1], T(-20), T(20)))
        omega_hat = one(T) / inv_omega_hat
        beta_hat = omega_hat .* gamma_hat

        # Delta Jacobian ∂(β, ω)/∂(γ, log(1/ω)) = [ω I_K  -β; 0  -ω]
        J_trans = zeros(T, K + 1, K + 1)
        @inbounds for k in 1:K
            J_trans[k, k] = omega_hat
            J_trans[k, K + 1] = -beta_hat[k]
        end
        J_trans[K + 1, K + 1] = -omega_hat
        vcov_mat = Matrix{T}(Symmetric(J_trans * res_opt.V_theta * J_trans'))
        npar = K + 1
    elseif normalize === :omega_one
        # ---- Case C: normalize == :omega_one (ω = 1 fixed) ----
        contribs_c = _ri_build_contribs_fn(choices_vec, menu_flats, nothing, obs_by_env,
                                           state_vec, priors_mat, J, S, K, M, :omega_one)
        starts_c = Vector{T}[
            copy(mnl.gamma_mnl),
            T(0.5) .* mnl.gamma_mnl,
            T(1.5) .* mnl.gamma_mnl,
            zeros(T, K),
        ]
        res_opt = _ri_optimize_and_vcov(contribs_c, starts_c, cov_type, tol_T, max_iter)
        beta_hat = Vector{T}(res_opt.theta_hat)
        omega_hat = one(T)
        vcov_mat = zeros(T, K + 1, K + 1)
        vcov_mat[1:K, 1:K] .= res_opt.V_theta
        npar = K
    else
        # ---- Unidentified fallback: base_payoff === nothing && normalize === :none ----
        contribs_u = _ri_build_contribs_fn(choices_vec, menu_flats, nothing, obs_by_env,
                                           state_vec, priors_mat, J, S, K, M, :unidentified)
        starts_u = Vector{T}[
            vcat(mnl.gamma_mnl, zero(T)),
            vcat(T(0.5) .* mnl.gamma_mnl, log(T(0.5))),
            vcat(T(2) .* mnl.gamma_mnl, log(T(2))),
        ]
        res_opt = _ri_optimize_and_vcov(contribs_u, starts_u, cov_type, tol_T, max_iter)
        th = res_opt.theta_hat
        beta_hat = Vector{T}(th[1:K])
        omega_hat = exp(clamp(th[K + 1], T(-20), T(20)))
        D_jac = Diagonal(vcat(ones(T, K), omega_hat))
        vcov_mat = Matrix{T}(Symmetric(D_jac * res_opt.V_theta * D_jac))
        npar = K + 1
    end

    # Compute final equilibrium solutions per market
    solutions = Vector{RIDiscreteChoiceSolution{T}}(undef, M)
    ll_exact = zero(T)
    for m in 1:M
        U_m = K > 0 ? reshape(menu_flats[m] * beta_hat, J, S) : zeros(T, J, S)
        if base_arr !== nothing
            @inbounds for j in 1:J, s in 1:S
                U_m[j, s] += base_arr[j, s, m]
            end
        end
        sol_m = solve_ri_discrete(U_m, @view(priors_mat[:, m]); omega=omega_hat, tol=T(1e-12))
        solutions[m] = sol_m
        for i in obs_by_env[m]
            p_i = state_vec === nothing ? sol_m.uncond_probs[choices_vec[i]] :
                                          sol_m.cond_probs[choices_vec[i], state_vec[i]]
            ll_exact += log(max(p_i, T(1e-15)))
        end
    end

    ll_null = _ri_null_loglik(choices_vec, J, T)
    pseudo_r2 = ll_null == zero(T) ? zero(T) : one(T) - ll_exact / ll_null
    aic_val = -T(2) * ll_exact + T(2) * T(npar)
    bic_val = -T(2) * ll_exact + log(T(N)) * T(npar)
    lr_stat = T(2) * (ll_exact - mnl.loglik_mnl)
    df_lr = max(1, abs(mnl.n_params - npar))
    lr_pval = lr_stat > zero(T) ? T(ccdf(Chisq(df_lr), lr_stat)) : one(T)

    RILogitModel{T}(
        choices_vec, menu_attrs, base_arr, normalize,
        env_vec, state_vec, priors_mat,
        beta_hat, omega_hat, vcov_mat, solutions,
        ll_exact, mnl.loglik_mnl, lr_stat, lr_pval,
        aic_val, bic_val, pseudo_r2, cov_type, vn,
        res_opt.converged, res_opt.iterations, N
    )
end

# 3D base_payoff-only convenience methods (K = 0)
function estimate_ri_logit(choices::AbstractVector{<:Integer},
                           base_payoff::AbstractArray{<:Real,3},
                           env_id::AbstractVector{<:Integer};
                           kwargs...)
    J, S, M = size(base_payoff)
    Tf = float(eltype(base_payoff))
    Tf <: AbstractFloat || (Tf = Float64)
    menu_zero = zeros(Tf, J, S, 0, M)
    estimate_ri_logit(choices, menu_zero, env_id;
                      base_payoff=base_payoff, normalize=:none, kwargs...)
end

function estimate_ri_logit(choices::AbstractVector{<:Integer},
                           env_id::AbstractVector{<:Integer},
                           base_payoff::AbstractArray{<:Real,3};
                           kwargs...)
    estimate_ri_logit(choices, base_payoff, env_id; kwargs...)
end

function estimate_ri_logit(choices::AbstractVector{<:Integer},
                           base_payoff::AbstractArray{<:Real,3};
                           kwargs...)
    M = size(base_payoff, 3)
    M == 1 || throw(ArgumentError("env_id must be provided when base_payoff has M=$M > 1 environments"))
    estimate_ri_logit(choices, base_payoff, ones(Int, length(choices)); kwargs...)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(m::RILogitModel) = vcat(m.beta, m.omega)
StatsAPI.vcov(m::RILogitModel) = m.vcov_mat
StatsAPI.stderror(m::RILogitModel{T}) where {T} = sqrt.(max.(diag(m.vcov_mat), zero(T)))
StatsAPI.loglikelihood(m::RILogitModel) = m.loglik
StatsAPI.nobs(m::RILogitModel) = m.n_obs
StatsAPI.dof(m::RILogitModel) = m.normalize === :none ? length(m.beta) + 1 : length(m.beta)
StatsAPI.dof_residual(m::RILogitModel) = m.n_obs - dof(m)
StatsAPI.aic(m::RILogitModel) = m.aic
StatsAPI.bic(m::RILogitModel) = m.bic
StatsAPI.r2(m::RILogitModel) = m.pseudo_r2
StatsAPI.islinear(::RILogitModel) = false

function StatsAPI.confint(m::RILogitModel{T}; level::Real=DEFAULT_CI_MICRO) where {T}
    lv = _validate_conf_level(level)
    se = stderror(m)
    theta = coef(m)
    crit = T(_conf_z(lv))
    hcat(theta .- crit .* se, theta .+ crit .* se)
end

"""
    StatsAPI.fitted(m::RILogitModel{T}) -> Matrix{T}

Return the `N × J` matrix of fitted choice probabilities (`P(j | s_i)` when
`state_id` was observed, or `P(j)` otherwise).
"""
function StatsAPI.fitted(m::RILogitModel{T}) where {T}
    N = m.n_obs
    J = size(m.menu_attributes, 1)
    P = Matrix{T}(undef, N, J)
    if m.state_id !== nothing
        @inbounds for i in 1:N
            sol = m.solutions[m.env_id[i]]
            s = m.state_id[i]
            for j in 1:J
                P[i, j] = sol.cond_probs[j, s]
            end
        end
    else
        @inbounds for i in 1:N
            sol = m.solutions[m.env_id[i]]
            for j in 1:J
                P[i, j] = sol.uncond_probs[j]
            end
        end
    end
    P
end

"""
    StatsAPI.predict(m::RILogitModel{T}; kind=:auto) -> Matrix{T}

Return the `N × J` matrix of predicted choice probabilities.
`kind` selects `:auto` (conditional if `state_id` observed, else unconditional),
`:conditional` (requires `state_id !== nothing`), or `:unconditional`.
"""
function StatsAPI.predict(m::RILogitModel{T}; kind::Symbol=:auto) where {T}
    kind in (:auto, :conditional, :unconditional) ||
        throw(ArgumentError("kind must be :auto, :conditional, or :unconditional; got :$kind"))
    if kind === :auto
        return fitted(m)
    elseif kind === :conditional
        m.state_id === nothing && throw(ArgumentError("Conditional prediction requires state_id !== nothing"))
        return fitted(m)
    else
        N = m.n_obs
        J = size(m.menu_attributes, 1)
        P = Matrix{T}(undef, N, J)
        @inbounds for i in 1:N
            sol = m.solutions[m.env_id[i]]
            for j in 1:J
                P[i, j] = sol.uncond_probs[j]
            end
        end
        return P
    end
end

"""
    StatsAPI.predict(m::RILogitModel{T}, menu_attrs_new::AbstractArray{<:Real,4},
                     env_id_new::AbstractVector{<:Integer};
                     state_id=nothing, priors=nothing, base_payoff=nothing) -> Matrix{T}

Predict `N_new × J` choice probabilities for out-of-sample environments.
"""
function StatsAPI.predict(m::RILogitModel{T},
                          menu_attrs_new::AbstractArray{<:Real,4},
                          env_id_new::AbstractVector{<:Integer};
                          state_id::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          priors::Union{Nothing,AbstractMatrix}=nothing,
                          base_payoff::Union{Nothing,AbstractArray{<:Real,3}}=nothing) where {T}
    J, S, K, M_new = size(menu_attrs_new)
    K == length(m.beta) || throw(ArgumentError("Expected K=$(length(m.beta)) attributes, got $K"))
    N_new = length(env_id_new)
    all(e -> 1 <= e <= M_new, env_id_new) || throw(ArgumentError("env_id_new must be in 1:$M_new"))
    if state_id !== nothing
        length(state_id) == N_new || throw(ArgumentError("state_id must have length $N_new"))
        all(s -> 1 <= s <= S, state_id) || throw(ArgumentError("state_id must be in 1:$S"))
    end
    priors_new = priors === nothing ? fill(one(T) / T(S), S, M_new) : Matrix{T}(priors)
    size(priors_new) == (S, M_new) || throw(ArgumentError("priors must have size ($S, $M_new)"))
    base_new = base_payoff === nothing ? nothing : Array{T,3}(base_payoff)
    if base_new !== nothing
        size(base_new) == (J, S, M_new) || throw(ArgumentError("base_payoff must have size ($J, $S, $M_new)"))
    end

    sols = Vector{RIDiscreteChoiceSolution{T}}(undef, M_new)
    attrs = Array{T,4}(menu_attrs_new)
    for m_idx in 1:M_new
        U_m = K > 0 ? reshape(reshape(attrs[:, :, :, m_idx], J * S, K) * m.beta, J, S) : zeros(T, J, S)
        if base_new !== nothing
            U_m .+= @view(base_new[:, :, m_idx])
        end
        sols[m_idx] = solve_ri_discrete(U_m, @view(priors_new[:, m_idx]); omega=m.omega)
    end

    P = Matrix{T}(undef, N_new, J)
    @inbounds for i in 1:N_new
        sol = sols[env_id_new[i]]
        if state_id !== nothing
            s = state_id[i]
            for j in 1:J
                P[i, j] = sol.cond_probs[j, s]
            end
        else
            for j in 1:J
                P[i, j] = sol.uncond_probs[j]
            end
        end
    end
    P
end

"""
    StatsAPI.predict(m::RILogitModel{T}, menu_attrs_new::AbstractArray{<:Real,3};
                     mu=nothing, base_payoff=nothing, kind=:conditional)

Predict choice probabilities for a single new `J × S × K` market (`J × S` matrix if
`kind === :conditional`, or length-`J` vector if `kind === :unconditional`).
"""
function StatsAPI.predict(m::RILogitModel{T},
                          menu_attrs_new::AbstractArray{<:Real,3};
                          mu::Union{Nothing,AbstractVector}=nothing,
                          base_payoff::Union{Nothing,AbstractMatrix}=nothing,
                          kind::Symbol=:conditional) where {T}
    J, S, K = size(menu_attrs_new)
    K == length(m.beta) || throw(ArgumentError("Expected K=$(length(m.beta)) attributes, got $K"))
    kind in (:conditional, :unconditional) ||
        throw(ArgumentError("kind must be :conditional or :unconditional; got :$kind"))
    mu_vec = mu === nothing ? fill(one(T) / T(S), S) : Vector{T}(mu)
    U_m = K > 0 ? reshape(reshape(Array{T,3}(menu_attrs_new), J * S, K) * m.beta, J, S) : zeros(T, J, S)
    if base_payoff !== nothing
        size(base_payoff) == (J, S) || throw(ArgumentError("base_payoff must have size ($J, $S)"))
        U_m .+= Matrix{T}(base_payoff)
    end
    sol = solve_ri_discrete(U_m, mu_vec; omega=m.omega)
    kind === :conditional ? sol.cond_probs : sol.uncond_probs
end

# =============================================================================
# Display & Report
# =============================================================================

function Base.show(io::IO, sol::RIDiscreteChoiceSolution{T}) where {T}
    J, S = size(sol.payoffs)
    cset_str = isempty(sol.consideration_set) ? "∅" : "{" * join(sol.consideration_set, ", ") * "}"
    spec = Any[
        "Alternatives (J)"   J;
        "States (S)"         S;
        "Shadow cost ω"      _fmt(sol.omega);
        "Consideration set"  cset_str;
        "Expected utility"   _fmt(sol.expected_utility);
        "Mutual info (bits)" _fmt(sol.mutual_info_bits);
        "Mutual info (nats)" _fmt(sol.mutual_info_nats);
        "Net welfare"        _fmt(sol.net_welfare);
        "Converged"          _yesno(sol.converged);
        "Iterations"         sol.iterations
    ]
    _pretty_table(io, spec;
        title="Rational Inattention Discrete-Choice Solution",
        column_labels=["Quantity", "Value"],
        alignment=[:l, :r],
    )

    alt_data = Matrix{Any}(undef, J, 4)
    cset = Set(sol.consideration_set)
    for j in 1:J
        alt_data[j, 1] = "Alt $j"
        alt_data[j, 2] = _fmt(sol.uncond_probs[j])
        alt_data[j, 3] = _fmt(sol.kkt_multipliers[j])
        alt_data[j, 4] = (j in cset) ? "Yes" : "No"
    end
    _pretty_table(io, alt_data;
        title="Choice Probabilities & KKT Certificate",
        column_labels=["Alternative", "Uncond. P(j)", "KKT ζ_j", "Active"],
        alignment=[:l, :r, :r, :c],
    )
end

report(sol::RIDiscreteChoiceSolution) = show(stdout, sol)
report(io::IO, sol::RIDiscreteChoiceSolution) = show(io, sol)

function Base.show(io::IO, d::RIIIADiagnostic{T}) where {T}
    J = size(d.base_solution.payoffs, 1)
    spec = Any[
        "Base alternatives"       J;
        "Base consideration set"  "{" * join(d.base_solution.consideration_set, ", ") * "}";
        "Exp. consideration set"  "{" * join(d.expanded_solution.consideration_set, ", ") * "}";
        "Max uncond. odds shift"  _fmt(maximum(d.uncond_odds_ratio_shift));
        "Max cond. odds shift"    _fmt(d.cond_odds_ratio_max_diff);
        "Max IIA violation"       _fmt(d.max_iia_violation)
    ]
    _pretty_table(io, spec;
        title="Rational Inattention IIA Breakdown Diagnostic",
        column_labels=["Diagnostic", "Value"],
        alignment=[:l, :r],
    )
    labels = ["Alt $j" for j in 1:J]
    _matrix_table(io, d.uncond_odds_ratio_shift, "Unconditional Odds-Ratio Relative Shift";
                  row_labels=labels, col_labels=labels)
end

report(d::RIIIADiagnostic) = show(stdout, d)
report(io::IO, d::RIIIADiagnostic) = show(io, d)

function Base.show(io::IO, m::RILogitModel{T}) where {T}
    J, S, K, M = size(m.menu_attributes)
    spec = Any[
        "Model"              "Rational Inattention Logit";
        "Observations"       m.n_obs;
        "Environments (M)"   M;
        "Alternatives (J)"   J;
        "States (S)"         S;
        "State observed"     _yesno(m.state_id !== nothing);
        "Baseline payoff"    _yesno(m.base_payoff !== nothing);
        "Normalization"      string(m.normalize);
        "Covariates (K)"     K;
        "Free parameters"    dof(m);
        "Attention cost ω"   _fmt(m.omega);
        "Log-lik."           _fmt(m.loglik; digits=2);
        "MNL benchmark LL"   _fmt(m.loglik_mnl; digits=2);
        "LR vs MNL"          _fmt(m.lr_vs_mnl; digits=2);
        "LR vs MNL p-val"    _format_pvalue(m.lr_pval);
        "Pseudo R-sq."       _fmt(m.pseudo_r2);
        "AIC"                _fmt(m.aic; digits=2);
        "BIC"                _fmt(m.bic; digits=2);
        "Cov. type"          _label(m.cov_type);
        "Converged"          _yesno(m.converged);
        "Iterations"         m.iterations
    ]
    _pretty_table(io, spec;
        title="Rational Inattention Discrete-Choice Logit",
        column_labels=["Specification", ""],
        alignment=[:l, :r],
    )

    pnames = vcat(m.varnames, ["omega"])
    _coef_table(io, "Parameters (Preferences & Attention Cost)",
                pnames, coef(m), stderror(m); dist=:z)
    _degenerate_fit_banner(io, coef(m))
    _sig_legend(io)
end

report(m::RILogitModel) = show(stdout, m)
report(io::IO, m::RILogitModel) = show(io, m)
