# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random
using Distributions
using ForwardDiff
using StatsAPI

const _LLGMM_FAST = get(ENV, "MACRO_FAST_TESTS", "0") == "1"

# ---------------------------------------------------------------------------
# Local Simulation Helpers (deterministic given rng)
# ---------------------------------------------------------------------------

function _sim_ll_var(rng::AbstractRNG, T_obs::Int, B0::AbstractMatrix{T},
                      A_list::Vector{Matrix{T}};
                      nu::AbstractVector{T}=zeros(T, size(B0, 1)),
                      burnin::Int=200,
                      shock_fn=nothing) where {T<:AbstractFloat}
    n = size(B0, 1)
    p = length(A_list)
    tot = T_obs + burnin
    E = if shock_fn !== nothing
        shock_fn(rng, tot, n)
    else
        # Standardized t(12) shocks: Var(t_df) = df / (df - 2)
        d = TDist(12)
        sc = sqrt(T(12) / T(10))
        T.(rand(rng, d, tot, n) ./ sc)
    end
    Y = zeros(T, tot, n)
    for t in (p + 1):tot
        u_t = B0 * @view(E[t, :])
        y_t = nu .+ u_t
        for l in 1:p
            y_t .+= A_list[l] * @view(Y[t - l, :])
        end
        @views Y[t, :] .= y_t
    end
    return Y[(burnin + 1):end, :], E[(burnin + 1):end, :]
end

function _sim_common_sv_shocks(rng::AbstractRNG, tot::Int, n::Int; rho::Float64=0.88, sigma_v::Float64=0.45)
    # Asymmetric non-Gaussian standardized innovations z_{it} (E[z]=0, E[z^2]=1, E[z^3] != 0)
    # multiplied by a common stationary stochastic volatility factor s_t normalized so E[s_t^2] = 1
    # => E[e_i^2] = 1, E[e_i e_j] = 0, E[e_i^3 e_j] = 0, BUT E[e_i^2 e_j^2] = E[s_t^4] > 1!
    h = zeros(tot)
    h[1] = sigma_v / sqrt(1 - rho^2) * randn(rng)
    for t in 2:tot
        h[t] = rho * h[t - 1] + sigma_v * randn(rng)
    end
    s_raw = exp.(0.5 .* h)
    s = s_raw ./ sqrt(mean(s_raw .^ 2))
    # Distinct standardized non-Gaussian shocks (kappa_1 != kappa_2 eliminates pi/4 spurious asym root)
    g1 = Gamma(1.5, 1.0)
    z1 = (rand(rng, g1, tot) .- 1.5) ./ sqrt(1.5)
    z2 = (rand(rng, tot) .- 0.5) .* sqrt(12.0)  # Uniform(-sqrt(3), sqrt(3)), kappa_2 = -1.2
    return hcat(z1, z2) .* s
end

function _rot2(theta::Real)
    c, s = cos(theta), sin(theta)
    [c -s; s c]
end

function _col_signed_frob(B_hat::AbstractMatrix, B_true::AbstractMatrix)
    # Match columns up to signed permutation (Pham-Garat canonical benchmark)
    n = size(B_true, 1)
    best = Inf
    for p in MacroEconometricModels._permutations(n)
        for signs in Iterators.product(ntuple(_ -> (-1.0, 1.0), n)...)
            cand = hcat([signs[j] * B_hat[:, p[j]] for j in 1:n]...)
            d = norm(cand - B_true) / max(norm(B_true), 1e-12)
            d < best && (best = d)
        end
    end
    best
end

@testset "Lanne-Luoto (2021) Co-Kurtosis GMM & Extensions [NGID-04]" begin

    @testset "Analytical Jacobian and CUE Gradient Verification" begin
        rng = Random.Xoshiro(101)
        for p in (0, 1, 2)
            n = 2
            B0 = [1.2 -0.35; 0.40 0.95]
            A_list = p == 0 ? Matrix{Float64}[] :
                     [0.25 * (0.5^(l - 1)) * [1.0 0.15; -0.10 0.85] for l in 1:p]
            Y, _ = _sim_ll_var(rng, 180, B0, A_list; nu=[0.1, -0.2])
            mod = MacroEconometricModels._llgmm_var_for_Y(Y, p; check_stability=false)
            Y_eff, X = MacroEconometricModels._llgmm_construct_matrices(mod)
            asym = MacroEconometricModels._llgmm_all_asym_pairs(n)
            sym = MacroEconometricModels._llgmm_all_sym_pairs(n)

            for form in (:b_model, :a_model)
                th = MacroEconometricModels._llgmm_init_theta(mod, Y_eff, X, form, asym, sym, 1, rng)
                # Perturb slightly away from OLS initialization for non-trivial residual moments
                th = th .+ 0.03 * randn(rng, length(th))

                _, gb, G_an, ok = MacroEconometricModels._llgmm_jacobian_impl(
                    th, Y_eff, X, n, p, form, asym, sym
                )
                @test ok

                G_fd = ForwardDiff.jacobian(t -> begin
                    M_t, _ = MacroEconometricModels._llgmm_moment_matrix(
                        t, Y_eff, X, n, p, form, asym, sym
                    )
                    vec(mean(M_t; dims=1))
                end, th)
                @test G_an ≈ G_fd rtol=1e-9 atol=1e-10
                @test size(G_an) == (MacroEconometricModels._llgmm_n_base_moments(n, p) + length(asym) + length(sym),
                                     MacroEconometricModels._llgmm_n_params(n, p))

                # Analytical CUE gradient vs finite differences (fixed bw=2 Bartlett)
                G_cue = zeros(length(th))
                q_val = MacroEconometricModels._llgmm_cue_eval_fg!(
                    G_cue, th, Y_eff, X, n, p, form, asym, sym, :bartlett, 2
                )
                @test isfinite(q_val) && q_val >= 0

                cue_scalar(t) = MacroEconometricModels._llgmm_cue_eval_fg!(
                    nothing, t, Y_eff, X, n, p, form, asym, sym, :bartlett, 2
                )
                eps_fd = 1e-6
                g_num = similar(th)
                for idx in eachindex(th)
                    tp = copy(th); tp[idx] += eps_fd
                    tm = copy(th); tm[idx] -= eps_fd
                    g_num[idx] = (cue_scalar(tp) - cue_scalar(tm)) / (2 * eps_fd)
                end
                @test G_cue ≈ g_num rtol=1e-5 atol=1e-6
            end
        end
    end

    @testset "Pham-Garat Normalization & Gaussian Rank Deficiency" begin
        rng = Random.Xoshiro(202)
        # Un-permuted vs permuted matrix recovers canonical positive diagonal
        B_true = [1.4 0.3; -0.2 1.1]
        B_perm = hcat(-B_true[:, 2], B_true[:, 1])
        pg = MacroEconometricModels._llgmm_pham_garat(B_perm)
        @test all(diag(pg.B) .> 0)
        @test pg.B ≈ B_true atol=1e-12

        # Under-identified (q < k) throws ArgumentError
        B_gauss = [1.1 -0.3; 0.25 0.9]
        A_gauss = [[0.4 0.05; -0.05 0.35]]
        Y_g, _ = _sim_ll_var(rng, 400, B_gauss, A_gauss;
                             shock_fn=(rng, Ttot, nn) -> randn(rng, Ttot, nn))
        @test_throws ArgumentError estimate_lanne_luoto_gmm(Y_g, 1;
                                                            moment_selection=:none,
                                                            asym_pairs=Tuple{Int,Int}[],
                                                            sym_pairs=Tuple{Int,Int}[],
                                                            seed=202)

        # Under Gaussian shocks, co-kurtosis Jacobian is near-singular (very small min singular value)
        res_g = estimate_lanne_luoto_gmm(Y_g, 1;
                                         moment_selection=:none,
                                         asym_pairs=[(1, 2)],
                                         sym_pairs=Tuple{Int,Int}[],
                                         seed=202)
        @test svdvals(res_g.G_jac)[end] < 0.25

        # Verify DEFICIENT warning renders when rank_ok is false
        vals_def = Any[getfield(res_g, fn) for fn in fieldnames(LanneLuotoGMMResult)]
        idx_rk = findfirst(==(:jacobian_rank), fieldnames(LanneLuotoGMMResult))
        idx_ok = findfirst(==(:rank_ok), fieldnames(LanneLuotoGMMResult))
        vals_def[idx_rk] = res_g.n_params - 1
        vals_def[idx_ok] = false
        res_def = LanneLuotoGMMResult{Float64}(vals_def...)
        io = IOBuffer()
        show(io, res_def)
        @test occursin("DEFICIENT", String(take!(io)))
    end

    @testset "Common Stochastic Volatility: Symmetric Fails, MSC+RMSC Drops Symmetric" begin
        rng = Random.Xoshiro(303)
        n = 2
        p = 1
        B_true = [1.20 -0.45; 0.35 1.05]
        A_true = [[0.45 0.08; -0.06 0.40]]
        Y_sv, _ = _sim_ll_var(rng, 2000, B_true, A_true; shock_fn=_sim_common_sv_shocks)

        # Including symmetric co-kurtosis moment fails overidentification test under common SV
        res_all = estimate_lanne_luoto_gmm(Y_sv, p;
                                           asym_pairs=:auto,
                                           sym_pairs=:auto,
                                           moment_selection=:none,
                                           weighting=:two_step,
                                           seed=303)
        @test res_all.rank_ok
        @test length(res_all.sym_pairs) == 1
        @test res_all.J_pvalue < 0.05

        # Two-stage MSC + RMSC selects only valid asymmetric co-kurtosis conditions
        sel = select_gmm_moments(Y_sv, p; weighting=:two_step, method=:msc_rmsc, seed=303)
        @test sel isa GMMMomentSelectionResult{Float64}
        @test isempty(sel.sym_pairs)
        @test length(sel.asym_pairs) >= 1
        @test sel.J_pvalue > 0.05
        @test !isempty(sel.candidates)
        @test sel.bandwidth >= 0

        res_sel = estimate_lanne_luoto_gmm(Y_sv, p;
                                           weighting=:two_step,
                                           moment_selection=:msc_rmsc,
                                           seed=303)
        @test res_sel.selection !== nothing
        @test isempty(res_sel.sym_pairs)
        @test res_sel.J_pvalue > 0.05
        @test _col_signed_frob(res_sel.B0, B_true) < 0.16
    end

    @testset "Lanne-Luoto Table 1 Design: Estimator, CUE vs Iterated, and GMM LR" begin
        # Lanne & Luoto (2021) Table 1: n=2, B0 = rotation(theta = -pi/5), t(12), T = 500
        theta_rot = -pi / 5
        B_true = _rot2(theta_rot)
        A_true = [[0.50 0.10; -0.05 0.45]]

        # Deterministic T = 500 run
        rng_det = Random.Xoshiro(404)
        Y_det, _ = _sim_ll_var(rng_det, 500, B_true, A_true)
        res_2s = estimate_lanne_luoto_gmm(Y_det, 1; weighting=:two_step, moment_selection=:none, seed=404)
        res_it = estimate_lanne_luoto_gmm(Y_det, 1; weighting=:iterated, moment_selection=:none, seed=404)
        res_cue = estimate_lanne_luoto_gmm(Y_det, 1; weighting=:cue, moment_selection=:none, seed=404)

        @test res_2s.rank_ok
        @test res_it.rank_ok
        @test res_cue.rank_ok
        @test res_cue.converged
        @test res_2s.J_pvalue > 0.01
        @test _col_signed_frob(res_2s.B0, B_true) < 0.35

        # Genuine CUE differs from iterated GMM in finite sample and improves CUE objective
        @test maximum(abs.(res_cue.theta .- res_it.theta)) > 1e-6
        @test maximum(abs.(res_cue.theta .- res_2s.theta)) > 1e-6

        # Test true restriction H0: B_{11} = cos(-pi/5) (linear restriction on B0[1,1])
        R_true = zeros(1, 4)
        R_true[1, 1] = 1.0
        r_true = [cos(theta_rot)]
        t_lr_true = test_svar_hypothesis(res_2s, R_true, r_true; method=:gmm_lr)
        t_w_true = test_svar_hypothesis(res_2s, R_true, r_true; method=:wald)
        @test t_lr_true isa MacroEconometricModels.SVARHypothesisTest{Float64}
        @test t_lr_true.statistic >= 0.0
        @test t_w_true.statistic >= 0.0
        @test t_lr_true.df == 1
        @test t_lr_true.pvalue > 0.05

        # False zero restriction H0: B_{12} = 0 tested under larger T (also verifies consistency)
        Y_pow, _ = _sim_ll_var(Random.Xoshiro(405), 1500, B_true, A_true)
        res_pow = estimate_lanne_luoto_gmm(Y_pow, 1; weighting=:two_step, moment_selection=:none, seed=405)
        @test _col_signed_frob(res_pow.B0, B_true) < 0.18
        t_lr_false = test_svar_hypothesis(res_pow, [NaN 0.0; NaN NaN]; method=:gmm_lr)
        @test t_lr_false.statistic > 5.0
        @test t_lr_false.pvalue < 0.05

        # Monte Carlo verification of Table 1 benchmark
        n_mc = _LLGMM_FAST ? 40 : 200
        mc_rng = Random.Xoshiro(12345)
        b_errs = zeros(n_mc)
        b11_biases = zeros(n_mc)
        j_rejects = 0
        lr_rejects = 0
        wald_rejects = 0

        for m in 1:n_mc
            Ym, _ = _sim_ll_var(mc_rng, 500, B_true, A_true)
            est = estimate_lanne_luoto_gmm(Ym, 1;
                                           weighting=:two_step,
                                           moment_selection=:none,
                                           n_starts=3,
                                           rng=Random.Xoshiro(10_000 + m))
            # Match canonical sign/perm before bias computation
            best_b = copy(est.B0)
            best_d = Inf
            for p_idx in MacroEconometricModels._permutations(2)
                for sgn in Iterators.product((-1.0, 1.0), (-1.0, 1.0))
                    cand = hcat(sgn[1] * est.B0[:, p_idx[1]], sgn[2] * est.B0[:, p_idx[2]])
                    d = norm(cand - B_true)
                    if d < best_d
                        best_d = d
                        best_b = cand
                    end
                end
            end
            b_errs[m] = best_d
            b11_biases[m] = best_b[1, 1] - B_true[1, 1]
            est.J_pvalue < 0.05 && (j_rejects += 1)
            t_lr = test_svar_hypothesis(est, R_true, r_true; method=:gmm_lr)
            t_lr.pvalue < 0.05 && (lr_rejects += 1)
            t_w = test_svar_hypothesis(est, R_true, r_true; method=:wald)
            t_w.pvalue < 0.05 && (wald_rejects += 1)
        end

        if _LLGMM_FAST
            @test abs(mean(b11_biases)) < 0.07
            @test abs(median(b11_biases)) < 0.05
            @test j_rejects / n_mc <= 0.20
            @test lr_rejects / n_mc <= 0.20
            @test lr_rejects <= wald_rejects
        else
            @test abs(mean(b11_biases)) < 0.05
            @test abs(median(b11_biases)) < 0.03
            @test j_rejects / n_mc <= 0.15
            @test lr_rejects / n_mc <= 0.15
            @test lr_rejects < wald_rejects
        end
    end

    @testset "A-Model vs B-Model Parameterization & Matrix Auto-Lag" begin
        rng = Random.Xoshiro(505)
        B_true = [1.3 -0.4; 0.35 0.9]
        A_true = [[0.35 0.05; -0.05 0.30]]
        Y, _ = _sim_ll_var(rng, 450, B_true, A_true)

        res_b = estimate_lanne_luoto_gmm(Y, nothing; form=:b_model, moment_selection=:none, seed=505)
        res_a = estimate_lanne_luoto_gmm(Y, 1; form=:a_model, moment_selection=:none, seed=505)

        @test res_b.form === :b_model
        @test res_a.form === :a_model
        @test diag(res_a.A0) ≈ ones(2) atol=1e-12
        @test all(res_a.sigma .> 0)
        @test res_a.A0 * res_a.B0 ≈ Diagonal(res_a.sigma) rtol=1e-8 atol=1e-8
        @test _col_signed_frob(res_a.B0, res_b.B0) < 0.08
        @test res_a.rank_ok && res_b.rank_ok

        # Argument validation checks
        @test_throws ArgumentError estimate_lanne_luoto_gmm(Y, 1; form=:invalid)
        @test_throws ArgumentError estimate_lanne_luoto_gmm(Y, 1; weighting=:invalid)
        @test_throws ArgumentError estimate_lanne_luoto_gmm(Y, 1; moment_selection=:invalid)
        @test_throws ArgumentError estimate_lanne_luoto_gmm(Y, 1; asym_pairs=[(1, 1)])
        @test_throws ArgumentError estimate_lanne_luoto_gmm(Y, 1; asym_pairs=[(1, 2), (1, 2)])
    end

    @testset "Higher Dimension (n=3 Exhaustive & n=4 Sequential RMSC)" begin
        # n = 3 exhaustive search
        rng3 = Random.Xoshiro(606)
        B3 = [1.2 -0.3 0.15; 0.25 1.05 -0.2; -0.15 0.3 0.95]
        A3 = [[0.30 0.0 0.0; 0.0 0.25 0.0; 0.0 0.0 0.20]]
        Y3, _ = _sim_ll_var(rng3, 450, B3, A3)
        res3 = estimate_lanne_luoto_gmm(Y3, 1; moment_selection=:msc_rmsc, n_starts=2, seed=606)
        @test res3.rank_ok
        @test length(res3.asym_pairs) >= 3
        @test res3.selection !== nothing
        @test res3.selection.method === :msc_rmsc

        # n = 4 triggers sequential backward-elimination RMSC automatically
        rng4 = Random.Xoshiro(707)
        B4 = [1.1 -0.25 0.1 0.05;
              0.2 1.0 -0.2 0.1;
              -0.1 0.25 0.95 -0.15;
              0.05 -0.1 0.2 1.05]
        A4 = [0.25 * Matrix{Float64}(I, 4, 4)]
        Y4, _ = _sim_ll_var(rng4, 400, B4, A4)

        sel4 = select_gmm_moments(Y4, 1; method=:msc_rmsc, n_starts=2, seed=707)
        @test sel4.method === :rmsc_seq
        @test length(sel4.asym_pairs) >= 6
        @test !isempty(sel4.candidates)

        res4 = estimate_lanne_luoto_gmm(Y4, 1; moment_selection=:rmsc_seq, n_starts=2, seed=707)
        @test res4.rank_ok
        @test length(res4.asym_pairs) >= 6
        io4 = IOBuffer()
        show(io4, sel4)
        @test occursin("GMM Co-Kurtosis Moment Selection", String(take!(io4)))
        report(IOBuffer(), sel4)
    end

    @testset "Structural Hypothesis Tests (Short-Run, Long-Run, Horizon, Nonlinear)" begin
        rng = Random.Xoshiro(810)
        # Recursiveness holds: B_true is strictly lower triangular; use t(5) for strong power
        B_rec = [1.2 0.0; 0.5 0.95]
        A_rec = [[0.45 0.0; 0.15 0.40]]
        t5_sc = sqrt(5.0 / 3.0)
        Y_rec, _ = _sim_ll_var(rng, 800, B_rec, A_rec;
                               shock_fn=(rng, tot, nn) -> rand(rng, TDist(5), tot, nn) ./ t5_sc)
        res_rec = estimate_lanne_luoto_gmm(Y_rec, 1; moment_selection=:none, seed=810)

        # Short-run recursiveness H0: B_{12} = 0 (true)
        h_rec = recursiveness_hypothesis(2)
        t_rec_lr = test_svar_hypothesis(res_rec, h_rec; method=:gmm_lr)
        t_rec_w = test_svar_hypothesis(res_rec, h_rec; method=:wald)
        @test t_rec_lr.statistic >= 0.0
        @test t_rec_w.statistic >= 0.0
        @test t_rec_lr.df == 1
        @test t_rec_lr.pvalue > 0.05
        @test t_rec_w.pvalue > 0.05

        # impact_zero_hypothesis false element H0: B_{21} = 0 (true value is 0.5)
        h_false = impact_zero_hypothesis(2, [(2, 1)])
        t_false_lr = test_svar_hypothesis(res_rec, h_false; method=:gmm_lr)
        @test t_false_lr.statistic > 5.0
        @test t_false_lr.pvalue < 0.05

        # impact_linear_hypothesis acts on off-diagonal β = vecd⁰(B) (normalized by column diag)
        R_lin = [1.0 0.0]
        r_lin = [B_rec[2, 1] / B_rec[1, 1]]
        h_lin = impact_linear_hypothesis(R_lin, r_lin)
        t_lin = test_svar_hypothesis(res_rec, h_lin; method=:gmm_lr)
        @test t_lin.df == 1 && t_lin.statistic >= 0.0 && t_lin.pvalue > 0.05

        # Matrix (R, r) overload acting directly on vec(B0)
        t_vecb = test_svar_hypothesis(res_rec, [1.0 0.0 0.0 -1.0], [B_rec[1, 1] - B_rec[2, 2]]; method=:gmm_lr)
        @test t_vecb.df == 1 && t_vecb.statistic >= 0.0 && t_vecb.pvalue > 0.05

        # long_run_hypothesis (upper right entry of Phi(1) = (I - A_1)^{-1} B_0 is 0 here since both A_1 and B_0 are lower triangular!)
        h_lr = long_run_hypothesis(2, [(1, 2)])
        t_long_lr = test_svar_hypothesis(res_rec, h_lr; method=:gmm_lr)
        t_long_w = test_svar_hypothesis(res_rec, h_lr; method=:wald)
        @test t_long_lr.statistic >= 0.0 && t_long_lr.df == 1 && t_long_lr.pvalue > 0.05
        @test t_long_w.statistic >= 0.0 && t_long_w.df == 1 && t_long_w.pvalue > 0.05

        # horizon_irf_hypothesis, mixed_svar_hypothesis, nonlinear_svar_hypothesis
        h_hor = horizon_irf_hypothesis(2, [(1, 1, 2)])
        t_hor = test_svar_hypothesis(res_rec, h_hor; method=:gmm_lr)
        @test t_hor.df == 1 && t_hor.statistic >= 0.0

        h_mix = mixed_svar_hypothesis([h_rec, h_lr])
        t_mix = test_svar_hypothesis(res_rec, h_mix; method=:gmm_lr)
        @test t_mix.df == 2 && t_mix.statistic >= 0.0 && t_mix.pvalue > 0.05

        h_nl = nonlinear_svar_hypothesis((_, vecB) -> [vecB[1]^2 - vecB[4]^2 - (B_rec[1, 1]^2 - B_rec[2, 2]^2)], 1)
        t_nl = test_svar_hypothesis(res_rec, h_nl; method=:gmm_lr)
        @test t_nl.df == 1 && t_nl.statistic >= 0.0 && t_nl.pvalue > 0.05

        @test_throws ArgumentError test_svar_hypothesis(res_rec, h_rec; method=:invalid_method)
    end

    @testset "Ecosystem Integration, StatsAPI, Float32 & RSER-01 Serialization Check" begin
        rng = Random.Xoshiro(909)
        B0 = [1.1 -0.25; 0.30 0.95]
        A1 = [[0.4 0.05; -0.05 0.35]]
        Y, _ = _sim_ll_var(rng, 320, B0, A1)
        res = estimate_lanne_luoto_gmm(Y, 1; moment_selection=:msc_rmsc, seed=909)

        # StatsAPI & property aliases
        @test StatsAPI.nobs(res) == size(res.shocks, 1)
        @test length(StatsAPI.coef(res)) == res.n_params
        @test size(StatsAPI.vcov(res)) == (res.n_params, res.n_params)
        @test length(StatsAPI.stderror(res)) == res.n_params
        @test size(StatsAPI.residuals(res)) == size(res.shocks)
        @test StatsAPI.dof(res) == res.n_params
        @test StatsAPI.dof_residual(res) == size(res.shocks, 1) * 2 - res.n_params
        @test StatsAPI.islinear(res) === false
        ci = StatsAPI.confint(res; level=0.95)
        @test size(ci) == (res.n_params, 2) && all(ci[:, 1] .<= res.theta .<= ci[:, 2])
        @test res.se === res.se_B0 && res.B === res.B0 && res.J === res.J_stat && res.residuals === res.shocks

        # IRF, FEVD, Historical Decomposition
        ir = irf(res, 10)
        @test size(ir.values, 1) == 10
        @test ir.values[1, :, :] ≈ res.B0 atol=1e-10

        fv = fevd(res, 8)
        @test size(fv.proportions) == (2, 2, 8)
        @test all(isapprox.(sum(fv.proportions; dims=2), 1.0; atol=1e-10))

        hd = historical_decomposition(res)
        @test verify_decomposition(hd)

        # label_shocks and test_overidentification
        labeled = label_shocks(res; by=:max_impact, shock_names=["Supply", "Demand"])
        @test labeled.shock_names == ["Supply", "Demand"]
        oid = test_overidentification(res.var_ols, res)
        @test oid.statistic ≈ res.J_stat rtol=1e-10
        @test oid.details[:df] == res.df

        # show and report
        io = IOBuffer()
        show(io, res)
        txt = String(take!(io))
        @test occursin("Lanne–Luoto (2021)", txt)
        report(IOBuffer(), res)

        # Float32 precision check (VARModel{Float32} preserves Float32 end-to-end; raw Float32 Y delegates cleanly)
        m = res.var_ols
        m32 = VARModel(Float32.(m.Y), m.p, Float32.(m.B), Float32.(m.U),
                       Float32.(m.Sigma), Float32(m.aic), Float32(m.bic),
                       Float32(m.hqic), m.varnames)
        res32 = estimate_lanne_luoto_gmm(m32; moment_selection=:none, seed=909)
        @test res32 isa LanneLuotoGMMResult{Float32}
        @test eltype(res32.B0) === Float32
        @test eltype(res32.vcov) === Float32
        res32_raw = estimate_lanne_luoto_gmm(Float32.(Y), 1; moment_selection=:none, seed=909)
        @test all(isfinite, res32_raw.B0)

        # RSER-01 exact positional {T} constructor check (including Any[] tuple-vector decoding)
        sel = res.selection
        sel_re = GMMMomentSelectionResult{Float64}(
            Any[collect(x) for x in sel.asym_pairs],
            Any[],
            Any[collect(x) for x in sel.msc_asym_pairs],
            Any[],
            sel.msc_score,
            sel.rmsc_score,
            sel.J_stat,
            sel.J_pvalue,
            sel.df,
            sel.n_moments,
            sel.n_params,
            sel.bandwidth,
            sel.method,
            Any[(asym_pairs=Any[collect(p) for p in c.asym_pairs],
                 sym_pairs=Any[collect(p) for p in c.sym_pairs],
                 n_moments=c.n_moments, df=c.df, J_stat=c.J_stat,
                 J_pvalue=c.J_pvalue, msc=c.msc, rmsc=c.rmsc, rank_ok=c.rank_ok)
                for c in sel.candidates]
        )
        @test sel_re isa GMMMomentSelectionResult{Float64}

        vals = [getfield(res, fn) for fn in fieldnames(LanneLuotoGMMResult)]
        res_re = LanneLuotoGMMResult{Float64}(vals...)
        @test res_re isa LanneLuotoGMMResult{Float64}
        @test res_re.B0 ≈ res.B0
    end
end
