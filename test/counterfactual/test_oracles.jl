# CF-23 (#403): cross-method oracle suite — theorem-level identities that
# hold exactly in linear laboratories. Every equality below is guaranteed by
# theory; a failure is an implementation bug, not statistical noise. All
# oracles are self-generated from package solvers (no replication numbers).
using Test
using LinearAlgebra
using Random
using Statistics
using Distributions
using MacroEconometricModels

const MEM = MacroEconometricModels

include(joinpath(@__DIR__, "oracles_fixtures.jl"))

@testset "Cross-method oracles (CF-23)" begin
    H = 40

    @testset "1. Proposition-1 exact recovery (RANK)" begin
        # MW Prop. 1: news-implemented rule counterfactuals == direct re-solve.
        sol_A, ce, base = orc_nk_inputs(H)

        # Taylor-variant closure via _respec
        direct = orc_direct_paths(orc_respec(ORC_NK, Dict(:φπ => 3.0)), H)
        pc = policy_counterfactual(base, ce,
                                   taylor_rule(H; rho=0.0, phi_pi=3.0, phi_y=0.0,
                                               pi_var=:pi, y_var=:ygap,
                                               outcomes=[:pi, :ygap], instruments=[:rate]))
        @test pc.x_cf[1] ≈ direct.pi atol = 1e-6
        @test pc.x_cf[2] ≈ direct.y atol = 1e-6
        @test pc.z_cf[1] ≈ direct.i atol = 1e-6

        # NGDP-level target closure (policy equation replaced)
        direct_n = orc_direct_paths(ORC_NK_NGDP, H)
        pc_n = policy_counterfactual(base, ce,
                                     ngdp_rule(H; pi_var=:pi, y_var=:ygap,
                                               outcomes=[:pi, :ygap], instruments=[:rate]))
        @test pc_n.x_cf[1] ≈ direct_n.pi atol = 1e-6
        @test pc_n.x_cf[2] ≈ direct_n.y atol = 1e-6
        @test pc_n.z_cf[1] ≈ direct_n.i atol = 1e-6

        # A rate PEG is indeterminate under RE (no direct solve exists) — the
        # news construction still enforces it exactly; assert enforcement.
        pc_p = policy_counterfactual(base, ce,
                                     rate_peg_rule(H; outcomes=[:pi, :ygap],
                                                   instruments=[:rate]))
        @test norm(pc_p.z_cf[1]) < 1e-8
        @test norm(pc_p.error_path) < 1e-8
    end

    @testset "2. Proposition-1 exact recovery (HA, administered rate)" begin
        # Under the administered-rate closure, pegging the rate against a wage
        # baseline must strip exactly the J_Cr·dr term: C_cf == J_Cw · dw.
        spec = MEM._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=100)
        ss = compute_steady_state(spec)
        Th = 40
        Hh = 12
        J_Cr = sequence_jacobian(spec, ss, :r, :C; T_horizon=Th)
        J_Cw = sequence_jacobian(spec, ss, :w, :C; T_horizon=Th)
        dw = [0.9^(t - 1) for t in 1:Th]
        dr = 0.3 .* [0.8^(t - 1) for t in 1:Th]        # any r-path in the baseline
        C_base = J_Cr * dr + J_Cw * dw
        ce = policy_causal_effects(spec, ss; outcomes=[:cons => :C],
                                   instruments=[:rate => :r], H=Hh, T_horizon=Th,
                                   rule_closure=:administered)
        base = MEM.BaselinePath{Float64}([:cons], [:rate],
                                         [C_base[1:Hh]], [dr[1:Hh]],
                                         nothing, nothing, Hh, "wage shock")
        pc = policy_counterfactual(base, ce,
                                   rate_peg_rule(Hh; outcomes=[:cons], instruments=[:rate]))
        # the peg strips exactly the controllable J_Cr·dr part over the window;
        # the beyond-window dr tail couples in through J_Cr's anticipation block
        C_direct = (J_Cw * dw)[1:Hh] + J_Cr[1:Hh, Hh+1:Th] * dr[Hh+1:Th]
        @test pc.z_cf[1] ≈ zeros(Hh) atol = 1e-8
        @test pc.x_cf[1] ≈ C_direct atol = 1e-8
    end

    @testset "3. Rule immateriality (menus from different closures)" begin
        # MW/CMW: Θ_ν differs across baseline closures, counterfactuals do not.
        _, ce_a, base = orc_nk_inputs(H)
        spec_b = orc_respec(ORC_NK, Dict(:φπ => 3.0))
        ce_b = policy_news_matrix(spec_b, :eps_i, [:pi => :π, :ygap => :y],
                                  [:rate => :i]; H=H)
        base_b = baseline_path(irf(solve(spec_b), H), "eps_d",
                               [:pi => "π", :ygap => "y"], [:rate => "i"]; H=H)
        @test !(ce_a.Theta_x[1] ≈ ce_b.Theta_x[1])          # menus differ...
        rule = ngdp_rule(H; pi_var=:pi, y_var=:ygap,
                         outcomes=[:pi, :ygap], instruments=[:rate])
        pc_a = policy_counterfactual(base, ce_a, rule)
        pc_b = policy_counterfactual(base_b, ce_b, rule)
        @test pc_a.x_cf[1] ≈ pc_b.x_cf[1] atol = 1e-8       # ...counterfactuals do not
        @test pc_a.x_cf[2] ≈ pc_b.x_cf[2] atol = 1e-8
        @test pc_a.z_cf[1] ≈ pc_b.z_cf[1] atol = 1e-8
    end

    @testset "4. Optimal-policy circle" begin
        # MW Prop. 2: the optimum is itself a rule CF-10 can enforce; FOC ⊥.
        _, ce, base = orc_nk_inputs(H)
        loss = policy_loss([:pi, :ygap], H; lambda=[1.0, 0.25], beta=0.99)
        po = optimal_policy(base, ce, loss)
        @test po.foc_norm < 1e-8
        pc = MEM._suppress_warnings() do
            policy_counterfactual(base, ce, optimal_rule(ce, loss))
        end
        @test pc.x_cf[1] ≈ po.x_cf[1] atol = 1e-8
        @test pc.x_cf[2] ≈ po.x_cf[2] atol = 1e-8
        @test pc.z_cf[1] ≈ po.z_cf[1] atol = 1e-8
    end

    @testset "5. OPP ≡ optimal projection" begin
        # BM Prop. 2 / MW eq. 27: OPP with a forecast base is the CF-11 solve.
        rng = MersenneTwister(23)
        _, ce, _ = orc_nk_inputs(H)
        v1, v2 = randn(rng, H), randn(rng, H)
        loss = policy_loss([:pi, :ygap], H; lambda=[1.0, 0.5], beta=0.98)
        fc = MEM.PolicyForecast{Float64}([:pi, :ygap], [v1, v2], nothing, H, "t")
        r = MEM._suppress_warnings() do
            opp(fc, ce, loss)
        end
        base_fc = MEM.BaselinePath{Float64}([:pi, :ygap], [:rate], [v1, v2],
                                            [zeros(H)], nothing, nothing, H, "fc")
        po = MEM._suppress_warnings() do
            optimal_policy(base_fc, ce, loss)
        end
        @test r.delta ≈ po.nu atol = 1e-10
        @test r.Y_opp[1] ≈ po.x_cf[1] atol = 1e-10
        @test r.loss_opp ≈ po.loss_cf atol = 1e-10
    end

    @testset "6. Lucas-robustness contrast (Sims–Zha ≠ MW)" begin
        # Naive period-by-period re-shocking enforces the rule ex post but is
        # NOT the rule counterfactual in a forward-looking model...
        # target rule: i = 3π (a Taylor variant — NOT strict π-targeting, which
        # pins y too via the NKPC). Persistent demand state: multi-period rule
        # violations are what separate the two constructions.
        ce = policy_news_matrix(ORC_NK_PERS, :eps_i, [:pi => :π, :ygap => :y],
                                [:rate => :i]; H=H)
        base = baseline_path(irf(solve(ORC_NK_PERS), H), "eps_d",
                             [:pi => "π", :ygap => "y"], [:rate => "i"]; H=H)
        phi = 3.0
        col_pi = ce.Theta_x[1][:, 1]
        col_y = ce.Theta_x[2][:, 1]
        col_i = ce.Theta_z[1][:, 1]
        pi_p = copy(base.x[1])
        y_p = copy(base.x[2])
        i_p = copy(base.z[1])
        c_imp = col_i[1] - phi * col_pi[1]
        for t in 1:H
            m = -(i_p[t] - phi * pi_p[t]) / c_imp        # date-t surprise, ex-post fix
            for s in t:H
                pi_p[s] += m * col_pi[s-t+1]
                y_p[s] += m * col_y[s-t+1]
                i_p[s] += m * col_i[s-t+1]
            end
        end
        pc = policy_counterfactual(base, ce,
                                   taylor_rule(H; rho=0.0, phi_pi=phi, phi_y=0.0,
                                               pi_var=:pi, y_var=:ygap,
                                               outcomes=[:pi, :ygap], instruments=[:rate]))
        @test maximum(abs.(i_p - phi .* pi_p)) < 1e-8     # both enforce the rule ex post...
        @test maximum(abs.(pc.z_cf[1] - phi .* pc.x_cf[1])) < 1e-8
        @test maximum(abs.(y_p - pc.x_cf[2])) > 1e-3      # ...but the paths differ

        # ...while in a purely backward model the two coincide (free sanity check)
        Hb = 20
        ce_b = policy_news_matrix(ORC_AR, :eps_i, [:x => :x], [:z => :z]; H=Hb)
        base_b = baseline_path(irf(solve(ORC_AR), Hb), "eps_d",
                               [:x => "x"], [:z => "z"]; H=Hb)
        xb = copy(base_b.x[1])
        zb = copy(base_b.z[1])
        cx = ce_b.Theta_x[1][:, 1]
        cz = ce_b.Theta_z[1][:, 1]
        for t in 1:Hb
            m = -xb[t] / cx[1]
            for s in t:Hb
                xb[s] += m * cx[s-t+1]
                zb[s] += m * cz[s-t+1]
            end
        end
        pc_b = policy_counterfactual(base_b, ce_b,
                                     inflation_target_rule(Hb; pi_var=:x,
                                                           outcomes=[:x], instruments=[:z]))
        @test xb ≈ pc_b.x_cf[1] atol = 1e-8
        @test zb ≈ pc_b.z_cf[1] atol = 1e-8
    end

    @testset "7. Historical evolution end-to-end" begin
        # CMW A.3: revision recursion == direct simulation under the new rule.
        rng = MersenneTwister(7)
        sol_A = solve(ORC_NK)
        sol_At = solve(orc_respec(ORC_NK, Dict(:φπ => 3.0)))
        t1, t2, T_all = 4, 18, 20
        E = zeros(T_all, 2)
        E[t1:t2, 2] = randn(rng, t2 - t1 + 1)
        simn(sol) = begin
            Y = zeros(T_all, 3)
            for t in 1:T_all
                prev = t == 1 ? zeros(3) : Y[t-1, :]
                Y[t, :] = sol.G1 * prev + sol.impact * E[t, :]
            end
            Y
        end
        Y_A = simn(sol_A)
        Y_At = simn(sol_At)
        m = VARModel(Y_A, 1, vcat(zeros(1, 3), Matrix(sol_A.G1')),
                     zeros(T_all - 1, 3), Matrix(sol_A.impact * sol_A.impact' + 1e-12I),
                     0.0, 0.0, 0.0, ["π", "y", "i"])
        ce = policy_news_matrix(ORC_NK, :eps_i, [:pi => :π, :ygap => :y],
                                [:rate => :i]; H=H)
        ch = counterfactual_history(m, Y_A, t1:t2, ce,
                                    taylor_rule(H; rho=0.0, phi_pi=3.0, phi_y=0.0,
                                                pi_var=:pi, y_var=:ygap,
                                                outcomes=[:pi, :ygap], instruments=[:rate]);
                                    outcomes=[:pi => "π", :ygap => "y"],
                                    instruments=[:rate => "i"], H=H)
        for (j, t) in enumerate(t1:t2), v in 1:3
            @test ch.cf[j, v] ≈ Y_At[t, v] atol = 1e-6
        end
    end

    @testset "8. Rotation invariance of second moments" begin
        # CMW A.2: an orthogonal rotation of the Wold input cancels in Σ_cf.
        rng = MersenneTwister(8)
        Hm = 30
        Theta = zeros(Hm, 2, 2)
        for h in 1:Hm
            Theta[h, :, :] = 0.6^(h - 1) .* [1.0 0.2; 0.1 0.8]
        end
        w = MEM.WoldRepresentation{Float64}(Theta, Matrix{Float64}(I, 2, 2),
                                            ["x", "z"], nothing)
        Q = Matrix(qr(randn(rng, 2, 2)).Q)
        Theta_r = similar(Theta)
        for h in 1:Hm
            Theta_r[h, :, :] = Theta[h, :, :] * Q
        end
        w_r = MEM.WoldRepresentation{Float64}(Theta_r, w.Sigma_u, w.varnames, nothing)
        ce = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                 Theta_x=[randn(rng, Hm, 2)], Theta_z=[randn(rng, Hm, 2)])
        loss = policy_loss([:x], Hm; lambda=[1.0])
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, loss; outcomes=[:x => 1],
                                   instruments=[:z => 2], warn_invertibility=false)
        end
        cm_r = MEM._suppress_warnings() do
            counterfactual_moments(w_r, ce, loss; outcomes=[:x => 1],
                                   instruments=[:z => 2], warn_invertibility=false)
        end
        @test cm.Sigma_cf ≈ cm_r.Sigma_cf atol = 1e-10
    end

    @testset "9. Model-averaging degeneracy" begin
        # Two identical members: probs = 1/2 each; pooled bands = member bands.
        rng = MersenneTwister(9)
        Hq = 6
        ce0 = policy_news_matrix(ORC_NK, :eps_i, [:pi => :π]; H=Hq)
        menus = [ce0 for _ in 1:40]
        mk() = MEM.ModelBankMember{Float64}("m", [:k], 0.1 .+ 0.01 .* randn(MersenneTwister(9), 40, 1),
                                            fill(-3.0, 40), -10.0, menus, 0.3, 3)
        mA, mB = mk(), mk()
        probs = posterior_model_probs([mA, mB])
        @test probs ≈ [0.5, 0.5] atol = 1e-12
        pooled = model_average([mA, mB], probs; n_pool=60, rng=rng)
        @test pooled.source == :pooled
        # identical menus everywhere → pooled draws identical to any member menu
        @test pooled.Theta_x[1] ≈ ce0.Theta_x[1] atol = 1e-12
        @test maximum(abs.(pooled.Theta_x_draws[1][:, :, 5] - ce0.Theta_x[1])) < 1e-12
    end

    @testset "10. Second-moment consistency vs analytic moments" begin
        # Pure-forward NK: G1 ≈ 0, so Σ_y = impact·impact′ under any closure.
        # The CF-12 construction enforces the alternative rule WITHOUT policy
        # deviations, so the analytic target is the demand-shock-only
        # covariance under Ã (the policy-shock column is driven to the rule).
        sol_A = solve(ORC_NK)
        sol_At = solve(orc_respec(ORC_NK, Dict(:φπ => 3.0)))
        @test maximum(abs.(sol_A.G1)) < 1e-10
        imp_d = sol_At.impact[:, 2]                      # eps_d column
        Sigma_target = imp_d * imp_d'
        Hm = 12
        m = VARModel(zeros(30, 3), 1, vcat(zeros(1, 3), Matrix(sol_A.G1')),
                     zeros(29, 3), Matrix(sol_A.impact * sol_A.impact' + 1e-12I),
                     0.0, 0.0, 0.0, ["π", "y", "i"])
        w = wold_representation(m; H=Hm)
        ce = policy_news_matrix(ORC_NK, :eps_i, [:pi => :π, :ygap => :y],
                                [:rate => :i]; H=Hm)
        rule = taylor_rule(Hm; rho=0.0, phi_pi=3.0, phi_y=0.0, pi_var=:pi,
                           y_var=:ygap, outcomes=[:pi, :ygap], instruments=[:rate])
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, rule;
                                   outcomes=[:pi => "π", :ygap => "y"],
                                   instruments=[:rate => "i"],
                                   warn_invertibility=false)
        end
        # varnames order: pi, ygap, rate ↔ π, y, i
        for (a, va) in enumerate((1, 2, 3)), (b, vb) in enumerate((1, 2, 3))
            @test cm.Sigma_cf[a, b] ≈ Sigma_target[va, vb] atol = 1e-4
        end
    end
end
