# CF-17 (#397): model-bank IRF matching, marginal likelihoods, averaging.
# Budget: tiny specs + short chains — machinery testing, not inference quality.
using Test
using LinearAlgebra
using Random
using Statistics
using Distributions
using MacroEconometricModels

const MEM = MacroEconometricModels

# 3-eq NK, one policy shock, parameter = the Taylor coefficient. Two target
# variables (12 rows) against H_news = 3 free GLS coefficients: genuinely
# over-determined, so a distorted (behavioral) menu CANNOT be re-fit — the
# discrimination the model bank is about. (A one-variable Fisher menu is too
# sparse: its columns have only c nonzero rows and any 3-column bank member
# fits the 3 effective dimensions exactly.)
const CF17_NK = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
    endogenous: π, y, i
    exogenous: eps_i
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1])
    i[t] = φπ * π[t] + eps_i[t]
end

const CF17_H = 6
# IDENTIFICATION NOTE (found the hard way): a policy-rule coefficient (φπ) is
# fundamentally UNIDENTIFIED by rule-free IRF matching — the GLS news re-fit
# absorbs the closure rule exactly (MW rule immateriality). The estimated
# parameter must be non-policy: the Phillips slope κ is sharply identified.
_cf17_menu(psi) = policy_news_matrix(
    MEM._respec(CF17_NK, merge(CF17_NK.param_values, Dict(:κ => psi[1]))),
    :eps_i, [:pi => :π, :y => :y]; H=CF17_H)
# mis-specified member: cognitive discounting m = 0.3 distorts the menu
_cf17_menu_b(psi) = behavioral(_cf17_menu(psi); m=0.3)
# φπ-parameterized builder: determinacy fails below φπ = 1 — used to exercise
# the failure-counting path (φπ itself is unidentified, which is fine there)
_cf17_menu_phi(psi) = policy_news_matrix(
    MEM._respec(CF17_NK, merge(CF17_NK.param_values, Dict(:φπ => psi[1]))),
    :eps_i, [:pi => :π, :y => :y]; H=CF17_H)

# Non-diagonal target covariance (CTW-damped) around the true menu.
function _cf17_target(kappa0; H_news=3, noise=0.0, rng=MersenneTwister(17))
    ce0 = _cf17_menu([kappa0])
    nu_true = [0.5, -0.2, 0.1][1:H_news]
    theta_hat = vcat(ce0.Theta_x[1][:, 1:H_news] * nu_true,
                     ce0.Theta_x[2][:, 1:H_news] * nu_true)
    if noise > 0
        theta_hat += noise .* randn(rng, length(theta_hat))
    end
    m = 2 * CF17_H
    # tight, non-diagonal V: the behavioral distortion must be decisive
    # relative to the target uncertainty for the bank to discriminate
    V0 = [5e-6 * 0.9^abs(i - j) for i in 1:m, j in 1:m]
    V = ctw_covariance(V0, CF17_H; bandwidth=4).V   # non-diagonal, PSD-repaired
    index = vcat([(var=:pi, shock=1, h=h) for h in 1:CF17_H],
                 [(var=:y, shock=1, h=h) for h in 1:CF17_H])
    return (theta_hat=theta_hat, V_bar=V, index=index), nu_true
end

@testset "Model bank (CF-17)" begin

    @testset "restricted GLS closed form" begin
        rng = MersenneTwister(171)
        H, H_news = 6, 3
        ce = PolicyCausalEffects(outcomes=[:pi], Theta_x=[randn(rng, H, H)])
        theta_hat = randn(rng, H)
        V0 = [0.02 * 0.9^abs(i - j) for i in 1:H, j in 1:H]
        V = ctw_covariance(V0, H; bandwidth=4).V
        index = [(var=:pi, shock=1, h=h) for h in 1:H]
        target = (theta_hat=theta_hat, V_bar=V, index=index)
        rows, n_blocks = MEM._bank_index_maps(ce, target.index)
        @test n_blocks == 1
        Phi = MEM._bank_phi(ce, rows, n_blocks, H_news)
        # H_news restriction keeps exactly the first H_news menu columns
        @test Phi == ce.Theta_x[1][:, 1:H_news]
        # GLS fit against the direct formula
        prec = MEM.precision_of(target.V_bar)
        Vinv = prec.precision
        Cw = MEM._pp_weight_factor(Vinv)
        btil = Cw * (-target.theta_hat)
        res = MEM._policy_projection(Cw * Phi, btil; method=:ls)
        nu_ref = (Phi' * Vinv * Phi) \ (Phi' * Vinv * target.theta_hat)
        @test res.nu ≈ nu_ref atol = 1e-8
        # level constant present: loglik(V) − loglik(cV) matches the analytic shift
        c = 5.0
        ll_V = MEM._bank_loglik(ce, rows, n_blocks, H_news, Cw, btil,
                                -prec.logdet / 2 - H * log(2pi) / 2)
        prec_c = MEM.precision_of(c .* target.V_bar)
        Cw_c = MEM._pp_weight_factor(prec_c.precision)
        ll_cV = MEM._bank_loglik(ce, rows, n_blocks, H_news, Cw_c,
                                 Cw_c * (-target.theta_hat),
                                 -prec_c.logdet / 2 - H * log(2pi) / 2)
        rss = sum(abs2, res.error_path)
        @test ll_V - ll_cV ≈ -(1 - 1 / c) * rss / 2 + (H / 2) * log(c) atol = 1e-8
    end

    @testset "self-recovery + model probabilities" begin
        kappa0 = 0.1
        target, _ = _cf17_target(kappa0; H_news=3, noise=1e-3)
        priors = [truncated(Normal(0.1, 0.05), 0.01, 0.5)]
        kw = (; H_news=3, n_adapt=400, n_burn=200, n_keep=800, thin=8,
              proposal_scale=5.66)      # 2.38²/d with d = 1
        mA = irf_match(_cf17_menu, target, priors, [:kappa];
                       name="RE", rng=MersenneTwister(1), kw...)
        mB = irf_match(_cf17_menu_b, target, priors, [:kappa];
                       name="behavioral", rng=MersenneTwister(2), kw...)
        @test mA isa ModelBankMember{Float64}
        @test size(mA.theta_draws, 1) == 100
        @test 0.05 <= mA.acceptance_rate <= 0.7
        @test isfinite(mA.log_marglik)
        # posterior concentrates near the generating kappa
        @test abs(median(mA.theta_draws[:, 1]) - kappa0) < 0.05
        # the well-specified member wins decisively
        probs = posterior_model_probs([mA, mB])
        @test probs[1] > 0.9
        @test sum(probs) ≈ 1.0 atol = 1e-12

        # determinism under a fixed rng
        mA2 = irf_match(_cf17_menu, target, priors, [:kappa];
                        name="RE", rng=MersenneTwister(1), kw...)
        @test mA2.theta_draws == mA.theta_draws
        @test mA2.log_marglik == mA.log_marglik

        # equal members split evenly
        p2 = posterior_model_probs([mA, mA2])
        @test p2 ≈ [0.5, 0.5] atol = 1e-12

        # model averaging over the pooled bank
        pooled = model_average([mA, mA2], p2; n_pool=50, rng=MersenneTwister(3))
        @test pooled isa PolicyCausalEffects{Float64}
        @test pooled.source == :pooled
        @test MEM.n_draws(pooled) == 50
        @test is_square(pooled)
        # pooled point stays close to a single member's menu median
        med_ref = median([m.Theta_x[1][2, 1] for m in mA.menu_draws])
        @test isapprox(pooled.Theta_x[1][2, 1], med_ref; atol=0.05)

        # subset pooling
        pooled_A = model_average([mA, mB], probs; n_pool=20, subset=[1],
                                 rng=MersenneTwister(4))
        @test MEM.n_draws(pooled_A) == 20

        # T_store truncation
        mT = irf_match(_cf17_menu, target, priors, [:kappa];
                       name="trunc", rng=MersenneTwister(5), T_store=4, kw...)
        @test mT.menu_draws[1].H == 4
        @test is_square(mT.menu_draws[1])
    end

    @testset "failure counting + validation" begin
        target, _ = _cf17_target(0.1; H_news=3)
        # prior mass below the determinacy threshold phi > 1 => -Inf builds occur
        # (phi is unidentified by rule-free matching — irrelevant here: this
        # testset only exercises the failure-counting mechanics)
        priors_wide = [truncated(Normal(1.1, 0.3), 0.5, 3.0)]
        m = @test_logs (:info, r"menu builds failed") match_mode = :any begin
            irf_match(_cf17_menu_phi, target, priors_wide, [:phi];
                      H_news=3, n_adapt=200, n_burn=100, n_keep=300, thin=6,
                      rng=MersenneTwister(6))
        end
        @test m isa ModelBankMember{Float64}
        @test all(isfinite, m.log_post)

        priors = [truncated(Normal(0.1, 0.05), 0.01, 0.5)]
        @test_throws ArgumentError irf_match(_cf17_menu, target, priors, [:a, :b];
                                             rng=MersenneTwister(7))
        @test_throws ArgumentError irf_match(_cf17_menu, (; theta_hat=[1.0]),
                                             priors, [:phi]; rng=MersenneTwister(8))
        # H_news larger than the menu
        @test_throws ArgumentError irf_match(_cf17_menu, target, priors, [:phi];
                                             H_news=CF17_H + 1, n_adapt=10,
                                             n_burn=5, n_keep=20, thin=2,
                                             rng=MersenneTwister(9))
        # thin/keep guards
        @test_throws ArgumentError irf_match(_cf17_menu, target, priors, [:phi];
                                             n_keep=2, thin=5, rng=MersenneTwister(10))
        # probs validation
        @test_throws ArgumentError posterior_model_probs(ModelBankMember{Float64}[])
    end
end
