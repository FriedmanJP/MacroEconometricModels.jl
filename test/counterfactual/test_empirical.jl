# CF-04 (#384): empirical causal-effect adapters — round-trips from
# VAR/BVAR/sign-set/LP IRFs, baseline paths, Wold representation, stacking.
using Test
using LinearAlgebra
using Random
using Statistics
using MacroEconometricModels

const MEM = MacroEconometricModels

# Small stable 3-variable VAR(1) DGP, deterministic seed.
function _cf04_data(rng; T_obs=250, n=3)
    A = [0.5 0.1 0.0; 0.0 0.4 0.1; 0.1 0.0 0.3][1:n, 1:n]
    Y = zeros(T_obs, n)
    for t in 2:T_obs
        Y[t, :] = A * Y[t-1, :] + randn(rng, n)
    end
    return Y
end

@testset "Empirical adapters (CF-04)" begin
    rng = MersenneTwister(20260805)
    Y = _cf04_data(rng)
    m = estimate_var(Y, 2)

    @testset "frequentist round-trip" begin
        ir = irf(m, 12; ci_type=:bootstrap, reps=50, rng=MersenneTwister(1))
        H = 10
        ce = policy_causal_effects(ir, [3], [:y1 => 1, :y2 => ir.variables[2]],
                                   [:rate => 3]; H=H)
        @test ce isa PolicyCausalEffects{Float64}
        @test ce.source == :var
        @test ce.H == H
        @test ce.shock_labels == [ir.shocks[3]]
        # point values equal ir.values slices exactly
        @test ce.Theta_x[1] == ir.values[1:H, 1, 3:3]
        @test ce.Theta_x[2] == ir.values[1:H, 2, 3:3]
        @test ce.Theta_z[1] == ir.values[1:H, 3, 3:3]
        # draw count and hand-indexed layout check
        @test MEM.n_draws(ce) == size(ir._draws, 1)
        @test ce.Theta_x_draws[1][4, 1, 7] == ir._draws[7, 4, 1, 3]
        @test ce.Theta_z_draws[1][2, 1, 33] == ir._draws[33, 2, 3, 3]

        # H exceeding the stored horizon errors with the re-run hint
        err = try
            policy_causal_effects(ir, [3], [:y1 => 1]; H=13)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("longer horizon", err.msg)

        # no draws when ci_type = :none
        ir0 = irf(m, 12)
        ce0 = policy_causal_effects(ir0, [3], [:y1 => 1])
        @test ce0.Theta_x_draws === nothing
        @test MEM.n_draws(ce0) == 0

        # unknown selector errors
        @test_throws ArgumentError policy_causal_effects(ir, ["nope"], [:y1 => 1])
        @test_throws ArgumentError policy_causal_effects(ir, [3], [:y1 => "nope"])
    end

    @testset "Bayesian round-trip" begin
        post = estimate_bvar(Y, 2; n_draws=200)
        bir = irf(post, 12)
        H = 8
        ce = policy_causal_effects(bir, [1, 3], [:y1 => 1, :y2 => 2], [:rate => 3]; H=H)
        @test ce.source == :bvar
        @test MEM.n_shocks(ce) == 2
        @test MEM.n_draws(ce) == size(bir._draws, 1)
        @test ce.Theta_x[1] == bir.point_estimate[1:H, 1, [1, 3]]
        @test ce.Theta_x_draws[2][3, 2, 11] == bir._draws[11, 3, 2, 3]
    end

    @testset "instrument-impact normalization" begin
        ir = irf(m, 12; ci_type=:bootstrap, reps=40, rng=MersenneTwister(2))
        ce = policy_causal_effects(ir, [3], [:y1 => 1, :y2 => 2], [:rate => 3];
                                   H=10, normalize=:instrument_impact)
        @test ce.Theta_z[1][1, 1] ≈ 1.0 atol = 1e-14
        @test all(abs.(ce.Theta_z_draws[1][1, 1, :] .- 1.0) .< 1e-12)
        # ratios preserved: normalized outcome = raw outcome / raw instrument impact
        raw = policy_causal_effects(ir, [3], [:y1 => 1, :y2 => 2], [:rate => 3]; H=10)
        cscale = raw.Theta_z[1][1, 1]
        @test ce.Theta_x[1] ≈ raw.Theta_x[1] ./ cscale atol = 1e-12

        # normalization without instruments errors
        @test_throws ArgumentError policy_causal_effects(ir, [3], [:y1 => 1];
                                                         H=10, normalize=:instrument_impact)

        # near-zero-impact draws are dropped and counted
        vals = 0.5 .* ones(4, 2, 1)
        draws = ones(3, 4, 2, 1)
        draws[2, 1, 2, 1] = 1e-12          # draw 2: instrument impact ~ 0
        ir_syn = MEM.ImpulseResponse(vals, zeros(4, 2, 1), zeros(4, 2, 1), 4,
                                     ["x", "z"], ["s"], :bootstrap, draws, 0.9, nothing)
        ce_syn = @test_logs (:warn, r"dropped 1 of 3") match_mode = :any begin
            policy_causal_effects(ir_syn, [1], [:x => 1], [:z => 2];
                                  H=4, normalize=:instrument_impact)
        end
        @test MEM.n_draws(ce_syn) == 2
        @test all(abs.(ce_syn.Theta_z_draws[1][1, 1, :] .- 1.0) .< 1e-12)
    end

    @testset "sign-set route" begin
        check = irfarr -> irfarr[1, 1, 1] > 0
        s = identify_sign(m, 12, check; max_draws=300, store_all=true,
                          rng=MersenneTwister(3))
        med = irf_median(s)
        ce = policy_causal_effects(s, [2], [:y1 => 1], [:rate => 3]; H=9)
        @test ce.source == :sign_set
        @test ce.Theta_x[1] == med[1:9, 1, 2:2]
        @test MEM.n_draws(ce) == s.n_accepted
        @test ce.Theta_x_draws[1][5, 1, 4] == s.irf_draws[4, 5, 1, 2]
    end

    @testset "LP route" begin
        mlp = estimate_lp(Y, 1, 8)
        lpr = lp_irf(mlp)
        nresp = length(lpr.response_vars)
        @test nresp >= 1
        ce = policy_causal_effects(lpr, [:resp1 => 1]; n_draws=4000,
                                   rng=MersenneTwister(4))
        @test ce.source == :lp
        @test MEM.n_shocks(ce) == 1
        @test ce.shock_labels == [lpr.shock_var]
        @test ce.Theta_x[1][:, 1] == lpr.values[1:size(lpr.values, 1), 1]
        # sampled draws reproduce the se (independent-normal approximation)
        h = 3
        sd_est = std(ce.Theta_x_draws[1][h, 1, :])
        @test isapprox(sd_est, lpr.se[h, 1]; rtol=0.1)

        # convenience dispatch on the LP model itself
        ce2 = policy_causal_effects(mlp, [:resp1 => 1]; n_draws=10,
                                    rng=MersenneTwister(5))
        @test ce2.Theta_x[1] == ce.Theta_x[1]
    end

    @testset "baseline_path" begin
        ir = irf(m, 12; ci_type=:bootstrap, reps=30, rng=MersenneTwister(6))
        H = 10
        bp = baseline_path(ir, 2, [:y1 => 1, :y3 => 3], [:rate => 3]; H=H)
        @test bp isa BaselinePath{Float64}
        @test bp.H == H
        @test bp.label == ir.shocks[2]
        @test bp.x[1] == ir.values[1:H, 1, 2]
        @test bp.x[2] == ir.values[1:H, 3, 2]
        @test bp.z[1] == ir.values[1:H, 3, 2]
        @test size(bp.x_draws[1]) == (H, size(ir._draws, 1))
        @test bp.x_draws[1][4, 9] == ir._draws[9, 4, 1, 2]
        @test MEM.n_draws(bp) == size(ir._draws, 1)

        bn = baseline_path(ir, 2, [:y1 => 1]; H=H, negate=true)
        @test bn.x[1] == -bp.x[1]
        @test bn.x_draws[1] == -bp.x_draws[1]
        @test occursin("negated", bn.label)

        # Bayesian variant
        post = estimate_bvar(Y, 2; n_draws=100)
        bir = irf(post, 12)
        bpb = baseline_path(bir, 1, [:y2 => 2]; H=6)
        @test bpb.x[1] == bir.point_estimate[1:6, 2, 1]
        @test size(bpb.x_draws[1], 2) == size(bir._draws, 1)
    end

    @testset "Wold representation" begin
        Y2 = _cf04_data(MersenneTwister(11); T_obs=400, n=2)[:, 1:2]
        m1 = estimate_var(Y2, 1)
        H = 6
        w = wold_representation(m1; H=H)
        @test w isa WoldRepresentation{Float64}
        A1 = Matrix(m1.B[2:end, :]')
        L = Matrix(cholesky(Hermitian(m1.Sigma)).L)
        @test w.Theta[1, :, :] ≈ L atol = 1e-10
        @test w.Theta[2, :, :] ≈ A1 * L atol = 1e-10
        @test w.Theta[3, :, :] ≈ A1 * A1 * L atol = 1e-10
        @test w.Sigma_u == m1.Sigma
        @test w.draws === nothing

        # variance identity: sum_h Psi_h Sigma Psi_h' == sum_h Theta_h Theta_h'
        V_psi = zeros(2, 2)
        V_theta = zeros(2, 2)
        Ph = Matrix{Float64}(I, 2, 2)
        for h in 1:H
            V_psi .+= Ph * m1.Sigma * Ph'
            V_theta .+= w.Theta[h, :, :] * w.Theta[h, :, :]'
            Ph = A1 * Ph
        end
        @test V_psi ≈ V_theta atol = 1e-8

        # reduced-form option
        w0 = wold_representation(m1; H=H, orthogonalize=:none)
        @test w0.Theta[1, :, :] ≈ Matrix{Float64}(I, 2, 2) atol = 1e-14

        # posterior variant: dims + one hand-checked draw
        post = estimate_bvar(Y2, 1; n_draws=80)
        wb = wold_representation(post; H=4, max_draws=50)
        @test size(wb.draws) == (4, 2, 2, 50)
        Bd = post.B_draws[1, :, :]
        Sd = post.Sigma_draws[1, :, :]
        A1d = Matrix(Bd[2:end, :]')
        Ld = Matrix(cholesky(Hermitian(Sd)).L)
        @test wb.draws[1, :, :, 1] ≈ Ld atol = 1e-10
        @test wb.draws[2, :, :, 1] ≈ A1d * Ld atol = 1e-10

        @test_throws ArgumentError wold_representation(m1; H=0)
        @test_throws ArgumentError wold_representation(m1; H=4, orthogonalize=:ordering)
    end

    @testset "_stack" begin
        blocks = Dict(:a => [1.0, 2.0], :b => [3.0, 4.0])
        @test MEM._stack(blocks, [:b, :a]) == [3.0, 4.0, 1.0, 2.0]
        mats = Dict(:u => ones(2, 2), :v => zeros(1, 2))
        @test MEM._stack(mats, [:u, :v]) == [1.0 1.0; 1.0 1.0; 0.0 0.0]
        @test_throws ArgumentError MEM._stack(blocks, [:a, :missing])
        @test_throws ArgumentError MEM._stack(blocks, Symbol[])
    end
end
