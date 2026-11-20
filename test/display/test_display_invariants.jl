# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Display invariants (T176 / #275). Renders every fixture through `_render` and
# asserts the RAW-string properties that the Stage-11 display batch (#260–#274)
# established. Unlike the goldens, these are regex/`occursin` checks — they are
# cross-platform robust and fail loudly the moment a future edit reintroduces a
# cropped table, a `-0.0`, a raw boolean, or a stray blank band. Each invariant
# names the audit defect it locks.

using Test
using MacroEconometricModels

isdefined(@__MODULE__, :build_display_fixtures) || include(joinpath(@__DIR__, "display_helpers.jl"))

# S1 — non-TTY cropping is disabled: no PrettyTables "… columns/rows omitted" footer.
_no_crop(s)     = !occursin("omitted", s)
# S2 — `_fmt` normalizes negative zero: no `-0.0`, `-0.00`, `-0.0000` cell (a value
# like `-0.0283` is fine — only an all-zero-after-the-point token is a defect).
_no_neg_zero(s) = !occursin(r"-0\.0+(?![0-9])", s)
# S2 — no raw un-`_fmt`'d float leaking in scientific form. `_fmt`'s `%.3g` fallback
# for tiny/huge cells caps at 2 mantissa decimals (e.g. `4.46e-12`); a token with ≥3
# mantissa decimals before the exponent is an unformatted `round()`/`string(float)` leak.
_no_raw_exp(s)  = !occursin(r"\.\d{3,}[eE][-+]?\d", s)
# S7 — empty column-label rows are suppressed, so no degenerate blank band. Section
# separators legitimately span up to two blank lines; three or more never occur.
_no_blank_band(s) = !occursin("\n\n\n\n", s)
# S8 — booleans render as Yes/No, never raw `true`/`false`, anywhere in the table.
_no_raw_bool(s) = !occursin(r"\b(?:true|false)\b", s)
# S8 (#407) — `cov_type` renders through `_label`, never `string(sym)`. Every label is
# capitalized or parenthesized, so a value that is a bare lower-case internal token
# (`cluster`, `driscoll_kraay`, `pcse`) is a raw-symbol leak.
function _no_raw_cov_type(s)
    for line in split(s, '\n')
        occursin("Cov. type", line) || continue
        val = strip(replace(line, r"^.*Cov\. type" => ""))
        occursin(r"^[a-z][a-z0-9_]*$", val) && return false
    end
    return true
end

@testset "Display invariants (T176/#275)" begin
    fixtures = display_fixtures()
    @test length(fixtures) >= 15

    @testset "$(f.name)" for f in fixtures
        s = _render(f.obj)

        @test !isempty(strip(s))            # the fixture actually rendered something
        @test _no_crop(s)                   # S1
        @test _no_neg_zero(s)               # S2
        @test _no_raw_exp(s)                # S2
        @test _no_blank_band(s)             # S7
        @test _no_raw_bool(s)               # S8
        @test _no_raw_cov_type(s)           # S8 / #407

        if f.stars
            # A high-SNR estimate → the coefficient table must carry a significance star.
            @test occursin("*", s)
        end

        if f.ref
            # S6 — the reference row is labelled `(ref)`, carries dashes for the
            # estimate/stat/CI, and never prints a computed significance star.
            @test occursin("—", s)
            reflines = filter(l -> occursin("(ref)", l), split(s, '\n'))
            @test length(reflines) == 1
            @test occursin("—", reflines[1]) && !occursin("*", reflines[1])
        end
    end
end

@testset "v1.1.0 display, report & refs invariants" begin
    rng = Xoshiro(9911)
    sim_q = dgp_quantile_factors(rng; T_obs=80, N=12)
    qfm = estimate_qfm(sim_q.X, 2, 0.25; se=:none, seed=11)
    qfm_ic = ic_criteria_qfm(sim_q.X, 4, 0.25)
    gar = growth_at_risk(sim_q.X, 1, 1; r=2, taus=[0.1, 0.5, 0.9], density=:normal, seed=11)
    Y_qv = randn(rng, 80, 2)
    qvar = estimate_qvar(Y_qv, 1, [0.25, 0.75]; se=:asymptotic)

    sim_g = dgp_infinite_gdfm(rng; T_obs=100, N=12, q=2)
    fhlz = estimate_gdfm_fhlz(sim_g.X, 2; p_max=2, n_perm=4, bandwidth=3, seed=11)
    gdfm_base = estimate_gdfm(sim_g.X, 2; r=3)
    bp = bandpass_gdfm(gdfm_base, (0.0, π / 4))
    ah = ahn_horenstein(sim_g.X, 4)

    sim_sv = dgp_ssvarma(rng; T=120, Phi=[0.35 0.0; 0.0 0.3], Theta=zeros(2, 2, 0),
                         C=[1.0 0.2; -0.15 0.9], dist=:student_t, nu=5.0)
    var_sv = estimate_var(sim_sv.Y, 1)
    lms = estimate_lms_svar(var_sv; distributions=[:student_t, :student_t],
                            method=:three_step, n_starts=1, max_iter=50, seed=11)
    ht = test_svar_hypothesis(lms, recursiveness_hypothesis(2); method=:wald)

    sys_ab = ABCDSystem([0.7 0.0; 0.0 0.5], [1.0 0.0; 0.0 1.0],
                        [1.0 0.3; 0.0 0.8], [1.0 0.1; 0.0 1.0])
    inv_res = poor_mans_invertibility(sys_ab)
    Y_dv = randn(rng, 60, 2)
    dv_prof = select_dsge_var_lambda(sys_ab, Y_dv, 1; lambda_grid=[0.5, 2.0, Inf])

    prob_ri = RationalInattentionProblem([0.85;;], [1.0;;], [1.0;;]; capacity=1.0)
    sol_ri = solve_ri_kalman(prob_ri)
    ssk_ri = MacroEconometricModels.SteadyStateKalman(
        sol_ri.Sigma_prior, sol_ri.Sigma_post, sol_ri.K_gain,
        [0.85;;] - sol_ri.K_gain * [1.0;;], ComplexF64[0.3 + 0.0im])
    U_dc = randn(rng, 3, 2)
    dc_ri = solve_ri_discrete(U_dc, [0.4, 0.6]; omega=1.0)

    objs = (qfm, qfm_ic, gar, qvar, fhlz, bp, ah, lms, ht, sys_ab, inv_res, dv_prof, prob_ri, sol_ri, ssk_ri, dc_ri)
    for obj in objs
        txt = _render(obj)
        @test !isempty(strip(txt))
        @test _no_crop(txt)
        @test _no_blank_band(txt)
        for b in (:text, :latex, :html)
            rtxt = with_display_backend(b) do
                sprint(io -> report(io, obj))
            end
            @test !isempty(strip(rtxt))
        end
        for f in (:text, :latex, :html, :bibtex)
            reftxt = sprint(io -> refs(io, obj; format=f))
            @test !isempty(strip(reftxt))
        end
    end
end
