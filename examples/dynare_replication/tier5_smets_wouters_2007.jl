# Tier 5: Smets & Wouters (2007) — Medium-Scale New Keynesian Model
# Dynare source: DSGE_mod/Smets_Wouters_2007/Smets_Wouters_2007.mod
# Reference: Smets, Frank and Wouters, Rafael (2007): "Shocks and Frictions
#   in US Business Cycles: A Bayesian DSGE Approach",
#   American Economic Review, 97(3), 586-606.
#
# Canonical medium-scale NK DSGE with 40 endogenous variables, 7 shocks,
# and model(linear) — all equations are pre-linearized around the balanced
# growth path. Variables represent log-deviations from steady state.
#
# This is the first model in our replication suite that uses `linear: true`.
using MacroEconometricModels, MAT, Printf, LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════
# Parameters — posterior mode values from usmodel_mode.mat
# ═══════════════════════════════════════════════════════════════════════════

# Fixed parameters
const ctou     = 0.025
const clandaw  = 1.5
const cg       = 0.18
const curvp    = 10.0
const curvw    = 10.0

# Estimated structural parameters (posterior mode)
const calfa       = 0.19280045641815527
const csigma      = 1.3951928979514387
const cfc         = 1.6149795879763305
const cgy         = 0.5261212194708431
const csadjcost   = 5.488197009060616
const chabb       = 0.7124006351787524
const cprobw      = 0.737541323772002
const csigl       = 1.9198838416864004
const cprobp      = 0.6562662602975502
const cindw       = 0.5919983094973862
const cindp       = 0.22835401911534914
const czcap       = 0.5472131292389921
const crpi        = 2.029467403441132
const crr         = 0.8153248720213849
const cry         = 0.08468690532858184
const crdy        = 0.22292570806394757

# Estimated AR parameters
const crhoa       = 0.9587740953362461
const crhob       = 0.18243934512556007
const crhog       = 0.9761614150464989
const crhoqs      = 0.7095693238736019
const crhoms      = 0.1271314763130675
const crhopinf    = 0.9038073405580113
const crhow       = 0.9718537740244471

# Estimated MA and observation parameters
const cmap        = 0.7448718466831306
const cmaw        = 0.8881459266182488
const constepinf  = 0.8179822205381722
const constebeta  = 0.16065411471321542
const constelab   = -0.10306516698580762
const ctrend      = 0.43202637481051603

# Shock standard deviations
const stderr_ea     = 0.45178828166212176
const stderr_eb     = 0.24246070101377046
const stderr_eg     = 0.5200103192082884
const stderr_eqs    = 0.45010690608083065
const stderr_em     = 0.23983932548400175
const stderr_epinf  = 0.141123850778673
const stderr_ew     = 0.24439160123349973

# ═══════════════════════════════════════════════════════════════════════════
# Derived compound parameters (Dynare #-variables)
# ═══════════════════════════════════════════════════════════════════════════
const cpie     = 1.0 + constepinf / 100.0
const cgamma   = 1.0 + ctrend / 100.0
const cbeta    = 1.0 / (1.0 + constebeta / 100.0)

const clandap  = cfc
const cbetabar = cbeta * cgamma^(-csigma)
const cr       = cpie / (cbeta * cgamma^(-csigma))
const crk      = cbeta^(-1) * cgamma^csigma - (1.0 - ctou)
const cw       = (calfa^calfa * (1.0 - calfa)^(1.0 - calfa) /
                  (clandap * crk^calfa))^(1.0 / (1.0 - calfa))
const cikbar   = 1.0 - (1.0 - ctou) / cgamma
const cik      = (1.0 - (1.0 - ctou) / cgamma) * cgamma
const clk      = ((1.0 - calfa) / calfa) * (crk / cw)
const cky      = cfc * clk^(calfa - 1.0)
const ciy      = cik * cky
const ccy      = 1.0 - cg - cik * cky
const crkky    = crk * cky
const cwhlc    = (1.0 / clandaw) * (1.0 - calfa) / calfa * crk * cky / ccy
const cwly     = 1.0 - crk * cky
const conster  = (cr - 1.0) * 100.0

# ═══════════════════════════════════════════════════════════════════════════
# Build DSGESpec programmatically (40 vars, 7 shocks)
# ═══════════════════════════════════════════════════════════════════════════
#
# We build the spec directly because the @dsge macro cannot interpolate
# runtime values. The equations are the same as in the .mod file.

# Variable ordering (must match Dynare for easy comparison)
endog = [:labobs, :robs, :pinfobs, :dy, :dc, :dinve, :dw,
         :ewma, :epinfma,
         :zcapf, :rkf, :kf, :pkf, :cf, :invef, :yf, :labf, :wf, :rrf,
         :mc, :zcap, :rk, :k, :pk, :c, :inve, :y, :lab, :pinf, :w, :r,
         :a, :b, :g, :qs, :ms, :spinf, :sw,
         :kpf, :kp]

exog = [:ea, :eb, :eg, :eqs, :em, :epinf_sh, :ew_sh]

# Parameter names and values — include both deep and derived params
param_names = [:ctou_p, :clandaw_p, :cg_p, :curvp_p, :curvw_p,
               :calfa_p, :csigma_p, :cfc_p, :cgy_p,
               :csadjcost_p, :chabb_p, :cprobw_p, :csigl_p, :cprobp_p,
               :cindw_p, :cindp_p, :czcap_p,
               :crpi_p, :crr_p, :cry_p, :crdy_p,
               :crhoa_p, :crhob_p, :crhog_p, :crhoqs_p, :crhoms_p,
               :crhopinf_p, :crhow_p,
               :cmap_p, :cmaw_p,
               :constepinf_p, :constebeta_p, :constelab_p, :ctrend_p,
               :cbetabar_p, :cgamma_p, :crk_p, :cikbar_p,
               :ccy_p, :ciy_p, :crkky_p, :cwhlc_p, :conster_p]

param_values = Dict{Symbol,Float64}(
    :ctou_p => ctou, :clandaw_p => clandaw, :cg_p => cg,
    :curvp_p => curvp, :curvw_p => curvw,
    :calfa_p => calfa, :csigma_p => csigma, :cfc_p => cfc, :cgy_p => cgy,
    :csadjcost_p => csadjcost, :chabb_p => chabb,
    :cprobw_p => cprobw, :csigl_p => csigl, :cprobp_p => cprobp,
    :cindw_p => cindw, :cindp_p => cindp, :czcap_p => czcap,
    :crpi_p => crpi, :crr_p => crr, :cry_p => cry, :crdy_p => crdy,
    :crhoa_p => crhoa, :crhob_p => crhob, :crhog_p => crhog,
    :crhoqs_p => crhoqs, :crhoms_p => crhoms,
    :crhopinf_p => crhopinf, :crhow_p => crhow,
    :cmap_p => cmap, :cmaw_p => cmaw,
    :constepinf_p => constepinf, :constebeta_p => constebeta,
    :constelab_p => constelab, :ctrend_p => ctrend,
    :cbetabar_p => cbetabar, :cgamma_p => cgamma,
    :crk_p => crk, :cikbar_p => cikbar,
    :ccy_p => ccy, :ciy_p => ciy, :crkky_p => crkky,
    :cwhlc_p => cwhlc, :conster_p => conster
)

# Variable index helpers
vi = Dict(s => i for (i, s) in enumerate(endog))
ei = Dict(s => i for (i, s) in enumerate(exog))

# Build residual functions: f_i(y_t, y_lag, y_lead, eps, theta) -> scalar
# Each equation is written as LHS - RHS = 0

function _build_sw07_residual_fns(vi, ei)
    fns = Function[]

    # Helper to access parameter values
    θ(p, th) = th[p]

    # ── (1) Production function: a = calfa*rkf + (1-calfa)*wf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:a]] - th[:calfa_p] * yt[vi[:rkf]] - (1.0 - th[:calfa_p]) * yt[vi[:wf]])

    # ── (2) Capital utilization (flex): zcapf = (1/(czcap/(1-czcap)))*rkf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:zcapf]] - (1.0 / (th[:czcap_p] / (1.0 - th[:czcap_p]))) * yt[vi[:rkf]])

    # ── (3) Rental rate (flex): rkf = wf + labf - kf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:rkf]] - yt[vi[:wf]] - yt[vi[:labf]] + yt[vi[:kf]])

    # ── (4) Effective capital (flex): kf = kpf(-1) + zcapf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:kf]] - yl[vi[:kpf]] - yt[vi[:zcapf]])

    # ── (5) Investment Euler (flex) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        cbg = th[:cbetabar_p] * th[:cgamma_p]
        yt[vi[:invef]] -
        (1.0 / (1.0 + cbg)) * yl[vi[:invef]] -
        (cbg / (1.0 + cbg)) * yle[vi[:invef]] -
        (1.0 / (th[:cgamma_p]^2 * th[:csadjcost_p])) / (1.0 + cbg) * yt[vi[:pkf]] -
        yt[vi[:qs]]
    end)

    # ── (6) Value of capital (flex) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        hab = th[:chabb_p] / th[:cgamma_p]
        c3inv = 1.0 / ((1.0 - hab) / (th[:csigma_p] * (1.0 + hab)))
        crk_val = th[:crk_p]
        yt[vi[:pkf]] + yt[vi[:rrf]] -
        c3inv * yt[vi[:b]] -
        (crk_val / (crk_val + 1.0 - th[:ctou_p])) * yle[vi[:rkf]] -
        ((1.0 - th[:ctou_p]) / (crk_val + 1.0 - th[:ctou_p])) * yle[vi[:pkf]]
    end)

    # ── (7) Consumption Euler (flex) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        hab = th[:chabb_p] / th[:cgamma_p]
        yt[vi[:cf]] -
        hab / (1.0 + hab) * yl[vi[:cf]] -
        (1.0 / (1.0 + hab)) * yle[vi[:cf]] -
        ((th[:csigma_p] - 1.0) * th[:cwhlc_p] / (th[:csigma_p] * (1.0 + hab))) * (yt[vi[:labf]] - yle[vi[:labf]]) +
        (1.0 - hab) / (th[:csigma_p] * (1.0 + hab)) * yt[vi[:rrf]] -
        yt[vi[:b]]
    end)

    # ── (8) Resource constraint (flex): yf = ccy*cf + ciy*invef + g + crkky*zcapf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:yf]] - th[:ccy_p] * yt[vi[:cf]] - th[:ciy_p] * yt[vi[:invef]] -
        yt[vi[:g]] - th[:crkky_p] * yt[vi[:zcapf]])

    # ── (9) Production function (flex): yf = cfc*(calfa*kf + (1-calfa)*labf + a) ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:yf]] - th[:cfc_p] * (th[:calfa_p] * yt[vi[:kf]] +
        (1.0 - th[:calfa_p]) * yt[vi[:labf]] + yt[vi[:a]]))

    # ── (10) Real wage (flex) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        hab = th[:chabb_p] / th[:cgamma_p]
        yt[vi[:wf]] - th[:csigl_p] * yt[vi[:labf]] -
        (1.0 / (1.0 - hab)) * yt[vi[:cf]] +
        (hab / (1.0 - hab)) * yl[vi[:cf]]
    end)

    # ── (11) Capital accumulation (flex): kpf = (1-cikbar)*kpf(-1) + cikbar*invef + cikbar*cgamma^2*csadjcost*qs ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:kpf]] - (1.0 - th[:cikbar_p]) * yl[vi[:kpf]] -
        th[:cikbar_p] * yt[vi[:invef]] -
        th[:cikbar_p] * th[:cgamma_p]^2 * th[:csadjcost_p] * yt[vi[:qs]])

    # ── (12) Marginal cost: mc = calfa*rk + (1-calfa)*w - a ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:mc]] - th[:calfa_p] * yt[vi[:rk]] - (1.0 - th[:calfa_p]) * yt[vi[:w]] + yt[vi[:a]])

    # ── (13) Capital utilization: zcap = (1/(czcap/(1-czcap)))*rk ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:zcap]] - (1.0 / (th[:czcap_p] / (1.0 - th[:czcap_p]))) * yt[vi[:rk]])

    # ── (14) Rental rate: rk = w + lab - k ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:rk]] - yt[vi[:w]] - yt[vi[:lab]] + yt[vi[:k]])

    # ── (15) Effective capital: k = kp(-1) + zcap ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:k]] - yl[vi[:kp]] - yt[vi[:zcap]])

    # ── (16) Investment Euler (sticky) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        cbg = th[:cbetabar_p] * th[:cgamma_p]
        yt[vi[:inve]] -
        (1.0 / (1.0 + cbg)) * yl[vi[:inve]] -
        (cbg / (1.0 + cbg)) * yle[vi[:inve]] -
        (1.0 / (th[:cgamma_p]^2 * th[:csadjcost_p])) / (1.0 + cbg) * yt[vi[:pk]] -
        yt[vi[:qs]]
    end)

    # ── (17) Value of capital (sticky) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        hab = th[:chabb_p] / th[:cgamma_p]
        c3inv = 1.0 / ((1.0 - hab) / (th[:csigma_p] * (1.0 + hab)))
        crk_val = th[:crk_p]
        yt[vi[:pk]] + yt[vi[:r]] - yle[vi[:pinf]] -
        c3inv * yt[vi[:b]] -
        (crk_val / (crk_val + 1.0 - th[:ctou_p])) * yle[vi[:rk]] -
        ((1.0 - th[:ctou_p]) / (crk_val + 1.0 - th[:ctou_p])) * yle[vi[:pk]]
    end)

    # ── (18) Consumption Euler (sticky) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        hab = th[:chabb_p] / th[:cgamma_p]
        yt[vi[:c]] -
        hab / (1.0 + hab) * yl[vi[:c]] -
        (1.0 / (1.0 + hab)) * yle[vi[:c]] -
        ((th[:csigma_p] - 1.0) * th[:cwhlc_p] / (th[:csigma_p] * (1.0 + hab))) * (yt[vi[:lab]] - yle[vi[:lab]]) +
        (1.0 - hab) / (th[:csigma_p] * (1.0 + hab)) * (yt[vi[:r]] - yle[vi[:pinf]]) -
        yt[vi[:b]]
    end)

    # ── (19) Resource constraint (sticky): y = ccy*c + ciy*inve + g + crkky*zcap ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:y]] - th[:ccy_p] * yt[vi[:c]] - th[:ciy_p] * yt[vi[:inve]] -
        yt[vi[:g]] - th[:crkky_p] * yt[vi[:zcap]])

    # ── (20) Production function (sticky): y = cfc*(calfa*k + (1-calfa)*lab + a) ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:y]] - th[:cfc_p] * (th[:calfa_p] * yt[vi[:k]] +
        (1.0 - th[:calfa_p]) * yt[vi[:lab]] + yt[vi[:a]]))

    # ── (21) NKPC (inflation) ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        cbg = th[:cbetabar_p] * th[:cgamma_p]
        denom = 1.0 + cbg * th[:cindp_p]
        slope = ((1.0 - th[:cprobp_p]) * (1.0 - cbg * th[:cprobp_p]) / th[:cprobp_p]) /
                ((th[:cfc_p] - 1.0) * th[:curvp_p] + 1.0)
        yt[vi[:pinf]] -
        (1.0 / denom) * (cbg * yle[vi[:pinf]] + th[:cindp_p] * yl[vi[:pinf]] + slope * yt[vi[:mc]]) -
        yt[vi[:spinf]]
    end)

    # ── (22) Wage Phillips curve ──
    push!(fns, (yt, yl, yle, eps, th) -> begin
        cbg = th[:cbetabar_p] * th[:cgamma_p]
        hab = th[:chabb_p] / th[:cgamma_p]
        wage_slope = (1.0 - th[:cprobw_p]) * (1.0 - cbg * th[:cprobw_p]) /
                     ((1.0 + cbg) * th[:cprobw_p]) *
                     (1.0 / ((th[:clandaw_p] - 1.0) * th[:curvw_p] + 1.0))
        yt[vi[:w]] -
        (1.0 / (1.0 + cbg)) * yl[vi[:w]] -
        (cbg / (1.0 + cbg)) * yle[vi[:w]] -
        (th[:cindw_p] / (1.0 + cbg)) * yl[vi[:pinf]] +
        (1.0 + cbg * th[:cindw_p]) / (1.0 + cbg) * yt[vi[:pinf]] -
        (cbg / (1.0 + cbg)) * yle[vi[:pinf]] -
        wage_slope * (th[:csigl_p] * yt[vi[:lab]] +
                      (1.0 / (1.0 - hab)) * yt[vi[:c]] -
                      (hab / (1.0 - hab)) * yl[vi[:c]] -
                      yt[vi[:w]]) -
        yt[vi[:sw]]
    end)

    # ── (23) Taylor rule ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:r]] -
        th[:crpi_p] * (1.0 - th[:crr_p]) * yt[vi[:pinf]] -
        th[:cry_p] * (1.0 - th[:crr_p]) * (yt[vi[:y]] - yt[vi[:yf]]) -
        th[:crdy_p] * (yt[vi[:y]] - yt[vi[:yf]] - yl[vi[:y]] + yl[vi[:yf]]) -
        th[:crr_p] * yl[vi[:r]] -
        yt[vi[:ms]])

    # ── (24) Capital accumulation (sticky): kp = (1-cikbar)*kp(-1) + cikbar*inve + cikbar*cgamma^2*csadjcost*qs ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:kp]] - (1.0 - th[:cikbar_p]) * yl[vi[:kp]] -
        th[:cikbar_p] * yt[vi[:inve]] -
        th[:cikbar_p] * th[:cgamma_p]^2 * th[:csadjcost_p] * yt[vi[:qs]])

    # ── (25) Technology: a = crhoa*a(-1) + ea ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:a]] - th[:crhoa_p] * yl[vi[:a]] - eps[ei[:ea]])

    # ── (26) Risk premium: b = crhob*b(-1) + eb ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:b]] - th[:crhob_p] * yl[vi[:b]] - eps[ei[:eb]])

    # ── (27) Government spending: g = crhog*g(-1) + eg + cgy*ea ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:g]] - th[:crhog_p] * yl[vi[:g]] - eps[ei[:eg]] - th[:cgy_p] * eps[ei[:ea]])

    # ── (28) Investment-specific technology: qs = crhoqs*qs(-1) + eqs ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:qs]] - th[:crhoqs_p] * yl[vi[:qs]] - eps[ei[:eqs]])

    # ── (29) Monetary policy: ms = crhoms*ms(-1) + em ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:ms]] - th[:crhoms_p] * yl[vi[:ms]] - eps[ei[:em]])

    # ── (30) Price markup (ARMA): spinf = crhopinf*spinf(-1) + epinf - cmap*epinfma(-1) ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:spinf]] - th[:crhopinf_p] * yl[vi[:spinf]] -
        eps[ei[:epinf_sh]] + th[:cmap_p] * yl[vi[:epinfma]])

    # ── (31) Price markup MA auxiliary: epinfma = epinf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:epinfma]] - eps[ei[:epinf_sh]])

    # ── (32) Wage markup (ARMA): sw = crhow*sw(-1) + ew - cmaw*ewma(-1) ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:sw]] - th[:crhow_p] * yl[vi[:sw]] -
        eps[ei[:ew_sh]] + th[:cmaw_p] * yl[vi[:ewma]])

    # ── (33) Wage markup MA auxiliary: ewma = ew ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:ewma]] - eps[ei[:ew_sh]])

    # ── (34) Output growth: dy = y - y(-1) + ctrend ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:dy]] - yt[vi[:y]] + yl[vi[:y]] - th[:ctrend_p])

    # ── (35) Consumption growth: dc = c - c(-1) + ctrend ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:dc]] - yt[vi[:c]] + yl[vi[:c]] - th[:ctrend_p])

    # ── (36) Investment growth: dinve = inve - inve(-1) + ctrend ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:dinve]] - yt[vi[:inve]] + yl[vi[:inve]] - th[:ctrend_p])

    # ── (37) Wage growth: dw = w - w(-1) + ctrend ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:dw]] - yt[vi[:w]] + yl[vi[:w]] - th[:ctrend_p])

    # ── (38) Inflation observation: pinfobs = pinf + constepinf ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:pinfobs]] - yt[vi[:pinf]] - th[:constepinf_p])

    # ── (39) Interest rate observation: robs = r + conster ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:robs]] - yt[vi[:r]] - th[:conster_p])

    # ── (40) Hours observation: labobs = lab + constelab ──
    push!(fns, (yt, yl, yle, eps, th) ->
        yt[vi[:labobs]] - yt[vi[:lab]] - th[:constelab_p])

    return fns
end

residual_fns = _build_sw07_residual_fns(vi, ei)

# Verify equation count
@assert length(residual_fns) == length(endog) "Need $(length(endog)) equations, got $(length(residual_fns))"

# ── Determine forward-looking equations ──
# Equations with [t+1] terms: (5) invef, (6) pkf, (7) cf, (9) yf (implicit via labf+1?),
# Actually let me identify from the residual_fns directly:
forward_indices = Int[]
n = length(endog)
y_test = zeros(n)
ε_test = zeros(length(exog))
for (i, fn) in enumerate(residual_fns)
    # Check if changing y_lead changes the residual
    is_fwd = false
    for j in 1:n
        y_lead_plus = copy(y_test)
        y_lead_plus[j] = 1.0
        r1 = fn(y_test, y_test, y_test, ε_test, param_values)
        r2 = fn(y_test, y_test, y_lead_plus, ε_test, param_values)
        if abs(r2 - r1) > 1e-12
            is_fwd = true
            break
        end
    end
    if is_fwd
        push!(forward_indices, i)
    end
end
n_expect = length(forward_indices)
println("Forward-looking equations: $n_expect")
println("  Indices: $forward_indices")

# Equations as placeholder expressions (not used for linear models)
equations = Expr[:(0 + 0) for _ in 1:n]

# Build spec
spec = MacroEconometricModels.DSGESpec{Float64}(
    endog, exog, param_names, param_values,
    equations, residual_fns,
    n_expect, forward_indices, Float64[], nothing;
    linear=true
)

println("\nModel built: $(spec.n_endog) endogenous, $(spec.n_exog) exogenous")
println("linear = ", spec.linear)

# ═══════════════════════════════════════════════════════════════════════════
# Solve
# ═══════════════════════════════════════════════════════════════════════════
sol = solve(spec; method=:gensys)
println("\nis_determined = ", is_determined(sol))
println("is_stable     = ", is_stable(sol))

# Check that model variables have zero SS
model_var_start = vi[:zcapf]  # first model variable after obs/aux
model_ss = sol.spec.steady_state[model_var_start:end]
println("All model-variable SS = 0? ", all(model_ss .== 0.0))

# ═══════════════════════════════════════════════════════════════════════════
# Compare with Dynare steady state
# ═══════════════════════════════════════════════════════════════════════════
dynare_mat = joinpath(@__DIR__, "dynare_results", "smets_wouters_2007.mat")
if !isfile(dynare_mat)
    println("\n  (No Dynare .mat file found — skipping comparison)")
    exit(1)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
d_names_raw = data["endo_names"]
if d_names_raw isa Matrix
    d_names = vec([strip(string(d_names_raw[i,1])) for i in 1:size(d_names_raw, 1)])
else
    d_names = vec([strip(string(x)) for x in d_names_raw])
end

# Build Dynare variable name → index map
dynare_idx = Dict{String,Int}()
for (i, name) in enumerate(d_names)
    dynare_idx[name] = i
end

# The effective SS = (I - G1)^{-1} * C_sol
effective_ss = (I - sol.G1) \ sol.C_sol

our_names = string.(spec.endog)

println("\n" * "=" ^ 70)
println("  Smets & Wouters (2007) — Steady State Comparison")
println("=" ^ 70)
@printf("  %-12s %14s %14s %10s %s\n", "Variable", "Julia", "Dynare", "diff", "")
all_ss_pass = true
for (i, name) in enumerate(our_names)
    di = get(dynare_idx, name, 0)
    di == 0 && continue
    j_val = effective_ss[i]
    d_val = d_ss[di]
    diff = abs(j_val - d_val)
    ok = diff < 1e-4
    global all_ss_pass = all_ss_pass && ok
    @printf("  %-12s %14.8f %14.8f %10.2e %s\n",
            name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
end
println("  Steady State: ", all_ss_pass ? "ALL PASS" : "SOME FAIL")

# ═══════════════════════════════════════════════════════════════════════════
# Generate and compare IRFs (self-consistency check)
# ═══════════════════════════════════════════════════════════════════════════
H = 40
ir = irf(sol, H)

# Print sample IRFs for key variables to technology shock (ea)
println("\n=== Sample IRFs to technology shock (ea, 1 std dev = $stderr_ea) ===")
ea_idx = ei[:ea]
@printf("  %-8s  %10s %10s %10s %10s %10s\n", "Horizon", "y", "c", "inve", "pinf", "r")
for h in [1, 2, 5, 10, 20, 40]
    @printf("  %-8d  %10.6f %10.6f %10.6f %10.6f %10.6f\n",
            h,
            ir.values[h, vi[:y], ea_idx] * stderr_ea,
            ir.values[h, vi[:c], ea_idx] * stderr_ea,
            ir.values[h, vi[:inve], ea_idx] * stderr_ea,
            ir.values[h, vi[:pinf], ea_idx] * stderr_ea,
            ir.values[h, vi[:r], ea_idx] * stderr_ea)
end

# Monetary policy shock
println("\n=== Sample IRFs to monetary policy shock (em, 1 std dev = $stderr_em) ===")
em_idx = ei[:em]
@printf("  %-8s  %10s %10s %10s %10s %10s\n", "Horizon", "y", "c", "inve", "pinf", "r")
for h in [1, 2, 5, 10, 20, 40]
    @printf("  %-8d  %10.6f %10.6f %10.6f %10.6f %10.6f\n",
            h,
            ir.values[h, vi[:y], em_idx] * stderr_em,
            ir.values[h, vi[:c], em_idx] * stderr_em,
            ir.values[h, vi[:inve], em_idx] * stderr_em,
            ir.values[h, vi[:pinf], em_idx] * stderr_em,
            ir.values[h, vi[:r], em_idx] * stderr_em)
end

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("  Summary — Smets & Wouters (2007)")
println("  Endogenous:  $(spec.n_endog)")
println("  Exogenous:   $(spec.n_exog)")
println("  Forward:     $n_expect")
println("  Determined:  $(is_determined(sol))")
println("  Stable:      $(is_stable(sol))")
println("  SS match:    $(all_ss_pass ? "PASS" : "FAIL")")
println("=" ^ 70)
