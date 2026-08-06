# Shared fixtures for the CF-23 cross-method oracle suite (#403).
# All oracles are self-generated from package solvers — no replication numbers.

# Forward-looking 3-eq NK (NKPC + Euler + Taylor closure), demand + policy
# shocks. φπ is a parameter, so alternative Taylor closures come from _respec.
const ORC_NK = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
    endogenous: π, y, i
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
    i[t] = φπ * π[t] + eps_i[t]
end

# The same economy closed by a STRICT NGDP-level target (the policy equation
# is replaced; i is free to enforce it) — the direct solve for oracle 1.
const ORC_NK_NGDP = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0
    endogenous: π, y, i
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
    π[t] + y[t] - y[t-1] = 0 * eps_i[t]
end

# NK with a PERSISTENT demand state: the Lucas contrast (oracle 6) needs
# multi-period rule violations — in the static NK every baseline is
# one-period and naive/news constructions trivially coincide.
const ORC_NK_PERS = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5, ρd = 0.7
    endogenous: π, y, i, d
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + d[t]
    i[t] = φπ * π[t] + eps_i[t]
    d[t] = ρd * d[t-1] + eps_d[t]
end

# Purely backward spec: makes the Lucas-contrast mechanism visible (naive
# re-shocking and the news construction coincide without forward-looking terms).
const ORC_AR = @dsge begin
    parameters: ρ = 0.8, θ = 0.5
    endogenous: x, z
    exogenous: eps_i, eps_d
    x[t] = ρ * x[t-1] - θ * z[t] + eps_d[t]
    z[t] = 0.5 * x[t-1] + eps_i[t]
end

orc_respec(spec, changes::Dict{Symbol,Float64}) =
    MacroEconometricModels._respec(spec, merge(spec.param_values, changes))

# Baseline (rule A) objects shared across oracles.
function orc_nk_inputs(H)
    sol_A = solve(ORC_NK)
    ce = policy_news_matrix(ORC_NK, :eps_i, [:pi => :π, :ygap => :y],
                            [:rate => :i]; H=H)
    base = baseline_path(irf(sol_A, H), "eps_d",
                         [:pi => "π", :ygap => "y"], [:rate => "i"]; H=H)
    return sol_A, ce, base
end

# Direct-solve IRF paths of (π, y, i) to eps_d under a given spec.
function orc_direct_paths(spec, H)
    ir = irf(solve(spec), H)
    si = findfirst(==("eps_d"), ir.shocks)
    idx(v) = findfirst(==(v), ir.variables)
    return (pi=ir.values[1:H, idx("π"), si],
            y=ir.values[1:H, idx("y"), si],
            i=ir.values[1:H, idx("i"), si])
end
