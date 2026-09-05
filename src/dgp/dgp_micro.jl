# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-14 (#803) / DGP-15 (#804) / DGP-16 (#805): micro
# simulators — cross-section (14 kinds), linear/nonlinear panel with
# correlated effects, and staggered difference-in-differences.


_logistic(x) = 1 / (1 + exp(-x))

"""
    dgp_cross_section(rng; kind, beta, n, hetero, cluster_rho, G, endog_rho,
                      pi1, overid_k, invalid_k, cutpoints, dispersion, censor,
                      select_rho, iia_rho) -> NamedTuple

Cross-section DGP, `kind ∈ :ols|:hc|:cluster|:iv|:logit|:probit|:ordered|`
`:mlogit|:poisson|:nb|:tobit|:truncreg|:heckman|:qreg|:rdd`.
Returns `(y, X, beta, …)` plus kind-specific truth: `:cluster` → `clust`
ids and Moulton `rho`; `:iv` → `Z` instruments (first `invalid_k` correlated
with the error); `:ordered` → `cutpoints`; `:nb` → `dispersion`;
`:heckman` → `select_rho`; `:rdd` → `cutoff`, `tau`; discrete choice →
closed-form `ame` at the truth. Heteroskedastic binary choice
(`kind = :logit/:probit` with `hetero = true`, `Var(ε|x)` varying) is the
sandwich-direction design.
"""
function dgp_cross_section(rng::AbstractRNG; kind::Symbol=:ols,
                           beta=[1.0, 0.5], n::Int=1000,
                           hetero::Bool=false, cluster_rho::Float64=0.5,
                           G::Int=50, endog_rho::Float64=0.6,
                           pi1::Float64=1.0, overid_k::Int=2,
                           invalid_k::Int=0,
                           cutpoints=[-0.5, 0.5, 1.5],
                           dispersion::Float64=1.5, censor::Float64=0.0,
                           select_rho::Float64=0.5, iia_rho::Float64=0.0)
    be = Vector{Float64}(beta)
    k = length(be)
    X = hcat(ones(n), randn(rng, n, k - 1))
    extras = Dict{Symbol,Any}()
    if kind === :ols || kind === :hc
        e = hetero ? (0.5 .+ abs.(X[:, 2])) .* randn(rng, n) : randn(rng, n)
        y = X * be + e
    elseif kind === :cluster
        cl = rand(rng, 1:G, n)
        ce = randn(rng, G)
        e = sqrt(cluster_rho) .* ce[cl] + sqrt(1 - cluster_rho) .* randn(rng, n)
        y = X * be + e
        extras[:clust] = cl
        extras[:rho] = cluster_rho
    elseif kind === :iv
        m = 1 + overid_k
        Z = hcat(ones(n), randn(rng, n, m - 1))
        v = randn(rng, n)
        u = endog_rho .* v + sqrt(1 - endog_rho^2) .* randn(rng, n)
        if invalid_k > 0  # first invalid_k (non-constant) instruments load on u
            Z[:, 2:(1 + invalid_k)] .+= 0.8 .* u
        end
        x_endog = Z[:, 2] * pi1 + v
        XX = k > 2 ? hcat(ones(n), x_endog, X[:, 3:end]) : hcat(ones(n), x_endog)
        y = XX * be[1:size(XX, 2)] + u
        extras[:Z] = Z
        extras[:pi1] = pi1
        return merge((y=y, X=XX, beta=be[1:size(XX, 2)]), extras)
    elseif kind === :logit || kind === :probit
        eta = X * be
        F = kind === :logit ? _logistic : x -> cdf(Normal(), x)
        sc = hetero ? (0.5 .+ abs.(X[:, 2])) : ones(n)
        p = F.(eta ./ sc)
        y = rand(rng, n) .< p
        f = kind === :logit ? (@. p * (1 - p)) : (@. pdf(Normal(), eta))
        extras[:ame] = fill(mean(f), length(be)) .* be
        extras[:hetero] = hetero
    elseif kind === :ordered
        cp = Vector{Float64}(cutpoints)
        eta = X * be + randn(rng, n)
        y = [searchsortedfirst(cp, eta[i]) for i in 1:n]
        extras[:cutpoints] = cp
    elseif kind === :mlogit
        J = 3
        Bj = [be, be .* 0.5 .+ (iia_rho != 0.0 ? iia_rho : 0.0), -be .* 0.5]
        U = hcat([X * Bj[j] + rand(rng, Gumbel(), n) for j in 1:J]...)
        if iia_rho != 0.0  # nested-logit-style correlated errors break IIA
            U[:, 2] += iia_rho * U[:, 1]
        end
        y = [argmax(U[i, :]) for i in 1:n]
        extras[:iia_rho] = iia_rho
    elseif kind === :poisson || kind === :nb
        lam = exp.(X * be)
        y = kind === :poisson ? rand.(rng, Poisson.(lam)) :
                                rand.(rng, NegativeBinomial.(dispersion,
                                    dispersion ./ (dispersion .+ lam)))
        extras[:dispersion] = dispersion
    elseif kind === :tobit || kind === :truncreg
        ystar = X * be + randn(rng, n)
        y = kind === :tobit ? max.(ystar, censor) : ystar[ystar .> censor]
        extras[:censor] = censor
        kind === :truncreg && (X = X[ystar .> censor, :])
    elseif kind === :heckman
        g = [0.3, 0.8]
        Zg = hcat(ones(n), X[:, 2])
        v = randn(rng, n)
        u = select_rho .* v + sqrt(1 - select_rho^2) .* randn(rng, n)
        s = (Zg * g + v) .> 0
        y = fill(NaN, n)
        y[s] = (X * be + u)[s]
        extras[:select_rho] = select_rho
        extras[:selected] = s
    elseif kind === :qreg
        e = (1 .+ 0.5 .* abs.(X[:, 2])) .* randn(rng, n)
        y = X * be + e
    elseif kind === :rdd
        r = 2 .* rand(rng, n) .- 1
        cutoff = 0.0
        tau = 1.0
        y = 0.5 .* r + tau .* (r .>= cutoff) + randn(rng, n)
        extras[:cutoff] = cutoff
        extras[:tau] = tau
        extras[:r] = r
    else
        throw(ArgumentError("unknown kind :$kind"))
    end
    return merge((y=y, X=X, beta=be), extras)
end

"""
    dgp_panel(rng; N, T, beta, sigma_u, sigma_e, corr_alpha_x, rho_ar,
              common_shock, hetero, dynamic_rho, kind) -> NamedTuple

Linear panel `y_{it} = α_i + x_{it}′β + e_{it}` with Mundlak correlated
effects `α_i = x̄_i′ξ + u_i` (`corr_alpha_x` scales `ξ`; 0 = random
effects), AR(1) within-entity errors (`rho_ar`), an optional common time
shock, and an optional lagged dependent variable (`dynamic_rho > 0`,
Nickell-bias design). `kind ∈ :linear|:logit|:probit` (nonlinear kinds
use a RE-probit/logit link with planted `sigma_u`). Returns
`(df, beta, sigma_u, sigma_e, alpha, mundlak)` with a long DataFrame.
"""
function dgp_panel(rng::AbstractRNG; N::Int=200, T::Int=20, beta=[1.0, 0.5],
                   sigma_u::Float64=1.0, sigma_e::Float64=1.0,
                   corr_alpha_x::Float64=0.0, rho_ar::Float64=0.0,
                   common_shock::Bool=false, hetero::Bool=false,
                   dynamic_rho::Float64=0.0, kind::Symbol=:linear)
    be = Vector{Float64}(beta)
    k = length(be)
    id = repeat(1:N, inner=T)
    time = repeat(1:T, outer=N)
    X = randn(rng, N * T, k)
    xi = corr_alpha_x .* ones(k)
    alpha = zeros(N)
    lam = common_shock ? randn(rng, T) : zeros(T)
    y = zeros(N * T)
    e_prev = zeros(N)
    for i in 1:N
        rows = ((i - 1) * T + 1):(i * T)
        xbar = vec(mean(X[rows, :], dims=1))
        alpha[i] = dot(xbar, xi) + sigma_u * randn(rng)
        for (s, t) in enumerate(rows)
            e_prev[i] = rho_ar * e_prev[i] +
                        sqrt(1 - rho_ar^2) * sigma_e * randn(rng)
            sc = hetero ? (0.5 + abs(X[t, 1])) : 1.0
            idx = alpha[i] + dot(X[t, :], be) + lam[s] + sc * e_prev[i]
            if kind === :linear
                y[t] = idx + (dynamic_rho > 0 && s > 1 ? dynamic_rho * y[t - 1] : 0.0)
            else
                p = kind === :logit ? _logistic(idx) : cdf(Normal(), idx)
                y[t] = rand(rng) < p ? 1.0 : 0.0
            end
        end
    end
    df = DataFrame(id=id, time=time, y=y)
    for j in 1:k
        df[!, Symbol("x$j")] = X[:, j]
    end
    return (df=df, beta=be, sigma_u=sigma_u, sigma_e=sigma_e, alpha=alpha,
            mundlak=xi, rho_ar=rho_ar, dynamic_rho=dynamic_rho)
end

"""
    dgp_staggered_did(rng; cohorts, tau, never_treated_share, N, T,
                      violate_pt, covariate_effect, cluster_rho) -> NamedTuple

Staggered DiD: units in `cohorts` adopt at `g`, never-treated share
`never_treated_share`; `tau(g, e)` is the effect for cohort `g` at event
time `e` (heterogeneous by default). `violate_pt > 0` adds a pre-trend
slope (parallel-trends violation). Returns `(df, att_by_event_time,
att_by_cohort, overall_att, cohort_of)` with a long DataFrame.
"""
function dgp_staggered_did(rng::AbstractRNG; cohorts=[6, 11, 16],
                           tau=(g, e) -> 1.0 + 0.1 * e + 0.05 * (g - 6),
                           never_treated_share::Float64=0.3, N::Int=300,
                           T::Int=25, violate_pt::Float64=0.0,
                           covariate_effect::Float64=0.0,
                           cluster_rho::Float64=0.0)
    G = length(cohorts)
    g_of = Vector{Union{Int,Nothing}}(undef, N)
    for i in 1:N
        g_of[i] = rand(rng) < never_treated_share ? nothing :
                  cohorts[rand(rng, 1:G)]
    end
    alpha = randn(rng, N)
    lambda = randn(rng, T)
    att_e = Dict{Int,Vector{Float64}}()
    att_c = Dict{Union{Int,Nothing},Float64}()
    num, den = 0.0, 0
    id, time, y, D = Int[], Int[], Float64[], Int[]
    for i in 1:N, t in 1:T
        g = g_of[i]
        e = g === nothing ? -999 : t - g
        te = (g === nothing || e < 0) ? 0.0 : tau(g, e)
        pre = (g !== nothing && e < 0) ? violate_pt * e : 0.0
        ce = covariate_effect * (i / N)
        ei = sqrt(cluster_rho) * alpha[i] + sqrt(1 - cluster_rho) * randn(rng)
        push!(id, i)
        push!(time, t)
        push!(y, alpha[i] + lambda[t] + te + pre + ce + ei)
        push!(D, (g !== nothing && t >= g) ? 1 : 0)
        if g !== nothing && e >= 0
            push!(get!(att_e, e, Float64[]), te)
            num += te
            den += 1
        end
    end
    att_by_e = Dict{Int,Float64}(e => mean(v) for (e, v) in att_e)
    for g in cohorts
        vals = Float64[]
        for i in 1:N
            g_of[i] == g || continue
            for t in g:T
                push!(vals, tau(g, t - g))
            end
        end
        att_c[g] = mean(vals)
    end
    df = DataFrame(id=id, time=time, y=y, D=D,
                   cohort=[g === nothing ? 0 : g for g in g_of[id]])
    return (df=df, att_by_event_time=att_by_e, att_by_cohort=att_c,
            overall_att=num / max(den, 1), cohort_of=g_of)
end
