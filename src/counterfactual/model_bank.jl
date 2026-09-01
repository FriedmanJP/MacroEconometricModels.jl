# Counterfactual module — CMW model bank (CF-17, #397)
#
# CMW (2025) discipline structural models by limited-information IRF
# matching: a Gaussian quasi-likelihood of the CF-06 stacked target at the
# model-implied target, where the model side is the news menu Φ(ψ) times a
# RESTRICTED-GLS best-fit news vector (only the first H_news news shocks per
# empirical shock dimension — rule-free: the estimator matches the induced
# instrument path, never a single time-0 shock under an assumed rule).
# Posteriors via RWMH; marginal likelihoods via the existing Geweke MHM
# (_geweke_mhm); posterior model probabilities and (model, draw) pooling make
# model uncertainty a first-class part of the counterfactual bands.

# --- Φ construction: menu -> target-ordered design ---------------------------

# Precompute, once, how target.index rows pull elements out of a menu
# container: (which Theta list, entry, menu row) per target row, and the
# empirical-shock block each row belongs to.
function _bank_index_maps(ce::PolicyCausalEffects{T}, index) where {T<:AbstractFloat}
    shocks = sort(unique(e.shock for e in index))
    blk = Dict(s => i for (i, s) in enumerate(shocks))
    rows = Vector{NTuple{4,Int}}(undef, length(index))   # (is_z, entry, h, block)
    for (r, e) in enumerate(index)
        i = findfirst(==(e.var), ce.outcomes)
        k = i === nothing ? findfirst(==(e.var), ce.instruments) : nothing
        (i === nothing && k === nothing) && throw(ArgumentError(
            "target variable :$(e.var) not found in the model menu (outcomes $(ce.outcomes), instruments $(ce.instruments))"))
        e.h <= ce.H || throw(ArgumentError(
            "target horizon $(e.h) exceeds the model menu H = $(ce.H)"))
        rows[r] = i === nothing ? (1, k, e.h, blk[e.shock]) : (0, i, e.h, blk[e.shock])
    end
    return rows, length(shocks)
end

# Fill the restricted design Φ_H (m × n_blocks·H_news) from a menu container.
function _bank_phi(ce::PolicyCausalEffects{T}, rows::Vector{NTuple{4,Int}},
                   n_blocks::Int, H_news::Int) where {T<:AbstractFloat}
    H_news <= n_shocks(ce) || throw(ArgumentError(
        "H_news = $H_news exceeds the menu's news horizon count $(n_shocks(ce))"))
    m = length(rows)
    Phi = zeros(T, m, n_blocks * H_news)
    for (r, (is_z, i, h, b)) in enumerate(rows)
        M = is_z == 1 ? ce.Theta_z[i] : ce.Theta_x[i]
        for j in 1:H_news
            Phi[r, (b-1)*H_news+j] = M[h, j]
        end
    end
    return Phi
end

# Quasi-log-likelihood at one menu: restricted GLS fit + Gaussian level
# constant (the missing-logdet bug class — CMW's own fix).
function _bank_loglik(ce::PolicyCausalEffects{T}, rows, n_blocks::Int,
                      H_news::Int, Cw::Matrix{T}, btil::Vector{T},
                      const_term::T) where {T<:AbstractFloat}
    Phi = _bank_phi(ce, rows, n_blocks, H_news)
    res = _policy_projection(Cw * Phi, btil; method=:ls)
    return -sum(abs2, res.error_path) / 2 + const_term
end

"""
    irf_match(menu_builder, target, priors, param_names;
              name="model", H_news=25, n_adapt=20_000, n_burn=10_000,
              n_keep=20_000, thin=20, init=nothing, proposal_scale=0.36,
              T_store=0, rng=Random.default_rng()) -> ModelBankMember

Estimate one model-bank member by limited-information IRF matching (CMW §4):

- `menu_builder(ψ) -> PolicyCausalEffects` — a user closure composing the
  model's **square** news menu at parameter ψ (e.g. CF-07's
  `policy_news_matrix` on a re-parameterized `@dsge` spec, optionally through
  CF-09's `behavioral`). The bank is model-agnostic by design.
- `target = (; theta_hat, V_bar, index)` from [`stacked_irf_target`](@ref)
  (+ [`ctw_covariance`](@ref)); the **non-diagonal** V is load-bearing — a
  diagonal V makes model probabilities artificially decisive.
- The model-implied target is `Φ_H·ν̃*` with the restricted GLS
  `ν̃* = (Φ_H'V⁻¹Φ_H)⁻¹Φ_H'V⁻¹θ̂` fit per empirical shock block over the
  first `H_news` news columns (overfitting guard).
- The quasi-likelihood includes the Gaussian level constant
  `−½·logdet(V) − (m/2)·log 2π`, so marginal likelihoods are comparable
  across members.
- Sampler: adaptive-then-frozen RWMH (proposal covariance from the adaptive
  history, scaled by `proposal_scale`; frozen before burn-in — the #137
  discipline). Menu-build failures (indeterminacy, solver errors,
  `ArgumentError`s from the builder) score `−Inf` and are counted.
- `T_store > 0` truncates the stored posterior menus to a `T_store×T_store`
  horizon (memory honesty: a 1000-draw 3-variable 200×200 bank is ≈ 1 GB in
  Float64); `0` keeps the full menu horizon.
"""
function irf_match(menu_builder::Function, target::NamedTuple,
                   priors::AbstractVector{<:Distribution},
                   param_names::AbstractVector{Symbol};
                   name::AbstractString="model",
                   H_news::Int=25,
                   n_adapt::Int=20_000, n_burn::Int=10_000,
                   n_keep::Int=20_000, thin::Int=20,
                   init::Union{Nothing,AbstractVector{<:Real}}=nothing,
                   proposal_scale::Real=0.36,
                   T_store::Int=0,
                   seed::Union{Integer,Nothing}=nothing,
                   rng::AbstractRNG=Random.default_rng())
    rng = _resolve_repro_rng(rng, seed)
    n_p = length(priors)
    n_p == length(param_names) || throw(ArgumentError(
        "priors/param_names length mismatch: $(n_p) vs $(length(param_names))"))
    (H_news >= 1 && thin >= 1 && n_keep >= thin) || throw(ArgumentError(
        "expected H_news >= 1, thin >= 1 and n_keep >= thin"))
    haskey(target, :theta_hat) && haskey(target, :V_bar) && haskey(target, :index) ||
        throw(ArgumentError("target must carry (theta_hat, V_bar, index) — build it with stacked_irf_target/ctw_covariance"))

    T = Float64
    theta_hat = Vector{T}(target.theta_hat)
    m = length(theta_hat)
    prec = precision_of(Matrix{T}(target.V_bar))
    Cw = _pp_weight_factor(prec.precision)           # V⁻¹ = Cw'Cw
    btil = Cw * (-theta_hat)
    const_term = -prec.logdet / 2 - m * log(2 * T(pi)) / 2

    # probe build at the initial point: index maps + validation
    psi0 = init === nothing ? T[quantile(pr, 0.5) for pr in priors] : Vector{T}(init)
    ce0 = menu_builder(psi0)
    ce0 isa PolicyCausalEffects || throw(ArgumentError(
        "menu_builder must return a PolicyCausalEffects, got $(typeof(ce0))"))
    is_square(ce0) || throw(ArgumentError(
        "menu_builder must return a SQUARE menu (n_s = H); got n_s = $(n_shocks(ce0)) < H = $(ce0.H)"))
    rows, n_blocks = _bank_index_maps(ce0, target.index)

    n_fail = Ref(0)
    function logpost(psi::Vector{Float64})
        lp = zero(T)
        for i in 1:n_p
            lp += T(logpdf(priors[i], psi[i]))
        end
        isfinite(lp) || return T(-Inf)
        ce = try
            menu_builder(psi)
        catch e
            if e isa ArgumentError || _benign_solve_error(e)
                n_fail[] += 1
                return T(-Inf)
            end
            rethrow()
        end
        ll = _suppress_warnings() do
            _bank_loglik(ce, rows, n_blocks, H_news, Cw, btil, const_term)
        end
        return ll + lp
    end

    # --- adaptive-then-frozen RWMH ---
    scale0 = T[(quantile(pr, 0.84) - quantile(pr, 0.16)) / 2 for pr in priors]
    prop_chol = Matrix{T}(Diagonal(max.(scale0, sqrt(eps(T))) .* sqrt(T(proposal_scale))))
    psi = copy(psi0)
    lp_cur = logpost(psi)
    isfinite(lp_cur) || throw(ArgumentError(
        "irf_match: the initial point (init/prior medians) has zero posterior mass — supply a valid init"))
    history = Matrix{T}(undef, n_adapt, n_p)
    for s in 1:n_adapt
        prop = psi + prop_chol * randn(rng, T, n_p)
        lp_prop = logpost(prop)
        if log(rand(rng, T)) < lp_prop - lp_cur
            psi = prop
            lp_cur = lp_prop
        end
        history[s, :] = psi
        if s % 200 == 0 && s >= max(50, 10 * n_p)
            S = cov(@view(history[1:s, :])) .* T(proposal_scale) +
                T(1e-10) * I
            prop_chol = Matrix{T}(safe_cholesky(Matrix{T}(S)))
        end
    end
    for _ in 1:n_burn                          # frozen-proposal burn-in
        prop = psi + prop_chol * randn(rng, T, n_p)
        lp_prop = logpost(prop)
        if log(rand(rng, T)) < lp_prop - lp_cur
            psi = prop
            lp_cur = lp_prop
        end
    end
    n_kept = n_keep ÷ thin
    theta_draws = Matrix{T}(undef, n_kept, n_p)
    log_post = Vector{T}(undef, n_kept)
    n_acc = 0
    kept = 0
    for s in 1:n_keep
        prop = psi + prop_chol * randn(rng, T, n_p)
        lp_prop = logpost(prop)
        if log(rand(rng, T)) < lp_prop - lp_cur
            psi = prop
            lp_cur = lp_prop
            n_acc += 1
        end
        if s % thin == 0 && kept < n_kept
            kept += 1
            theta_draws[kept, :] = psi
            log_post[kept] = lp_cur
        end
    end
    acc = T(n_acc) / T(n_keep)
    n_fail[] > 0 && @info "irf_match($(name)): $(n_fail[]) menu builds failed (indeterminacy/solver errors) and scored -Inf"

    lml = _geweke_mhm(theta_draws, log_post; p=0.5)

    # rebuild (and optionally truncate) the stored menus at the kept draws
    menus = Vector{PolicyCausalEffects{T}}(undef, kept)
    for i in 1:kept
        ce_i = menu_builder(Vector{T}(theta_draws[i, :]))
        menus[i] = T_store > 0 && T_store < ce_i.H ? _bank_truncate(ce_i, T_store) : ce_i
    end

    result = ModelBankMember{T}(String(name), collect(Symbol, param_names),
                                theta_draws[1:kept, :], log_post[1:kept], T(lml),
                                menus, acc, H_news)
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("n_keep" => n_keep, "thin" => thin,
                                  "n_adapt" => n_adapt, "n_burn" => n_burn)))
end

# Truncate a square menu container to a T_store×T_store horizon.
function _bank_truncate(ce::PolicyCausalEffects{T}, T_store::Int) where {T<:AbstractFloat}
    PolicyCausalEffects{T}(copy(ce.outcomes), copy(ce.instruments),
                           [M[1:T_store, 1:T_store] for M in ce.Theta_x],
                           [M[1:T_store, 1:T_store] for M in ce.Theta_z],
                           nothing, nothing, T_store,
                           ce.shock_labels[1:T_store], ce.source)
end

"""
    posterior_model_probs(members; prior=fill(1/n, n)) -> Vector{Float64}

Posterior model probabilities `p(M_j|θ̂) ∝ exp(log_marglik_j)·prior_j`,
normalized with a log-sum-exp (CMW `get_posterior_probs.m`).
"""
function posterior_model_probs(members::AbstractVector{<:ModelBankMember};
                               prior::AbstractVector{<:Real}=fill(1 / length(members), length(members)))
    n = length(members)
    n >= 1 || throw(ArgumentError("members: expected at least one member"))
    length(prior) == n || throw(ArgumentError(
        "prior: expected $n weights, got $(length(prior))"))
    all(p -> p >= 0, prior) && sum(prior) > 0 || throw(ArgumentError(
        "prior: expected nonnegative weights with positive sum"))
    l = [m.log_marglik + log(float(prior[j])) for (j, m) in enumerate(members)]
    any(isnan, l) && throw(ArgumentError(
        "posterior_model_probs: a member has NaN log_marglik (chain too short for Geweke MHM)"))
    l0 = maximum(l)
    w = exp.(l .- l0)
    return w ./ sum(w)
end

"""
    model_average(members, probs; n_pool=1000, subset=nothing,
                  rng=Random.default_rng()) -> PolicyCausalEffects

Pool the model bank (CMW `sample_from_models.m`): draw `n_pool`
(model, parameter) pairs — the model from `probs`, the parameter uniformly
from the member's kept draws — and stack the sampled menus into a pooled
`PolicyCausalEffects` (`source = :pooled`; point = element-wise median over
the pooled draws). `subset` restricts pooling to a member subset with
renormalized probabilities (CMW's RE-only/behavioral-only variants).
"""
function model_average(members::AbstractVector{<:ModelBankMember},
                       probs::AbstractVector{<:Real};
                       n_pool::Int=1000,
                       subset::Union{Nothing,AbstractVector{Int}}=nothing,
                       seed::Union{Integer,Nothing}=nothing,
                       rng::AbstractRNG=Random.default_rng())
    rng = _resolve_repro_rng(rng, seed)
    n = length(members)
    length(probs) == n || throw(ArgumentError(
        "probs: expected $n probabilities, got $(length(probs))"))
    n_pool >= 1 || throw(ArgumentError("n_pool: expected >= 1, got $n_pool"))
    idxs = subset === nothing ? collect(1:n) : collect(subset)
    all(j -> 1 <= j <= n, idxs) || throw(ArgumentError(
        "subset: member indices out of range 1:$n"))
    w = float.(probs[idxs])
    sum(w) > 0 || throw(ArgumentError("subset has zero total probability"))
    w ./= sum(w)

    ref = members[idxs[1]].menu_draws[1]
    T = eltype(ref)
    H = ref.H
    for j in idxs, ceq in (members[j].menu_draws[1],)
        (ceq.H == H && ceq.outcomes == ref.outcomes && ceq.instruments == ref.instruments) ||
            throw(ArgumentError(
                "model_average: member \"$(members[j].name)\" menus are incompatible with \"$(members[idxs[1]].name)\" (H/outcomes/instruments must match — align T_store and variable maps)"))
    end

    counts = zeros(Int, n)
    n_x = length(ref.outcomes)
    n_z = length(ref.instruments)
    Dx = [Array{T,3}(undef, H, H, n_pool) for _ in 1:n_x]
    Dz = [Array{T,3}(undef, H, H, n_pool) for _ in 1:n_z]
    cw = cumsum(w)
    for d in 1:n_pool
        u = rand(rng)
        j = idxs[something(findfirst(>=(u), cw), length(cw))]
        counts[j] += 1
        mem = members[j]
        ce_d = mem.menu_draws[rand(rng, 1:length(mem.menu_draws))]
        for i in 1:n_x
            Dx[i][:, :, d] = ce_d.Theta_x[i]
        end
        for k in 1:n_z
            Dz[k][:, :, d] = ce_d.Theta_z[k]
        end
    end
    @info "model_average: pooled $(n_pool) draws — " *
          join(("$(members[j].name): $(counts[j])" for j in idxs), ", ")

    med(D) = Matrix{T}([median(@view(D[h, s, :])) for h in 1:H, s in 1:H])
    PolicyCausalEffects{T}(copy(ref.outcomes), copy(ref.instruments),
                           [med(D) for D in Dx], [med(D) for D in Dz],
                           n_x > 0 ? Dx : nothing, n_z > 0 ? Dz : nothing,
                           H, copy(ref.shock_labels), :pooled)
end
