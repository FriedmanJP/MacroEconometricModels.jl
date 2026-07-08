# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Identification diagnostics for DSGE models.

Provides:
- `identification_diagnostics` — Iskrev (2010) local-identification rank test
  on the Jacobian of observable means + autocovariances w.r.t. the parameters
- `learning_rate_check` — Koop-Pesaran-Smith (2013) posterior-variance
  learning-rate check across nested subsamples
- `prior_posterior_overlap` — prior/posterior overlap coefficients flagging
  parameters whose posterior is essentially the prior

References:
- Iskrev, N. (2010). Local Identification in DSGE Models.
  *Journal of Monetary Economics*, 57(2), 189-202.
- Koop, G., Pesaran, M. H. & Smith, R. P. (2013). On Identification of
  Bayesian DSGE Models. *Journal of Business & Economic Statistics*, 31(3).
"""

# =============================================================================
# Iskrev (2010) rank test
# =============================================================================

"""
    IdentificationDiagnostics{T}

Result of the Iskrev (2010) local-identification rank test.

Fields:
- `param_names::Vector{Symbol}` — parameters checked
- `theta::Vector{T}` — evaluation point
- `rank::Int` — numerical rank of the moment Jacobian `J(θ)`
- `n_params::Int` — number of parameters (full column rank required)
- `n_moments::Int` — number of differentiated moments
- `n_lags::Int` — autocovariance lags included in the moment vector
- `singular_values::Vector{T}` — singular values of `J(θ)`
- `tol::T` — rank tolerance used (relative to the largest singular value)
- `null_space::Matrix{T}` — `n_params × k` basis of unidentified directions
  (empty when identified)
- `identified::Bool` — `rank == n_params`
"""
struct IdentificationDiagnostics{T<:AbstractFloat}
    param_names::Vector{Symbol}
    theta::Vector{T}
    rank::Int
    n_params::Int
    n_moments::Int
    n_lags::Int
    singular_values::Vector{T}
    tol::T
    null_space::Matrix{T}
    identified::Bool
end

"""
    _identification_moments(spec, param_names, theta, observables, n_lags, solver,
                            solver_kwargs) → Vector or nothing

Iskrev moment map `m(θ)`: observable steady-state means, the lower triangle of
the contemporaneous covariance, and the full autocovariance matrices at lags
`1..n_lags` of the observables, computed from the first-order solution via the
discrete Lyapunov equation. Returns `nothing` when the model fails to solve.
"""
function _identification_moments(spec::DSGESpec{T}, param_names::Vector{Symbol},
                                 theta::AbstractVector{T},
                                 observables::Vector{Symbol}, n_lags::Int,
                                 solver::Symbol,
                                 solver_kwargs::NamedTuple) where {T<:AbstractFloat}
    try
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = _respec(spec, new_pv)
        new_spec = compute_steady_state(new_spec)
        sol = solve(new_spec; method=solver, solver_kwargs...)
        is_determined(sol) || return nothing

        G1 = Matrix{T}(sol.G1)
        impact = Matrix{T}(sol.impact)
        obs_idx = [findfirst(==(o), new_spec.endog) for o in observables]

        # Effective mean of the observables (linear=true constant models included)
        ss = if new_spec.linear && !all(iszero, sol.C_sol)
            (I - G1) \ sol.C_sol
        else
            new_spec.steady_state
        end

        Sigma = solve_lyapunov(G1, impact)
        n_o = length(obs_idx)

        m = T[]
        append!(m, T[ss[j] for j in obs_idx])
        S0 = Sigma[obs_idx, obs_idx]
        for i in 1:n_o, j in i:n_o
            push!(m, S0[i, j])
        end
        Gh = Sigma
        for _ in 1:n_lags
            Gh = G1 * Gh                       # Γ_h = G1^h Σ
            Sh = Gh[obs_idx, obs_idx]
            append!(m, vec(Sh))
        end
        return m
    catch
        return nothing
    end
end

"""
    identification_diagnostics(spec, param_names; theta=nothing,
                               observables=Symbol[], n_lags=2,
                               tol_rel=sqrt(eps()), solver=:gensys,
                               solver_kwargs=NamedTuple()) → IdentificationDiagnostics

Iskrev (2010) local-identification rank test at the point `θ`.

Builds the Jacobian `J(θ) = ∂m(θ)/∂θ'` of the **data moments** — the observable
steady-state means, contemporaneous covariance (lower triangle), and the
autocovariance matrices at lags `1..n_lags` computed from the first-order
state-space solution — by central finite differences, then inspects its column
rank via SVD. The parameters are locally identified from these moments iff
`J(θ)` has full column rank. When rank-deficient, the right null space of `J`
names the unidentified parameter directions, and a warning lists the involved
parameters.

Increase `n_lags` until `n_moments ≥ n_params` with slack; Iskrev recommends
checking several lag horizons.

# Arguments
- `spec::DSGESpec` — model specification
- `param_names::Vector{Symbol}` — parameters to check

# Keywords
- `theta=nothing` — evaluation point (default: the spec's current parameter values)
- `observables::Vector{Symbol}=Symbol[]` — observed variables (default: all endogenous)
- `n_lags::Int=2` — autocovariance lags in the moment vector
- `tol_rel::Real=sqrt(eps())` — rank tolerance relative to the largest singular value
- `solver::Symbol=:gensys`, `solver_kwargs::NamedTuple=()` — solution method

# References
- Iskrev, N. (2010). Local Identification in DSGE Models. *JME*, 57(2).
"""
function identification_diagnostics(spec::DSGESpec{T}, param_names::Vector{Symbol};
                                    theta::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                    observables::Vector{Symbol}=Symbol[],
                                    n_lags::Int=2,
                                    tol_rel::Real=sqrt(eps(Float64)),
                                    solver::Symbol=:gensys,
                                    solver_kwargs::NamedTuple=NamedTuple()) where {T<:AbstractFloat}
    d = length(param_names)
    d > 0 || throw(ArgumentError("param_names must be non-empty"))
    theta_v = theta === nothing ? T[spec.param_values[p] for p in param_names] : T.(theta)
    length(theta_v) == d ||
        throw(ArgumentError("theta has length $(length(theta_v)), expected $d"))
    if isempty(observables)
        observables = copy(spec.endog)
    end
    for o in observables
        o in spec.endog ||
            throw(ArgumentError("observable :$o is not an endogenous variable of the spec"))
    end

    m0 = _identification_moments(spec, param_names, theta_v, observables, n_lags,
                                 solver, solver_kwargs)
    m0 === nothing &&
        throw(ArgumentError("model does not solve at θ = $theta_v; cannot run the rank test"))
    n_m = length(m0)

    # Jacobian by ForwardDiff where possible, central differences otherwise
    moment_map = th -> begin
        m = _identification_moments(spec, param_names, Vector{T}(th), observables,
                                    n_lags, solver, solver_kwargs)
        m === nothing ? fill(T(NaN), n_m) : m
    end
    J = try
        ForwardDiff.jacobian(moment_map, theta_v)
    catch
        step = T(1e-6) * max(one(T), maximum(abs, theta_v))
        numerical_gradient(moment_map, theta_v; eps=step)
    end
    all(isfinite, J) ||
        throw(ArgumentError("moment Jacobian contains non-finite entries — the model " *
                            "fails to solve in a neighborhood of θ; move θ inside the " *
                            "determinacy region"))

    F = svd(Matrix{T}(J); full=true)           # full V so the null space is complete
    sv = F.S
    tol = T(tol_rel) * (isempty(sv) ? one(T) : maximum(sv))
    r = count(>(tol), sv)
    # Right null space: columns of V for singular values ≤ tol (plus missing ones
    # when n_moments < n_params)
    null_cols = [k for k in 1:d if k > length(sv) || sv[k] <= tol]
    Nsp = Matrix{T}(F.V[:, null_cols])

    ident = r == d
    if !ident
        involved = Set{Symbol}()
        for c in 1:size(Nsp, 2)
            w = abs.(Nsp[:, c])
            thr = T(0.1) * maximum(w)
            for i in 1:d
                w[i] >= thr && push!(involved, param_names[i])
            end
        end
        @warn "identification_diagnostics: rank(J) = $r < $d — parameter(s) " *
              join(sort!(collect(string.(involved))), ", ") *
              " appear locally unidentified from the chosen observables/moments " *
              "(Iskrev 2010)"
    end

    return IdentificationDiagnostics{T}(copy(param_names), theta_v, r, d, n_m,
                                        n_lags, Vector{T}(sv), tol, Nsp, ident)
end

function Base.show(io::IO, idd::IdentificationDiagnostics{T}) where {T}
    header = Any[
        "Parameters"           idd.n_params;
        "Moments"              idd.n_moments;
        "Autocov. lags"        idd.n_lags;
        "rank(J)"              idd.rank;
        "Smallest sing. value" round(minimum(idd.singular_values); sigdigits=4);
        "Rank tolerance"       round(idd.tol; sigdigits=4);
        "Locally identified"   idd.identified ? "yes" : "NO";
    ]
    _pretty_table(io, header;
        title="Iskrev (2010) Identification Rank Test",
        column_labels=["", ""],
        alignment=[:l, :r])

    if !idd.identified
        for c in 1:size(idd.null_space, 2)
            w = idd.null_space[:, c]
            thr = 0.1 * maximum(abs, w)
            parts = String[]
            for i in 1:idd.n_params
                abs(w[i]) >= thr &&
                    push!(parts, "$(round(w[i]; digits=3))·$(idd.param_names[i])")
            end
            println(io, "Unidentified direction $c: ", join(parts, " + "))
        end
    end
end

# =============================================================================
# Koop-Pesaran-Smith (2013) learning-rate check
# =============================================================================

"""
    LearningRateCheck{T}

Result of the Koop-Pesaran-Smith (2013) posterior-variance learning-rate check.

Fields:
- `param_names::Vector{Symbol}` — parameters checked
- `sample_sizes::Vector{Int}` — nested subsample sizes used
- `post_vars::Matrix{T}` — n_params × n_subsamples posterior variances
- `learning_rate::Vector{T}` — estimated rate `α` in `var ∝ T^{-α}` per parameter
- `flagged::Vector{Bool}` — `α < threshold` (posterior barely updates with T)
- `threshold::T` — flag threshold
"""
struct LearningRateCheck{T<:AbstractFloat}
    param_names::Vector{Symbol}
    sample_sizes::Vector{Int}
    post_vars::Matrix{T}
    learning_rate::Vector{T}
    flagged::Vector{Bool}
    threshold::T
end

"""
    learning_rate_check(result::BayesianDSGE; fractions=[0.5, 1.0], n_smc=300,
                        threshold=0.2, rng=Random.default_rng()) → LearningRateCheck

Koop-Pesaran-Smith (2013) identification check: for an identified parameter the
posterior variance shrinks at the `1/T` rate, so re-estimating on nested
subsamples `⌊f·T⌋` and regressing `log var` on `log T` should give a slope near
`−1`. A weakly identified parameter's posterior barely updates (`α ≈ 0`).

Each subsample is re-estimated with a fresh SMC run (`n_smc` particles) using
the stored estimation context, so the variances are comparable across sample
sizes regardless of the original sampler. Parameters with `α < threshold` are
flagged with a warning. This is a Monte Carlo procedure — expect noise in `α`;
it is a screening device, not a hypothesis test.

# References
- Koop, G., Pesaran, M. H. & Smith, R. P. (2013). *JBES*, 31(3), 300-314.
"""
function learning_rate_check(result::BayesianDSGE{T};
                             fractions::Vector{<:Real}=[0.5, 1.0],
                             n_smc::Int=300,
                             threshold::Real=0.2,
                             rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    isempty(result.data) &&
        throw(ArgumentError("result carries no estimation data (hand-constructed?); " *
                            "learning_rate_check needs the stored sample"))
    length(fractions) >= 2 ||
        throw(ArgumentError("need at least two subsample fractions"))
    fr = sort(T.(fractions))
    (first(fr) > 0 && last(fr) <= 1) ||
        throw(ArgumentError("fractions must lie in (0, 1]"))

    T_obs = size(result.data, 2)
    d = length(result.param_names)
    priors_dict = Dict{Symbol,Distribution}(result.param_names[i] =>
        result.priors.distributions[i] for i in 1:d)
    theta0 = vec(mean(result.theta_draws; dims=1))

    sizes = [max(2 * d + 2, floor(Int, f * T_obs)) for f in fr]
    post_vars = zeros(T, d, length(fr))
    for (k, Tk) in enumerate(sizes)
        sub = result.data[:, 1:Tk]
        fit = estimate_dsge_bayes(result.spec, sub, theta0;
                                  priors=priors_dict, method=:smc, n_smc=n_smc,
                                  observables=result.observables,
                                  measurement_error=result.measurement_error,
                                  solver=result.solver,
                                  solver_kwargs=result.solver_kwargs, rng=rng)
        post_vars[:, k] = [var(fit.theta_draws[:, i]) for i in 1:d]
    end

    # Slope of log var on log T (least squares over the subsamples), α = −slope
    lt = log.(T.(sizes))
    lt_c = lt .- mean(lt)
    denom = sum(abs2, lt_c)
    alpha = zeros(T, d)
    for i in 1:d
        lv = log.(max.(post_vars[i, :], floatmin(T)))
        alpha[i] = -sum(lt_c .* (lv .- mean(lv))) / denom
    end

    flagged = alpha .< T(threshold)
    if any(flagged)
        @warn "learning_rate_check: posterior variance of " *
              join(string.(result.param_names[flagged]), ", ") *
              " does not shrink with the sample size (α < $threshold) — " *
              "weak identification suspected (Koop-Pesaran-Smith 2013)"
    end

    return LearningRateCheck{T}(copy(result.param_names), sizes, post_vars,
                                alpha, Vector{Bool}(flagged), T(threshold))
end

function Base.show(io::IO, lrc::LearningRateCheck{T}) where {T}
    d = length(lrc.param_names)
    data = Matrix{Any}(undef, d, 3 + length(lrc.sample_sizes))
    for i in 1:d
        data[i, 1] = string(lrc.param_names[i])
        for k in 1:length(lrc.sample_sizes)
            data[i, 1+k] = round(lrc.post_vars[i, k]; sigdigits=4)
        end
        data[i, 2+length(lrc.sample_sizes)] = round(lrc.learning_rate[i]; digits=3)
        data[i, 3+length(lrc.sample_sizes)] = lrc.flagged[i] ? "WEAK" : "ok"
    end
    _pretty_table(io, data;
        title="Koop-Pesaran-Smith Learning-Rate Check (var ∝ T^-α; identified ⇒ α ≈ 1)",
        column_labels=vcat(["Parameter"],
                           ["var (T=$(t))" for t in lrc.sample_sizes],
                           ["α", "Flag"]),
        alignment=vcat([:l], fill(:r, length(lrc.sample_sizes) + 2)))
end

# =============================================================================
# Prior/posterior overlap
# =============================================================================

"""
    PriorPosteriorOverlap{T}

Per-parameter prior/posterior overlap coefficients.

Fields:
- `param_names::Vector{Symbol}` — parameters
- `overlap::Vector{T}` — overlap coefficient `∫ min(π(θ), p(θ|Y)) dθ ∈ [0, 1]`
- `flagged::Vector{Bool}` — overlap ≥ threshold (posterior ≈ prior)
- `threshold::T` — flag threshold
"""
struct PriorPosteriorOverlap{T<:AbstractFloat}
    param_names::Vector{Symbol}
    overlap::Vector{T}
    flagged::Vector{Bool}
    threshold::T
end

"""
    prior_posterior_overlap(result::BayesianDSGE; n_grid=0, threshold=0.8)
        → PriorPosteriorOverlap

Overlap coefficient `∫ min(π(θᵢ), p(θᵢ|Y)) dθᵢ` between each parameter's prior
density and its posterior marginal (histogram estimate on a common grid;
`n_grid=0` picks `≈√N` bins so the histogram density is smooth at the available
draw count). An overlap near 1 means the data barely moved the prior — the
practical symptom of weak identification. Parameters with overlap ≥ `threshold`
are flagged with a warning.
"""
function prior_posterior_overlap(result::BayesianDSGE{T};
                                 n_grid::Int=0,
                                 threshold::Real=0.8) where {T<:AbstractFloat}
    d = length(result.param_names)
    n_draws = size(result.theta_draws, 1)
    if n_grid <= 0
        n_grid = clamp(round(Int, sqrt(n_draws)), 10, 512)
    end
    ovl = zeros(T, d)
    for i in 1:d
        draws = result.theta_draws[:, i]
        dist = result.priors.distributions[i]
        lo = min(T(quantile(dist, 0.001)), minimum(draws))
        hi = max(T(quantile(dist, 0.999)), maximum(draws))
        hi > lo || (ovl[i] = one(T); continue)
        edges = range(lo, hi; length=n_grid + 1)
        width = step(edges)
        counts = zeros(T, n_grid)
        for x in draws
            b = clamp(1 + floor(Int, (x - lo) / width), 1, n_grid)
            counts[b] += 1
        end
        counts ./= (length(draws) * width)      # posterior density estimate
        s = zero(T)
        for b in 1:n_grid
            c = (edges[b] + edges[b+1]) / 2
            s += min(counts[b], T(pdf(dist, c))) * width
        end
        ovl[i] = min(s, one(T))
    end

    flagged = ovl .>= T(threshold)
    if any(flagged)
        @warn "prior_posterior_overlap: posterior of " *
              join(string.(result.param_names[flagged]), ", ") *
              " is essentially the prior (overlap ≥ $threshold) — the data are " *
              "not informative about these parameters"
    end

    return PriorPosteriorOverlap{T}(copy(result.param_names), ovl,
                                    Vector{Bool}(flagged), T(threshold))
end

function Base.show(io::IO, ppo::PriorPosteriorOverlap{T}) where {T}
    d = length(ppo.param_names)
    data = Matrix{Any}(undef, d, 3)
    for i in 1:d
        data[i, 1] = string(ppo.param_names[i])
        data[i, 2] = round(ppo.overlap[i]; digits=3)
        data[i, 3] = ppo.flagged[i] ? "WEAK" : "ok"
    end
    _pretty_table(io, data;
        title="Prior/Posterior Overlap (≥ $(ppo.threshold) ⇒ data uninformative)",
        column_labels=["Parameter", "Overlap", "Flag"],
        alignment=[:l, :r, :r])
end
