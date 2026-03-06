# Panel Regression & Ordered/Multinomial Models — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement panel regression (FE/RE/FD/Between/CRE, panel IV, panel Probit/Logit, dynamic panel) and ordered/multinomial discrete choice models.

**Architecture:** Two parallel workstreams — #77 (ordered/multinomial in `src/reg/`) and #66 (panel regression in `src/panel_reg/`). Both follow the existing codebase pattern: struct types with StatsAPI compliance, Newton-Raphson MLE or OLS/GLS estimation, sandwich covariance, `report()`/`refs()` display.

**Tech Stack:** Julia, StatsAPI.jl, Distributions.jl, LinearAlgebra, PrettyTables.jl. Existing infrastructure: `_reg_vcov()`, `_coef_table()`, `_gauss_hermite_nodes_weights()`, `robust_inv()`, `PanelData{T}`.

**Design doc:** `docs/plans/2026-03-07-panel-reg-ordered-multinomial-design.md`

---

## Workstream A: Ordered & Multinomial Models (#77)

### Task 1: Ordered Logit/Probit Types & Estimation

**Files:**
- Create: `src/reg/ordered.jl`
- Modify: `src/MacroEconometricModels.jl` (add includes + exports)
- Test: `test/reg/test_ordered.jl`

**Step 1: Write failing tests for ordered logit**

Create `test/reg/test_ordered.jl`:

```julia
using Random, Distributions

@testset "Ordered Logit" begin
    @testset "estimate_ologit — coefficient recovery" begin
        rng = MersenneTwister(42)
        n = 2000
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        X = hcat(x1, x2)
        beta_true = [1.0, -0.5]
        cutpoints_true = [-1.0, 1.0]  # 3 categories
        eta = X * beta_true
        # Generate ordered outcomes via latent variable
        y = Vector{Int}(undef, n)
        for i in 1:n
            u = rand(rng)
            p1 = 1.0 / (1.0 + exp(-(cutpoints_true[1] - eta[i])))
            p2 = 1.0 / (1.0 + exp(-(cutpoints_true[2] - eta[i])))
            if u < p1
                y[i] = 1
            elseif u < p2
                y[i] = 2
            else
                y[i] = 3
            end
        end

        m = estimate_ologit(y, X)
        @test m isa OrderedLogitModel
        @test length(coef(m)) == 2
        @test length(m.cutpoints) == 2
        @test all(abs.(coef(m) .- beta_true) .< 0.15)
        @test all(abs.(m.cutpoints .- cutpoints_true) .< 0.15)
        @test m.converged
        @test m.n_categories == 3
        @test m.pseudo_r2 > 0
        @test m.loglik > m.loglik_null
    end

    @testset "estimate_ologit — 5 categories" begin
        rng = MersenneTwister(99)
        n = 3000
        X = randn(rng, n, 3)
        beta_true = [0.8, -0.4, 0.6]
        cutpoints_true = [-2.0, -0.5, 0.5, 2.0]  # 5 categories
        eta = X * beta_true
        y = Vector{Int}(undef, n)
        for i in 1:n
            u = rand(rng)
            cum_p = 0.0
            y[i] = 5
            for j in 1:4
                cum_p = 1.0 / (1.0 + exp(-(cutpoints_true[j] - eta[i])))
                if u < cum_p
                    y[i] = j
                    break
                end
            end
        end
        m = estimate_ologit(y, X)
        @test m.n_categories == 5
        @test length(m.cutpoints) == 4
        @test issorted(m.cutpoints)  # cutpoints must be ordered
        @test all(abs.(coef(m) .- beta_true) .< 0.15)
    end

    @testset "estimate_ologit — StatsAPI interface" begin
        rng = MersenneTwister(55)
        n = 500
        X = randn(rng, n, 2)
        y = rand(rng, 1:3, n)
        m = estimate_ologit(y, X; varnames=["x1", "x2"])
        @test length(coef(m)) == 2
        @test size(vcov(m)) == (4, 4)  # 2 beta + 2 cutpoints
        @test nobs(m) == n
        @test dof(m) == 4  # 2 betas + 2 cutpoints
        @test length(stderror(m)) == 4  # joint SEs
        @test isfinite(loglikelihood(m))
        @test isfinite(aic(m))
        @test isfinite(bic(m))
        @test m.varnames == ["x1", "x2"]
    end

    @testset "estimate_ologit — robust SEs" begin
        rng = MersenneTwister(77)
        n = 500
        X = randn(rng, n, 2)
        y = rand(rng, 1:3, n)
        m_ols = estimate_ologit(y, X; cov_type=:ols)
        m_hc1 = estimate_ologit(y, X; cov_type=:hc1)
        @test m_ols.cov_type == :ols
        @test m_hc1.cov_type == :hc1
        # Robust SEs generally differ from classical
        @test vcov(m_ols) != vcov(m_hc1)
    end
end

@testset "Ordered Probit" begin
    @testset "estimate_oprobit — coefficient recovery" begin
        rng = MersenneTwister(43)
        n = 2000
        X = randn(rng, n, 2)
        beta_true = [0.8, -0.6]
        cutpoints_true = [-0.5, 0.5]
        eta = X * beta_true
        y = Vector{Int}(undef, n)
        for i in 1:n
            u = rand(rng)
            p1 = cdf(Normal(), cutpoints_true[1] - eta[i])
            p2 = cdf(Normal(), cutpoints_true[2] - eta[i])
            if u < p1
                y[i] = 1
            elseif u < p2
                y[i] = 2
            else
                y[i] = 3
            end
        end
        m = estimate_oprobit(y, X)
        @test m isa OrderedProbitModel
        @test all(abs.(coef(m) .- beta_true) .< 0.15)
        @test all(abs.(m.cutpoints .- cutpoints_true) .< 0.15)
        @test m.converged
    end
end
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/declarative-kindling-wozniak && julia --project=. -e 'using Test, MacroEconometricModels; include("test/reg/test_ordered.jl")'`

Expected: FAIL — `estimate_ologit` not defined.

**Step 3: Implement ordered logit/probit estimation**

Create `src/reg/ordered.jl`:

```julia
# ── Ordered Logit Model ──────────────────────────────────────────────

struct OrderedLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}
    X::Matrix{T}
    beta::Vector{T}
    cutpoints::Vector{T}
    vcov_mat::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    n_categories::Int
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

struct OrderedProbitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}
    X::Matrix{T}
    beta::Vector{T}
    cutpoints::Vector{T}
    vcov_mat::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    n_categories::Int
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

# ── Link functions ───────────────────────────────────────────────────

_ordered_logit_cdf(z::T) where {T} = one(T) / (one(T) + exp(-z))
_ordered_logit_pdf(z::T) where {T} = (p = _ordered_logit_cdf(z); p * (one(T) - p))
_ordered_probit_cdf(z::T) where {T} = T(cdf(Normal(), z))
_ordered_probit_pdf(z::T) where {T} = T(pdf(Normal(), z))

# ── Category probabilities ──────────────────────────────────────────

function _ordered_probs(eta::T, cutpoints::Vector{T}, cdf_fn) where {T}
    J = length(cutpoints) + 1
    probs = Vector{T}(undef, J)
    prev = zero(T)
    for j in 1:(J-1)
        cum = cdf_fn(cutpoints[j] - eta)
        probs[j] = max(cum - prev, eps(T))
        prev = cum
    end
    probs[J] = max(one(T) - prev, eps(T))
    return probs
end

# ── Log-likelihood ──────────────────────────────────────────────────

function _ordered_loglik(beta::Vector{T}, cutpoints::Vector{T},
                         y::Vector{Int}, X::Matrix{T}, cdf_fn) where {T}
    n = length(y)
    eta = X * beta
    ll = zero(T)
    for i in 1:n
        probs = _ordered_probs(eta[i], cutpoints, cdf_fn)
        ll += log(probs[y[i]])
    end
    return ll
end

# ── Newton-Raphson MLE ──────────────────────────────────────────────

function _estimate_ordered(y::Vector{Int}, X::Matrix{T}, cdf_fn, pdf_fn;
                           maxiter::Int=100, tol::T=T(1e-8)) where {T}
    n, k = size(X)
    cats = sort(unique(y))
    J = length(cats)
    # Remap y to 1:J
    cat_map = Dict(c => i for (i, c) in enumerate(cats))
    y_mapped = [cat_map[yi] for yi in y]

    # Initialize: cutpoints from empirical quantiles, beta = 0
    beta = zeros(T, k)
    cutpoints = Vector{T}(undef, J - 1)
    cum_frac = cumsum([count(==(j), y_mapped) for j in 1:J]) ./ n
    for j in 1:(J-1)
        # Inverse CDF at empirical cumulative fraction
        cutpoints[j] = if cdf_fn === _ordered_logit_cdf
            log(cum_frac[j] / (one(T) - cum_frac[j]))  # logit quantile
        else
            T(quantile(Normal(), cum_frac[j]))  # probit quantile
        end
    end

    n_params = k + J - 1
    converged = false
    iterations = 0
    ll_old = T(-Inf)

    for iter in 1:maxiter
        iterations = iter
        # Compute gradient and Hessian
        grad = zeros(T, n_params)
        H = zeros(T, n_params, n_params)
        eta = X * beta
        ll = zero(T)

        for i in 1:n
            yi = y_mapped[i]
            probs = _ordered_probs(eta[i], cutpoints, cdf_fn)
            ll += log(probs[yi])

            # Derivatives of P(y=j) w.r.t. (beta, cutpoints)
            # P(y=j) = F(α_j - x'β) - F(α_{j-1} - x'β)
            # dP/dβ = -f(α_j - x'β) * x + f(α_{j-1} - x'β) * x
            # dP/dα_j = f(α_j - x'β) if j < J; dP/dα_{j-1} = -f(α_{j-1} - x'β)
            f_upper = yi < J ? pdf_fn(cutpoints[yi] - eta[i]) : zero(T)
            f_lower = yi > 1 ? pdf_fn(cutpoints[yi-1] - eta[i]) : zero(T)

            # Score contribution: (1/p_j) * dp_j/dθ
            inv_p = one(T) / probs[yi]

            # Gradient w.r.t. beta (indices 1:k)
            dbeta = (-f_upper + f_lower) * inv_p
            for l in 1:k
                grad[l] += dbeta * X[i, l]
            end

            # Gradient w.r.t. cutpoints (indices k+1:k+J-1)
            if yi < J
                grad[k + yi] += f_upper * inv_p
            end
            if yi > 1
                grad[k + yi - 1] += -f_lower * inv_p
            end

            # Hessian via outer product of gradient (BHHH approximation for robustness)
            score_i = zeros(T, n_params)
            score_i[1:k] .= dbeta .* X[i, :]
            if yi < J
                score_i[k + yi] += f_upper * inv_p
            end
            if yi > 1
                score_i[k + yi - 1] += -f_lower * inv_p
            end
            for a in 1:n_params, b in 1:n_params
                H[a, b] -= score_i[a] * score_i[b]
            end
        end

        # Check convergence
        if abs(ll - ll_old) / (abs(ll) + one(T)) < tol
            converged = true
            break
        end
        ll_old = ll

        # Newton step
        delta = -(Matrix{T}(robust_inv(Hermitian(H))) * grad)
        beta .+= delta[1:k]
        cutpoints .+= delta[(k+1):end]

        # Enforce cutpoint ordering
        for j in 2:(J-1)
            if cutpoints[j] <= cutpoints[j-1]
                cutpoints[j] = cutpoints[j-1] + T(0.01)
            end
        end
    end

    return beta, cutpoints, y_mapped, cats, converged, iterations
end

# ── Observed information (for classical and sandwich SEs) ───────────

function _ordered_info_matrix(beta::Vector{T}, cutpoints::Vector{T},
                              y::Vector{Int}, X::Matrix{T},
                              cdf_fn, pdf_fn) where {T}
    n, k = size(X)
    J = length(cutpoints) + 1
    n_params = k + J - 1
    # Compute score vectors for sandwich
    scores = zeros(T, n, n_params)
    H = zeros(T, n_params, n_params)
    eta = X * beta

    for i in 1:n
        yi = y[i]
        probs = _ordered_probs(eta[i], cutpoints, cdf_fn)
        inv_p = one(T) / probs[yi]
        f_upper = yi < J ? pdf_fn(cutpoints[yi] - eta[i]) : zero(T)
        f_lower = yi > 1 ? pdf_fn(cutpoints[yi-1] - eta[i]) : zero(T)

        dbeta = (-f_upper + f_lower) * inv_p
        scores[i, 1:k] .= dbeta .* X[i, :]
        if yi < J
            scores[i, k + yi] += f_upper * inv_p
        end
        if yi > 1
            scores[i, k + yi - 1] += -f_lower * inv_p
        end
        # BHHH Hessian
        for a in 1:n_params, b in 1:n_params
            H[a, b] -= scores[i, a] * scores[i, b]
        end
    end
    return H, scores
end

function _ordered_vcov(beta::Vector{T}, cutpoints::Vector{T},
                       y::Vector{Int}, X::Matrix{T},
                       cdf_fn, pdf_fn, cov_type::Symbol;
                       clusters=nothing) where {T}
    H, scores = _ordered_info_matrix(beta, cutpoints, y, X, cdf_fn, pdf_fn)
    Hinv = Matrix{T}(robust_inv(Hermitian(H)))

    if cov_type == :ols
        return -Hinv  # Classical MLE: V = -H^{-1}
    end

    # Sandwich: V = Hinv * B * Hinv
    n = size(scores, 1)
    n_params = size(scores, 2)

    if cov_type == :cluster && clusters !== nothing
        ugroups = unique(clusters)
        G = length(ugroups)
        B = zeros(T, n_params, n_params)
        for g in ugroups
            idx = findall(==(g), clusters)
            sg = vec(sum(scores[idx, :], dims=1))
            B .+= sg * sg'
        end
        B .*= T(G) / T(G - 1)
    else
        # HC1-style: individual score outer products with n/(n-k) correction
        B = scores' * scores
        k_eff = n_params
        B .*= T(n) / T(n - k_eff)
    end

    return Hinv * B * Hinv
end

# ── Null model log-likelihood ───────────────────────────────────────

function _ordered_null_loglik(y::Vector{Int}, J::Int, cdf_fn)
    n = length(y)
    T_val = Float64
    freqs = zeros(T_val, J)
    for yi in y
        freqs[yi] += one(T_val)
    end
    freqs ./= n
    ll = zero(T_val)
    for i in 1:n
        ll += log(max(freqs[y[i]], eps(T_val)))
    end
    return ll
end

# ── Public API ──────────────────────────────────────────────────────

function estimate_ologit(y::AbstractVector{<:Integer}, X::AbstractMatrix{T};
                         cov_type::Symbol=:ols,
                         varnames::Union{Nothing,Vector{String}}=nothing,
                         clusters::Union{Nothing,AbstractVector}=nothing,
                         maxiter::Int=100,
                         tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, k = size(X)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames

    beta, cutpoints, y_mapped, cats, converged, iterations =
        _estimate_ordered(y_mapped_placeholder_not_needed, X, _ordered_logit_cdf, _ordered_logit_pdf;
                          maxiter=maxiter, tol=tol)
    # NOTE: _estimate_ordered remaps y internally, we pass original y
    beta, cutpoints, y_mapped, cats, converged, iterations =
        _estimate_ordered(Vector{Int}(y), X, _ordered_logit_cdf, _ordered_logit_pdf;
                          maxiter=maxiter, tol=tol)

    J = length(cats)
    vcov_mat = _ordered_vcov(beta, cutpoints, y_mapped, X,
                             _ordered_logit_cdf, _ordered_logit_pdf, cov_type;
                             clusters=clusters)

    ll = _ordered_loglik(beta, cutpoints, y_mapped, X, _ordered_logit_cdf)
    ll_null = _ordered_null_loglik(y_mapped, J, _ordered_logit_cdf)
    pseudo_r2 = one(T) - ll / ll_null
    n_params = k + J - 1
    aic_val = -2ll + 2n_params
    bic_val = -2ll + n_params * log(T(n))

    return OrderedLogitModel{T}(y_mapped, X, beta, cutpoints, vcov_mat,
                                ll, ll_null, pseudo_r2, aic_val, bic_val,
                                J, vnames, converged, iterations, cov_type)
end

function estimate_ologit(y::AbstractVector{<:Integer}, X::AbstractMatrix; kwargs...)
    estimate_ologit(y, Matrix{Float64}(X); kwargs...)
end

function estimate_oprobit(y::AbstractVector{<:Integer}, X::AbstractMatrix{T};
                          cov_type::Symbol=:ols,
                          varnames::Union{Nothing,Vector{String}}=nothing,
                          clusters::Union{Nothing,AbstractVector}=nothing,
                          maxiter::Int=100,
                          tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, k = size(X)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames

    beta, cutpoints, y_mapped, cats, converged, iterations =
        _estimate_ordered(Vector{Int}(y), X, _ordered_probit_cdf, _ordered_probit_pdf;
                          maxiter=maxiter, tol=tol)

    J = length(cats)
    vcov_mat = _ordered_vcov(beta, cutpoints, y_mapped, X,
                             _ordered_probit_cdf, _ordered_probit_pdf, cov_type;
                             clusters=clusters)

    ll = _ordered_loglik(beta, cutpoints, y_mapped, X, _ordered_probit_cdf)
    ll_null = _ordered_null_loglik(y_mapped, J, _ordered_probit_cdf)
    pseudo_r2 = one(T) - ll / ll_null
    n_params = k + J - 1
    aic_val = -2ll + 2n_params
    bic_val = -2ll + n_params * log(T(n))

    return OrderedProbitModel{T}(y_mapped, X, beta, cutpoints, vcov_mat,
                                 ll, ll_null, pseudo_r2, aic_val, bic_val,
                                 J, vnames, converged, iterations, cov_type)
end

function estimate_oprobit(y::AbstractVector{<:Integer}, X::AbstractMatrix; kwargs...)
    estimate_oprobit(y, Matrix{Float64}(X); kwargs...)
end

# ── StatsAPI interface ──────────────────────────────────────────────

for MT in (:OrderedLogitModel, :OrderedProbitModel)
    @eval begin
        StatsAPI.coef(m::$MT) = m.beta
        StatsAPI.vcov(m::$MT) = m.vcov_mat
        StatsAPI.nobs(m::$MT) = length(m.y)
        StatsAPI.dof(m::$MT) = length(m.beta) + length(m.cutpoints)
        StatsAPI.loglikelihood(m::$MT) = m.loglik
        StatsAPI.aic(m::$MT) = m.aic
        StatsAPI.bic(m::$MT) = m.bic
        StatsAPI.islinear(m::$MT) = false

        function StatsAPI.stderror(m::$MT{T}) where {T}
            return [sqrt(max(m.vcov_mat[i,i], zero(T))) for i in 1:size(m.vcov_mat, 1)]
        end

        function StatsAPI.confint(m::$MT{T}; level::Real=0.95) where {T}
            se = stderror(m)
            z = T(quantile(Normal(), (1 + level) / 2))
            params = vcat(m.beta, m.cutpoints)
            hcat(params .- z .* se, params .+ z .* se)
        end
    end
end

# ── Predict: n × J matrix of probabilities ─────────────────────────

function StatsAPI.predict(m::OrderedLogitModel{T}, X_new::AbstractMatrix) where {T}
    n = size(X_new, 1)
    eta = X_new * m.beta
    probs = Matrix{T}(undef, n, m.n_categories)
    for i in 1:n
        probs[i, :] .= _ordered_probs(eta[i], m.cutpoints, _ordered_logit_cdf)
    end
    return probs
end

function StatsAPI.predict(m::OrderedProbitModel{T}, X_new::AbstractMatrix) where {T}
    n = size(X_new, 1)
    eta = X_new * m.beta
    probs = Matrix{T}(undef, n, m.n_categories)
    for i in 1:n
        probs[i, :] .= _ordered_probs(eta[i], m.cutpoints, _ordered_probit_cdf)
    end
    return probs
end

# ── Display ─────────────────────────────────────────────────────────

function Base.show(io::IO, m::OrderedLogitModel{T}) where {T}
    println(io, "Ordered Logit Model")
    println(io, "═══════════════════")
    println(io, "  Observations:    $(nobs(m))")
    println(io, "  Categories:      $(m.n_categories)")
    println(io, "  Log-likelihood:  $(_fmt(m.loglik))")
    println(io, "  Pseudo R²:       $(_fmt(m.pseudo_r2))")
    println(io, "  AIC:             $(_fmt(m.aic))")
    println(io, "  BIC:             $(_fmt(m.bic))")
    println(io, "  Converged:       $(m.converged) ($(m.iterations) iterations)")
    println(io, "  Cov. type:       $(m.cov_type)")
    println(io)
    _coef_table(io, "Coefficients", m.varnames, m.beta,
                stderror(m)[1:length(m.beta)]; dist=:z)
    println(io)
    cutpoint_names = ["cut$(j)" for j in 1:length(m.cutpoints)]
    k = length(m.beta)
    _coef_table(io, "Cutpoints", cutpoint_names, m.cutpoints,
                stderror(m)[(k+1):end]; dist=:z)
end

function Base.show(io::IO, m::OrderedProbitModel{T}) where {T}
    println(io, "Ordered Probit Model")
    println(io, "════════════════════")
    println(io, "  Observations:    $(nobs(m))")
    println(io, "  Categories:      $(m.n_categories)")
    println(io, "  Log-likelihood:  $(_fmt(m.loglik))")
    println(io, "  Pseudo R²:       $(_fmt(m.pseudo_r2))")
    println(io, "  AIC:             $(_fmt(m.aic))")
    println(io, "  BIC:             $(_fmt(m.bic))")
    println(io, "  Converged:       $(m.converged) ($(m.iterations) iterations)")
    println(io, "  Cov. type:       $(m.cov_type)")
    println(io)
    _coef_table(io, "Coefficients", m.varnames, m.beta,
                stderror(m)[1:length(m.beta)]; dist=:z)
    println(io)
    cutpoint_names = ["cut$(j)" for j in 1:length(m.cutpoints)]
    k = length(m.beta)
    _coef_table(io, "Cutpoints", cutpoint_names, m.cutpoints,
                stderror(m)[(k+1):end]; dist=:z)
end

report(m::OrderedLogitModel) = show(stdout, m)
report(m::OrderedProbitModel) = show(stdout, m)
```

**IMPORTANT:** The code above has a bug in `estimate_ologit` (duplicate call, placeholder variable). When implementing, the correct version calls `_estimate_ordered` once with `Vector{Int}(y)` directly. Remove the placeholder line.

**Step 4: Add includes and exports to main module**

In `src/MacroEconometricModels.jl`, after line 99 (`include("reg/predict.jl")`), add:
```julia
include("reg/ordered.jl")
```

In the exports section (after line 812), add:
```julia
export OrderedLogitModel, OrderedProbitModel
export estimate_ologit, estimate_oprobit
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/declarative-kindling-wozniak && julia --project=. -e 'using Test, MacroEconometricModels, Distributions; include("test/reg/test_ordered.jl")'`

Expected: PASS

**Step 6: Commit**

```bash
git add src/reg/ordered.jl test/reg/test_ordered.jl src/MacroEconometricModels.jl
git commit -m "feat(reg): add ordered logit/probit estimation (#77)"
```

---

### Task 2: Multinomial Logit Type & Estimation

**Files:**
- Create: `src/reg/multinomial.jl`
- Modify: `src/MacroEconometricModels.jl` (add includes + exports)
- Test: `test/reg/test_multinomial.jl`

**Step 1: Write failing tests for multinomial logit**

Create `test/reg/test_multinomial.jl`:

```julia
using Random, Distributions

@testset "Multinomial Logit" begin
    @testset "estimate_mlogit — coefficient recovery" begin
        rng = MersenneTwister(44)
        n = 3000
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        X = hcat(ones(n), x1, x2)  # with intercept
        # 3 categories, base = 1
        beta2_true = [0.5, 1.0, -0.5]   # coeffs for category 2
        beta3_true = [-0.5, -0.3, 0.8]  # coeffs for category 3
        eta2 = X * beta2_true
        eta3 = X * beta3_true
        y = Vector{Int}(undef, n)
        for i in 1:n
            denom = 1.0 + exp(eta2[i]) + exp(eta3[i])
            p1 = 1.0 / denom
            p2 = exp(eta2[i]) / denom
            u = rand(rng)
            if u < p1
                y[i] = 1
            elseif u < p1 + p2
                y[i] = 2
            else
                y[i] = 3
            end
        end

        m = estimate_mlogit(y, X)
        @test m isa MultinomialLogitModel
        @test size(m.beta) == (3, 2)  # K × (J-1)
        @test m.base_category == 1
        @test m.n_categories == 3
        @test m.converged
        @test all(abs.(m.beta[:, 1] .- beta2_true) .< 0.2)
        @test all(abs.(m.beta[:, 2] .- beta3_true) .< 0.2)
        @test m.pseudo_r2 > 0
    end

    @testset "estimate_mlogit — StatsAPI interface" begin
        rng = MersenneTwister(56)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        y = rand(rng, 1:4, n)  # 4 categories
        m = estimate_mlogit(y, X; varnames=["(Intercept)", "x1", "x2"])
        @test length(coef(m)) == 9  # 3 × (4-1) = 9 vectorized
        @test size(vcov(m)) == (9, 9)
        @test nobs(m) == n
        @test dof(m) == 9
        @test isfinite(loglikelihood(m))
        @test isfinite(aic(m))
        @test isfinite(bic(m))
    end

    @testset "estimate_mlogit — predict" begin
        rng = MersenneTwister(57)
        n = 200
        X = hcat(ones(n), randn(rng, n, 2))
        y = rand(rng, 1:3, n)
        m = estimate_mlogit(y, X)
        probs = predict(m, X)
        @test size(probs) == (n, 3)
        @test all(probs .>= 0)
        @test all(abs.(sum(probs, dims=2) .- 1.0) .< 1e-10)
    end

    @testset "estimate_mlogit — robust SEs" begin
        rng = MersenneTwister(58)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        y = rand(rng, 1:3, n)
        m_ols = estimate_mlogit(y, X; cov_type=:ols)
        m_hc1 = estimate_mlogit(y, X; cov_type=:hc1)
        @test vcov(m_ols) != vcov(m_hc1)
    end

    @testset "estimate_mlogit — display" begin
        rng = MersenneTwister(59)
        n = 200
        X = hcat(ones(n), randn(rng, n, 1))
        y = rand(rng, 1:3, n)
        m = estimate_mlogit(y, X; varnames=["(Intercept)", "x1"])
        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test occursin("Multinomial Logit", s)
        @test occursin("x1", s)
    end
end
```

**Step 2: Run tests to verify they fail**

Expected: FAIL — `estimate_mlogit` not defined.

**Step 3: Implement multinomial logit**

Create `src/reg/multinomial.jl`:

```julia
# ── Multinomial Logit Model ─────────────────────────────────────────

struct MultinomialLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}
    X::Matrix{T}
    beta::Matrix{T}             # K × (J-1)
    vcov_mat::Matrix{T}         # K(J-1) × K(J-1)
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    n_categories::Int
    base_category::Int
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

# ── Softmax probabilities ───────────────────────────────────────────

function _mlogit_probs(xi::AbstractVector{T}, beta::Matrix{T}) where {T}
    # beta is K × (J-1), base category has eta = 0
    J = size(beta, 2) + 1
    etas = zeros(T, J)
    for j in 1:(J-1)
        etas[j+1] = dot(xi, beta[:, j])
    end
    max_eta = maximum(etas)
    etas .-= max_eta  # log-sum-exp trick
    exp_etas = exp.(etas)
    return exp_etas ./ sum(exp_etas)
end

# ── Newton-Raphson MLE ──────────────────────────────────────────────

function _estimate_mlogit(y::Vector{Int}, X::Matrix{T};
                          maxiter::Int=100, tol::T=T(1e-8)) where {T}
    n, k = size(X)
    cats = sort(unique(y))
    J = length(cats)
    cat_map = Dict(c => i for (i, c) in enumerate(cats))
    y_mapped = [cat_map[yi] for yi in y]

    # Initialize beta = 0
    beta = zeros(T, k, J - 1)
    n_params = k * (J - 1)
    converged = false
    iterations = 0
    ll_old = T(-Inf)

    for iter in 1:maxiter
        iterations = iter
        grad = zeros(T, n_params)
        H = zeros(T, n_params, n_params)
        ll = zero(T)

        for i in 1:n
            probs = _mlogit_probs(view(X, i, :), beta)
            yi = y_mapped[i]
            ll += log(max(probs[yi], eps(T)))

            # Score and Hessian
            for j in 1:(J-1)
                ind_j = (yi == j + 1) ? one(T) : zero(T)
                resid_j = ind_j - probs[j+1]
                offset_j = (j - 1) * k
                for l in 1:k
                    grad[offset_j + l] += resid_j * X[i, l]
                end

                # Hessian blocks
                for j2 in 1:(J-1)
                    offset_j2 = (j2 - 1) * k
                    w_jj2 = if j == j2
                        -probs[j+1] * (one(T) - probs[j+1])
                    else
                        probs[j+1] * probs[j2+1]
                    end
                    for l1 in 1:k, l2 in 1:k
                        H[offset_j + l1, offset_j2 + l2] += w_jj2 * X[i, l1] * X[i, l2]
                    end
                end
            end
        end

        if abs(ll - ll_old) / (abs(ll) + one(T)) < tol
            converged = true
            break
        end
        ll_old = ll

        delta = -(Matrix{T}(robust_inv(Hermitian(H))) * grad)
        for j in 1:(J-1)
            offset = (j - 1) * k
            beta[:, j] .+= delta[(offset+1):(offset+k)]
        end
    end

    return beta, y_mapped, cats, converged, iterations
end

# ── Sandwich covariance ─────────────────────────────────────────────

function _mlogit_vcov(beta::Matrix{T}, y::Vector{Int}, X::Matrix{T},
                      cov_type::Symbol; clusters=nothing) where {T}
    n, k = size(X)
    J = size(beta, 2) + 1
    n_params = k * (J - 1)

    scores = zeros(T, n, n_params)
    H = zeros(T, n_params, n_params)

    for i in 1:n
        probs = _mlogit_probs(view(X, i, :), beta)
        yi = y[i]
        for j in 1:(J-1)
            ind_j = (yi == j + 1) ? one(T) : zero(T)
            resid_j = ind_j - probs[j+1]
            offset = (j - 1) * k
            for l in 1:k
                scores[i, offset + l] = resid_j * X[i, l]
            end
            for j2 in 1:(J-1)
                offset2 = (j2 - 1) * k
                w = j == j2 ? -probs[j+1] * (one(T) - probs[j+1]) : probs[j+1] * probs[j2+1]
                for l1 in 1:k, l2 in 1:k
                    H[offset + l1, offset2 + l2] += w * X[i, l1] * X[i, l2]
                end
            end
        end
    end

    Hinv = Matrix{T}(robust_inv(Hermitian(H)))

    if cov_type == :ols
        return -Hinv
    end

    if cov_type == :cluster && clusters !== nothing
        ugroups = unique(clusters)
        G = length(ugroups)
        B = zeros(T, n_params, n_params)
        for g in ugroups
            idx = findall(==(g), clusters)
            sg = vec(sum(scores[idx, :], dims=1))
            B .+= sg * sg'
        end
        B .*= T(G) / T(G - 1)
    else
        B = scores' * scores
        B .*= T(n) / T(n - n_params)
    end

    return Hinv * B * Hinv
end

# ── Public API ──────────────────────────────────────────────────────

function estimate_mlogit(y::AbstractVector{<:Integer}, X::AbstractMatrix{T};
                         cov_type::Symbol=:ols,
                         varnames::Union{Nothing,Vector{String}}=nothing,
                         clusters::Union{Nothing,AbstractVector}=nothing,
                         maxiter::Int=100,
                         tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, k = size(X)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames

    beta, y_mapped, cats, converged, iterations =
        _estimate_mlogit(Vector{Int}(y), Matrix{T}(X); maxiter=maxiter, tol=tol)

    J = length(cats)
    vcov_mat = _mlogit_vcov(beta, y_mapped, Matrix{T}(X), cov_type; clusters=clusters)

    # Log-likelihood
    ll = zero(T)
    for i in 1:n
        probs = _mlogit_probs(view(X, i, :), beta)
        ll += log(max(probs[y_mapped[i]], eps(T)))
    end
    # Null: equal probabilities
    ll_null = T(n) * log(one(T) / T(J))
    pseudo_r2 = one(T) - ll / ll_null
    n_params = k * (J - 1)
    aic_val = -2ll + 2n_params
    bic_val = -2ll + n_params * log(T(n))

    return MultinomialLogitModel{T}(y_mapped, Matrix{T}(X), beta, vcov_mat,
                                    ll, ll_null, pseudo_r2, aic_val, bic_val,
                                    J, 1, vnames, converged, iterations, cov_type)
end

function estimate_mlogit(y::AbstractVector{<:Integer}, X::AbstractMatrix; kwargs...)
    estimate_mlogit(y, Matrix{Float64}(X); kwargs...)
end

# ── StatsAPI interface ──────────────────────────────────────────────

StatsAPI.coef(m::MultinomialLogitModel) = vec(m.beta)
StatsAPI.vcov(m::MultinomialLogitModel) = m.vcov_mat
StatsAPI.nobs(m::MultinomialLogitModel) = length(m.y)
StatsAPI.dof(m::MultinomialLogitModel) = length(m.beta)
StatsAPI.loglikelihood(m::MultinomialLogitModel) = m.loglik
StatsAPI.aic(m::MultinomialLogitModel) = m.aic
StatsAPI.bic(m::MultinomialLogitModel) = m.bic
StatsAPI.islinear(m::MultinomialLogitModel) = false

function StatsAPI.stderror(m::MultinomialLogitModel{T}) where {T}
    [sqrt(max(m.vcov_mat[i,i], zero(T))) for i in 1:size(m.vcov_mat, 1)]
end

function StatsAPI.confint(m::MultinomialLogitModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    z = T(quantile(Normal(), (1 + level) / 2))
    params = vec(m.beta)
    hcat(params .- z .* se, params .+ z .* se)
end

# ── Predict: n × J probability matrix ──────────────────────────────

function StatsAPI.predict(m::MultinomialLogitModel{T}, X_new::AbstractMatrix) where {T}
    n = size(X_new, 1)
    probs = Matrix{T}(undef, n, m.n_categories)
    for i in 1:n
        probs[i, :] .= _mlogit_probs(view(X_new, i, :), m.beta)
    end
    return probs
end

# ── Display ─────────────────────────────────────────────────────────

function Base.show(io::IO, m::MultinomialLogitModel{T}) where {T}
    println(io, "Multinomial Logit Model")
    println(io, "═══════════════════════")
    println(io, "  Observations:    $(nobs(m))")
    println(io, "  Categories:      $(m.n_categories) (base = $(m.base_category))")
    println(io, "  Log-likelihood:  $(_fmt(m.loglik))")
    println(io, "  Pseudo R²:       $(_fmt(m.pseudo_r2))")
    println(io, "  AIC:             $(_fmt(m.aic))")
    println(io, "  BIC:             $(_fmt(m.bic))")
    println(io, "  Converged:       $(m.converged) ($(m.iterations) iterations)")
    println(io, "  Cov. type:       $(m.cov_type)")
    k = length(m.varnames)
    se_all = stderror(m)
    for j in 1:(m.n_categories - 1)
        println(io)
        offset = (j - 1) * k
        cat_label = j + 1  # base=1, so alternatives are 2, 3, ...
        _coef_table(io, "Category $cat_label vs $(m.base_category)",
                    m.varnames, m.beta[:, j], se_all[(offset+1):(offset+k)]; dist=:z)
    end
end

report(m::MultinomialLogitModel) = show(stdout, m)
```

**Step 4: Add includes and exports to main module**

After the ordered.jl include, add:
```julia
include("reg/multinomial.jl")
```

Exports:
```julia
export MultinomialLogitModel
export estimate_mlogit
```

**Step 5: Run tests, verify pass**

**Step 6: Commit**

```bash
git add src/reg/multinomial.jl test/reg/test_multinomial.jl src/MacroEconometricModels.jl
git commit -m "feat(reg): add multinomial logit estimation (#77)"
```

---

### Task 3: Marginal Effects for Ordered & Multinomial Models

**Files:**
- Modify: `src/reg/margins.jl` (add dispatches)
- Modify: `test/reg/test_ordered.jl`, `test/reg/test_multinomial.jl`

**Step 1: Write failing tests**

Add to `test/reg/test_ordered.jl`:
```julia
@testset "Ordered Logit — marginal effects" begin
    rng = MersenneTwister(60)
    n = 1000
    X = randn(rng, n, 2)
    beta_true = [1.0, -0.5]
    cutpoints_true = [-1.0, 1.0]
    eta = X * beta_true
    y = Vector{Int}(undef, n)
    for i in 1:n
        u = rand(rng)
        p1 = 1.0 / (1.0 + exp(-(cutpoints_true[1] - eta[i])))
        p2 = 1.0 / (1.0 + exp(-(cutpoints_true[2] - eta[i])))
        y[i] = u < p1 ? 1 : (u < p2 ? 2 : 3)
    end
    m = estimate_ologit(y, X; varnames=["x1", "x2"])
    me = marginal_effects(m)
    @test me isa Matrix  # K × J matrix of AMEs
    @test size(me) == (2, 3)
    # AMEs across categories sum to zero for each variable
    @test all(abs.(sum(me, dims=2)) .< 0.05)
end
```

Add to `test/reg/test_multinomial.jl`:
```julia
@testset "Multinomial Logit — marginal effects" begin
    rng = MersenneTwister(61)
    n = 1000
    X = hcat(ones(n), randn(rng, n, 2))
    y = rand(rng, 1:3, n)
    m = estimate_mlogit(y, X; varnames=["(Intercept)", "x1", "x2"])
    me = marginal_effects(m)
    @test me isa Matrix  # K × J matrix of AMEs
    @test size(me) == (3, 3)
    # AMEs across categories sum to zero for each variable
    @test all(abs.(sum(me, dims=2)) .< 0.05)
end
```

**Step 2: Implement marginal effects dispatches**

In `src/reg/margins.jl` (or in `ordered.jl`/`multinomial.jl` — implementer's choice, but keep consistent):

For ordered models: AME for variable j on category c = (1/n) Σᵢ [f(αc - x'β) - f(αc₋₁ - x'β)] * (-βⱼ)

For multinomial: AME for variable j on category c = (1/n) Σᵢ pᵢc [βⱼc - Σₘ pᵢₘ βⱼₘ]

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git commit -m "feat(reg): add marginal effects for ordered/multinomial models (#77)"
```

---

### Task 4: Brant Test & Hausman IIA Test

**Files:**
- Modify: `src/reg/ordered.jl` (add `brant_test`)
- Modify: `src/reg/multinomial.jl` (add `hausman_iia`)
- Modify: `test/reg/test_ordered.jl`, `test/reg/test_multinomial.jl`

**Step 1: Write failing tests**

```julia
# In test_ordered.jl
@testset "Brant test" begin
    rng = MersenneTwister(70)
    n = 1000
    X = randn(rng, n, 2)
    y = rand(rng, 1:3, n)  # random data — parallel assumption likely holds
    m = estimate_ologit(y, X)
    bt = brant_test(m)
    @test bt isa NamedTuple
    @test haskey(bt, :statistic)
    @test haskey(bt, :pvalue)
    @test haskey(bt, :df)
    @test bt.df == 2  # K variables, J-2 free binary comparisons
end

# In test_multinomial.jl
@testset "Hausman IIA test" begin
    rng = MersenneTwister(71)
    n = 1000
    X = hcat(ones(n), randn(rng, n, 2))
    y = rand(rng, 1:3, n)
    m = estimate_mlogit(y, X)
    ht = hausman_iia(m; omit_category=3)
    @test ht isa NamedTuple
    @test haskey(ht, :statistic)
    @test haskey(ht, :pvalue)
    @test haskey(ht, :df)
end
```

**Step 2: Implement**

- `brant_test`: Fit J-1 separate binary logits (y ≤ j vs y > j), test equality of coefficients across equations using Wald statistic.
- `hausman_iia`: Re-estimate model excluding one category, compare coefficients via Hausman χ² statistic.

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git commit -m "feat(reg): add Brant test and Hausman IIA test (#77)"
```

---

### Task 5: refs() Dispatches for Ordered & Multinomial

**Files:**
- Modify: `src/summary_refs.jl` (add refs dispatches and bibliography entries)
- Test: verify with `show(io, refs(m))` in existing test files

**Step 1: Add refs entries**

Add to `_TYPE_REFS` dict and `_BIBLIOGRAPHY`:
- `:OrderedLogitModel` → `[:mccullagh1980, :brant1990, :wooldridge2010]`
- `:OrderedProbitModel` → `[:mccullagh1980, :wooldridge2010]`
- `:MultinomialLogitModel` → `[:mcfadden1974, :hausman_mcfadden1984, :wooldridge2010]`

**Step 2: Add dispatch methods**

```julia
refs(io::IO, ::OrderedLogitModel; kw...) = refs(io, _TYPE_REFS[:OrderedLogitModel]; kw...)
refs(io::IO, ::OrderedProbitModel; kw...) = refs(io, _TYPE_REFS[:OrderedProbitModel]; kw...)
refs(io::IO, ::MultinomialLogitModel; kw...) = refs(io, _TYPE_REFS[:MultinomialLogitModel]; kw...)
```

**Step 3: Commit**

```bash
git commit -m "feat(reg): add refs() for ordered/multinomial models (#77)"
```

---

### Task 6: Register Tests in runtests.jl

**Files:**
- Modify: `test/runtests.jl`

Add `"reg/test_ordered.jl"` and `"reg/test_multinomial.jl"` to Group 5 (after `"reg/test_reg.jl"`), and add corresponding `@testset` entries in the serial fallback section.

**Commit:**
```bash
git commit -m "test: register ordered/multinomial tests in runtests.jl (#77)"
```

---

## Workstream B: Panel Regression (#66)

### Task 7: Panel Regression Types

**Files:**
- Create: `src/panel_reg/types.jl`
- Modify: `src/MacroEconometricModels.jl` (add includes + exports)

**Step 1: Write types file**

Create `src/panel_reg/types.jl` with all 5 types from the design doc:
- `PanelRegModel{T}` — FE/RE/FD/Between/CRE
- `PanelIVModel{T}` — panel IV
- `PanelLogitModel{T}` — panel logit
- `PanelProbitModel{T}` — panel probit
- `PanelTestResult{T}` — specification tests

Plus StatsAPI interface implementations for each (coef, vcov, nobs, dof, etc.).

**Step 2: Add includes to main module**

After line 247 (pvar_lag_selection.jl), add a panel_reg block:
```julia
# ── Panel Regression ──
include("panel_reg/types.jl")
```

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add type definitions (#66)"
```

---

### Task 8: Panel Covariance Estimators

**Files:**
- Create: `src/panel_reg/covariance.jl`
- Test: `test/panel_reg/test_panel_reg.jl` (initial file)

**Step 1: Write failing tests**

```julia
@testset "Panel Covariance" begin
    @testset "entity cluster" begin
        rng = MersenneTwister(100)
        n, k = 500, 3
        X = randn(rng, n, k)
        resid = randn(rng, n)
        groups = repeat(1:50, inner=10)
        XtXinv = inv(X' * X)
        V = _panel_cluster_vcov(X, resid, XtXinv, groups)
        @test size(V) == (k, k)
        @test issymmetric(V) || isapprox(V, V', atol=1e-12)
        @test all(diag(V) .> 0)
    end

    @testset "two-way cluster" begin
        # Cameron-Gelbach-Miller: V = V_entity + V_time - V_entity×time
        rng = MersenneTwister(101)
        N, T_per = 20, 10
        n = N * T_per
        k = 2
        X = randn(rng, n, k)
        resid = randn(rng, n)
        entity = repeat(1:N, inner=T_per)
        time = repeat(1:T_per, outer=N)
        XtXinv = inv(X' * X)
        V = _panel_twoway_vcov(X, resid, XtXinv, entity, time)
        @test size(V) == (k, k)
        @test all(diag(V) .> 0)
    end

    @testset "Driscoll-Kraay" begin
        rng = MersenneTwister(102)
        N, T_per = 20, 25
        n = N * T_per
        k = 2
        X = randn(rng, n, k)
        resid = randn(rng, n)
        entity = repeat(1:N, inner=T_per)
        time = repeat(1:T_per, outer=N)
        XtXinv = inv(X' * X)
        V = _panel_driscoll_kraay_vcov(X, resid, XtXinv, entity, time; bandwidth=3)
        @test size(V) == (k, k)
        @test all(diag(V) .> 0)
    end
end
```

**Step 2: Implement covariance estimators**

- `_panel_cluster_vcov(X, resid, XtXinv, groups)` — entity cluster with G/(G-1) correction
- `_panel_time_cluster_vcov(X, resid, XtXinv, time_ids)` — time cluster
- `_panel_twoway_vcov(X, resid, XtXinv, groups, time_ids)` — V_g + V_t - V_gt
- `_panel_driscoll_kraay_vcov(X, resid, XtXinv, groups, time_ids; bandwidth)` — Newey-West on cross-sectional averages

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add panel covariance estimators (#66)"
```

---

### Task 9: Fixed Effects (Within) Estimation

**Files:**
- Create: `src/panel_reg/estimation.jl`
- Modify: `test/panel_reg/test_panel_reg.jl`

**Step 1: Write failing tests for FE**

```julia
@testset "estimate_xtreg — Fixed Effects" begin
    @testset "coefficient recovery with entity FE" begin
        rng = MersenneTwister(200)
        N, T_per = 50, 20
        beta_true = [1.5, -0.8]
        # Generate panel with entity fixed effects
        df = DataFrame()
        for i in 1:N
            alpha_i = randn(rng)  # entity effect
            x1 = randn(rng, T_per)
            x2 = randn(rng, T_per)
            y = alpha_i .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.5 .* randn(rng, T_per)
            append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, x2=x2))
        end
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
        @test m isa PanelRegModel
        @test m.method == :fe
        @test all(abs.(coef(m) .- beta_true) .< 0.1)
        @test m.n_groups == N
        @test m.n_obs == N * T_per
        @test m.r2_within > 0.8
        @test m.group_effects !== nothing
        @test length(m.group_effects) == N
    end

    @testset "two-way FE" begin
        rng = MersenneTwister(201)
        N, T_per = 30, 15
        beta_true = [1.0]
        df = DataFrame()
        time_effects = randn(rng, T_per)
        for i in 1:N
            alpha_i = randn(rng)
            x1 = randn(rng, T_per)
            y = alpha_i .+ time_effects .+ beta_true[1] .* x1 .+ 0.5 .* randn(rng, T_per)
            append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
        end
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:fe, twoway=true)
        @test m.twoway
        @test abs(coef(m)[1] - beta_true[1]) < 0.1
    end

    @testset "FE with clustered SEs" begin
        rng = MersenneTwister(202)
        N, T_per = 50, 20
        df = DataFrame()
        for i in 1:N
            x1 = randn(rng, T_per)
            y = randn(rng) .+ x1 .+ randn(rng, T_per)
            append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
        end
        pd = xtset(df, :id, :t)
        m_ols = estimate_xtreg(pd, :y, [:x1]; model=:fe, cov_type=:ols)
        m_cl = estimate_xtreg(pd, :y, [:x1]; model=:fe, cov_type=:cluster)
        @test vcov(m_ols) != vcov(m_cl)
    end
end
```

**Step 2: Implement FE estimator**

In `src/panel_reg/estimation.jl`:
- `estimate_xtreg(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol}; model=:fe, twoway=false, cov_type=:cluster)` — main dispatch
- `_estimate_fe(y, X, group_ids, time_ids; twoway, cov_type)` — within transformation + OLS
- Within transform: subtract group means (and time means, add grand mean for two-way)
- R² variants: within = 1 - SSR/TSS_within; between = corr(ȳᵢ, x̄ᵢ'β)²; overall = corr(yᵢₜ, x'β + α̂ᵢ)²

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git commit -m "feat(panel_reg): add FE (within) estimator (#66)"
```

---

### Task 10: Random Effects (GLS) Estimation

**Files:**
- Modify: `src/panel_reg/estimation.jl`
- Modify: `test/panel_reg/test_panel_reg.jl`

**Step 1: Write failing tests for RE**

```julia
@testset "estimate_xtreg — Random Effects" begin
    @testset "coefficient recovery" begin
        rng = MersenneTwister(210)
        N, T_per = 50, 20
        beta_true = [0.5, 1.0, -0.3]
        df = DataFrame()
        for i in 1:N
            alpha_i = 0.5 * randn(rng)  # RE: uncorrelated with X
            x1 = randn(rng, T_per)
            x2 = randn(rng, T_per)
            y = 1.0 .+ alpha_i .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ 0.3 .* randn(rng, T_per)
            append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, x2=x2))
        end
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)
        @test m.method == :re
        @test m.theta !== nothing
        @test m.sigma_u > 0
        @test m.sigma_e > 0
        @test 0 < m.rho < 1
        # RE should recover intercept + slopes
        @test all(abs.(coef(m)[2:3] .- beta_true[1:2]) .< 0.15)
    end

    @testset "RE variance components" begin
        rng = MersenneTwister(211)
        N, T_per = 100, 10
        sigma_u_true = 1.0
        sigma_e_true = 0.5
        df = DataFrame()
        for i in 1:N
            alpha_i = sigma_u_true * randn(rng)
            x1 = randn(rng, T_per)
            y = alpha_i .+ x1 .+ sigma_e_true .* randn(rng, T_per)
            append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
        end
        pd = xtset(df, :id, :t)
        m = estimate_xtreg(pd, :y, [:x1]; model=:re)
        @test abs(m.sigma_u - sigma_u_true) < 0.3
        @test abs(m.sigma_e - sigma_e_true) < 0.2
    end
end
```

**Step 2: Implement RE estimator**

- `_estimate_re(y, X, group_ids; cov_type)`:
  1. Run FE to get σ²ₑ = SSR_FE / (NT - N - K)
  2. Run Between to get σ²ᵤ = (σ²_between - σ²ₑ/T̄) (Swamy-Arora)
  3. θ = 1 - √(σ²ₑ / (T̄ᵢσ²ᵤ + σ²ₑ)) per group (or common θ for balanced)
  4. Quasi-demean: ỹᵢₜ = yᵢₜ - θȳᵢ, X̃ = Xᵢₜ - θX̄ᵢ
  5. OLS on quasi-demeaned data (with intercept)

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git commit -m "feat(panel_reg): add RE (GLS) estimator (#66)"
```

---

### Task 11: FD, Between, and CRE Estimators

**Files:**
- Modify: `src/panel_reg/estimation.jl`
- Modify: `test/panel_reg/test_panel_reg.jl`

**Step 1: Write tests for FD, Between, CRE**

```julia
@testset "estimate_xtreg — First Differences" begin
    rng = MersenneTwister(220)
    N, T_per = 50, 20
    beta_true = [1.0, -0.5]
    df = DataFrame()
    for i in 1:N
        alpha_i = randn(rng)
        x1 = cumsum(randn(rng, T_per))  # persistent X
        x2 = randn(rng, T_per)
        y = alpha_i .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, x2=x2))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fd)
    @test m.method == :fd
    @test all(abs.(coef(m) .- beta_true) .< 0.15)
end

@testset "estimate_xtreg — Between" begin
    rng = MersenneTwister(230)
    N, T_per = 100, 10
    df = DataFrame()
    for i in 1:N
        x1 = fill(randn(rng), T_per)  # time-invariant
        y = 0.5 .+ 2.0 .* x1 .+ 0.3 .* randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1]; model=:between)
    @test m.method == :between
    @test abs(coef(m)[2] - 2.0) < 0.2  # intercept + slope
end

@testset "estimate_xtreg — CRE (Mundlak)" begin
    rng = MersenneTwister(240)
    N, T_per = 50, 20
    beta_true = [1.0, -0.5]
    df = DataFrame()
    for i in 1:N
        alpha_i = randn(rng)
        x1 = alpha_i .+ randn(rng, T_per)  # correlated with FE
        x2 = randn(rng, T_per)
        y = alpha_i .+ beta_true[1] .* x1 .+ beta_true[2] .* x2 .+ randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, x2=x2))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:cre)
    @test m.method == :cre
    # CRE should recover same slopes as FE when FE is appropriate
    m_fe = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
    @test all(abs.(coef(m)[2:3] .- coef(m_fe)) .< 0.15)
end
```

**Step 2: Implement FD, Between, CRE**

- **FD:** First-difference y and X within groups (drop first obs per group), OLS with intercept on Δy, ΔX.
- **Between:** Compute group means ȳᵢ, X̄ᵢ, OLS on means with intercept. n_obs = N.
- **CRE (Mundlak):** Compute group means X̄ᵢ for all time-varying regressors, augment X with repeated X̄ᵢ, run RE.

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git commit -m "feat(panel_reg): add FD, Between, CRE estimators (#66)"
```

---

### Task 12: Specification Tests

**Files:**
- Create: `src/panel_reg/tests.jl`
- Create: `test/panel_reg/test_panel_tests.jl`

**Step 1: Write failing tests**

```julia
@testset "Hausman test (FE vs RE)" begin
    rng = MersenneTwister(300)
    N, T_per = 50, 20
    df = DataFrame()
    for i in 1:N
        alpha_i = randn(rng)
        x1 = alpha_i .+ randn(rng, T_per)  # correlated → FE preferred
        y = alpha_i .+ x1 .+ randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m_fe = estimate_xtreg(pd, :y, [:x1]; model=:fe)
    m_re = estimate_xtreg(pd, :y, [:x1]; model=:re)
    ht = hausman_test(m_fe, m_re)
    @test ht isa PanelTestResult
    @test ht.pvalue < 0.05  # should reject RE
end

@testset "Breusch-Pagan LM test" begin
    rng = MersenneTwister(301)
    N, T_per = 50, 20
    df = DataFrame()
    for i in 1:N
        alpha_i = 2.0 * randn(rng)  # large RE variance
        x1 = randn(rng, T_per)
        y = alpha_i .+ x1 .+ randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m_re = estimate_xtreg(pd, :y, [:x1]; model=:re)
    bp = breusch_pagan_test(m_re)
    @test bp isa PanelTestResult
    @test bp.pvalue < 0.05  # should reject pooled OLS
end

@testset "Pesaran CD test" begin
    rng = MersenneTwister(302)
    N, T_per = 20, 30
    common_shock = randn(rng, T_per)
    df = DataFrame()
    for i in 1:N
        x1 = randn(rng, T_per)
        y = x1 .+ 0.5 .* common_shock .+ randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1]; model=:fe)
    cd = pesaran_cd_test(m)
    @test cd isa PanelTestResult
    @test cd.pvalue < 0.05  # common shock → cross-sectional dependence
end

@testset "Wooldridge AR test" begin
    rng = MersenneTwister(303)
    N, T_per = 50, 20
    df = DataFrame()
    for i in 1:N
        x1 = randn(rng, T_per)
        e = zeros(T_per)
        e[1] = randn(rng)
        for t in 2:T_per
            e[t] = 0.5 * e[t-1] + randn(rng)  # AR(1) errors
        end
        y = randn(rng) .+ x1 .+ e
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1]; model=:fe)
    wt = wooldridge_ar_test(m)
    @test wt isa PanelTestResult
    @test wt.pvalue < 0.10  # AR(1) errors present
end

@testset "Modified Wald test" begin
    rng = MersenneTwister(304)
    N, T_per = 30, 20
    df = DataFrame()
    for i in 1:N
        sigma_i = 0.5 + 2.0 * rand(rng)  # heterogeneous variance
        x1 = randn(rng, T_per)
        y = randn(rng) .+ x1 .+ sigma_i .* randn(rng, T_per)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1]; model=:fe)
    mw = modified_wald_test(m)
    @test mw isa PanelTestResult
    @test mw.pvalue < 0.05  # groupwise heteroskedasticity present
end
```

**Step 2: Implement specification tests**

- `hausman_test(fe::PanelRegModel, re::PanelRegModel)` — Wald chi² on common coefficients
- `breusch_pagan_test(re::PanelRegModel)` — LM statistic using pooled OLS residuals
- `f_test_fe(fe::PanelRegModel)` — F-test: (SSR_pooled - SSR_fe) / (N-1) / (SSR_fe / (NT-N-K))
- `pesaran_cd_test(m::PanelRegModel)` — pairwise residual correlations
- `wooldridge_ar_test(m::PanelRegModel)` — FD residual regression
- `modified_wald_test(m::PanelRegModel)` — χ² = Σᵢ (σ̂²ᵢ - σ̂²)² / Var(σ̂²ᵢ)

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add specification tests (#66)"
```

---

### Task 13: Panel Predict & Display

**Files:**
- Create: `src/panel_reg/predict.jl`
- Create: `src/panel_reg/display.jl`
- Modify tests

**Step 1: Implement predict**

```julia
# Within prediction: X * beta (demeaned)
# Overall prediction: X * beta + alpha_i (for FE)
# Between prediction: X̄ᵢ * beta
StatsAPI.predict(m::PanelRegModel{T}) where {T} = m.fitted
StatsAPI.predict(m::PanelRegModel{T}, pd_new::PanelData) where {T} = ...
```

**Step 2: Implement display**

`show(io, m::PanelRegModel)` — Stata-style output:
- Header: model type, n_obs, n_groups, obs_per_group (min/avg/max)
- Coefficient table via `_coef_table()`
- Footer: sigma_u, sigma_e, rho (for RE), R² variants
- F-test line

`report(m::PanelRegModel) = show(stdout, m)`

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add predict and display (#66)"
```

---

### Task 14: Panel IV Estimation

**Files:**
- Create: `src/panel_reg/iv.jl`
- Create: `test/panel_reg/test_panel_iv.jl`

**Step 1: Write failing tests**

```julia
@testset "estimate_xtiv — FE-IV" begin
    rng = MersenneTwister(400)
    N, T_per = 50, 20
    beta_true = [1.5]
    df = DataFrame()
    for i in 1:N
        alpha_i = randn(rng)
        z1 = randn(rng, T_per)
        x1 = 0.7 .* z1 .+ 0.3 .* randn(rng, T_per)  # endogenous
        e = randn(rng, T_per)
        x1 .+= 0.5 .* e  # correlation with error
        y = alpha_i .+ beta_true[1] .* x1 .+ e
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, z1=z1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtiv(pd, :y, Symbol[], [:x1]; instruments=[:z1], model=:fe)
    @test m isa PanelIVModel
    @test m.method == :fe_iv
    @test abs(coef(m)[1] - beta_true[1]) < 0.3
    @test m.first_stage_f > 10  # strong instrument
end

@testset "estimate_xtiv — FD-IV" begin
    rng = MersenneTwister(401)
    N, T_per = 50, 20
    df = DataFrame()
    for i in 1:N
        z1 = randn(rng, T_per)
        x1 = 0.7 .* z1 .+ randn(rng, T_per)
        e = randn(rng, T_per)
        x1 .+= 0.3 .* e
        y = randn(rng) .+ x1 .+ e
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, z1=z1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtiv(pd, :y, Symbol[], [:x1]; instruments=[:z1], model=:fd)
    @test m.method == :fd_iv
    @test abs(coef(m)[1] - 1.0) < 0.3
end
```

**Step 2: Implement FE-IV, RE-IV, FD-IV, Hausman-Taylor**

- FE-IV: within-demean y, X, Z, then 2SLS (reuse IV logic from `src/reg/iv.jl`)
- RE-IV (EC2SLS): quasi-demean, use within + between transformed instruments
- FD-IV: first-difference all, 2SLS
- Hausman-Taylor: partition variables, construct instruments from group means + within deviations

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add panel IV estimators (#66)"
```

---

### Task 15: Panel Logit & Probit

**Files:**
- Create: `src/panel_reg/logit.jl`
- Create: `src/panel_reg/probit.jl`
- Create: `test/panel_reg/test_panel_nonlinear.jl`

**Step 1: Write failing tests**

```julia
@testset "estimate_xtlogit — pooled with cluster SEs" begin
    rng = MersenneTwister(500)
    N, T_per = 50, 10
    beta_true = [1.0, -0.5]
    df = DataFrame()
    for i in 1:N
        x1 = randn(rng, T_per)
        x2 = randn(rng, T_per)
        eta = beta_true[1] .* x1 .+ beta_true[2] .* x2
        prob = 1.0 ./ (1.0 .+ exp.(-eta))
        y = Float64.(rand(rng, T_per) .< prob)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1, x2=x2))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:pooled)
    @test m isa PanelLogitModel
    @test m.method == :pooled
    @test all(abs.(coef(m) .- beta_true) .< 0.3)
end

@testset "estimate_xtlogit — FE (conditional)" begin
    rng = MersenneTwister(501)
    N, T_per = 100, 10
    df = DataFrame()
    for i in 1:N
        alpha_i = randn(rng)
        x1 = randn(rng, T_per)
        eta = alpha_i .+ 1.0 .* x1
        prob = 1.0 ./ (1.0 .+ exp.(-eta))
        y = Float64.(rand(rng, T_per) .< prob)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtlogit(pd, :y, [:x1]; model=:fe)
    @test m.method == :fe
    @test abs(coef(m)[1] - 1.0) < 0.3
end

@testset "estimate_xtlogit — RE" begin
    rng = MersenneTwister(502)
    N, T_per = 50, 10
    df = DataFrame()
    for i in 1:N
        alpha_i = 0.5 * randn(rng)
        x1 = randn(rng, T_per)
        eta = alpha_i .+ 0.8 .* x1
        prob = 1.0 ./ (1.0 .+ exp.(-eta))
        y = Float64.(rand(rng, T_per) .< prob)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtlogit(pd, :y, [:x1]; model=:re)
    @test m.method == :re
    @test m.sigma_u !== nothing
    @test m.sigma_u > 0
end

@testset "estimate_xtprobit — RE" begin
    rng = MersenneTwister(510)
    N, T_per = 50, 10
    df = DataFrame()
    for i in 1:N
        alpha_i = 0.5 * randn(rng)
        x1 = randn(rng, T_per)
        eta = alpha_i .+ 0.8 .* x1
        prob = cdf.(Normal(), eta)
        y = Float64.(rand(rng, T_per) .< prob)
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtprobit(pd, :y, [:x1]; model=:re)
    @test m isa PanelProbitModel
    @test m.method == :re
    @test m.sigma_u > 0
end
```

**Step 2: Implement panel logit/probit**

- **Pooled:** Call `estimate_logit`/`estimate_probit` with `clusters=group_ids`, wrap in PanelLogitModel/PanelProbitModel.
- **FE Logit (conditional):** For each group with variation in y, condition on Σₜyᵢₜ. Use Newton-Raphson on conditional log-likelihood. Groups with all-0 or all-1 are dropped.
- **RE Logit/Probit:** Log-likelihood = Σᵢ log ∫ Πₜ F(xᵢₜ'β + σᵤu) φ(u) du. Use `_gauss_hermite_nodes_weights()` for integration. Joint optimize (β, log σᵤ).
- **CRE:** Add group means to X, then run RE.

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add panel logit/probit (#66)"
```

---

### Task 16: Panel Marginal Effects

**Files:**
- Create: `src/panel_reg/margins.jl`
- Modify: `test/panel_reg/test_panel_nonlinear.jl`

**Step 1: Tests**

```julia
@testset "Panel marginal effects" begin
    # ... estimate a panel logit/probit ...
    me = marginal_effects(m)
    @test me isa MarginalEffects
    @test length(me.effects) == length(coef(m))
end
```

**Step 2: Implement**

For pooled: same as cross-sectional marginal_effects.
For RE: AME = (1/n) Σᵢ ∫ f(x'β + σᵤu) β φ(u) du — numerical integration.
For FE logit: AME computed conditional on estimated effects.

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add panel marginal effects (#66)"
```

---

### Task 17: Dynamic Panel Wrappers

**Files:**
- Modify: `src/panel_reg/estimation.jl`
- Add tests in `test/panel_reg/test_panel_reg.jl`

**Step 1: Tests**

```julia
@testset "estimate_xtreg — Arellano-Bond" begin
    rng = MersenneTwister(600)
    N, T_per = 50, 15
    df = DataFrame()
    for i in 1:N
        y = zeros(T_per)
        x1 = randn(rng, T_per)
        y[1] = randn(rng)
        for t in 2:T_per
            y[t] = 0.3 * y[t-1] + 0.5 * x1[t] + randn(rng)
        end
        append!(df, DataFrame(id=fill(i, T_per), t=1:T_per, y=y, x1=x1))
    end
    pd = xtset(df, :id, :t)
    m = estimate_xtreg(pd, :y, [:x1]; model=:ab)
    @test m isa PanelRegModel || m isa PVARModel  # wrapped or direct
end
```

**Step 2: Implement wrappers**

`:ab` dispatches to `estimate_pvar(pd, 1; dependent_vars=..., exog_vars=..., transformation=:fd)`
`:bb` dispatches to `estimate_pvar(pd, 1; ..., system_instruments=true)`

Wrap result in PanelRegModel or return PVARModel directly with clear documentation.

**Step 3: Commit**

```bash
git commit -m "feat(panel_reg): add Arellano-Bond/Blundell-Bond wrappers (#66)"
```

---

### Task 18: refs() Dispatches for Panel Models

**Files:**
- Modify: `src/summary_refs.jl`

Add refs dispatches for PanelRegModel, PanelIVModel, PanelLogitModel, PanelProbitModel, PanelTestResult.

Key references by method:
- FE/RE: Wooldridge (2010), Baltagi (2021), Swamy & Arora (1972)
- FD: Wooldridge (2010)
- CRE: Mundlak (1978), Chamberlain (1982)
- Panel IV: Baltagi (1981), Hausman & Taylor (1981), Anderson & Hsiao (1982)
- Panel logit FE: Chamberlain (1980)
- Panel logit/probit RE: Butler & Moffitt (1982)
- Hausman test: Hausman (1978)
- Pesaran CD: Pesaran (2004)
- Driscoll-Kraay: Driscoll & Kraay (1998)
- Two-way cluster: Cameron, Gelbach & Miller (2011)

**Commit:**
```bash
git commit -m "feat(panel_reg): add refs() dispatches (#66)"
```

---

### Task 19: Register All Panel Tests & Final Module Wiring

**Files:**
- Modify: `src/MacroEconometricModels.jl` — complete all includes and exports
- Modify: `test/runtests.jl` — register new test files
- Create: `test/panel_reg/` directory

**Step 1: Complete module includes**

After existing panel_reg/types.jl include, add:
```julia
include("panel_reg/covariance.jl")
include("panel_reg/estimation.jl")
include("panel_reg/iv.jl")
include("panel_reg/logit.jl")
include("panel_reg/probit.jl")
include("panel_reg/tests.jl")
include("panel_reg/margins.jl")
include("panel_reg/predict.jl")
include("panel_reg/display.jl")
```

**Step 2: Complete exports**

```julia
export PanelRegModel, PanelIVModel, PanelLogitModel, PanelProbitModel, PanelTestResult
export estimate_xtreg, estimate_xtiv, estimate_xtlogit, estimate_xtprobit
export hausman_test, breusch_pagan_test, pesaran_cd_test, wooldridge_ar_test, modified_wald_test
```

**Step 3: Register in runtests.jl**

Add a new test group or append to Group 5:
```julia
"panel_reg/test_panel_reg.jl",
"panel_reg/test_panel_tests.jl",
"panel_reg/test_panel_iv.jl",
"panel_reg/test_panel_nonlinear.jl",
```

Also add `@testset` entries in the serial fallback section.

**Step 4: Commit**

```bash
git commit -m "chore: wire panel_reg module, exports, and test registration (#66)"
```

---

### Task 20: Full Test Suite Run

**Step 1: Run full test suite**

```bash
cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/declarative-kindling-wozniak
MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl
```

**Step 2: Fix any failures**

Address regressions, type mismatches, or edge cases.

**Step 3: Final commit**

```bash
git commit -m "fix: address test suite issues for #66 and #77"
```

---

## Implementation Order Summary

**Parallel Track A (#77):** Tasks 1 → 2 → 3 → 4 → 5 → 6
**Parallel Track B (#66):** Tasks 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19
**Integration:** Task 20 (depends on all above)

Tasks 1-6 and 7-13 can proceed in parallel since they touch different files. Tasks 14-19 depend on 7-13.
