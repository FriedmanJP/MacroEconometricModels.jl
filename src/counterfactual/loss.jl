# Counterfactual module — loss builders (CF-02, #382)
#
# Losses are H×H weight matrices per target variable, acting on stacked
# H-period paths. All weights are parameters (paper text and replication
# defaults differ — nothing is hardcoded).

"""
    weight_matrix([T=Float64,] H; lambda=1.0, beta=1.0) -> Matrix{T}

The discounted diagonal weight block `λ · Diagonal(β⁰, β¹, …, β^{H−1})`,
materialized as a dense `Matrix{T}`. This is the elementary building block of
every quadratic policy loss (Barnichon–Mesters `W = diag(β ⊗ λ)` per variable).
"""
function weight_matrix(::Type{T}, H::Int; lambda::Real=1.0, beta::Real=1.0) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("weight_matrix: expected H >= 1, got $H"))
    W = zeros(T, H, H)
    for t in 1:H
        W[t, t] = T(lambda) * T(beta)^(t - 1)
    end
    return W
end
weight_matrix(H::Int; kwargs...) = weight_matrix(Float64, H; kwargs...)

"""
    policy_loss(outcomes, H; lambda, beta=1.0, instruments=Symbol[], W_z=nothing,
                name="discounted diagonal")

[`PolicyLoss`](@ref) with one discounted diagonal block per outcome:
`W_x[i] = λᵢ · Diagonal(β⁰, …, β^{H−1})` — the Barnichon–Mesters
`W = diag(β ⊗ λ)` in per-variable-block form. `lambda` must have one weight per
outcome. Optional `W_z` (e.g. from [`smoothing_penalty`](@ref)) attaches
instrument penalties.
"""
function policy_loss(outcomes::AbstractVector{Symbol}, H::Int;
                     lambda::AbstractVector{<:Real},
                     beta::Real=1.0,
                     instruments::AbstractVector{Symbol}=Symbol[],
                     W_z::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                     name::AbstractString="discounted diagonal")
    length(lambda) == length(outcomes) || throw(ArgumentError(
        "policy_loss: lambda: expected $(length(outcomes)) weights (one per outcome), got $(length(lambda))"))
    T = float(promote_type(eltype(lambda), typeof(beta)))
    W_x = [weight_matrix(T, H; lambda=lam, beta=beta) for lam in lambda]
    PolicyLoss(outcomes=outcomes, instruments=instruments, W_x=W_x, W_z=W_z,
               lambda=lambda, beta=beta, name=name)
end

"""
    _ait_averaging_matrix(::Type{T}, H; delta, K) -> Matrix{T}

The `H × H` backward-averaging matrix `Π̄`: row `t` holds normalized exponential
weights `ω_ℓ ∝ e^{−δℓ}` on entries `t−ℓ`, `ℓ = 0…min(K, t−1)`. Rows near the top
truncate the window and renormalize to sum 1.
"""
function _ait_averaging_matrix(::Type{T}, H::Int; delta::Real, K::Int) where {T<:AbstractFloat}
    K >= 0 || throw(ArgumentError("_ait_averaging_matrix: expected K >= 0, got $K"))
    Pi_bar = zeros(T, H, H)
    for t in 1:H
        Lmax = min(K, t - 1)
        w = [exp(-T(delta) * l) for l in 0:Lmax]
        s = sum(w)
        for l in 0:Lmax
            Pi_bar[t, t-l] = w[l+1] / s
        end
    end
    return Pi_bar
end

"""
    ait_loss([T=Float64,] H; beta=1/1.01, lambda_avg=0.6, lambda_t=0.4,
             lambda_y=1.0, delta=0.1, K=19, pi_var=:infl, y_var=:ygap)

Average-inflation-targeting [`PolicyLoss`](@ref) (McKay–Wolf `set_polpref.m`):
with `W = weight_matrix(H; beta)` and the backward-averaging matrix `Π̄`
(exponential weights `ω_ℓ ∝ e^{−δℓ}` over an at-most-`K`-period window,
truncated and renormalized near the top),

`W_π = λ_avg · Π̄'WΠ̄ + λ_t · W`,  `W_y = λ_y · W`.

Defaults are the McKay–Wolf replication values (the paper's Fig.-5 text uses
`λ_π = λ_y = 1`); every weight is overridable. Note `Π̄'WΠ̄` is PSD-singular by
construction — which is why `PolicyLoss` never numerically verifies PSD-ness.
"""
function ait_loss(::Type{T}, H::Int; beta::Real=1/1.01, lambda_avg::Real=0.6,
                  lambda_t::Real=0.4, lambda_y::Real=1.0, delta::Real=0.1,
                  K::Int=19, pi_var::Symbol=:infl, y_var::Symbol=:ygap) where {T<:AbstractFloat}
    pi_var == y_var && throw(ArgumentError("ait_loss: pi_var and y_var must differ, both are :$pi_var"))
    W = weight_matrix(T, H; beta=beta)
    Pi_bar = _ait_averaging_matrix(T, H; delta=delta, K=K)
    W_pi = T(lambda_avg) * (Pi_bar' * W * Pi_bar) + T(lambda_t) * W
    W_y = T(lambda_y) * W
    PolicyLoss(outcomes=[pi_var, y_var], W_x=[W_pi, W_y],
               lambda=T[lambda_avg + lambda_t, lambda_y], beta=beta,
               name="AIT(λ_avg=$(lambda_avg), λ_t=$(lambda_t), λ_y=$(lambda_y), δ=$(delta), K=$(K))")
end
ait_loss(H::Int; kwargs...) = ait_loss(Float64, H; kwargs...)

"""
    smoothing_penalty([T=Float64,] H; lambda=1.0, beta=1.0, z_lag=0.0)
        -> (W_z::Matrix{T}, wedge_term::Vector{T})

Instrument-smoothing penalty `λ Σ_{t=1}^H β^{t−1} (z_t − z_{t−1})²` with the
pre-sample initial condition `z_0 = z_lag`. With `D = I − L` and
`d₀ = (z_lag, 0, …, 0)`:

`W_z = λ·D'WD`,  `wedge_term = λ·D'W·d₀ = λ·(z_lag, 0, …, 0)`,

and the penalty expands as

`z'·W_z·z − 2·wedge_term'·z + λ·z_lag²`.

Sign convention: `wedge_term` is the **positive** linear-term vector that
downstream normal equations (CF-11/CF-13/CF-18) consume directly — it equals
Caravello–McKay–Wolf's `wedge = λ_Δi·(i_hist_last, 0, …)` initial condition.
(The penalty expansion therefore carries the minus sign shown above.)
"""
function smoothing_penalty(::Type{T}, H::Int; lambda::Real=1.0, beta::Real=1.0,
                           z_lag::Real=0.0) where {T<:AbstractFloat}
    D = Matrix{T}(I, H, H) - _lag_shift(T, H)
    W = weight_matrix(T, H; beta=beta)
    W_z = T(lambda) * (D' * W * D)
    wedge_term = T(lambda) * (D' * (W * [t == 1 ? T(z_lag) : zero(T) for t in 1:H]))
    return (W_z=W_z, wedge_term=wedge_term)
end
smoothing_penalty(H::Int; kwargs...) = smoothing_penalty(Float64, H; kwargs...)
