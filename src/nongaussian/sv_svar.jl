# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Identification of SVARs by stochastic volatility (Bertsche–Braun 2022) via
frequentist EM maximum likelihood.

Each heteroskedastic shock follows an independent AR(1) log-volatility
`h_it = μ_i + φ_i (h_{i,t-1} − μ_i) + s_i ω_it` with `e_it ~ N(0, exp(h_it))`.
Time-varying variances identify `B` up to permutation/sign when SV paths are
linearly independent (order-invariant); with `hetero` selecting a strict
subset, the remaining columns are only partially identified (BB §3.5).

E-step (BB EM-2): Monte Carlo integration over KSC-mixture indicators with
Kalman smoothing (FFBS) per draw, reusing `_ksc_draw_indicators`/`_ksc_ffbs`
from `src/sv/`. The repo's Omori et al. (2007) 10-component mixture is used in
place of BB's KSC 7-component mixture (strictly more accurate; documented
here). An EKF/linearized smoother was considered and rejected: linearizing the
log-χ² state space adds approximation bias exactly where the mixture is exact.
M-step: (a) AR(1) SV parameters by OLS-type updates on smoothed states
(ignoring initial conditions, as in BB); (b) VAR slopes by per-shock weighted
least squares; (c) impact-matrix *rotation* by numerical maximization over
Givens angles with scales frozen at the OLS Cholesky factor (`B = L̂Q`).
Deviation from BB (deliberate, pilot evidence): joint `(B-scale, SV-level)`
estimation has a finite-sample ridge (scales wander ~40% while `σ_h²` is
preserved), which cascades into wrong `ε̂` mixtures and bad smoothed vols;
freezing scales at `L̂` (√T-consistent under SV by the WLLN for covariances)
breaks the cascade. Rotation is the identified object BB's theory concerns
(order-invariant columns) and all `compute_Q` needs. Scale is fixed by
`μ_i = −s_i²/(2(1−φ_i²))` (unit unconditional shock variance, BB (2.6)).

References:
- Bertsche, D. & Braun, R. (2022). "Identification of Structural VARs by
  Stochastic Volatility." Journal of Business & Economic Statistics 40(1),
  328–341. WP: Bank of England Staff WP 869 (2020). EM algorithms: §3.2,
  Appendix B.3 (EM-2: MC integration + Kalman smoothing).
"""

using LinearAlgebra, Statistics, Random

# =============================================================================
# Result Type
# =============================================================================

"""
    SVSVARResult{T} <: AbstractNonGaussianSVAR

Result from Bertsche–Braun (2022) SV-SVAR EM estimation.

Fields (homoskedastic shocks, where `hetero` is false, carry `NaN` SV entries):
- `B::Matrix{T}` — structural impact matrix (n × n)
- `A::Vector{Matrix{T}}` — lag coefficient matrices `A[1:p]`
- `c::Vector{T}` — intercepts
- `mus::Vector{T}` — SV levels (from the unit-variance restriction)
- `rhos::Vector{T}` — SV persistence `φ`
- `sigmas::Vector{T}` — SV innovation scales `s`
- `hetero::BitVector` — which shocks were given SV treatment
- `H_smooth::Matrix{T}` — T_eff × n smoothed log-vols (last E-step means)
- `loglik::Vector{T}` — expected complete-data log-lik path (up to constants)
- `converged::Bool`, `iters::Int`, `message::String`
"""
struct SVSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B::Matrix{T}
    A::Vector{Matrix{T}}
    c::Vector{T}
    mus::Vector{T}
    rhos::Vector{T}
    sigmas::Vector{T}
    hetero::BitVector
    H_smooth::Matrix{T}
    loglik::Vector{T}
    converged::Bool
    iters::Int
    message::String
end

function Base.show(io::IO, r::SVSVARResult{T}) where {T}
    n = size(r.B, 1)
    p = length(r.A)
    spec = Any[
        "Variables"      n;
        "Lags"           p;
        "Hetero shocks"  join(findall(r.hetero), ",");
        "Log-lik (last)" _fmt(r.loglik[end]);
        "EM iters"       r.iters;
        "Converged"      string(r.converged)
    ]
    _pretty_table(io, spec;
        title = "BB SV-SVAR Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    svrows = [("shock $i", r.mus[i], r.rhos[i], r.sigmas[i])
              for i in 1:n if r.hetero[i]]
    if !isempty(svrows)
        svmat = Matrix{Any}(undef, length(svrows), 4)
        for (k, row) in enumerate(svrows)
            svmat[k, 1] = row[1]
            svmat[k, 2] = row[2]
            svmat[k, 3] = row[3]
            svmat[k, 4] = row[4]
        end
        _pretty_table(io, svmat;
            title = "SV parameters",
            column_labels = ["", "μ", "ρ", "σ"],
            alignment = [:l, :r, :r, :r],
        )
    end
end

# =============================================================================
# E-step: MCEM smoothing via KSC-mixture Gibbs (BB EM-2)
# =============================================================================

"""One Gibbs run over mixture indicators + FFBS; returns `R` h-draws and the
warm-start h for the next E-step."""
function _svsvar_smooth(ystar::Vector{T}, mu::T, phi::T, sig::T,
        h_warm::Vector{T}, burn::Int, draws::Int,
        rng::AbstractRNG) where {T<:AbstractFloat}
    Te = length(ystar)
    h = copy(h_warm)
    H = Matrix{T}(undef, Te, draws)
    for s in 1:(burn + draws)
        ind = _ksc_draw_indicators(ystar, h, rng)
        h = _ksc_ffbs(ystar, ind, mu, phi, sig, rng)
        s > burn && (H[:, s - burn] = h)
    end
    return H, h
end

"""Monte Carlo sufficient stats from h-draws: mean, variance, lag-1 covariance,
and `E[exp(−h)]` (GLS precision weights)."""
function _svsvar_mcstats(H::Matrix{T}) where {T<:AbstractFloat}
    Te = size(H, 1)
    Eh = vec(mean(H; dims=2))
    Vh = vec(var(H; dims=2, corrected=false))
    Cvh = Vector{T}(undef, Te)
    Cvh[1] = zero(T)
    Einv = vec(mean(exp.(.-H); dims=2))
    for t in 2:Te
        Cvh[t] = mean((H[t, :] .- Eh[t]) .* (H[t - 1, :] .- Eh[t - 1]))
    end
    return Eh, Vh, Cvh, Einv
end

"""Full E-step over all shocks (BB EM-2): smooth each heteroskedastic shock by
KSC-mixture Gibbs + FFBS, filling `Eh/Vh/Cvh/Einv` and updating `H_last`
(warm starts). Homoskedastic shocks keep precision weight 1."""
function _svsvar_estep!(Eh::Matrix{T}, Vh::Matrix{T}, Cvh::Matrix{T},
        Einv::Matrix{T}, H_last::Matrix{T}, Eps::Matrix{T},
        mu::Vector{T}, phi::Vector{T}, sdv::Vector{T}, het::BitVector,
        cc::T, gibbs_burn::Int, gibbs_draws::Int,
        rng::AbstractRNG) where {T<:AbstractFloat}
    n = size(Eps, 2)
    for i in 1:n
        het[i] || continue
        ystar = log.(Eps[:, i] .^ 2 .+ cc)
        Hdraws, hnew = _svsvar_smooth(ystar, mu[i], phi[i], sdv[i],
            Vector{T}(H_last[:, i]), gibbs_burn, gibbs_draws, rng)
        H_last[:, i] = hnew
        e, v, cv, ei = _svsvar_mcstats(Hdraws)
        Eh[:, i] = e
        Vh[:, i] = v
        Cvh[:, i] = cv
        Einv[:, i] = ei
    end
    return nothing
end

# =============================================================================
# M-step (c): rotation update with frozen OLS-Cholesky scales
# =============================================================================

"""One rotation M-step: `min_θ ½Σ_{t,i} w_it ((Zt·Q(θ))_it)²` with whitened
residuals `Zt = L̂⁻¹Û` (Te×n) and precision weights `W`, from warm-start
`theta0`. Scales stay frozen at `L̂`, so no barrier/positivity issues arise."""
function _svsvar_rotation_step(Zt::Matrix{T}, W::Matrix{T},
        theta0::Vector{T}, b_iter::Int) where {T<:AbstractFloat}
    n = size(Zt, 2)
    obj = th -> begin
        Q = _givens_to_orthogonal(th, n)
        E = Zt * Q
        T(0.5) * sum(W .* E .^ 2)
    end
    res = Optim.optimize(obj, Vector{T}(theta0), Optim.LBFGS(),
        Optim.Options(iterations=b_iter))
    return Vector{T}(Optim.minimizer(res))
end

"""Grid-refined rotation start: exhaustive grid over Givens angles (up to a
capped total, deterministic) minimizing the same objective as
`_svsvar_rotation_step`, from fallback `theta0`. Globalizes the rotation
initialization so EM does not start in a bad basin (single Haar starts can
strand SV params at the φ ≈ 1 boundary). `per_angle ≤ 1` disables the grid."""
function _svsvar_grid_start(Zt::Matrix{T}, W::Matrix{T}, theta0::Vector{T},
        per_angle::Int) where {T<:AbstractFloat}
    n = size(Zt, 2)
    na = n * (n - 1) ÷ 2
    per_angle >= 2 || return copy(theta0)
    per = min(per_angle, 12)
    while per > 2 && per^na > 50000
        per -= 1
    end
    obj = th -> begin
        Q = _givens_to_orthogonal(th, n)
        E = Zt * Q
        T(0.5) * sum(W .* E .^ 2)
    end
    grid1d = range(zero(T), stop=2 * T(pi), length=per + 1)[1:per]
    best_th = copy(theta0)
    best_v = obj(best_th)
    for tup in Iterators.product(fill(grid1d, na)...)
        th = Vector{T}(collect(tup))
        v = obj(th)
        v < best_v && ((best_v, best_th) = (v, th))
    end
    return best_th
end

# =============================================================================
# Estimation
# =============================================================================

"""
    identify_sv_svar(Y, p; hetero=trues(n), smoother=:ksc, maxiter=500,
                     tol=1e-3, b_iter=5, gibbs_burn=5, gibbs_draws=100,
                     init=:ols_chol, phi_init=0.9, s_init=0.2,
                     theta_grid=12, c=1e-5,
                     rng=Random.default_rng()) -> SVSVARResult

Estimate the SV-SVAR by EM maximum likelihood (Bertsche–Braun 2022, EM-2).

`Y` is T×n levels; the estimator runs its own OLS/GLS on rows `p+1:T`
(`H_smooth` and the log-lik path use this effective sample; document alignment
when comparing with `model.Y/p` conventions). `hetero[i]` selects shocks with
AR(1) log-vol; the rest are homoskedastic (precision weight 1, `NaN` SV
entries) — partial identification in BB's sense. `init=:ols_chol`
(deterministic Cholesky; default) or `:haar` (BB default: Cholesky times
Haar-random rotation, rng-driven). Either way the first rotation step is
globalized by a deterministic Givens grid (`theta_grid` points per angle, 0 to
disable), since single Haar starts can strand SV params at the φ ≈ 1
boundary. `b_iter` is small by design (default 5): each rotation M-step
warm-starts from the previous `θ`, so a partial LBFGS step (GEM) suffices and
large values chase Monte Carlo weight noise, breaking convergence.
`smoother=:ksc` is the only supported smoother
(KSC-mixture FFBS = BB EM-2; anything else throws). B scales are frozen at the
OLS Cholesky factor throughout (see header note); EM refines the rotation, VAR
slopes, and SV params.

Convergence is declared on parameter stabilization (max change in `φ`, relative
change in `s`, and change in `Q` all below `tol`; default `1e-3`), NOT on the
`loglik` path: the recorded Q mixes E-step distributions and Monte Carlo
noise, so its relative change never reaches BB's deterministic-EM threshold.
`loglik` is kept as a diagnostic path. Returns the full `(A,B,SV)` object; the
`compute_Q` layer exposes only Q/B given `model.Sigma`.

# Examples
```julia
rng = Xoshiro(13)
T = 800
h = zeros(T, 2)
for t in 2:T, i in 1:2
    h[t, i] = 0.95 * h[t - 1, i] + 0.2 * randn(rng)
end
E = randn(rng, T, 2) .* exp.(h ./ 2)
B0 = [1.0 0.3; 0.2 1.0]
Y = Matrix((B0 * E')')
res = identify_sv_svar(Y, 1; maxiter=50, rng=Xoshiro(14))
res.B  # ≈ B0 up to signed permutation; check res.converged
```
"""
function identify_sv_svar(Y::AbstractMatrix, p::Int;
        hetero::AbstractVector{Bool}=trues(size(Y, 2)),
        smoother::Symbol=:ksc,
        maxiter::Int=500,
        tol::Real=1e-3,
        b_iter::Int=5,
        gibbs_burn::Int=5,
        gibbs_draws::Int=100,
        init::Symbol=:ols_chol,
        phi_init::Real=0.9,
        s_init::Real=0.2,
        theta_grid::Int=12,
        c::Real=1e-5,
        rng::AbstractRNG=Random.default_rng())
    smoother === :ksc || throw(ArgumentError(
        "smoother must be :ksc (KSC-mixture FFBS, BB EM-2); EKF linearization " *
        "rejected: it biases the log-χ² state space where the mixture is exact"))
    T = float(promote_type(eltype(Y), Float64))
    n = size(Y, 2)
    T0 = size(Y, 1)
    n >= 2 || throw(ArgumentError("SV-SVAR requires n ≥ 2 variables"))
    p >= 1 || throw(ArgumentError("p must be ≥ 1"))
    Te = T0 - p
    Te >= 100 || throw(ArgumentError("need ≥ 100 usable observations (got T=$T0, p=$p)"))
    length(hetero) == n || throw(ArgumentError("hetero must have length n=$n"))
    any(hetero) || throw(ArgumentError("at least one shock must be heteroskedastic"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol > 0 || throw(ArgumentError("tol must be > 0"))
    init === :haar || init === :ols_chol || throw(ArgumentError(
        "init must be :haar or :ols_chol, got :$init"))
    het = BitVector(collect(hetero))
    Ym = Matrix{T}(Y)

    # Design: rows t = p+1:T0, regressors [1, Y_{t-1}, …, Y_{t-p}]
    Xdes = Matrix{T}(undef, Te, 1 + n * p)
    Yt = Matrix{T}(undef, Te, n)
    for (r, t) in enumerate((p + 1):T0)
        Yt[r, :] = Ym[t, :]
        Xdes[r, 1] = one(T)
        for lag in 1:p, j in 1:n
            Xdes[r, 1 + (lag - 1) * n + j] = Ym[t - lag, j]
        end
    end
    k = 1 + n * p

    # Init: OLS slopes, Cholesky (× Haar) impact matrix, generic SV params
    XtX = Symmetric(Xdes' * Xdes)
    XtY = Xdes' * Yt
    beta = Matrix{T}(robust_inv(XtX) * XtY)'  # n × k
    Uhat = Yt - Xdes * beta'
    Sig = Symmetric(Matrix{T}(Uhat' * Uhat / Te))
    L0 = Matrix{T}(safe_cholesky(Sig))
    # Scales frozen at OLS-Cholesky L0 for the whole run (see header note);
    # only the rotation Q(θ) is EM-refined. theta persists across iterations.
    Lhat = copy(L0)
    B = init === :haar ? Matrix{T}(L0 * Matrix{T}(generate_Q(n; rng=rng))) : copy(L0)
    theta = init === :haar ?
        Vector{T}(_orthogonal_to_givens(Matrix{T}(Lhat \ B), n)) :
        zeros(T, n * (n - 1) ÷ 2)
    phi = fill(T(phi_init), n)
    sdv = fill(T(s_init), n)
    mu = -sdv .^ 2 ./ (2 * (1 .- phi .^ 2))
    h_warm = Matrix{T}(undef, Te, n)
    for i in 1:n
        h_warm[:, i] = mu[i] .+ (sdv[i] / sqrt(max(1 - phi[i]^2, T(1e-8)))) .* randn(rng, T, Te)
    end
    cc = T(c)

    loglik = T[]
    converged = false
    iters = 0
    # E-step buffers (reused)
    Eh = Matrix{T}(undef, Te, n)
    Vh = Matrix{T}(undef, Te, n)
    Cvh = Matrix{T}(undef, Te, n)
    Einv = ones(T, Te, n)
    H_last = copy(h_warm)

    # Grid-refined rotation start: one E-step at B_init, then a deterministic
    # global grid over Givens angles, so EM starts in a good rotation basin.
    Eps0 = Matrix{T}((B \ Uhat')')
    _svsvar_estep!(Eh, Vh, Cvh, Einv, H_last, Eps0, mu, phi, sdv, het, cc,
        gibbs_burn, gibbs_draws, rng)
    Zt0 = Matrix{T}((Lhat \ Uhat')')
    theta = _svsvar_grid_start(Zt0, Einv, theta, theta_grid)
    B = Matrix{T}(Lhat * _givens_to_orthogonal(theta, n))

    Qprev = Matrix{T}(_givens_to_orthogonal(theta, n))
    for iter in 1:maxiter
        oldphi = copy(phi)
        oldsdv = copy(sdv)
        # ---- E-step: smooth each hetero shock (BB EM-2) ----
        Eps = Matrix{T}((B \ Uhat')')
        _svsvar_estep!(Eh, Vh, Cvh, Einv, H_last, Eps, mu, phi, sdv, het, cc,
            gibbs_burn, gibbs_draws, rng)
        # ---- M-step (a): SV params (ignore initial conditions, BB B.3) ----
        for i in 1:n
            het[i] || continue
            mup = mu[i]
            Sxx = zero(T)
            Syy = zero(T)
            Sxy = zero(T)
            for t in 2:Te
                d0 = Eh[t - 1, i] - mup
                d1 = Eh[t, i] - mup
                Sxx += Vh[t - 1, i] + d0^2
                Syy += Vh[t, i] + d1^2
                Sxy += Cvh[t, i] + d1 * d0
            end
            ph = Sxx > 0 ? Sxy / Sxx : phi[i]
            sv = max((Syy - 2 * ph * Sxy + ph^2 * Sxx) / (Te - 1), T(1e-12))
            # Dampen toward previous values: the smoother leaks its assumed
            # persistence into the estimates, which ratchets φ to the boundary
            # undamped (verified in pilots). λ = 1/2 is fixed, not tuned.
            lam = T(0.5)
            ph = lam * phi[i] + (1 - lam) * clamp(ph, T(-0.999), T(0.999))
            sv = lam * sdv[i]^2 + (1 - lam) * sv
            phi[i] = ph
            sdv[i] = sqrt(sv)
            mu[i] = -sv / (2 * (1 - ph^2))
        end
        # ---- M-step (b): VAR slopes by per-shock WLS ----
        Yw = Matrix{T}((B \ Yt')')
        for i in 1:n
            sq = sqrt.(Einv[:, i])
            Xw = Xdes .* sq
            yw = Yw[:, i] .* sq
            beta[i, :] = Vector{T}(robust_inv(Symmetric(Matrix{T}(Xw' * Xw))) * Vector{T}(Xw' * yw))
        end
        Uhat = Yt - Xdes * beta'
        # ---- M-step (c): rotation with frozen scales (GEM) ----
        Zt = Matrix{T}((Lhat \ Uhat')')
        theta = _svsvar_rotation_step(Zt, Einv, theta, b_iter)
        B = Matrix{T}(Lhat * _givens_to_orthogonal(theta, n))
        # ---- Q-function value (up to additive constants) ----
        Eps2 = Matrix{T}((B \ Uhat')')
        qval = -Te * first(logabsdet(B)) - T(0.5) * sum(Einv .* Eps2 .^ 2)
        for i in 1:n
            het[i] || continue
            acc = zero(T)
            for t in 2:Te
                r1 = Eh[t, i] - mu[i] - phi[i] * (Eh[t - 1, i] - mu[i])
                acc += r1^2 + Vh[t, i] + phi[i]^2 * Vh[t - 1, i] -
                    2 * phi[i] * Cvh[t, i]
            end
            qval += -Te * log(sdv[i]) - acc / (2 * sdv[i]^2)
        end
        push!(loglik, qval)
        iters = iter
        Qnow = Matrix{T}(_givens_to_orthogonal(theta, n))
        dphi = maximum(abs.(phi .- oldphi))
        dsdv = maximum(abs.(sdv .- oldsdv) ./ max.(abs.(oldsdv), T(1e-6)))
        dQ = maximum(abs.(Qnow .- Qprev))
        Qprev = Qnow
        if iter >= 2 && max(dphi, dsdv, dQ) < tol
            converged = true
            break
        end
    end

    A = [Matrix{T}(beta[:, (1 + (l - 1) * n + 1):(1 + l * n)]) for l in 1:p]
    cvec = Vector{T}(beta[:, 1])
    Hout = Matrix{T}(undef, Te, n)
    for i in 1:n
        if het[i]
            Hout[:, i] = Eh[:, i]
        else
            Hout[:, i] .= T(NaN)
        end
    end
    mus = [het[i] ? mu[i] : T(NaN) for i in 1:n]
    rhos = [het[i] ? phi[i] : T(NaN) for i in 1:n]
    sigs = [het[i] ? sdv[i] : T(NaN) for i in 1:n]
    message = converged ? "converged in $iters iterations" :
        "reached maxiter=$maxiter (last ΔQ rel. " *
        string(round(abs(loglik[end] - loglik[end - 1]) / max(one(T), abs(loglik[end - 1])); sigdigits=3)) * ")"
    SVSVARResult{T}(B, A, cvec, mus, rhos, sigs, het, Hout, loglik,
        converged, iters, message)
end
