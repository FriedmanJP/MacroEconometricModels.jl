# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-06 (#795) / DGP-07 (#796): dynamic-factor and
# mixed-frequency (Mariano–Murasawa) simulators. Factors are VAR(p) — never
# iid — so dynamic-factor / FAVAR / GDFM tests have factor dynamics to recover.


"""
    dgp_dynamic_factors(rng; A, Lambda, r, p, N, T, Sigma_F, idio, idio_ar,
                        signal_share, blocks, burn) -> NamedTuple

Dynamic factor model: `F_t = A_1 F_{t-1} + … + A_p F_{t-p} + η_t`,
`X = F Λ′ + e` with idiosyncratic `e` iid (`idio = :iid`) or AR(1)
(`idio = :ar1` with `idio_ar`). `blocks::Dict{Int,Vector{Int}}` restricts
columns of `Λ` to row blocks (zero elsewhere). Loadings default to a
`signal_share ≈ 0.7` common component. Returns
`(X, F, Lambda, A, Sigma_F, idio_var, eps)` with `eps` the standardized
factor innovations (pre-`Sigma_F`-impact, same convention as `dgp_var`).
"""
function dgp_dynamic_factors(rng::AbstractRNG;
                             A=[0.6 0.15; 0.1 0.5], Lambda=nothing,
                             r::Int=2, p::Int=1, N::Int=40, T::Int=400,
                             Sigma_F=nothing, idio::Symbol=:iid,
                             idio_ar::Float64=0.5, idio_sd::Float64=1.0,
                             signal_share::Float64=0.7,
                             blocks=nothing, burn::Int=200)
    As = A isa AbstractMatrix ? [Matrix{Float64}(A)] : [Matrix{Float64}(a) for a in A]
    rr, pp = size(As[1], 1), length(As)
    SF = Sigma_F === nothing ? Matrix{Float64}(I, rr, rr) : Matrix{Float64}(Sigma_F)
    LF = cholesky(Symmetric(SF)).L
    Lam = if Lambda === nothing
        M = randn(rng, N, rr)
        M ./= sqrt.(sum(M .^ 2, dims=2))  # unit-norm rows
        # Stationary factor covariance via the companion Lyapunov equation,
        # so the average common share is signal_share by construction.
        rp = rr * pp
        Qc = zeros(rp, rp)
        Qc[1:rr, 1:rr] = SF
        Gf = lyapunov_gamma0(_companion(As), Qc)[1:rr, 1:rr]
        cbar = sum(diag(M * Gf * M')) / N
        M .* sqrt((signal_share * idio_sd^2 / (1 - signal_share)) / cbar)
    else
        Matrix{Float64}(Lambda)
    end
    if blocks !== nothing
        Z = zeros(size(Lam))
        for (col, rows) in blocks
            Z[rows, col] = Lam[rows, col]
        end
        Lam = Z
    end
    Nn = T + burn
    F = zeros(Nn, rr)
    Eps = zeros(Nn, rr)
    hist = [zeros(rr) for _ in 1:pp]
    for t in 1:Nn
        e = randn(rng, rr)
        Eps[t, :] .= e
        f = LF * e
        for i in 1:pp
            f += As[i] * hist[i]
        end
        for i in pp:-1:2
            hist[i] .= hist[i - 1]
        end
        hist[1] .= f
        F[t, :] .= f
    end
    NN = size(Lam, 1)
    E = zeros(Nn, NN)
    if idio === :iid
        E = randn(rng, Nn, NN)
    elseif idio === :ar1
        for j in 1:NN, t in 2:Nn
            E[t, j] = idio_ar * E[t - 1, j] + sqrt(1 - idio_ar^2) * randn(rng)
        end
    else
        throw(ArgumentError("unknown idio :$idio (iid|ar1)"))
    end
    X = F * Lam' + idio_sd * E
    keep = (burn + 1):Nn
    return (X=X[keep, :], F=F[keep, :], Lambda=Lam, A=As, Sigma_F=SF,
            idio_var=idio === :iid ? 1.0 : 1.0, eps=Eps[keep, :], r=rr, p=pp)
end

"""
    dgp_mixed_frequency_panel(rng; A, Lambda_M, Lambda_Q, r, nM, nQ, T,
                              idio_ar, blocks, ragged, ragged_target)
        -> NamedTuple

Monthly VAR(1) factors; monthly series `X_M = F Λ_M′ + e`; quarterly series
as the Mariano–Murasawa aggregate
`X_Q[t] = Λ_Q (F_t + 2F_{t-1} + 3F_{t-2} + 2F_{t-3} + F_{t-4}) + e_Q`,
observed at `t ≡ 0 (mod 3)`, `NaN` elsewhere. `ragged = k` blanks the last
`k` monthly releases (`ragged_target = true` also cuts the target quarter).
Returns `(Y, is_quarterly, F, Lambda_M, Lambda_Q, A, withheld)`.
"""
function dgp_mixed_frequency_panel(rng::AbstractRNG;
                                   A=[0.7 0.1; 0.05 0.6], Lambda_M=nothing,
                                   Lambda_Q=nothing, r::Int=2, nM::Int=10,
                                   nQ::Int=2, T::Int=240,
                                   idio_ar::Float64=0.0, blocks=nothing,
                                   ragged::Int=0, ragged_target::Bool=false,
                                   burn::Int=200)
    A1 = Matrix{Float64}(A)
    rr = size(A1, 1)
    LM = Lambda_M === nothing ? randn(rng, nM, rr) : Matrix{Float64}(Lambda_M)
    LQ = Lambda_Q === nothing ? randn(rng, nQ, rr) : Matrix{Float64}(Lambda_Q)
    if blocks !== nothing
        Z = zeros(size(LM))
        for (col, rows) in blocks
            Z[rows, col] = LM[rows, col]
        end
        LM = Z
    end
    N = T + burn
    F = zeros(N, rr)
    f = zeros(rr)
    for t in 1:N
        f = A1 * f + randn(rng, rr)
        F[t, :] .= f
    end
    agg_w = [1.0, 2.0, 3.0, 2.0, 1.0]
    XM = F * LM'
    XQ = zeros(N, nQ)
    for t in 5:N
        Fp = sum(agg_w[l] * F[t - l + 1, :] for l in 1:5)
        XQ[t, :] = LQ * Fp
    end
    EM = zeros(N, nM)
    EQ = zeros(N, nQ)
    if idio_ar == 0.0
        EM = randn(rng, N, nM)
        EQ = randn(rng, N, nQ)
    else
        s = sqrt(1 - idio_ar^2)
        for j in 1:nM, t in 2:N
            EM[t, j] = idio_ar * EM[t - 1, j] + s * randn(rng)
        end
        for j in 1:nQ, t in 2:N
            EQ[t, j] = idio_ar * EQ[t - 1, j] + s * randn(rng)
        end
    end
    XM += 0.2 .* EM
    XQ += 0.2 .* EQ
    Y = hcat(XM, XQ)
    isq = vcat(falses(nM), trues(nQ))
    for t in 1:N, j in (nM + 1):(nM + nQ)
        mod(t, 3) != 0 && (Y[t, j] = NaN)
    end
    withheld = Dict{String,Any}()
    if ragged > 0
        for k in 1:ragged
            t = N - k + 1
            for j in 1:(ragged_target ? nM + nQ : nM)
                isq[j] && mod(t, 3) != 0 && continue
                withheld["$t,$j"] = Y[t, j]
                Y[t, j] = NaN
            end
        end
    end
    keep = (burn + 1):N
    return (Y=Y[keep, :], is_quarterly=isq, F=F[keep, :], Lambda_M=LM,
            Lambda_Q=LQ, A=A1, agg_weights=agg_w, withheld=withheld)
end

"""
    dgp_quantile_factors(rng::AbstractRNG=Random.default_rng();
                         T_obs::Int=200, N::Int=80,
                         location_factors::Int=1, scale_factors::Int=1,
                         asym_factors::Int=0, df_idio::Real=2.0,
                         idio_scale::Real=1.0, loadings_scale::Real=1.0,
                         r::Union{Int,Nothing}=nothing,
                         seed::Union{Integer,Nothing}=nothing) -> NamedTuple

Generate a Chen–Dolado–Gonzalo (2021) quantile factor panel with location,
scale, and optional asymmetry factors:

```math
X_{it} = \\sum_{k=1}^{r_{\\text{loc}}} f_{t,k}^{\\text{loc}} \\lambda_{i,k}^{\\text{loc}}
+ \\sigma_0 \\left(\\sum_{k=1}^{r_{\\text{scale}}} f_{t,k}^{\\text{scale}} \\lambda_{i,k}^{\\text{scale}}\\right) e_{it}
+ \\sigma_0 \\left(\\sum_{m=1}^{r_{\\text{asym}}} f_{t,m}^{\\text{asym}} \\lambda_{i,m}^{\\text{asym}}\\right) \\psi(e_{it})
```

where ``e_{it} \\stackrel{\\text{iid}}{\\sim} t_{\\nu}`` (or ``\\mathcal{N}(0, 1)``
when `df_idio == Inf`), ``\\sigma_0`` is `idio_scale`, and
``\\psi(u) = \\min(u, 0) + 0.4 \\max(u, 0)``. Because the conditional quantile map
is strictly increasing in ``e_{it}`` with ``Q_{0.5}(e) = 0``, only location
factors carry nonzero loadings at the symmetric median (``\\tau = 0.5``), whereas
at tail quantiles (``\\tau \\neq 0.5``) the scale and asymmetry factors become
active.

# Keyword Arguments
- `T_obs::Int = 200` — number of time-series observations
- `N::Int = 80` — cross-sectional panel dimension
- `location_factors::Int = 1` — number of location (mean-shifting) factors
- `scale_factors::Int = 1` — number of positive scale (volatility) factors
- `asym_factors::Int = 0` — number of tail-asymmetry factors
- `df_idio::Real = 2.0` — Student-`t` degrees of freedom for idiosyncratic shocks (`Inf` for Gaussian)
- `idio_scale::Real = 1.0` — baseline idiosyncratic shock multiplier
- `loadings_scale::Real = 1.0` — scaling factor for factor loadings
- `r::Union{Int,Nothing} = nothing` — optional total factor count override across location/scale/asymmetry blocks
- `seed::Union{Integer,Nothing} = nothing` — optional seed overriding `rng` via `Xoshiro(seed)`

# Returns
A `NamedTuple` with fields `(X, F_loc, F_scale, F_asym, Lambda_loc, Lambda_scale,
Lambda_asym, true_factors, true_loadings, true_r, tau_grid, df_idio, idio_scale,
loadings_scale, r, location_factors, scale_factors, asym_factors, seed)`.
"""
function dgp_quantile_factors(rng::AbstractRNG=Random.default_rng();
                              T_obs::Int=200, N::Int=80,
                              location_factors::Int=1,
                              scale_factors::Int=1,
                              asym_factors::Int=0,
                              df_idio::Real=2.0,
                              idio_scale::Real=1.0,
                              loadings_scale::Real=1.0,
                              r::Union{Int,Nothing}=nothing,
                              seed::Union{Integer,Nothing}=nothing)
    T_obs >= 2 || throw(ArgumentError("T_obs must be >= 2, got $T_obs"))
    N >= 2 || throw(ArgumentError("N must be >= 2, got $N"))
    location_factors >= 0 || throw(ArgumentError("location_factors must be >= 0, got $location_factors"))
    scale_factors >= 0 || throw(ArgumentError("scale_factors must be >= 0, got $scale_factors"))
    asym_factors >= 0 || throw(ArgumentError("asym_factors must be >= 0, got $asym_factors"))
    (!(df_idio isa Bool) && !isnan(df_idio) && df_idio > 0) ||
        throw(ArgumentError("df_idio must be > 0, got $df_idio"))
    (!(idio_scale isa Bool) && isfinite(idio_scale) && idio_scale > 0) ||
        throw(ArgumentError("idio_scale must be positive and finite, got $idio_scale"))
    (!(loadings_scale isa Bool) && isfinite(loadings_scale) && loadings_scale > 0) ||
        throw(ArgumentError("loadings_scale must be positive and finite, got $loadings_scale"))

    n_loc, n_scale, n_asym = if r === nothing
        (location_factors, scale_factors, asym_factors)
    else
        r >= 1 || throw(ArgumentError("r must be >= 1, got $r"))
        if r == location_factors + scale_factors + asym_factors
            (location_factors, scale_factors, asym_factors)
        else
            nl = min(location_factors, r)
            na = min(asym_factors, r - nl)
            ns = r - nl - na
            (nl, ns, na)
        end
    end
    (n_loc + n_scale + n_asym) >= 1 ||
        throw(ArgumentError("Total number of factors must be >= 1"))

    rng_use = _resolve_repro_rng(rng, seed)
    df_f = Float64(df_idio)
    sig0 = Float64(idio_scale)
    lscale = Float64(loadings_scale)

    F_loc = n_loc > 0 ? randn(rng_use, T_obs, n_loc) : zeros(T_obs, 0)
    Lambda_loc = n_loc > 0 ? (1.6 * lscale) .* randn(rng_use, N, n_loc) : zeros(N, 0)

    F_scale = n_scale > 0 ? (exp.(0.65 .* randn(rng_use, T_obs, n_scale)) .+ 0.3) : zeros(T_obs, 0)
    Lambda_scale = n_scale > 0 ? (0.45 * lscale) .* (0.7 .+ 0.8 .* rand(rng_use, N, n_scale)) : zeros(N, 0)

    F_asym = n_asym > 0 ? (exp.(0.60 .* randn(rng_use, T_obs, n_asym)) .+ 0.3) : zeros(T_obs, 0)
    Lambda_asym = n_asym > 0 ? (0.45 * lscale) .* (0.7 .+ 0.8 .* rand(rng_use, N, n_asym)) : zeros(N, 0)

    innov = isinf(df_f) ? randn(rng_use, T_obs, N) :
            rand(rng_use, Distributions.TDist(df_f), T_obs, N)

    X = zeros(T_obs, N)
    if n_loc > 0
        X .+= F_loc * Lambda_loc'
    end
    if n_scale > 0
        X .+= sig0 .* (F_scale * Lambda_scale') .* innov
    end
    if n_asym > 0
        psi_innov = @. min(innov, 0.0) + 0.4 * max(innov, 0.0)
        X .+= sig0 .* (F_asym * Lambda_asym') .* psi_innov
    end
    if n_scale == 0 && n_asym == 0
        X .+= sig0 .* innov
    end

    _q_innov(tau::Real) = begin
        tf = Float64(tau)
        (!(tau isa Bool) && isfinite(tf) && 0.0 < tf < 1.0) ||
            throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
        isinf(df_f) ? Float64(Distributions.quantile(Distributions.Normal(), tf)) :
                      Float64(Distributions.quantile(Distributions.TDist(df_f), tf))
    end

    true_r(tau::Real) = begin
        qt = _q_innov(tau)
        sw = sig0 * qt
        aw = sig0 * (min(qt, 0.0) + 0.4 * max(qt, 0.0))
        nr = n_loc
        n_scale > 0 && abs(sw) > 1e-12 && (nr += n_scale)
        n_asym > 0 && abs(aw) > 1e-12 && (nr += n_asym)
        nr
    end

    true_factors(tau::Real) = begin
        qt = _q_innov(tau)
        sw = sig0 * qt
        aw = sig0 * (min(qt, 0.0) + 0.4 * max(qt, 0.0))
        cols = Matrix{Float64}[]
        n_loc > 0 && push!(cols, F_loc)
        (n_scale > 0 && abs(sw) > 1e-12) && push!(cols, F_scale)
        (n_asym > 0 && abs(aw) > 1e-12) && push!(cols, F_asym)
        isempty(cols) ? zeros(T_obs, 0) : hcat(cols...)
    end

    true_loadings(tau::Real) = begin
        qt = _q_innov(tau)
        sw = sig0 * qt
        aw = sig0 * (min(qt, 0.0) + 0.4 * max(qt, 0.0))
        cols = Matrix{Float64}[]
        n_loc > 0 && push!(cols, Lambda_loc)
        (n_scale > 0 && abs(sw) > 1e-12) && push!(cols, sw .* Lambda_scale)
        (n_asym > 0 && abs(aw) > 1e-12) && push!(cols, aw .* Lambda_asym)
        isempty(cols) ? zeros(N, 0) : hcat(cols...)
    end

    return (
        X=X,
        F_loc=F_loc,
        F_scale=F_scale,
        F_asym=F_asym,
        Lambda_loc=Lambda_loc,
        Lambda_scale=Lambda_scale,
        Lambda_asym=Lambda_asym,
        true_factors=true_factors,
        true_loadings=true_loadings,
        true_r=true_r,
        tau_grid=[0.1, 0.25, 0.5, 0.75, 0.9],
        df_idio=df_f,
        idio_scale=sig0,
        loadings_scale=lscale,
        r=n_loc + n_scale + n_asym,
        location_factors=n_loc,
        scale_factors=n_scale,
        asym_factors=n_asym,
        seed=seed === nothing ? nothing : Int(seed),
    )
end

"""
    dgp_qfavar(rng::AbstractRNG=Random.default_rng();
               T_obs::Int=250, N::Int=40, r::Int=2, p::Int=1, n_key::Int=1,
               df_idio::Real=4.0, tail_amplification::Real=1.5, burn::Int=100,
               seed::Union{Integer,Nothing}=nothing) -> NamedTuple

Generate a Quantile Factor-Augmented VAR (Korobilis & Schröder 2025) panel and
key observed series ``(X_{\\text{panel}}, Y_{\\text{key}})`` with tail-amplifying
quantile-dependent autoregressive transmission:

```math
A_0 Z_t = c(\\tau) + \\sum_{\\ell=1}^{p} A_{\\ell}(\\tau) Z_{t-\\ell} + u_t(\\tau),
\\qquad
Z_t = [F_t';\\, Y_{\\text{key}, t}']'
```

where ``A_0`` is unit lower-triangular (slow latent factors precede fast key
variables), lower-tail transmission from ``Y_{\\text{key}, t-1}`` to ``F_t``
is scaled up by `tail_amplification` for ``\\tau \\le 0.10`` relative to the
median and upper tail, and the first `n_key` columns of `X_panel` equal `Y_key`.

# Keyword Arguments
- `T_obs::Int = 250` — post-burn-in sample length
- `N::Int = 40` — total number of panel variables (including `n_key` key variables)
- `r::Int = 2` — number of latent slow-moving factors
- `p::Int = 1` — VAR lag order (`>= 1`)
- `n_key::Int = 1` — number of observed key variables placed in columns `1:n_key` of `X_panel`
- `df_idio::Real = 4.0` — Student-`t` degrees of freedom for factor/panel innovations (`Inf` for Gaussian)
- `tail_amplification::Real = 1.5` — lower-tail multiplier (`>= 1.0`) on lag-1 `Y_key` transmission to factors
- `burn::Int = 100` — burn-in periods discarded before returning `T_obs` observations
- `seed::Union{Integer,Nothing} = nothing` — optional seed overriding `rng` via `Xoshiro(seed)`

# Returns
A `NamedTuple` containing `(X_panel, X, Y_key, Y_key_indices, true_factors, F,
loadings, Lambda, Lambda_y, A0, B_median, B_lower, B_upper, B_by_tau, true_B,
taus, true_qirf_impact, true_qirf, r, p, n_key, df_idio, tail_amplification,
seed)`.
"""
function dgp_qfavar(rng::AbstractRNG=Random.default_rng();
                    T_obs::Int=250, N::Int=40, r::Int=2, p::Int=1, n_key::Int=1,
                    df_idio::Real=4.0, tail_amplification::Real=1.5,
                    burn::Int=100, seed::Union{Integer,Nothing}=nothing)
    T_obs >= p + 5 || throw(ArgumentError("T_obs must be >= p + 5, got $T_obs"))
    r >= 1 || throw(ArgumentError("r must be >= 1, got $r"))
    p >= 1 || throw(ArgumentError("p must be >= 1, got $p"))
    n_key >= 1 || throw(ArgumentError("n_key must be >= 1, got $n_key"))
    N >= r + n_key + 1 || throw(ArgumentError("N must be >= r + n_key + 1 = $(r + n_key + 1), got $N"))
    (!(df_idio isa Bool) && !isnan(df_idio) && df_idio > 0) ||
        throw(ArgumentError("df_idio must be > 0, got $df_idio"))
    (!(tail_amplification isa Bool) && isfinite(tail_amplification) && tail_amplification > 0) ||
        throw(ArgumentError("tail_amplification must be positive and finite, got $tail_amplification"))
    burn >= 0 || throw(ArgumentError("burn must be >= 0, got $burn"))

    rng_use = _resolve_repro_rng(rng, seed)
    df_f = Float64(df_idio)
    amp = Float64(tail_amplification)
    n_aug = r + n_key
    k_lag = 1 + n_aug * p
    sig_f = 0.40
    sig_y = 0.90

    A0 = Matrix{Float64}(I, n_aug, n_aug)
    for k in 1:n_key, j in 1:r
        A0[r + k, j] = -0.25 / r
    end
    A0_inv = inv(A0)

    _w_tail(tau::Float64) = clamp((0.50 - tau) / 0.40, 0.0, 1.0)
    _q_f(tau::Float64) = isinf(df_f) ? Float64(Distributions.quantile(Distributions.Normal(), tau)) :
                                       Float64(Distributions.quantile(Distributions.TDist(df_f), tau))
    _q_y(tau::Float64) = Float64(Distributions.quantile(Distributions.Normal(), tau))

    function _build_lag_mats(tau::Float64)
        As = [zeros(n_aug, n_aug) for _ in 1:p]
        tau_boost = clamp(_w_tail(tau) * (amp - 1.0), -1.0, 1.5)
        f_diag = clamp(0.45 + 0.20 * tau_boost, 0.10, 0.78)
        for j in 1:r
            As[1][j, j] = f_diag
            if r > 1
                for j2 in 1:r
                    j2 != j && (As[1][j, j2] = 0.05 / (r - 1))
                end
            end
        end
        fy_coef = -0.35 * (1.0 + (amp - 1.0) * _w_tail(tau)) / n_key
        for j in 1:r, k in 1:n_key
            As[1][j, r + k] = fy_coef
        end
        y_diag = clamp(0.45 + 0.20 * tau_boost, 0.10, 0.78)
        for k in 1:n_key
            As[1][r + k, r + k] = y_diag
        end
        for lag in 2:p
            scale_l = 0.08 / (lag^2)
            for idx in 1:n_aug
                As[lag][idx, idx] = scale_l
            end
        end
        return As
    end

    true_B(tau::Real) = begin
        tf = Float64(tau)
        (!(tau isa Bool) && isfinite(tf) && 0.0 < tf < 1.0) ||
            throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
        B = zeros(k_lag, n_aug)
        qf = sig_f * _q_f(tf)
        qy = sig_y * _q_y(tf)
        for j in 1:r
            B[1, j] = qf
        end
        for k in 1:n_key
            B[1, r + k] = qy
        end
        As = _build_lag_mats(tf)
        for lag in 1:p
            row_rng = (1 + (lag - 1) * n_aug + 1):(1 + lag * n_aug)
            B[row_rng, :] .= As[lag]'
        end
        B
    end

    N_tot = T_obs + burn + p
    Z = zeros(N_tot, n_aug)
    for t in (p + 1):N_tot
        # Slow factors first
        for j in 1:r
            u = clamp(rand(rng_use), 1e-6, 1.0 - 1e-6)
            As_u = _build_lag_mats(u)
            val = sig_f * _q_f(u)
            for lag in 1:p, col in 1:n_aug
                val += As_u[lag][j, col] * Z[t - lag, col]
            end
            Z[t, j] = val
        end
        # Fast key variables react contemporaneously via A0
        for k in 1:n_key
            eq = r + k
            u = clamp(rand(rng_use), 1e-6, 1.0 - 1e-6)
            As_u = _build_lag_mats(u)
            val = sig_y * _q_y(u)
            for j in 1:r
                val -= A0[eq, j] * Z[t, j]
            end
            for lag in 1:p, col in 1:n_aug
                val += As_u[lag][eq, col] * Z[t - lag, col]
            end
            Z[t, eq] = val
        end
    end

    keep = (N_tot - T_obs + 1):N_tot
    F_true = Z[keep, 1:r]
    Y_key = Z[keep, (r + 1):n_aug]

    # Panel loadings with #528 key-variable identity in columns 1:n_key
    Lambda = zeros(N, r)
    # Non-singular leading r×r block in non-key rows for clean BBE identification
    Lambda[(n_key + 1):(n_key + r), :] .= Matrix{Float64}(I, r, r) .+ 0.2 .* randn(rng_use, r, r)
    if N > n_key + r
        Lambda[(n_key + r + 1):N, :] .= 0.9 .* randn(rng_use, N - n_key - r, r)
    end

    Lambda_y = zeros(N, n_key)
    Lambda_y[1:n_key, 1:n_key] .= Matrix{Float64}(I, n_key, n_key)
    Lambda_y[(n_key + 1):N, :] .= 0.3 .* randn(rng_use, N - n_key, n_key)

    n_nonkey = N - n_key
    idio_nonkey = 0.18 .* (isinf(df_f) ? randn(rng_use, T_obs, n_nonkey) :
                                        rand(rng_use, Distributions.TDist(df_f), T_obs, n_nonkey))
    X_nonkey = F_true * @view(Lambda[(n_key + 1):N, :])' .+
               Y_key * @view(Lambda_y[(n_key + 1):N, :])' .+
               idio_nonkey
    X_panel = hcat(Y_key, X_nonkey)

    taus = [0.10, 0.50, 0.90]
    B_lower = true_B(0.10)
    B_median = true_B(0.50)
    B_upper = true_B(0.90)
    B_by_tau = Dict{Float64,Matrix{Float64}}(
        0.10 => B_lower,
        0.50 => B_median,
        0.90 => B_upper,
    )

    true_qirf(tau::Real, H::Int) = begin
        H >= 1 || throw(ArgumentError("H must be >= 1, got $H"))
        tf = Float64(tau)
        (!(tau isa Bool) && isfinite(tf) && 0.0 < tf < 1.0) ||
            throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
        As = _build_lag_mats(tf)
        vals = zeros(H, n_aug, n_aug)
        vals[1, :, :] .= A0_inv
        for h in 2:H
            acc = zeros(n_aug, n_aug)
            for lag in 1:min(p, h - 1)
                acc .+= (A0_inv * As[lag]) * @view(vals[h - lag, :, :])
            end
            vals[h, :, :] .= acc
        end
        vals
    end

    return (
        X_panel=X_panel,
        X=X_panel,
        Y_key=Y_key,
        Y_key_indices=collect(1:n_key),
        true_factors=F_true,
        F=F_true,
        loadings=Lambda,
        Lambda=Lambda,
        Lambda_y=Lambda_y,
        A0=A0,
        B_median=B_median,
        B_lower=B_lower,
        B_upper=B_upper,
        B_by_tau=B_by_tau,
        true_B=true_B,
        taus=taus,
        true_qirf_impact=A0_inv,
        true_qirf=true_qirf,
        r=r,
        p=p,
        n_key=n_key,
        df_idio=df_f,
        tail_amplification=amp,
        seed=seed === nothing ? nothing : Int(seed),
    )
end
