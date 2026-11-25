# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

# GDFM-08 (#900): Infinite-dimensional Generalized Dynamic Factor Model DGP
# (Forni, Hallin, Lippi & Zaffaroni 2015, 2017; Barigozzi, Lippi & Luciani 2021).

"""
    dgp_infinite_gdfm(rng=Random.default_rng(); T_obs=200, N=50, q=2,
                      a_coef=0.5, b_coef=0.3, n_permanent=0, signal_share=0.75,
                      idio_ar=0.0, burn=100, seed=nothing) -> NamedTuple

Simulate a panel from the infinite-dimensional Generalized Dynamic Factor Model
(Forni, Hallin, Lippi & Zaffaroni 2015, 2017) with optional ``I(1)`` cointegrated
common trends (Barigozzi, Lippi & Luciani 2021):

```math
X_{i,t} = \\chi_{i,t} + e_{i,t}, \\qquad
\\chi_{i,t} = \\sum_{k=1}^{n_{\\text{perm}}} \\frac{b_{i,k}}{(1 - L)(1 - \\alpha_{i,k} L)} u_{k,t}
+ \\sum_{k=n_{\\text{perm}}+1}^{q} \\frac{b_{i,k}}{1 - \\alpha_{i,k} L} u_{k,t}
```

where:
- ``X_{i,t}`` is the ``(t, i)`` observed panel entry
- ``\\chi_{i,t}`` is the infinite-dimensional common component (``r = \\infty`` when ``\\alpha_{i,k}`` are heterogeneous)
- ``e_{i,t}`` is the weakly correlated (or AR(1) when `idio_ar != 0`) idiosyncratic component
- ``b_{i,k} \\sim N(0, b_{\\text{coef}}^2)`` are impact filter numerators
- ``\\alpha_{i,k} \\sim \\text{Uniform}(-|a_{\\text{coef}}|, |a_{\\text{coef}}|)`` are heterogeneous rational poles with ``|\\alpha_{i,k}| < 1``
- ``u_{k,t} \\sim \\text{i.i.d. } N(0, 1)`` are orthonormal dynamic common innovations
- ``n_{\\text{perm}}`` (`n_permanent`) is the number of permanent stochastic trends (``0 \\le n_{\\text{perm}} \\le q``)

When ``n_{\\text{permanent}} = 0``, `e_{i,t}` is scaled so that the population average
common variance share equals `signal_share`. When ``n_{\\text{permanent}} > 0``,
the stationary ``I(0)`` idiosyncratic component `e_{i,t}` is scaled so that the
first-difference common variance share ``\\text{Var}(\\Delta \\chi) / \\text{Var}(\\Delta X)``
equals `signal_share`.

# Arguments
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for reproducible draws

# Keyword Arguments
- `T_obs::Int=200`: Sample length after burn-in (``\\ge 2``)
- `N::Int=50`: Number of cross-sectional series (``\\ge 1``)
- `q::Int=2`: Number of dynamic common shocks (``\\ge 1``)
- `a_coef::Real=0.5`: Maximum pole magnitude ``|\\alpha_{i,k}| \\le |a_{\\text{coef}}| < 1``
- `b_coef::Real=0.3`: Standard deviation of rational numerator loadings ``b_{i,k}``
- `n_permanent::Int=0`: Number of ``I(1)`` permanent stochastic trends (``0 \\le n_{\\text{permanent}} \\le q``)
- `signal_share::Real=0.75`: Target common variance share in ``(0, 1]``
- `idio_ar::Real=0.0`: Idiosyncratic AR(1) persistence in ``(-1, 1)``
- `burn::Int=100`: Burn-in observations discarded before the retained sample (``\\ge 0``)
- `seed::Union{Integer,Nothing}=nothing`: Optional integer seed overriding `rng` via `Random.Xoshiro(seed)`

# Returns
A `NamedTuple` with fields:
- `X::Matrix{Float64}`: Observed ``T_{\\text{obs}} \\times N`` panel `common + idiosyncratic`
- `common::Matrix{Float64}`: True common component ``\\chi`` (``T_{\\text{obs}} \\times N``)
- `idiosyncratic::Matrix{Float64}`: Idiosyncratic component ``e`` (``T_{\\text{obs}} \\times N``)
- `innovations::Matrix{Float64}`: Orthonormal dynamic common shocks ``u`` (``T_{\\text{obs}} \\times q``)
- `q::Int`: Number of dynamic common shocks
- `n_permanent::Int`: Number of permanent stochastic trends
- `irf_true::Array{Float64,3}`: Exact population impulse response tensor (``41 \\times N \\times q``, horizons ``h = 0, \\dots, 40``)
- `spectral_density::Array{ComplexF64,3}`: Exact population spectral density of ``X_t`` (or ``\\Delta X_t`` when ``n_{\\text{permanent}} > 0``) on `frequencies` (``N \\times N \\times 33``)
- `spectral_density_common::Array{ComplexF64,3}`: Exact population common spectral density (``N \\times N \\times 33``)
- `frequencies::Vector{Float64}`: Angular frequency grid in ``[0, \\pi]`` (length 33)
- `alpha::Matrix{Float64}`: Drawn pole parameters ``\\alpha_{i,k}`` (``N \\times q``)
- `b::Matrix{Float64}`: Drawn impact coefficients ``b_{i,k}`` (``N \\times q``)

# References
- Barigozzi, M., Lippi, M., & Luciani, M. (2021). Large-Dimensional Dynamic Factor Models: Estimation of Impulse-Response Functions With ``I(1)`` Cointegrated Factors. *Journal of Business & Economic Statistics*, 39(1), 323–333. [DOI](https://doi.org/10.1080/07350015.2019.1660179)
- Forni, M., Hallin, M., Lippi, M., & Zaffaroni, P. (2015). Dynamic factor models with infinite-dimensional factor spaces: One-sided representations. *Journal of Econometrics*, 185(2), 359–371. [DOI](https://doi.org/10.1016/j.jeconom.2014.08.013)
- Forni, M., Hallin, M., Lippi, M., & Zaffaroni, P. (2017). Dynamic factor models with infinite-dimensional factor spaces: Asymptotic analysis. *Journal of Econometrics*, 199(1), 74–92. [DOI](https://doi.org/10.1016/j.jeconom.2017.04.002)
"""
function dgp_infinite_gdfm(rng::AbstractRNG=Random.default_rng();
                           T_obs::Int=200,
                           N::Int=50,
                           q::Int=2,
                           a_coef::Real=0.5,
                           b_coef::Real=0.3,
                           n_permanent::Int=0,
                           signal_share::Real=0.75,
                           idio_ar::Real=0.0,
                           burn::Int=100,
                           seed::Union{Integer,Nothing}=nothing)
    T_obs >= 2 || throw(ArgumentError("T_obs must be >= 2, got $T_obs"))
    N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
    q >= 1 || throw(ArgumentError("q must be >= 1, got $q"))
    0 <= n_permanent <= q || throw(ArgumentError(
        "n_permanent must satisfy 0 <= n_permanent <= q ($q), got $n_permanent"))
    a_scale = abs(Float64(a_coef))
    (isfinite(a_scale) && a_scale < 1.0) || throw(ArgumentError(
        "|a_coef| must be < 1 for rational filter stability, got $a_coef"))
    b_scale = Float64(b_coef)
    isfinite(b_scale) || throw(ArgumentError("b_coef must be finite, got $b_coef"))
    sh = Float64(signal_share)
    (isfinite(sh) && 0.0 < sh <= 1.0) || throw(ArgumentError(
        "signal_share must lie in (0, 1], got $signal_share"))
    rho = Float64(idio_ar)
    (isfinite(rho) && abs(rho) < 1.0) || throw(ArgumentError(
        "|idio_ar| must be < 1 for stationary idiosyncratic noise, got $idio_ar"))
    burn >= 0 || throw(ArgumentError("burn must be >= 0, got $burn"))

    rng_eff = _resolve_repro_rng(rng, seed)

    # Reproducible rational filter parameters: |α_{i,k}| <= |a_coef| < 1
    alphas = a_scale .* (2.0 .* rand(rng_eff, N, q) .- 1.0)
    b_mat = b_scale .* randn(rng_eff, N, q)

    T_tot = T_obs + burn
    U = randn(rng_eff, T_tot, q)

    # Exact recursive rational state z_{i,k,t} = α_{i,k} z_{i,k,t-1} + b_{i,k} u_{k,t}
    common = zeros(Float64, T_obs, N)
    @inbounds for k in 1:q
        is_perm = k <= n_permanent
        for i in 1:N
            a_ik = alphas[i, k]
            b_ik = b_mat[i, k]
            z_state = 0.0
            for t in 1:burn
                z_state = a_ik * z_state + b_ik * U[t, k]
            end
            level_acc = 0.0
            for s in 1:T_obs
                z_state = a_ik * z_state + b_ik * U[burn + s, k]
                if is_perm
                    level_acc += z_state
                    common[s, i] += level_acc
                else
                    common[s, i] += z_state
                end
            end
        end
    end

    # Idiosyncratic unit-variance stationary AR(1) / i.i.d. component
    e_unit = Matrix{Float64}(undef, T_obs, N)
    if rho == 0.0
        for s in 1:burn
            randn(rng_eff, N)
        end
        e_unit .= randn(rng_eff, T_obs, N)
    else
        innov_sd = sqrt(1.0 - rho * rho)
        e_cur = randn(rng_eff, N)
        for _ in 1:burn
            e_cur .= rho .* e_cur .+ innov_sd .* randn(rng_eff, N)
        end
        for s in 1:T_obs
            e_cur .= rho .* e_cur .+ innov_sd .* randn(rng_eff, N)
            e_unit[s, :] .= e_cur
        end
    end

    # Match exact population variance ratio to signal_share
    sigma_e = if sh == 1.0
        0.0
    elseif n_permanent == 0
        v_chi = 0.0
        @inbounds for i in 1:N
            vi = 0.0
            for k in 1:q
                vi += (b_mat[i, k]^2) / (1.0 - alphas[i, k]^2)
            end
            v_chi += vi
        end
        v_chi /= N
        v_chi > 0.0 ? sqrt(v_chi * (1.0 - sh) / sh) : 1.0
    else
        # For I(1) panels, scale I(0) idiosyncratic noise against Δχ variance
        v_dchi = 0.0
        @inbounds for i in 1:N
            vi = 0.0
            for k in 1:q
                if k <= n_permanent
                    vi += (b_mat[i, k]^2) / (1.0 - alphas[i, k]^2)
                else
                    vi += 2.0 * (b_mat[i, k]^2) / (1.0 + alphas[i, k])
                end
            end
            v_dchi += vi
        end
        v_dchi /= N
        v_dchi > 0.0 ? sqrt(v_dchi * (1.0 - sh) / (sh * 2.0 * (1.0 - rho))) : 1.0
    end

    idiosyncratic = sigma_e .* e_unit
    X = common .+ idiosyncratic
    innovations = U[(burn + 1):T_tot, :]

    # Exact population IRF tensor for horizons h = 0..40 (41 × N × q)
    H_irf = 41
    irf_true = zeros(Float64, H_irf, N, q)
    @inbounds for k in 1:q
        is_perm = k <= n_permanent
        for i in 1:N
            a_ik = alphas[i, k]
            b_ik = b_mat[i, k]
            if is_perm
                acc = 0.0
                term = b_ik
                for h in 1:H_irf
                    acc += term
                    irf_true[h, i, k] = acc
                    term *= a_ik
                end
            else
                term = b_ik
                for h in 1:H_irf
                    irf_true[h, i, k] = term
                    term *= a_ik
                end
            end
        end
    end

    # Exact population spectral density on [0, π] (matches general lag-window phase convention)
    n_freq = 33
    freqs = Float64[π * (h - 1) / (n_freq - 1) for h in 1:n_freq]
    S_common = Array{ComplexF64,3}(undef, N, N, n_freq)
    S_total = Array{ComplexF64,3}(undef, N, N, n_freq)
    inv_twoπ = 1.0 / (2.0 * π)
    B_theta = Matrix{ComplexF64}(undef, N, q)
    @inbounds for (h, θ) in enumerate(freqs)
        z = cis(θ)
        diff_fac = 1.0 - z
        for k in 1:q, i in 1:N
            tf = b_mat[i, k] / (1.0 - alphas[i, k] * z)
            if n_permanent > 0 && k > n_permanent
                tf *= diff_fac
            end
            B_theta[i, k] = tf
        end
        Sc = (B_theta * B_theta') .* inv_twoπ
        Sc = (Sc + Sc') .* 0.5
        S_common[:, :, h] = Sc

        se_theta = (sigma_e^2 * (1.0 - rho * rho) * inv_twoπ) / abs2(1.0 - rho * z)
        if n_permanent > 0
            se_theta *= abs2(diff_fac)
        end
        S_total[:, :, h] = Sc + Diagonal(fill(ComplexF64(se_theta), N))
    end

    return (
        X=X,
        common=common,
        idiosyncratic=idiosyncratic,
        innovations=innovations,
        q=q,
        n_permanent=n_permanent,
        irf_true=irf_true,
        spectral_density=S_total,
        spectral_density_common=S_common,
        frequencies=freqs,
        alpha=alphas,
        b=b_mat,
    )
end
