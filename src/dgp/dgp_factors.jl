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
