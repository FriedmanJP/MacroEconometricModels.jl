# ARIMA estimation driver for X-13

function _x13_pack_params!(estprm::AbstractVector{Float64}, model::_X13ARIMAModel)
    spec = model.spec; idx = 0
    @inbounds for i in 1:spec.p; idx += 1; estprm[idx] = model.ar[i]; end
    @inbounds for i in 1:spec.q; idx += 1; estprm[idx] = model.ma[i]; end
    @inbounds for i in 1:spec.P; idx += 1; estprm[idx] = model.sar[i]; end
    @inbounds for i in 1:spec.Q; idx += 1; estprm[idx] = model.sma[i]; end
    return idx
end

function _x13_unpack_params!(model::_X13ARIMAModel, estprm::AbstractVector{Float64})
    spec = model.spec; idx = 0
    @inbounds for i in 1:spec.p; idx += 1; model.ar[i] = estprm[idx]; end
    @inbounds for i in 1:spec.q; idx += 1; model.ma[i] = estprm[idx]; end
    @inbounds for i in 1:spec.P; idx += 1; model.sar[i] = estprm[idx]; end
    @inbounds for i in 1:spec.Q; idx += 1; model.sma[i] = estprm[idx]; end
    return nothing
end

function _x13_estimate!(model::_X13ARIMAModel, y::Vector{Float64}, X::Matrix{Float64};
                        maxiter::Int=200, tol::Float64=1e-5)
    spec = model.spec; nobs_orig = length(y); ncx = size(X, 2)
    nestpm = spec.p + spec.q + spec.P + spec.Q
    if nestpm == 0
        yd = copy(y)
        (spec.d > 0 || spec.D > 0) && (yd = _x13_difference!(yd, spec.d, spec.D, spec.frequency))
        nobs = length(yd)
        Xd = if ncx > 0
            cols = [let col = copy(X[:, j]); (spec.d > 0 || spec.D > 0) && (col = _x13_difference!(col, spec.d, spec.D, spec.frequency)); col; end for j in 1:ncx]
            hcat(cols...)
        else
            Matrix{Float64}(undef, nobs, 0)
        end
        if ncx > 0
            Xy = hcat(Xd, yd); nr, nc = size(Xy)
            b = zeros(ncx); pxpx = nc * (nc + 1) ÷ 2; chlxpx = zeros(pxpx)
            _x13_olsreg!(Xy, nr, nc, nc, b, chlxpx, pxpx)
            rsd = zeros(nr); _x13_compute_residuals!(Xy, nr, nc, nc, 1, ncx, -1.0, b, rsd)
            apa = _x13_yprmy(rsd)
        else
            apa = _x13_yprmy(yd)
        end
        model.sigma2 = apa / nobs; k = ncx
        model.loglik = -(nobs * (log(2π * model.sigma2) + 1.0)) / 2.0
        model.aic = -2.0 * model.loglik + 2.0 * k
        model.aicc = (nobs - k - 1 > 0) ? model.aic + 2.0 * k * (k + 1.0) / (nobs - k - 1) : model.aic
        model.converged = true; model.niter = 0
        return model
    end
    all_zero = all(v -> v == 0.0, model.ar) && all(v -> v == 0.0, model.ma) &&
               all(v -> v == 0.0, model.sar) && all(v -> v == 0.0, model.sma)
    if all_zero
        fill!(model.ar, 0.1); fill!(model.ma, 0.1)
        fill!(model.sar, 0.1); fill!(model.sma, 0.1)
    end
    estprm = zeros(nestpm); _x13_pack_params!(estprm, model)
    yw_trial = copy(y); Xw_trial = copy(X)
    _, _, na_trial, _, info_trial = _x13_armafl!(yw_trial, model, Xw_trial)
    if info_trial != 0
        fill!(model.ar, 0.05); fill!(model.ma, 0.05)
        fill!(model.sar, 0.05); fill!(model.sma, 0.05)
        _x13_pack_params!(estprm, model)
        yw_trial = copy(y); Xw_trial = copy(X)
        _, _, na_trial, _, _ = _x13_armafl!(yw_trial, model, Xw_trial)
    end
    na = na_trial
    na <= 0 && error("_x13_estimate!: no effective observations after filtering")
    lndtcv_ref = Ref(0.0)
    function lm_callback!(m::Integer, n::Integer, x::AbstractVector{Float64},
                          fvec::AbstractVector{Float64}, iflag::Integer)
        LRGRSD = 1.0e6
        _x13_unpack_params!(model, x)
        yw = copy(y); Xw = copy(X)
        filtered_y, filtered_X, nout, lndtcv, info = _x13_armafl!(yw, model, Xw)
        if info != 0 || nout <= 0
            @inbounds for i in 1:m; fvec[i] = LRGRSD; end
            return iflag
        end
        lndtcv_ref[] = lndtcv
        if ncx > 0
            Xy_f = hcat(filtered_X, filtered_y); nr_f, nc_f = size(Xy_f)
            b_f = zeros(ncx); pxpx_f = nc_f * (nc_f + 1) ÷ 2; chlxpx_f = zeros(pxpx_f)
            ols_info = _x13_olsreg!(Xy_f, nr_f, nc_f, nc_f, b_f, chlxpx_f, pxpx_f)
            if ols_info != 0
                nfill = min(m, nout)
                @inbounds for i in 1:nfill; fvec[i] = filtered_y[i]; end
                @inbounds for i in (nfill+1):m; fvec[i] = 0.0; end
                return iflag
            end
            rsd_f = zeros(nr_f)
            _x13_compute_residuals!(Xy_f, nr_f, nc_f, nc_f, 1, ncx, -1.0, b_f, rsd_f)
            nfill = min(m, nr_f)
            @inbounds for i in 1:nfill; fvec[i] = rsd_f[i]; end
            @inbounds for i in (nfill+1):m; fvec[i] = 0.0; end
        else
            nfill = min(m, nout)
            @inbounds for i in 1:nfill; fvec[i] = filtered_y[i]; end
            @inbounds for i in (nfill+1):m; fvec[i] = 0.0; end
        end
        # NOTE (#205, false positive): `lndtcv ≡ 0` here — `_x13_armafl!` is a pure CSS filter
        # and never sets a transformation-covariance determinant, so this scaling branch is dead
        # code (kept as the hook for a future exact-ML transformation term). No determinant is
        # double-counted.
        if lndtcv != 0.0 && nout > 0
            fac = exp(lndtcv / (2.0 * nout))
            @inbounds for i in 1:m; fvec[i] *= fac; end
        end
        return iflag
    end
    m_lm = max(na, nestpm); n_lm = nestpm
    fvec = zeros(m_lm); fjac = zeros(m_lm, n_lm)
    ipvt = zeros(Int, n_lm); qtf = zeros(n_lm); diag_lm = ones(n_lm)
    wa1 = zeros(n_lm); wa2 = zeros(n_lm); wa3 = zeros(n_lm); wa4 = zeros(m_lm)
    info_lm, nliter, _ = _x13_lmdif!(lm_callback!, m_lm, n_lm, estprm, fvec,
                                      tol, tol, 0.0, maxiter, 0.0, diag_lm, 1, 100.0,
                                      fjac, m_lm, ipvt, qtf, wa1, wa2, wa3, wa4)
    _x13_unpack_params!(model, estprm)
    final_fvec = zeros(m_lm); lm_callback!(m_lm, n_lm, estprm, final_fvec, 1)
    apa = 0.0; @inbounds for i in 1:m_lm; apa += final_fvec[i]^2; end
    lndtcv = lndtcv_ref[]
    model.sigma2 = apa / na
    # `lndtcv ≡ 0` (CSS, not exact-ML): the `+ lndtcv` term adds nothing — it is the reserved
    # slot for a future exact-ML transformation-covariance log-determinant (#205, false positive).
    model.loglik = -(lndtcv + na * (log(2π * model.sigma2) + 1.0)) / 2.0
    k = nestpm + ncx
    model.aic = -2.0 * model.loglik + 2.0 * k
    model.aicc = (na - k - 1 > 0) ? model.aic + 2.0 * k * (k + 1.0) / (na - k - 1) : model.aic
    model.converged = (1 <= info_lm <= 4); model.niter = nliter
    return model
end
