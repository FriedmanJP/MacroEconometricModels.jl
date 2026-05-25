# Levenberg-Marquardt optimizer ported from MINPACK Fortran sources

function _x13_fdjac2!(fcn!, m::Integer, n::Integer,
                      x::AbstractVector{Float64}, fvec::AbstractVector{Float64},
                      fjac::AbstractMatrix{Float64}, ldfjac::Integer,
                      epsfcn::Float64, wa::AbstractVector{Float64})
    epsmch = _X13_MACH_EPS
    eps_fd = sqrt(max(epsfcn, epsmch))
    iflag = 2
    for j in 1:n
        temp = x[j]
        h = eps_fd * abs(temp)
        _x13_dpeq(h, 0.0) && (h = eps_fd)
        x[j] = temp + h
        iflag = fcn!(m, n, x, wa, iflag)
        iflag < 0 && return iflag
        x[j] = temp
        @inbounds for i in 1:m
            fjac[i, j] = (wa[i] - fvec[i]) / h
        end
    end
    return iflag
end

function _x13_qrsolv!(n::Integer, r::AbstractMatrix{Float64}, ldr::Integer,
                      ipvt::AbstractVector{Int}, diag::AbstractVector{Float64},
                      qtb::AbstractVector{Float64}, x::AbstractVector{Float64},
                      sdiag::AbstractVector{Float64}, wa::AbstractVector{Float64})
    p5 = 0.5; p25 = 0.25
    for j in 1:n
        @inbounds for i in j:n; r[i, j] = r[j, i]; end
        x[j] = r[j, j]; wa[j] = qtb[j]
    end
    for j in 1:n
        l = ipvt[j]
        if !_x13_dpeq(diag[l], 0.0)
            for k in j:n; sdiag[k] = 0.0; end
            sdiag[j] = diag[l]
            qtbpj = 0.0
            for k in j:n
                if !_x13_dpeq(sdiag[k], 0.0)
                    if abs(r[k, k]) >= abs(sdiag[k])
                        tangnt = sdiag[k] / r[k, k]
                        cosine = p5 / sqrt(p25 + p25 * tangnt^2)
                        sine = cosine * tangnt
                    else
                        cotan = r[k, k] / sdiag[k]
                        sine = p5 / sqrt(p25 + p25 * cotan^2)
                        cosine = sine * cotan
                    end
                    r[k, k] = cosine * r[k, k] + sine * sdiag[k]
                    temp = cosine * wa[k] + sine * qtbpj
                    qtbpj = -sine * wa[k] + cosine * qtbpj
                    wa[k] = temp
                    kp1 = k + 1
                    if n >= kp1
                        @inbounds for i in kp1:n
                            temp = cosine * r[i, k] + sine * sdiag[i]
                            sdiag[i] = -sine * r[i, k] + cosine * sdiag[i]
                            r[i, k] = temp
                        end
                    end
                end
            end
        end
        sdiag[j] = r[j, j]; r[j, j] = x[j]
    end
    nsing = n
    for j in 1:n
        _x13_dpeq(sdiag[j], 0.0) && nsing == n && (nsing = j - 1)
        nsing < n && (wa[j] = 0.0)
    end
    if nsing >= 1
        for k in 1:nsing
            j = nsing - k + 1
            summ = 0.0
            jp1 = j + 1
            if nsing >= jp1
                @inbounds for i in jp1:nsing; summ += r[i, j] * wa[i]; end
            end
            wa[j] = (wa[j] - summ) / sdiag[j]
        end
    end
    for j in 1:n; l = ipvt[j]; x[l] = wa[j]; end
    return nothing
end

function _x13_lmpar!(n::Integer, r::AbstractMatrix{Float64}, ldr::Integer,
                     ipvt::AbstractVector{Int}, diag::AbstractVector{Float64},
                     qtb::AbstractVector{Float64}, delta::Float64,
                     par::Float64, x::AbstractVector{Float64},
                     sdiag::AbstractVector{Float64},
                     wa1::AbstractVector{Float64}, wa2::AbstractVector{Float64})
    p1 = 0.1; p001 = 0.001; dwarf = 3.834e-20
    nsing = n
    for j in 1:n
        wa1[j] = qtb[j]
        _x13_dpeq(r[j, j], 0.0) && nsing == n && (nsing = j - 1)
        nsing < n && (wa1[j] = 0.0)
    end
    if nsing >= 1
        for k in 1:nsing
            j = nsing - k + 1
            wa1[j] = wa1[j] / r[j, j]
            temp = wa1[j]; jm1 = j - 1
            if jm1 >= 1
                @inbounds for i in 1:jm1; wa1[i] = wa1[i] - r[i, j] * temp; end
            end
        end
    end
    for j in 1:n; l = ipvt[j]; x[l] = wa1[j]; end
    iter = 0
    for j in 1:n; wa2[j] = diag[j] * x[j]; end
    dxnorm = _x13_enorm(n, wa2)
    fp = dxnorm - delta
    if fp <= p1 * delta
        iter == 0 && (par = 0.0)
        return par
    end
    parl = 0.0
    if nsing >= n
        for j in 1:n; l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); end
        for j in 1:n
            sumv = 0.0; jm1 = j - 1
            jm1 >= 1 && (@inbounds for i in 1:jm1; sumv += r[i, j] * wa1[i]; end)
            wa1[j] = (wa1[j] - sumv) / r[j, j]
        end
        temp = _x13_enorm(n, wa1)
        parl = ((fp / delta) / temp) / temp
    end
    for j in 1:n
        sumv = 0.0
        @inbounds for i in 1:j; sumv += r[i, j] * qtb[i]; end
        l = ipvt[j]; wa1[j] = sumv / diag[l]
    end
    gnorm = _x13_enorm(n, wa1)
    paru = gnorm / delta
    _x13_dpeq(paru, 0.0) && (paru = dwarf / min(delta, p1))
    par = max(par, parl); par = min(par, paru)
    _x13_dpeq(par, 0.0) && (par = gnorm / dxnorm)
    while true
        iter += 1
        _x13_dpeq(par, 0.0) && (par = max(dwarf, p001 * paru))
        temp = sqrt(par)
        for j in 1:n; wa1[j] = temp * diag[j]; end
        _x13_qrsolv!(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2)
        for j in 1:n; wa2[j] = diag[j] * x[j]; end
        dxnorm = _x13_enorm(n, wa2); temp = fp; fp = dxnorm - delta
        (abs(fp) <= p1 * delta || (_x13_dpeq(parl, 0.0) && fp <= temp && temp < 0.0) || iter == 10) && break
        for j in 1:n; l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); end
        for j in 1:n
            wa1[j] = wa1[j] / sdiag[j]; temp = wa1[j]; jp1 = j + 1
            n >= jp1 && (@inbounds for i in jp1:n; wa1[i] = wa1[i] - r[i, j] * temp; end)
        end
        temp = _x13_enorm(n, wa1)
        parc = ((fp / delta) / temp) / temp
        fp > 0.0 && (parl = max(parl, par))
        fp < 0.0 && (paru = min(paru, par))
        par = max(parl, par + parc)
    end
    iter == 0 && (par = 0.0)
    return par
end

function _x13_lmdif!(fcn!, m::Integer, n::Integer,
                     x::AbstractVector{Float64}, fvec::AbstractVector{Float64},
                     ftol::Float64, xtol::Float64, gtol::Float64,
                     maxiter::Integer, epsfcn::Float64,
                     diag::AbstractVector{Float64}, mode::Integer,
                     factor::Float64,
                     fjac::AbstractMatrix{Float64}, ldfjac::Integer,
                     ipvt::AbstractVector{Int}, qtf::AbstractVector{Float64},
                     wa1::AbstractVector{Float64}, wa2::AbstractVector{Float64},
                     wa3::AbstractVector{Float64}, wa4::AbstractVector{Float64})
    epsmch = _X13_MACH_EPS
    ONE = 1.0; P1 = 0.1; P5 = 0.5; P25 = 0.25; P75 = 0.75
    P0001 = 0.0001; MONE = -1.0; ZERO = 0.0
    info = 0; iflag = 0; nliter = 0; nfev = 0
    (n <= 0 || m < n || ldfjac < m || ftol < ZERO || xtol < ZERO ||
     gtol < ZERO || maxiter < 0 || factor <= ZERO) && return (info, nliter, nfev)
    maxfev = max(maxiter, 200) * (n + 1)
    if mode == 2
        for j in 1:n; diag[j] <= ZERO && return (info, nliter, nfev); end
    end
    iflag = 1
    iflag = fcn!(m, n, x, fvec, iflag)
    nfev += 1
    iflag < 0 && (info = iflag; return (info, nliter, nfev))
    fnorm = _x13_enorm(m, fvec)
    par = ZERO; delta = ZERO; xnorm = ZERO; ratio = ZERO
    first_iter = true
    while true
        iflag = _x13_fdjac2!(fcn!, m, n, x, fvec, fjac, ldfjac, epsfcn, wa4)
        nfev += n
        iflag < 0 && (info = iflag; return (info, nliter, nfev))
        _x13_qrfac!(m, n, fjac, ldfjac, true, ipvt, n, wa1, wa2, wa3)
        if first_iter
            if mode != 2
                for j in 1:n
                    diag[j] = wa2[j]; _x13_dpeq(wa2[j], ZERO) && (diag[j] = ONE)
                end
            end
            for j in 1:n; wa3[j] = diag[j] * x[j]; end
            xnorm = _x13_enorm(n, wa3)
            delta = factor * xnorm; _x13_dpeq(delta, ZERO) && (delta = factor)
        end
        for i in 1:m; wa4[i] = fvec[i]; end
        for j in 1:n
            if !_x13_dpeq(fjac[j, j], ZERO)
                sumv = ZERO
                @inbounds for i in j:m; sumv += fjac[i, j] * wa4[i]; end
                temp = -sumv / fjac[j, j]
                @inbounds for i in j:m; wa4[i] = wa4[i] + fjac[i, j] * temp; end
            end
            fjac[j, j] = wa1[j]; qtf[j] = wa4[j]
        end
        gnorm = ZERO
        if !_x13_dpeq(fnorm, ZERO)
            for j in 1:n
                l = ipvt[j]
                if !_x13_dpeq(wa2[l], ZERO)
                    sumv = ZERO
                    @inbounds for i in 1:j; sumv += fjac[i, j] * (qtf[i] / fnorm); end
                    gnorm = max(gnorm, abs(sumv / wa2[l]))
                end
            end
        end
        gnorm <= gtol && (info = 4; return (info, nliter, nfev))
        mode != 2 && (for j in 1:n; diag[j] = max(diag[j], wa2[j]); end)
        while true
            par = _x13_lmpar!(n, fjac, ldfjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4)
            for j in 1:n; wa1[j] = -wa1[j]; wa2[j] = x[j] + wa1[j]; wa3[j] = diag[j] * wa1[j]; end
            pnorm = _x13_enorm(n, wa3)
            first_iter && (delta = min(delta, pnorm))
            iflag = 1
            iflag = fcn!(m, n, wa2, wa4, iflag)
            nfev += 1
            iflag < 0 && (info = iflag; return (info, nliter, nfev))
            fnorm1 = _x13_enorm(m, wa4)
            actred = MONE
            P1 * fnorm1 < fnorm && (actred = ONE - (fnorm1 / fnorm)^2)
            for j in 1:n
                wa3[j] = ZERO; l = ipvt[j]; temp = wa1[l]
                @inbounds for i in 1:j; wa3[i] = wa3[i] + fjac[i, j] * temp; end
            end
            temp1 = _x13_enorm(n, wa3) / fnorm
            temp2 = (sqrt(par) * pnorm) / fnorm
            prered = temp1^2 + temp2^2 / P5
            dirder = -(temp1^2 + temp2^2)
            ratio = ZERO
            !_x13_dpeq(prered, ZERO) && (ratio = actred / prered)
            if ratio <= P25
                actred >= ZERO && (temp = P5)
                actred < ZERO && (temp = P5 * dirder / (dirder + P5 * actred))
                (P1 * fnorm1 >= fnorm || temp < P1) && (temp = P1)
                delta = temp * min(delta, pnorm / P1); par = par / temp
            elseif _x13_dpeq(par, ZERO) || ratio >= P75
                delta = pnorm / P5; par = P5 * par
            end
            if ratio >= P0001
                for j in 1:n; x[j] = wa2[j]; wa2[j] = diag[j] * x[j]; end
                for i in 1:m; fvec[i] = wa4[i]; end
                xnorm = _x13_enorm(n, wa2); fnorm = fnorm1; nliter += 1
            end
            abs(actred) <= ftol && prered <= ftol && P5 * ratio <= ONE && (info = 1)
            delta <= xtol * xnorm && (info = 2)
            abs(actred) <= ftol && prered <= ftol && P5 * ratio <= ONE && info == 2 && (info = 3)
            info != 0 && return (info, nliter, nfev)
            maxiter > 0 && nliter >= maxiter && (info = 5)
            nfev >= maxfev && (info = 5)
            abs(actred) <= epsmch && prered <= epsmch && P5 * ratio <= ONE && (info = 6)
            delta <= epsmch * xnorm && (info = 7)
            gnorm <= epsmch && (info = 8)
            info != 0 && return (info, nliter, nfev)
            ratio >= P0001 && break
        end
        first_iter = false
    end
    return (info, nliter, nfev)
end
