# Likelihood and regression utilities ported from X-13ARIMA-SEATS Fortran sources

function _x13_yprmy(y::AbstractVector{Float64})
    s = 0.0
    @inbounds for i in eachindex(y); s += y[i]^2; end
    return s
end

function _x13_olsreg!(Xy::AbstractMatrix{Float64}, nrxy::Integer, ncxy::Integer,
                      pcxy::Integer, b::AbstractVector{Float64},
                      chlxpx::AbstractVector{Float64}, pxpx::Integer)
    needed = ncxy * (ncxy + 1) ÷ 2
    needed > pxpx && error("_x13_olsreg!: packed storage too small")
    nb = ncxy - 1
    fill!(chlxpx, 0.0)
    ielt = 0
    @inbounds for i in 1:ncxy
        for j in 1:i
            ielt += 1; s = 0.0
            for k in 1:nrxy; s += Xy[k, i] * Xy[k, j]; end
            chlxpx[ielt] = s
        end
    end
    info = _x13_dppfa!(chlxpx, ncxy)
    if info <= 0 || info == ncxy
        xelt = nb * ncxy ÷ 2
        @inbounds for i in 1:nb; b[i] = chlxpx[xelt + i]; end
        @inbounds for i in nb:-1:1
            b[i] /= chlxpx[xelt]; xelt -= i
            for jj in 1:(i - 1); b[jj] -= b[i] * chlxpx[xelt + jj]; end
        end
        info = 0
    end
    return info
end

function _x13_compute_residuals!(Xy::AbstractMatrix{Float64}, nr::Integer,
                                 nc::Integer, pc::Integer, begcol::Integer,
                                 endcol::Integer, fac::Float64,
                                 b::AbstractVector{Float64},
                                 rsd::AbstractVector{Float64})
    if nc == 0 || endcol + 1 == begcol
        @inbounds for i in 1:nr; rsd[i] = Xy[i, pc]; end
        return nothing
    end
    addsub = sign(fac); addsub == 0.0 && (addsub = 1.0)
    @inbounds for i in 1:nr; rsd[i] = Xy[i, pc]; end
    @inbounds for icol in begcol:endcol
        ab = addsub * b[icol]
        for i in 1:nr; rsd[i] += ab * Xy[i, icol]; end
    end
    return nothing
end
