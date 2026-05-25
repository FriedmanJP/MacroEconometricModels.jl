# Linear algebra routines ported from LINPACK/MINPACK Fortran sources

const _X13_MACH_EPS = eps(Float64)
const _X13_DPEQ_DELTA = 3.834e-20

@inline function _x13_dpeq(x::Float64, target::Float64)
    return abs(x - target) < _X13_DPEQ_DELTA
end

function _x13_enorm(n::Integer, x::AbstractVector{Float64})
    RDWARF = 3.834e-20
    RGIANT = 1.304e19
    s1 = 0.0; s2 = 0.0; s3 = 0.0
    x1max = 0.0; x3max = 0.0
    n <= 0 && return 0.0
    floatn = Float64(n)
    agiant = RGIANT / floatn
    @inbounds for i in 1:n
        xabs = abs(x[i])
        if xabs > RDWARF && xabs < agiant
            s2 += xabs^2
        elseif xabs <= RDWARF
            if xabs <= x3max
                !_x13_dpeq(xabs, 0.0) && (s3 += (xabs / x3max)^2)
            else
                s3 = 1.0 + s3 * (x3max / xabs)^2
                x3max = xabs
            end
        else
            if xabs <= x1max
                s1 += (xabs / x1max)^2
            else
                s1 = 1.0 + s1 * (x1max / xabs)^2
                x1max = xabs
            end
        end
    end
    if !_x13_dpeq(s1, 0.0)
        return x1max * sqrt(s1 + (s2 / x1max) / x1max)
    elseif _x13_dpeq(s2, 0.0) && s3 > 0.0
        return x3max * sqrt(s3)
    else
        (_x13_dpeq(x3max, 0.0) && _x13_dpeq(sqrt(s2), 0.0)) && return 0.0
        if s2 >= x3max
            return sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)))
        else
            return sqrt(x3max * ((s2 / x3max) + (x3max * s3)))
        end
    end
end

function _x13_dppfa!(ap::AbstractVector{Float64}, n::Integer)
    mprec = _X13_MACH_EPS
    jj = 0
    for j in 1:n
        info = j
        s = 0.0
        jm1 = j - 1
        kj = jj
        kk = 0
        if jm1 >= 1
            for k in 1:jm1
                kj += 1
                t = ap[kj]
                @inbounds for i in 1:(k-1)
                    t -= ap[kk + i] * ap[jj + i]
                end
                kk += k
                t /= ap[kk]
                ap[kj] = t
                s += t * t
            end
        end
        jj += j
        s = ap[jj] - s
        if s > 0.0
            ap[jj] = sqrt(s)
        else
            s >= -mprec && (ap[jj] = 0.0)
            return info
        end
    end
    return 0
end

function _x13_dppsl!(ap::AbstractVector{Float64}, n::Integer,
                     b::AbstractVector{Float64}; alt::Bool=false)
    kk = 0
    for k in 1:n
        t = 0.0
        @inbounds for i in 1:(k-1)
            t += ap[kk + i] * b[i]
        end
        kk += k
        b[k] = (b[k] - t) / ap[kk]
    end
    alt && return nothing
    for kb in 1:n
        k = n + 1 - kb
        b[k] /= ap[kk]
        kk -= k
        t = -b[k]
        @inbounds for i in 1:(k-1)
            b[i] += t * ap[kk + i]
        end
    end
    return nothing
end

function _x13_qrfac!(m::Integer, n::Integer,
                     a::AbstractMatrix{Float64}, lda::Integer,
                     pivot::Bool,
                     ipvt::AbstractVector{Int}, lipvt::Integer,
                     rdiag::AbstractVector{Float64},
                     acnorm::AbstractVector{Float64},
                     wa::AbstractVector{Float64})
    epsmch = _X13_MACH_EPS
    p05 = 5.0e-2
    for j in 1:n
        acnorm[j] = _x13_enorm(m, @view(a[1:m, j]))
        rdiag[j] = acnorm[j]
        wa[j] = rdiag[j]
        pivot && (ipvt[j] = j)
    end
    minmn = min(m, n)
    for j in 1:minmn
        if pivot
            kmax = j
            for k in j:n
                rdiag[k] > rdiag[kmax] && (kmax = k)
            end
            if kmax != j
                @inbounds for i in 1:m
                    temp = a[i, j]; a[i, j] = a[i, kmax]; a[i, kmax] = temp
                end
                rdiag[kmax] = rdiag[j]; wa[kmax] = wa[j]
                k = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = k
            end
        end
        ajnorm = _x13_enorm(m - j + 1, @view(a[j:m, j]))
        if !_x13_dpeq(ajnorm, 0.0)
            a[j, j] < 0.0 && (ajnorm = -ajnorm)
            @inbounds for i in j:m; a[i, j] /= ajnorm; end
            a[j, j] += 1.0
            jp1 = j + 1
            if n >= jp1
                for k in jp1:n
                    sum_val = 0.0
                    @inbounds for i in j:m; sum_val += a[i, j] * a[i, k]; end
                    temp = sum_val / a[j, j]
                    @inbounds for i in j:m; a[i, k] -= temp * a[i, j]; end
                    if pivot && !_x13_dpeq(rdiag[k], 0.0)
                        temp = a[j, k] / rdiag[k]
                        rdiag[k] *= sqrt(max(0.0, 1.0 - temp^2))
                        if p05 * (rdiag[k] / wa[k])^2 <= epsmch
                            rdiag[k] = _x13_enorm(m - j, @view(a[jp1:m, k]))
                            wa[k] = rdiag[k]
                        end
                    end
                end
            end
        end
        rdiag[j] = -ajnorm
    end
    return nothing
end
