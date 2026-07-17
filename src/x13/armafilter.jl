# ARMA filtering routines ported from X-13ARIMA-SEATS Fortran sources

function _x13_difference!(y::AbstractVector{Float64}, lag::Int)
    n = length(y)
    newlen = n - lag
    @inbounds for i in 1:newlen
        y[i] = y[i + lag] - y[i]
    end
    return resize!(y, newlen)
end

function _x13_difference!(y::AbstractVector{Float64}, d::Int, D::Int, s::Int)
    for _ in 1:d; _x13_difference!(y, 1); end
    for _ in 1:D; _x13_difference!(y, s); end
    return y
end

function _x13_expand_arma_polynomial(ar::AbstractVector{Float64},
                                     sar::AbstractVector{Float64}, s::Int)
    p = length(ar); P = length(sar)
    p == 0 && P == 0 && return Float64[], Int[]
    maxlag_total = p + P * s
    product = zeros(Float64, maxlag_total + 1)
    product[1] = 1.0
    for k in 1:p; product[k + 1] = -ar[k]; end
    for j in 1:P
        lag = j * s
        for i in (maxlag_total + 1):-1:1
            i - lag >= 1 && (product[i] -= sar[j] * product[i - lag])
        end
    end
    full_coeffs = Float64[]; full_lags = Int[]
    for i in 2:(maxlag_total + 1)
        if product[i] != 0.0
            push!(full_coeffs, -product[i]); push!(full_lags, i - 1)
        end
    end
    return full_coeffs, full_lags
end

function _x13_check_roots(coeffs::AbstractVector{Float64}, kind::Symbol)
    p = length(coeffs)
    p == 0 && return true
    companion = zeros(Float64, p, p)
    for j in 1:p; companion[1, j] = coeffs[j]; end
    for i in 2:p; companion[i, i - 1] = 1.0; end
    eigenvalues = eigvals(companion)
    for ev in eigenvalues; abs(ev) >= 1.0 && return false; end
    return true
end

function _x13_check_roots_expanded(coeffs::AbstractVector{Float64},
                                   lags::AbstractVector{Int})
    isempty(coeffs) && return true
    maxlag = maximum(lags)
    dense = zeros(Float64, maxlag)
    for k in eachindex(coeffs); dense[lags[k]] = coeffs[k]; end
    return _x13_check_roots(dense, :ar)
end

function _x13_armafl!(y::AbstractVector{Float64},
                      model::_X13ARIMAModel,
                      X::AbstractMatrix{Float64})
    spec = model.spec
    # `lndtcv` (log transformation-covariance determinant) is a reserved slot: this routine is a
    # pure CSS filter and leaves it at 0, so the exact-ML determinant term is never accumulated
    # here (#205, false positive).
    n = length(y); ncols = size(X, 2); lndtcv = 0.0
    if ncols > 0 && size(X, 1) != n
        error("X must have same number of rows as y")
    end
    if spec.p == 0 && spec.d == 0 && spec.q == 0 &&
       spec.P == 0 && spec.D == 0 && spec.Q == 0
        return copy(y), copy(X), n, 0.0, 0
    end
    ar_coeffs, ar_lags = _x13_expand_arma_polynomial(model.ar, model.sar, spec.frequency)
    ma_coeffs, ma_lags = _x13_expand_arma_polynomial(model.ma, model.sma, spec.frequency)
    nc = 1 + ncols
    nelta = n * nc
    data = Vector{Float64}(undef, nelta)
    @inbounds for i in 1:n
        data[(i - 1) * nc + 1] = y[i]
        for j in 1:ncols; data[(i - 1) * nc + 1 + j] = X[i, j]; end
    end
    for _ in 1:spec.d
        mxlag_scaled = nc; newlen = nelta - mxlag_scaled
        @inbounds for i in 1:newlen; data[i] = data[i + mxlag_scaled] - data[i]; end
        nelta = newlen
    end
    for _ in 1:spec.D
        mxlag_scaled = spec.frequency * nc; newlen = nelta - mxlag_scaled
        @inbounds for i in 1:newlen; data[i] = data[i + mxlag_scaled] - data[i]; end
        nelta = newlen
    end
    if !isempty(ar_coeffs)
        scaled_ar_lags = ar_lags .* nc
        mxlag_scaled = maximum(scaled_ar_lags); newlen = nelta - mxlag_scaled
        @inbounds for i in 1:newlen
            off = i + mxlag_scaled; tmp = data[off]
            for k in eachindex(ar_coeffs); tmp -= ar_coeffs[k] * data[off - scaled_ar_lags[k]]; end
            data[i] = tmp
        end
        nelta = newlen
    end
    if !isempty(ma_coeffs)
        maxma = maximum(ma_lags); max_scaled = maxma * nc
        work = zeros(Float64, nelta + max_scaled)
        @inbounds for i in 1:nelta; work[max_scaled + i] = data[i]; end
        @inbounds for i in 1:nelta
            idx = max_scaled + i; tmp = work[idx]
            for k in eachindex(ma_coeffs)
                prev_idx = idx - ma_lags[k] * nc
                prev_idx >= 1 && (tmp += ma_coeffs[k] * work[prev_idx])
            end
            work[idx] = tmp
        end
        drop = maxma * nc
        if drop < nelta
            new_nelta = nelta - drop
            @inbounds for i in 1:new_nelta; data[i] = work[max_scaled + drop + i]; end
            nelta = new_nelta
        end
    end
    na = nelta ÷ nc
    filtered_y = Vector{Float64}(undef, na)
    filtered_X = Matrix{Float64}(undef, na, ncols)
    @inbounds for i in 1:na
        filtered_y[i] = data[(i - 1) * nc + 1]
        for j in 1:ncols; filtered_X[i, j] = data[(i - 1) * nc + 1 + j]; end
    end
    return filtered_y, filtered_X, na, lndtcv, 0
end
