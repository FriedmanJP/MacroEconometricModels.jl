# Automatic ARIMA model identification (TRAMO-style search)

function _x13_auto_model(y::Vector{Float64}, frequency::Int;
                         maxorder::Tuple{Int,Int}=(3, 2),
                         maxdiff::Tuple{Int,Int}=(2, 1))
    n = length(y); X = Matrix{Float64}(undef, n, 0)
    max_pq, max_PQ = maxorder; max_d, max_D = maxdiff
    frequency <= 1 && (max_D = 0; max_PQ = 0)
    best_d = min(1, max_d); best_D = frequency > 1 ? min(1, max_D) : 0
    best_aicc = Inf; best_model = nothing
    for d in 0:max_d, D in 0:max_D
        lost = d + D * frequency
        n - lost < max(10, 2 * frequency) && continue
        q = min(1, max_pq); Q = (frequency > 1 && D >= 0) ? min(1, max_PQ) : 0
        spec = _X13ARIMASpec(0, d, q, 0, D, Q, frequency)
        model = _X13ARIMAModel(spec)
        success = _x13_try_estimate!(model, y, X)
        if success && isfinite(model.aicc) && model.aicc < best_aicc
            best_aicc = model.aicc; best_d = d; best_D = D
            best_model = _x13_copy_model(model)
        end
    end
    d = best_d; D = best_D
    candidates = _x13_build_candidates(d, D, frequency, max_pq, max_PQ)
    for (p, q, P, Q) in candidates
        p + q > max_pq && continue; P + Q > max_PQ && continue
        lost = d + D * frequency; nparams = p + q + P + Q
        n - lost < max(10, nparams + 5) && continue
        spec = _X13ARIMASpec(p, d, q, P, D, Q, frequency)
        model = _X13ARIMAModel(spec)
        success = _x13_try_estimate!(model, y, X)
        if success && isfinite(model.aicc) && model.aicc < best_aicc
            best_aicc = model.aicc; best_model = _x13_copy_model(model)
        end
    end
    best_model === nothing && (best_model = _x13_fallback_model(y, frequency, n, X))
    return (best_model, _X13Outlier[])
end

function _x13_build_candidates(d::Int, D::Int, frequency::Int, max_pq::Int, max_PQ::Int)
    candidates = NTuple{4,Int}[]
    base = if frequency > 1
        [(0,1,0,1),(1,0,0,1),(0,1,1,0),(1,0,1,0),(1,1,0,1),(0,1,1,1),(1,1,1,1),
         (2,0,0,1),(0,2,0,1),(2,1,0,1),(1,2,0,1),(0,1,0,2),(0,1,2,0),(0,0,0,1),
         (1,0,0,0),(0,0,1,0),(0,1,0,0),(1,1,1,0),(2,0,1,0),(0,2,1,0),(2,0,1,1),(0,2,1,1)]
    else
        [(0,1,0,0),(1,0,0,0),(1,1,0,0),(2,0,0,0),(0,2,0,0),(2,1,0,0),(1,2,0,0),
         (3,0,0,0),(0,3,0,0),(2,2,0,0),(3,1,0,0),(1,3,0,0),(0,0,0,0)]
    end
    for (p, q, P, Q) in base
        p + q <= max_pq && P + Q <= max_PQ && push!(candidates, (p, q, P, Q))
    end
    return candidates
end

function _x13_try_estimate!(model::_X13ARIMAModel, y::Vector{Float64}, X::Matrix{Float64})
    try _x13_estimate!(model, y, X); return model.converged
    catch; return false; end
end

function _x13_copy_model(model::_X13ARIMAModel)
    _X13ARIMAModel(model.spec, copy(model.ar), copy(model.ma),
                   copy(model.sar), copy(model.sma),
                   model.sigma2, model.loglik, model.aic, model.aicc,
                   model.converged, model.niter)
end

function _x13_fallback_model(y::Vector{Float64}, frequency::Int, n::Int, X::Matrix{Float64})
    fallbacks = frequency > 1 ?
        [_X13ARIMASpec(0,1,0,0,0,0,frequency), _X13ARIMASpec(0,0,0,0,1,0,frequency), _X13ARIMASpec(0,0,0,0,0,0,frequency)] :
        [_X13ARIMASpec(0,1,0,0,0,0,1), _X13ARIMASpec(1,0,0,0,0,0,1), _X13ARIMASpec(0,0,0,0,0,0,1)]
    for spec in fallbacks
        model = _X13ARIMAModel(spec)
        _x13_try_estimate!(model, y, X) && return model
    end
    spec = _X13ARIMASpec(0, 0, 0, 0, 0, 0, frequency)
    model = _X13ARIMAModel(spec); _x13_try_estimate!(model, y, X); return model
end
