# Automatic transformation selection (log vs level)

function _x13_auto_transform(y::Vector{Float64}, frequency::Int;
                              aicc_threshold::Float64=2.0)
    n = length(y)
    any(v -> v <= 0.0, y) && return (_X13_NONE, 1.0)
    spec = _X13ARIMASpec(0, 1, 1, 0, 1, 1, frequency)
    X = Matrix{Float64}(undef, n, 0)
    model_none = _X13ARIMAModel(spec)
    _x13_estimate!(model_none, y, X)
    aicc_none = model_none.aicc
    logy = log.(y)
    model_log = _X13ARIMAModel(spec)
    _x13_estimate!(model_log, logy, X)
    jacobian = 2.0 * sum(logy)
    aicc_log = model_log.aicc + jacobian
    if aicc_log + aicc_threshold < aicc_none
        return (_X13_LOG, 0.0)
    else
        return (_X13_NONE, 1.0)
    end
end
