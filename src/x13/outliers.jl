# Outlier detection ported from X-13ARIMA-SEATS

function _x13_make_outlier_regressor(type::_X13OutlierType, position::Int, n::Int; tc_rate::Float64=0.7)
    x = zeros(n)
    if type == _X13_AO
        x[position] = 1.0
    elseif type == _X13_LS
        @inbounds for t in position:n; x[t] = 1.0; end
    elseif type == _X13_TC
        @inbounds for t in position:n; x[t] = tc_rate^(t - position); end
    end
    return x
end

function _x13_default_critical_value(n::Int)
    n <= 50 && return 3.0
    n <= 200 && return 3.5
    n <= 400 && return 3.8
    return 4.0
end

function _x13_detect_outliers!(model::_X13ARIMAModel, y::Vector{Float64}, X::Matrix{Float64};
                               types::Vector{_X13OutlierType}=_X13OutlierType[_X13_AO, _X13_LS, _X13_TC],
                               critical_value::Float64=0.0, tc_rate::Float64=0.7)
    n = length(y)
    cv = critical_value > 0.0 ? critical_value : _x13_default_critical_value(n)
    outliers = _X13Outlier[]; Xcurrent = copy(X)
    max_outliers = div(n, 5)
    while length(outliers) < max_outliers
        _x13_estimate!(model, y, Xcurrent)
        residuals_vec = _x13_compute_outlier_residuals(model, y, Xcurrent)
        rbmse = 1.49 * median(abs.(residuals_vec))
        rbmse < 1e-15 && break
        best_t = 0.0; best_type = _X13_AO; best_pos = 0
        for t0 in 1:n, otype in types
            any(o -> o.position == t0 && o.type == otype, outliers) && continue
            x_otl = _x13_make_outlier_regressor(otype, t0, n; tc_rate=tc_rate)
            dot_xx = dot(x_otl, x_otl); dot_xx < 1e-15 && continue
            coef = dot(residuals_vec, x_otl) / dot_xx
            se = rbmse / sqrt(dot_xx); tstat = coef / se
            abs(tstat) > abs(best_t) && (best_t = tstat; best_type = otype; best_pos = t0)
        end
        if abs(best_t) > cv && best_pos > 0
            x_best = _x13_make_outlier_regressor(best_type, best_pos, n; tc_rate=tc_rate)
            coef = dot(residuals_vec, x_best) / dot(x_best, x_best)
            push!(outliers, _X13Outlier(best_type, best_pos, best_t, coef))
            Xcurrent = hcat(Xcurrent, x_best)
        else
            break
        end
    end
    return outliers
end

function _x13_compute_outlier_residuals(model::_X13ARIMAModel, y::Vector{Float64}, X::Matrix{Float64})
    spec = model.spec; n = length(y); ncx = size(X, 2)
    nestpm = spec.p + spec.q + spec.P + spec.Q
    if nestpm == 0
        yd = copy(y)
        (spec.d > 0 || spec.D > 0) && (yd = _x13_difference!(yd, spec.d, spec.D, spec.frequency))
        nobs = length(yd)
        if ncx > 0
            Xd = Matrix{Float64}(undef, nobs, ncx)
            for j in 1:ncx
                col = copy(X[:, j])
                (spec.d > 0 || spec.D > 0) && (col = _x13_difference!(col, spec.d, spec.D, spec.frequency))
                Xd[:, j] = col
            end
            Xy = hcat(Xd, yd); nr, nc = size(Xy)
            b = zeros(ncx); pxpx = nc * (nc + 1) ÷ 2; chlxpx = zeros(pxpx)
            _x13_olsreg!(Xy, nr, nc, nc, b, chlxpx, pxpx)
            rsd = zeros(nr); _x13_compute_residuals!(Xy, nr, nc, nc, 1, ncx, -1.0, b, rsd)
            return vcat(zeros(n - nobs), rsd)
        else
            return vcat(zeros(n - nobs), yd)
        end
    else
        yw = copy(y); Xw = copy(X)
        filtered_y, filtered_X, nout, _, info = _x13_armafl!(yw, model, Xw)
        (info != 0 || nout <= 0) && return copy(y)
        if ncx > 0 && size(filtered_X, 2) > 0
            Xy_f = hcat(filtered_X, filtered_y); nr_f, nc_f = size(Xy_f)
            b_f = zeros(ncx); pxpx_f = nc_f * (nc_f + 1) ÷ 2; chlxpx_f = zeros(pxpx_f)
            ols_info = _x13_olsreg!(Xy_f, nr_f, nc_f, nc_f, b_f, chlxpx_f, pxpx_f)
            if ols_info == 0
                rsd_f = zeros(nr_f)
                _x13_compute_residuals!(Xy_f, nr_f, nc_f, nc_f, 1, ncx, -1.0, b_f, rsd_f)
                return vcat(zeros(n - nr_f), rsd_f)
            end
        end
        return vcat(zeros(n - length(filtered_y)), filtered_y)
    end
end
