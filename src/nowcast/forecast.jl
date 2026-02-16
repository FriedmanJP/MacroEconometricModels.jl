"""
Nowcast and forecast dispatch for nowcasting models.
"""

# =============================================================================
# nowcast() — Extract current-quarter nowcast
# =============================================================================

"""
    nowcast(model::AbstractNowcastModel; target_var=nothing) -> NowcastResult{T}

Generate nowcast result from an estimated nowcasting model.

Returns the current-quarter nowcast and next-quarter forecast for the
target variable (default: last quarterly variable).

# Arguments
- `model::AbstractNowcastModel` — estimated nowcast model

# Keyword Arguments
- `target_var::Union{Int,Nothing}=nothing` — target variable index (default: last column)

# Returns
`NowcastResult{T}` with nowcast and forecast values.
"""
function nowcast(model::NowcastDFM{T}; target_var::Union{Int,Nothing}=nothing) where {T}
    N = size(model.X_sm, 2)
    T_obs = size(model.X_sm, 1)
    tv = something(target_var, N)

    # Current-quarter nowcast: last observation of smoothed data
    now_val = model.X_sm[T_obs, tv]

    # Next-quarter forecast: use state-space to project forward
    state_dim = size(model.A, 1)
    x_smooth_final = zeros(T, state_dim)

    # Run final smoother to get last state
    y_t = ((model.data .- model.Mx') ./ model.Wx')'
    x_sm, _, _, _ = _kalman_smoother_missing(y_t, model.A, model.C, model.Q, model.R,
                                              model.Z_0, model.V_0)
    x_last = x_sm[:, T_obs]

    # 3-step ahead forecast (one quarter = 3 months)
    x_fc = copy(x_last)
    for _ in 1:3
        x_fc = model.A * x_fc
    end
    fc_val = dot(model.C[tv, :], x_fc) * model.Wx[tv] + model.Mx[tv]

    NowcastResult{T}(model, model.X_sm, tv, now_val, fc_val, :dfm)
end

function nowcast(model::NowcastBVAR{T}; target_var::Union{Int,Nothing}=nothing) where {T}
    N = size(model.X_sm, 2)
    T_obs = size(model.X_sm, 1)
    tv = something(target_var, N)

    now_val = model.X_sm[T_obs, tv]

    # Forecast using VAR equation
    x_lag = ones(T, 1)
    for lag in 1:model.lags
        t_lag = T_obs - lag + 1
        if t_lag >= 1
            x_lag = vcat(x_lag, model.X_sm[t_lag, :])
        else
            x_lag = vcat(x_lag, zeros(T, N))
        end
    end
    y_fc = model.beta' * x_lag
    fc_val = y_fc[tv]

    NowcastResult{T}(model, model.X_sm, tv, now_val, fc_val, :bvar)
end

function nowcast(model::NowcastBridge{T}; target_var::Union{Int,Nothing}=nothing) where {T}
    n_quarters = length(model.Y_nowcast)
    N = size(model.X_sm, 2)
    tv = something(target_var, N)

    now_val = model.Y_nowcast[n_quarters]
    fc_val = isnan(now_val) ? now_val : now_val  # Bridge doesn't do multi-step forecast natively

    NowcastResult{T}(model, model.X_sm, tv, now_val, fc_val, :bridge)
end

# =============================================================================
# forecast() dispatch for nowcast models
# =============================================================================

"""
    forecast(model::NowcastDFM, h; target_var=nothing) -> Vector{T}

Generate h-step ahead monthly forecast from DFM nowcasting model.

# Arguments
- `model::NowcastDFM` — estimated model
- `h::Int` — forecast horizon (months)

# Keyword Arguments
- `target_var::Union{Int,Nothing}` — variable to forecast (default: all)

# Returns
Vector of h forecast values (if target_var specified) or Matrix h × N.
"""
function forecast(model::NowcastDFM{T}, h::Int;
                  target_var::Union{Int,Nothing}=nothing) where {T}
    h >= 1 || throw(ArgumentError("h must be >= 1"))
    N = size(model.data, 2)
    T_obs = size(model.data, 1)
    state_dim = size(model.A, 1)

    # Get last smoothed state
    y_t = ((model.data .- model.Mx') ./ model.Wx')'
    x_sm, _, _, _ = _kalman_smoother_missing(y_t, model.A, model.C, model.Q, model.R,
                                              model.Z_0, model.V_0)
    x_t = x_sm[:, T_obs]

    # Iterate forward
    fc = zeros(T, h, N)
    for step in 1:h
        x_t = model.A * x_t
        fc_std = model.C * x_t
        fc[step, :] = fc_std .* model.Wx .+ model.Mx
    end

    if target_var !== nothing
        return fc[:, target_var]
    end
    return fc
end

function forecast(model::NowcastBVAR{T}, h::Int;
                  target_var::Union{Int,Nothing}=nothing) where {T}
    h >= 1 || throw(ArgumentError("h must be >= 1"))
    N = size(model.data, 2)
    T_obs = size(model.X_sm, 1)

    fc = zeros(T, h, N)
    # Use X_sm for lagged values
    X_ext = vcat(model.X_sm, fc)

    for step in 1:h
        t = T_obs + step
        x_lag = ones(T, 1)
        for lag in 1:model.lags
            t_lag = t - lag
            x_lag = vcat(x_lag, X_ext[t_lag, :])
        end
        X_ext[t, :] = (model.beta' * x_lag)'
        fc[step, :] = X_ext[t, :]
    end

    if target_var !== nothing
        return fc[:, target_var]
    end
    return fc
end
