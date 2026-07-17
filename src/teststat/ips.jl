# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Im-Pesaran-Shin (2003) `W_tbar` first-generation panel unit root test.

Averages per-unit ADF t-statistics and standardizes them with the finite-sample
moments tabulated in IPS (2003, Table 3). Assumes cross-sectional independence;
route genuine cross-sectional dependence to [`pesaran_cips_test`](@ref).

References:
- Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in
  heterogeneous panels. Journal of Econometrics, 115(1), 53-74.
"""

# =============================================================================
# IPS (2003) Table 3 finite-sample moments E[t_iT] / Var[t_iT]
# Transcribed VERBATIM from the reference implementation `plm::purtest`
# (`adj.ips.wtbar`, components x1..x4, array dim c(10,9,2,2), dimnames
# T=c(10,15,20,25,30,40,50,60,70,100) × lags=0:8 × {mean,var} × {intercept,trend}).
# Column-major fill order (T fastest) verified against the raw plm literals; `NaN`
# marks the (T,lag) combinations plm leaves `NA` (interpolated over below).
# =============================================================================

const _IPS_T_GRID = Float64[10, 15, 20, 25, 30, 40, 50, 60, 70, 100]

# Each entry: 9×10 matrix indexed [lag+1, T], for the given (moment, deterministic).
const _IPS_MOMENTS = let
    nan = NaN
    mean_int = permutedims(reshape(Float64[
        -1.504,-1.514,-1.522,-1.520,-1.526,-1.523,-1.527,-1.519,-1.524,-1.532,
        -1.488,-1.503,-1.516,-1.514,-1.519,-1.520,-1.524,-1.519,-1.522,-1.530,
        -1.319,-1.387,-1.428,-1.443,-1.460,-1.476,-1.493,-1.490,-1.498,-1.514,
        -1.306,-1.366,-1.413,-1.433,-1.453,-1.471,-1.489,-1.486,-1.495,-1.512,
        -1.171,-1.260,-1.329,-1.363,-1.394,-1.428,-1.454,-1.458,-1.470,-1.495,
           nan,   nan,-1.313,-1.351,-1.384,-1.421,-1.451,-1.454,-1.467,-1.494,
           nan,   nan,   nan,-1.289,-1.331,-1.380,-1.418,-1.427,-1.444,-1.476,
           nan,   nan,   nan,-1.273,-1.319,-1.371,-1.411,-1.423,-1.441,-1.474,
           nan,   nan,   nan,-1.212,-1.266,-1.329,-1.377,-1.393,-1.415,-1.456],
        10, 9))
    var_int = permutedims(reshape(Float64[
        1.069,0.923,0.851,0.809,0.789,0.770,0.760,0.749,0.736,0.735,
        1.255,1.011,0.915,0.861,0.831,0.803,0.781,0.770,0.753,0.745,
        1.421,1.078,0.969,0.905,0.865,0.830,0.798,0.789,0.766,0.754,
        1.759,1.181,1.037,0.952,0.907,0.858,0.819,0.802,0.782,0.761,
        2.080,1.279,1.097,1.005,0.946,0.886,0.842,0.819,0.801,0.771,
          nan,  nan,1.171,1.055,0.980,0.912,0.863,0.839,0.814,0.781,
          nan,  nan,  nan,1.114,1.023,0.942,0.886,0.858,0.834,0.795,
          nan,  nan,  nan,1.164,1.062,0.968,0.910,0.875,0.851,0.806,
          nan,  nan,  nan,1.217,1.105,0.996,0.929,0.896,0.871,0.818],
        10, 9))
    mean_trend = permutedims(reshape(Float64[
        -2.166,-2.167,-2.168,-2.167,-2.172,-2.173,-2.176,-2.174,-2.174,-2.177,
        -2.173,-2.169,-2.172,-2.172,-2.173,-2.177,-2.180,-2.178,-2.176,-2.179,
        -1.914,-1.999,-2.047,-2.074,-2.095,-2.120,-2.137,-2.143,-2.146,-2.158,
        -1.922,-1.977,-2.032,-2.065,-2.091,-2.117,-2.137,-2.142,-2.146,-2.158,
        -1.750,-1.823,-1.911,-1.968,-2.009,-2.057,-2.091,-2.103,-2.114,-2.135,
           nan,   nan,-1.888,-1.955,-1.998,-2.051,-2.087,-2.101,-2.111,-2.135,
           nan,   nan,   nan,-1.868,-1.923,-1.995,-2.042,-2.065,-2.081,-2.113,
           nan,   nan,   nan,-1.851,-1.912,-1.986,-2.036,-2.063,-2.079,-2.112,
           nan,   nan,   nan,-1.761,-1.835,-1.925,-1.987,-2.024,-2.046,-2.088],
        10, 9))
    var_trend = permutedims(reshape(Float64[
        1.132,0.869,0.763,0.713,0.690,0.655,0.633,0.621,0.610,0.597,
        1.453,0.975,0.845,0.769,0.734,0.687,0.654,0.641,0.627,0.605,
        1.627,1.036,0.882,0.796,0.756,0.702,0.661,0.653,0.634,0.613,
        2.482,1.214,0.983,0.861,0.808,0.735,0.688,0.674,0.650,0.625,
        3.947,1.332,1.052,0.913,0.845,0.759,0.705,0.685,0.662,0.629,
          nan,  nan,1.165,0.991,0.899,0.792,0.730,0.705,0.673,0.638,
          nan,  nan,  nan,1.055,0.945,0.828,0.753,0.725,0.689,0.650,
          nan,  nan,  nan,1.145,1.009,0.872,0.786,0.747,0.713,0.661,
          nan,  nan,  nan,1.208,1.063,0.902,0.808,0.766,0.728,0.670],
        10, 9))
    Dict(
        (:constant, :mean) => mean_int, (:constant, :var) => var_int,
        (:trend, :mean) => mean_trend, (:trend, :var) => var_trend,
    )
end

"""Interpolate an IPS Table-3 moment (`:mean`/`:var`) in T for given lag & case,
skipping `NaN` grid entries."""
function _ips_moment(moment::Symbol, T_obs::Real, lag::Int, deterministic::Symbol)
    haskey(_IPS_MOMENTS, (deterministic, moment)) || throw(ArgumentError(
        "IPS moments available only for :constant or :trend"))
    row = clamp(lag, 0, 8) + 1
    vals = @view _IPS_MOMENTS[(deterministic, moment)][row, :]
    # Keep only non-NaN (T, value) pairs for this lag row.
    idx = findall(!isnan, vals)
    isempty(idx) && throw(ArgumentError("IPS: no tabulated moment for lag=$lag"))
    _interp_clamped(T_obs, _IPS_T_GRID[idx], vals[idx])
end

# =============================================================================
# IPS test
# =============================================================================

"""
    ips_test(X::AbstractMatrix{T}; deterministic=:constant, lags=:auto,
             max_lags=nothing, criterion=:aic, cs_demean=false) -> IPSResult{T}

Im-Pesaran-Shin (2003) `W_tbar` panel unit root test. `X` is `T×N`; a
`PanelData` method is also provided.

Runs a per-unit ADF regression on each column, averages the t-statistics to
`t̄`, and standardizes with the IPS (2003, Table 3) finite-sample moments
(linearly interpolated in T per unit's `(deterministic, lag)`):
`W_tbar = √N (t̄ − N⁻¹Σ E[t_iT]) / √(N⁻¹Σ Var[t_iT]) ~ N(0,1)`. Very negative
values reject H0 (all panels have a unit root).

# Keyword Arguments
- `deterministic`: `:constant` (default) or `:trend` (IPS tabulates only these)
- `lags`: common integer lag `p`, or `:auto` for per-unit IC selection (capped at 8,
  the last tabulated lag)
- `max_lags`, `criterion`: control `:auto` selection (default `:aic`)
- `cs_demean`: subtract the cross-sectional mean at each `t` (crude CSD mitigation;
  prefer [`pesaran_cips_test`](@ref) for genuine CSD)

# Example
```julia
X = randn(60, 20)                    # stationary panel
result = ips_test(X; deterministic=:constant)
result.pvalue                        # small ⇒ reject the panel unit root
```

# References
- Im, Pesaran & Shin (2003). Journal of Econometrics, 115(1), 53-74.
"""
function ips_test(X::AbstractMatrix{T};
                  deterministic::Symbol=:constant,
                  lags::Union{Int,Symbol}=:auto,
                  max_lags::Union{Int,Nothing}=nothing,
                  criterion::Symbol=:aic,
                  cs_demean::Bool=false) where {T<:AbstractFloat}
    deterministic in (:constant, :trend) || throw(ArgumentError(
        "IPS deterministic must be :constant or :trend, got :$deterministic"))
    Xw = cs_demean ? _cs_demean(X) : X
    T_obs, N = size(Xw)
    T_obs < 20 && throw(ArgumentError(
        "Time dimension T=$T_obs too small; need at least 20 observations"))
    N < 2 && throw(ArgumentError("IPS needs at least N=2 panel units, got N=$N"))
    mpx = isnothing(max_lags) ? nothing : max_lags

    t_i = Vector{T}(undef, N)
    p_i = Vector{Int}(undef, N)
    E_i = Vector{T}(undef, N)
    V_i = Vector{T}(undef, N)

    for i in 1:N
        y = collect(@view Xw[:, i])
        r = if lags === :auto
            adf_test(y; lags=criterion, max_lags=mpx, regression=deterministic)
        else
            adf_test(y; lags=lags::Int, regression=deterministic)
        end
        t_i[i] = r.statistic
        p_i[i] = r.lags
        E_i[i] = T(_ips_moment(:mean, T_obs, r.lags, deterministic))
        V_i[i] = T(_ips_moment(:var, T_obs, r.lags, deterministic))
    end

    tbar = mean(t_i)
    E_mean = mean(E_i)
    V_mean = mean(V_i)
    W = sqrt(T(N)) * (tbar - E_mean) / sqrt(V_mean)
    pval = T(cdf(Normal(), W))                # left-tailed unit-root null

    IPSResult{T}(W, pval, tbar, t_i, E_mean, V_mean, p_i, deterministic, T_obs, N)
end

ips_test(X::AbstractMatrix; kwargs...) = ips_test(Float64.(X); kwargs...)
ips_test(pd::PanelData; kwargs...) = ips_test(_panel_to_matrix(pd); kwargs...)
