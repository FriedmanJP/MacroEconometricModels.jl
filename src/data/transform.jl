"""
FRED transformation codes (tcodes 1–7) for MacroEconometricModels.jl.

| tcode | Name              | Formula                 |
|-------|-------------------|-------------------------|
| 1     | Level             | x_t                     |
| 2     | First difference  | Δx_t                    |
| 3     | Second difference | Δ²x_t                   |
| 4     | Log               | log(x_t)                |
| 5     | Diff of log       | Δlog(x_t)               |
| 6     | Second diff of log| Δ²log(x_t)              |
| 7     | Delta pct change  | Δ(x_t/x_{t-1} − 1)     |
"""

# =============================================================================
# Univariate apply_tcode
# =============================================================================

"""
    apply_tcode(y::AbstractVector{<:Real}, tcode::Int) -> Vector{Float64}

Apply FRED transformation code to a univariate series.

Codes 4–7 require strictly positive data.
The output vector is shorter than the input for difference-based codes:
- tcode 1: same length
- tcode 2, 4, 5: length T-1
- tcode 3, 6, 7: length T-2

# Examples
```julia
y = [100.0, 102.0, 105.0, 103.0, 108.0]
apply_tcode(y, 5)  # log first differences (approx growth rates)
```
"""
function apply_tcode(y::AbstractVector{<:Real}, tcode::Int)
    1 <= tcode <= 7 || throw(ArgumentError("tcode must be in 1:7, got $tcode"))
    x = Float64.(y)
    T = length(x)

    if tcode >= 4
        any(v -> v <= 0, x) && throw(ArgumentError(
            "tcode $tcode requires strictly positive data"))
    end

    if tcode == 1
        return x
    elseif tcode == 2
        return diff(x)
    elseif tcode == 3
        return diff(diff(x))
    elseif tcode == 4
        return log.(x)
    elseif tcode == 5
        return diff(log.(x))
    elseif tcode == 6
        return diff(diff(log.(x)))
    else  # tcode == 7
        pct = x[2:end] ./ x[1:end-1] .- 1.0
        return diff(pct)
    end
end

# =============================================================================
# Data container apply_tcode
# =============================================================================

"""
    apply_tcode(d::TimeSeriesData, tcodes::Vector{Int}) -> TimeSeriesData

Apply per-variable FRED transformation codes. Rows are trimmed consistently
(to the shortest transformed series).

# Examples
```julia
d = TimeSeriesData(rand(200, 3) .+ 1; varnames=["GDP","CPI","FFR"])
d2 = apply_tcode(d, [5, 5, 1])  # log-diff GDP and CPI, leave FFR in levels
```
"""
function apply_tcode(d::TimeSeriesData{T}, tcodes::Vector{Int}) where {T}
    n = d.n_vars
    length(tcodes) != n && throw(ArgumentError(
        "tcodes length ($(length(tcodes))) must match n_vars ($n)"))

    # Compute the number of rows lost per tcode
    function _rows_lost(tc::Int)
        tc == 1 && return 0
        tc ∈ (2, 4, 5) && return 1
        tc ∈ (3, 6, 7) && return 2
        return 0
    end

    max_lost = maximum(_rows_lost(tc) for tc in tcodes)
    new_T = d.T_obs - max_lost
    new_T < 1 && throw(ArgumentError("Not enough observations after transformation"))

    # Transform each column and align to common length
    new_data = Matrix{Float64}(undef, new_T, n)
    for j in 1:n
        col_transformed = apply_tcode(d.data[:, j], tcodes[j])
        # Take the last new_T elements (align to end)
        offset = length(col_transformed) - new_T
        new_data[:, j] = col_transformed[(offset + 1):end]
    end

    # Trim time_index to match
    new_ti = d.time_index[(max_lost + 1):end]

    TimeSeriesData(new_data;
                   varnames=copy(d.varnames),
                   frequency=d.frequency,
                   tcode=tcodes,
                   time_index=new_ti,
                   desc=desc(d),
                   vardesc=copy(d.vardesc),
                   source_refs=copy(d.source_refs))
end

"""
    apply_tcode(d::TimeSeriesData, tcode::Int) -> TimeSeriesData

Apply the same FRED transformation code to all variables.
"""
function apply_tcode(d::TimeSeriesData, tcode::Int)
    apply_tcode(d, fill(tcode, d.n_vars))
end

# =============================================================================
# Inverse transformation
# =============================================================================

"""
    inverse_tcode(y::AbstractVector{<:Real}, tcode::Int;
                  x_prev::Union{AbstractVector{<:Real},Nothing}=nothing) -> Vector{Float64}

Undo a FRED transformation to recover the original series.

For difference-based codes (2, 3, 5, 6, 7), `x_prev` must supply the initial values
needed for reconstruction:
- tcode 2: `x_prev = [x₀]` (1 value)
- tcode 3: `x_prev = [x₀, x₁]` (2 values, original levels)
- tcode 5: `x_prev = [x₀]` (1 value, original level)
- tcode 6: `x_prev = [x₀, x₁]` (2 values, original levels)
- tcode 7: `x_prev = [x₀, x₁]` (2 values, original levels)

# Examples
```julia
y = [100.0, 102.0, 105.0]
yd = apply_tcode(y, 2)                        # [2.0, 3.0]
inverse_tcode(yd, 2; x_prev=[100.0])          # [102.0, 105.0]
```
"""
function inverse_tcode(y::AbstractVector{<:Real}, tcode::Int;
                       x_prev::Union{AbstractVector{<:Real},Nothing}=nothing)
    1 <= tcode <= 7 || throw(ArgumentError("tcode must be in 1:7, got $tcode"))
    z = Float64.(y)
    T = length(z)

    if tcode == 1
        return z
    elseif tcode == 4
        return exp.(z)
    elseif tcode == 2
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 2"))
        length(x_prev) < 1 && throw(ArgumentError("x_prev must have at least 1 value"))
        out = Vector{Float64}(undef, T)
        prev = Float64(x_prev[end])
        for i in 1:T
            prev = prev + z[i]
            out[i] = prev
        end
        return out
    elseif tcode == 3
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 3"))
        length(x_prev) < 2 && throw(ArgumentError("x_prev must have at least 2 values"))
        x0 = Float64(x_prev[end-1])
        x1 = Float64(x_prev[end])
        d1_prev = x1 - x0
        d1 = Vector{Float64}(undef, T)
        prev_d = d1_prev
        for i in 1:T
            prev_d = prev_d + z[i]
            d1[i] = prev_d
        end
        out = Vector{Float64}(undef, T)
        prev_x = x1
        for i in 1:T
            prev_x = prev_x + d1[i]
            out[i] = prev_x
        end
        return out
    elseif tcode == 5
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 5"))
        length(x_prev) < 1 && throw(ArgumentError("x_prev must have at least 1 value"))
        log_prev = log(Float64(x_prev[end]))
        out = Vector{Float64}(undef, T)
        for i in 1:T
            log_prev = log_prev + z[i]
            out[i] = exp(log_prev)
        end
        return out
    elseif tcode == 6
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 6"))
        length(x_prev) < 2 && throw(ArgumentError("x_prev must have at least 2 values"))
        lx0 = log(Float64(x_prev[end-1]))
        lx1 = log(Float64(x_prev[end]))
        dlog_prev = lx1 - lx0
        out = Vector{Float64}(undef, T)
        log_prev = lx1
        for i in 1:T
            dlog_prev = dlog_prev + z[i]
            log_prev = log_prev + dlog_prev
            out[i] = exp(log_prev)
        end
        return out
    else  # tcode == 7
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 7"))
        length(x_prev) < 2 && throw(ArgumentError("x_prev must have at least 2 values"))
        xm2 = Float64(x_prev[end-1])
        xm1 = Float64(x_prev[end])
        pct_prev = xm1 / xm2 - 1.0
        out = Vector{Float64}(undef, T)
        prev_x = xm1
        for i in 1:T
            pct_prev = pct_prev + z[i]
            prev_x = prev_x * (1.0 + pct_prev)
            out[i] = prev_x
        end
        return out
    end
end
