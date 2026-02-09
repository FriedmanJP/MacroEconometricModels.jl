"""
Data validation and diagnostics for MacroEconometricModels.jl —
diagnose data issues, fix them, and validate compatibility with model types.
"""

# =============================================================================
# DataDiagnostic
# =============================================================================

"""
    DataDiagnostic

Result of `diagnose(d)` — per-variable issue counts and overall cleanliness.

# Fields
- `n_nan::Vector{Int}` — NaN count per variable
- `n_inf::Vector{Int}` — Inf count per variable
- `is_constant::Vector{Bool}` — true if variable has zero variance
- `is_short::Bool` — true if series has fewer than 10 observations
- `varnames::Vector{String}` — variable names
- `is_clean::Bool` — true if no issues detected
"""
struct DataDiagnostic
    n_nan::Vector{Int}
    n_inf::Vector{Int}
    is_constant::Vector{Bool}
    is_short::Bool
    varnames::Vector{String}
    is_clean::Bool
end

function Base.show(io::IO, d::DataDiagnostic)
    if d.is_clean
        println(io, "DataDiagnostic: clean (no issues detected)")
        return
    end
    println(io, "DataDiagnostic: issues detected")
    n = length(d.varnames)
    # Build table data
    rows = Any[]
    for i in 1:n
        has_issue = d.n_nan[i] > 0 || d.n_inf[i] > 0 || d.is_constant[i]
        if has_issue
            push!(rows, Any[d.varnames[i], d.n_nan[i], d.n_inf[i],
                           d.is_constant[i] ? "yes" : "no"])
        end
    end
    if !isempty(rows)
        data = Matrix{Any}(undef, length(rows), 4)
        for (k, r) in enumerate(rows)
            for j in 1:4
                data[k, j] = r[j]
            end
        end
        _pretty_table(io, data;
            column_labels = ["Variable", "NaN", "Inf", "Constant"],
            alignment = [:l, :r, :r, :c])
    end
    if d.is_short
        println(io, "  Warning: series has fewer than 10 observations")
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::DataDiagnostic)
    show(io, d)
end

# =============================================================================
# diagnose
# =============================================================================

"""
    diagnose(d::AbstractMacroData) -> DataDiagnostic

Scan data for NaN, Inf, constant columns, and very short series.

# Examples
```julia
d = TimeSeriesData(randn(100, 3))
diag = diagnose(d)
diag.is_clean  # true if no issues
```
"""
function diagnose(d::AbstractMacroData)
    mat = Matrix(d)
    T_obs, n = size(mat)
    vn = varnames(d)

    n_nan = [count(isnan, @view(mat[:, j])) for j in 1:n]
    n_inf = [count(isinf, @view(mat[:, j])) for j in 1:n]

    is_const = Vector{Bool}(undef, n)
    for j in 1:n
        col = @view(mat[:, j])
        finite_vals = filter(x -> isfinite(x), col)
        if length(finite_vals) <= 1
            is_const[j] = true
        else
            is_const[j] = all(x -> x == finite_vals[1], finite_vals)
        end
    end

    short = T_obs < 10

    clean = all(==(0), n_nan) && all(==(0), n_inf) && !any(is_const) && !short

    DataDiagnostic(n_nan, n_inf, is_const, short, copy(vn), clean)
end

# =============================================================================
# fix
# =============================================================================

"""
    fix(d::TimeSeriesData; method=:listwise) -> TimeSeriesData

Fix data issues and return a clean copy.

# Methods
- `:listwise` — drop rows with any NaN or Inf (default)
- `:interpolate` — linear interpolation for interior NaN, forward-fill edges
- `:mean` — replace NaN with column mean of finite values

Inf values are always replaced with NaN first (then handled by the chosen method).
Constant columns are dropped with a warning.

# Examples
```julia
d = TimeSeriesData([1.0 NaN; 2.0 3.0; 3.0 4.0])
d_clean = fix(d; method=:listwise)  # drops row 1
```
"""
function fix(d::TimeSeriesData{T}; method::Symbol=:listwise) where {T}
    method ∈ (:listwise, :interpolate, :mean) ||
        throw(ArgumentError("method must be :listwise, :interpolate, or :mean, got :$method"))

    mat = copy(d.data)
    T_obs, n = size(mat)

    # Replace Inf with NaN
    for j in 1:n, i in 1:T_obs
        if isinf(mat[i, j])
            mat[i, j] = T(NaN)
        end
    end

    if method == :listwise
        good_rows = [!any(isnan, @view(mat[i, :])) for i in 1:T_obs]
        mat = mat[good_rows, :]
        ti = d.time_index[good_rows]
    elseif method == :interpolate
        for j in 1:n
            _interpolate_column!(view(mat, :, j))
        end
        ti = copy(d.time_index)
    elseif method == :mean
        for j in 1:n
            col = @view(mat[:, j])
            finite_vals = filter(isfinite, col)
            if !isempty(finite_vals)
                m = mean(finite_vals)
                for i in 1:T_obs
                    if isnan(col[i])
                        col[i] = T(m)
                    end
                end
            end
        end
        ti = copy(d.time_index)
    end

    # Drop constant columns
    keep_cols = Int[]
    for j in 1:size(mat, 2)
        col = @view(mat[:, j])
        finite_vals = filter(isfinite, col)
        if length(finite_vals) > 1 && !all(x -> x == finite_vals[1], finite_vals)
            push!(keep_cols, j)
        else
            @warn "Dropping constant column '$(d.varnames[j])'"
        end
    end

    if isempty(keep_cols)
        throw(ArgumentError("All columns are constant after fixing — no data remaining"))
    end

    TimeSeriesData(mat[:, keep_cols];
                   varnames=d.varnames[keep_cols],
                   frequency=d.frequency,
                   tcode=d.tcode[keep_cols],
                   time_index=ti)
end

"""Linear interpolation for interior NaN, forward-fill for edges."""
function _interpolate_column!(col::AbstractVector{T}) where {T}
    n = length(col)
    n == 0 && return

    # Find first and last finite values
    first_finite = findfirst(isfinite, col)
    last_finite = findlast(isfinite, col)
    first_finite === nothing && return  # all NaN — nothing to do

    # Forward-fill leading NaN
    for i in 1:(first_finite - 1)
        col[i] = col[first_finite]
    end
    # Backward-fill trailing NaN
    for i in (last_finite + 1):n
        col[i] = col[last_finite]
    end

    # Linear interpolation for interior NaN
    i = first_finite + 1
    while i <= last_finite
        if isnan(col[i])
            # Find next finite value
            j = i + 1
            while j <= last_finite && isnan(col[j])
                j += 1
            end
            # Interpolate between col[i-1] and col[j]
            span = j - (i - 1)
            for k in i:(j - 1)
                frac = T(k - (i - 1)) / T(span)
                col[k] = col[i - 1] + frac * (col[j] - col[i - 1])
            end
            i = j + 1
        else
            i += 1
        end
    end
end

# =============================================================================
# validate_for_model
# =============================================================================

"""
    validate_for_model(d::AbstractMacroData, model_type::Symbol)

Check that data is compatible with the specified model type. Throws `ArgumentError` on mismatch.

# Model types requiring multivariate data (n_vars ≥ 2)
`:var`, `:vecm`, `:bvar`, `:factors`, `:dynamic_factors`, `:gdfm`

# Model types requiring univariate data (n_vars == 1)
`:arima`, `:ar`, `:ma`, `:arma`, `:arch`, `:garch`, `:egarch`, `:gjr_garch`, `:sv`,
`:hp_filter`, `:hamilton_filter`, `:beveridge_nelson`, `:baxter_king`, `:boosted_hp`,
`:adf`, `:kpss`, `:pp`, `:za`, `:ngperron`

# Model types accepting any dimensionality
`:lp`, `:lp_iv`, `:smooth_lp`, `:state_lp`, `:propensity_lp`, `:gmm`

# Examples
```julia
d = TimeSeriesData(randn(100, 3))
validate_for_model(d, :var)    # OK
validate_for_model(d, :arima)  # throws ArgumentError
```
"""
function validate_for_model(d::AbstractMacroData, model_type::Symbol)
    n = nvars(d)

    multivariate = (:var, :vecm, :bvar, :factors, :dynamic_factors, :gdfm)
    univariate = (:arima, :ar, :ma, :arma, :arch, :garch, :egarch, :gjr_garch, :sv,
                  :hp_filter, :hamilton_filter, :beveridge_nelson, :baxter_king, :boosted_hp,
                  :adf, :kpss, :pp, :za, :ngperron)
    flexible = (:lp, :lp_iv, :smooth_lp, :state_lp, :propensity_lp, :gmm)

    if model_type ∈ multivariate
        n < 2 && throw(ArgumentError(
            "Model :$model_type requires multivariate data (n_vars ≥ 2), got n_vars=$n"))
    elseif model_type ∈ univariate
        n != 1 && throw(ArgumentError(
            "Model :$model_type requires univariate data (n_vars == 1), got n_vars=$n"))
    elseif model_type ∉ flexible
        throw(ArgumentError("Unknown model type :$model_type"))
    end

    nothing
end
