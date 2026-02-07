"""
Historical Decomposition for frequentist and Bayesian VAR models.

Decomposes observed variable movements into contributions from individual structural shocks.
Theory: y_t = Σ_{s=0}^{t-1} Θ_s ε_{t-s} + initial_conditions
where Θ_s = Φ_s * P (structural MA coefficients) and P = L * Q (impact matrix).
"""

using LinearAlgebra, Statistics, MCMCChains, PrettyTables

# =============================================================================
# Abstract Type
# =============================================================================

"""Abstract supertype for historical decomposition results."""
abstract type AbstractHistoricalDecomposition <: AbstractAnalysisResult end

# =============================================================================
# Result Types
# =============================================================================

"""
    HistoricalDecomposition{T} <: AbstractHistoricalDecomposition

Frequentist historical decomposition result.

Fields:
- `contributions`: Shock contributions (T_eff × n_vars × n_shocks)
- `initial_conditions`: Initial condition component (T_eff × n_vars)
- `actual`: Actual data values (T_eff × n_vars)
- `shocks`: Structural shocks (T_eff × n_shocks)
- `T_eff`: Effective number of time periods
- `variables`: Variable names
- `shock_names`: Shock names
- `method`: Identification method used
"""
struct HistoricalDecomposition{T<:AbstractFloat} <: AbstractHistoricalDecomposition
    contributions::Array{T,3}      # T_eff × n_vars × n_shocks
    initial_conditions::Matrix{T}  # T_eff × n_vars
    actual::Matrix{T}              # T_eff × n_vars
    shocks::Matrix{T}              # T_eff × n_shocks
    T_eff::Int
    variables::Vector{String}
    shock_names::Vector{String}
    method::Symbol
end

"""
    BayesianHistoricalDecomposition{T} <: AbstractHistoricalDecomposition

Bayesian historical decomposition with posterior quantiles.

Fields:
- `quantiles`: Contribution quantiles (T_eff × n_vars × n_shocks × n_quantiles)
- `mean`: Mean contributions (T_eff × n_vars × n_shocks)
- `initial_quantiles`: Initial condition quantiles (T_eff × n_vars × n_quantiles)
- `initial_mean`: Mean initial conditions (T_eff × n_vars)
- `shocks_mean`: Mean structural shocks (T_eff × n_shocks)
- `actual`: Actual data values (T_eff × n_vars)
- `T_eff`: Effective number of time periods
- `variables`: Variable names
- `shock_names`: Shock names
- `quantile_levels`: Quantile levels (e.g., [0.16, 0.5, 0.84])
- `method`: Identification method used
"""
struct BayesianHistoricalDecomposition{T<:AbstractFloat} <: AbstractHistoricalDecomposition
    quantiles::Array{T,4}           # T_eff × n_vars × n_shocks × n_quantiles
    mean::Array{T,3}                # T_eff × n_vars × n_shocks
    initial_quantiles::Array{T,3}   # T_eff × n_vars × n_quantiles
    initial_mean::Matrix{T}         # T_eff × n_vars
    shocks_mean::Matrix{T}          # T_eff × n_shocks
    actual::Matrix{T}               # T_eff × n_vars
    T_eff::Int
    variables::Vector{String}
    shock_names::Vector{String}
    quantile_levels::Vector{T}
    method::Symbol
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
Compute structural MA coefficients Θ_s = Φ_s * P for s = 0, ..., horizon-1.
Returns Vector{Matrix{T}} of length horizon.

Uses `_compute_ma_coefficients` from identification.jl to avoid code duplication.
"""
function _compute_structural_ma_coefficients(model::VARModel{T}, Q::AbstractMatrix{T},
                                              horizon::Int) where {T<:AbstractFloat}
    L = safe_cholesky(model.Sigma)
    P = L * Q  # Impact matrix

    # Reuse existing function from identification.jl (returns Phi[1:horizon] with 0-indexing convention)
    Phi = _compute_ma_coefficients(model, horizon - 1)

    # Structural MA coefficients: Θ_s = Φ_s * P
    [Phi[s] * P for s in 1:horizon]
end

"""
Compute historical decomposition contributions from structural shocks and MA coefficients.
HD[t, i, j] = Σ_{s=0}^{t-1} Θ_s[i, j] * ε_j(t-s)
"""
function _compute_hd_contributions(shocks::AbstractMatrix{T}, Theta::Vector{Matrix{T}}) where {T<:AbstractFloat}
    T_eff, n_shocks = size(shocks)
    n_vars = size(Theta[1], 1)
    horizon = length(Theta)

    contributions = zeros(T, T_eff, n_vars, n_shocks)

    @inbounds for t in 1:T_eff
        for i in 1:n_vars
            for j in 1:n_shocks
                # Sum over lags s = 0, ..., min(t-1, horizon-1)
                for s in 0:min(t-1, horizon-1)
                    if t - s >= 1
                        contributions[t, i, j] += Theta[s+1][i, j] * shocks[t-s, j]
                    end
                end
            end
        end
    end

    contributions
end

"""
Compute initial conditions as residual: actual - total shock contributions.
"""
function _compute_initial_conditions(actual::Matrix{T},
                                      contributions::Array{T,3}) where {T<:AbstractFloat}
    T_eff, n_vars = size(actual)
    initial = zeros(T, T_eff, n_vars)

    @inbounds for t in 1:T_eff
        for i in 1:n_vars
            total_shock = sum(@view contributions[t, i, :])
            initial[t, i] = actual[t, i] - total_shock
        end
    end

    initial
end

# =============================================================================
# Frequentist Historical Decomposition
# =============================================================================

"""
    historical_decomposition(model::VARModel, horizon; method=:cholesky, ...) -> HistoricalDecomposition

Compute historical decomposition for a VAR model.

Decomposes observed data into contributions from each structural shock plus initial conditions.

# Arguments
- `model::VARModel`: Estimated VAR model
- `horizon::Int`: Maximum horizon for MA coefficient computation (typically T_eff)

# Keyword Arguments
- `method::Symbol=:cholesky`: Identification method
- `check_func=nothing`: Sign restriction check function (for method=:sign or :narrative)
- `narrative_check=nothing`: Narrative restriction check function (for method=:narrative)
- `max_draws::Int=1000`: Maximum draws for sign/narrative identification
- `transition_var=nothing`: Transition variable (for method=:smooth_transition)
- `regime_indicator=nothing`: Regime indicator (for method=:external_volatility)

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

# Returns
`HistoricalDecomposition` containing:
- `contributions`: Shock contributions (T_eff × n_vars × n_shocks)
- `initial_conditions`: Initial condition effects (T_eff × n_vars)
- `actual`: Actual data values
- `shocks`: Structural shocks

# Example
```julia
model = estimate_var(Y, 2)
hd = historical_decomposition(model, size(Y, 1) - 2)
verify_decomposition(hd)  # Check decomposition identity
```
"""
function historical_decomposition(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing,
    max_draws::Int=1000,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
) where {T<:AbstractFloat}

    n = nvars(model)
    T_eff = effective_nobs(model)

    # Ensure horizon doesn't exceed T_eff
    horizon = min(horizon, T_eff)

    # Get identification matrix Q
    Q = compute_Q(model, method, horizon, check_func, narrative_check;
                  max_draws=max_draws, transition_var=transition_var, regime_indicator=regime_indicator)

    # Compute structural shocks: ε_t = Q' L^{-1} u_t
    shocks = compute_structural_shocks(model, Q)

    # Get actual data (effective sample)
    actual = model.Y[(model.p + 1):end, :]

    # Compute structural MA coefficients
    Theta = _compute_structural_ma_coefficients(model, Q, horizon)

    # Compute contributions
    contributions = _compute_hd_contributions(shocks, Theta)

    # Compute initial conditions
    initial_conditions = _compute_initial_conditions(actual, contributions)

    HistoricalDecomposition{T}(
        contributions, initial_conditions, actual, shocks, T_eff,
        default_var_names(n), default_shock_names(n), method
    )
end

# =============================================================================
# Structural LP Historical Decomposition
# =============================================================================

"""
    historical_decomposition(slp::StructuralLP{T}, T_hd::Int) -> HistoricalDecomposition{T}

Compute historical decomposition from structural LP.

Uses LP-estimated IRFs as the structural MA coefficients Θ_h and the structural
shocks from the underlying VAR identification.

# Arguments
- `slp`: Structural LP result
- `T_hd`: Number of time periods for decomposition (≤ T_eff of underlying VAR)

# Returns
`HistoricalDecomposition{T}` with contributions, initial conditions, and actual data.
"""
function historical_decomposition(slp::StructuralLP{T}, T_hd::Int) where {T<:AbstractFloat}
    n = nvars(slp)
    H = size(slp.irf.values, 1)
    T_eff = effective_nobs(slp.var_model)
    T_hd = min(T_hd, T_eff)

    # LP IRFs as structural MA coefficients
    Theta = [Matrix{T}(slp.irf.values[s, :, :]) for s in 1:H]

    # Structural shocks (truncated to T_hd)
    shocks = slp.structural_shocks[1:T_hd, :]

    # Compute contributions
    contributions = _compute_hd_contributions(shocks, Theta)

    # Actual data from VAR's effective sample
    actual = slp.var_model.Y[(slp.var_model.p+1):(slp.var_model.p+T_hd), :]

    # Initial conditions
    initial_conditions = _compute_initial_conditions(actual, contributions)

    HistoricalDecomposition{T}(
        contributions, initial_conditions, actual, shocks, T_hd,
        default_var_names(n), default_shock_names(n), slp.method
    )
end

# =============================================================================
# Bayesian Historical Decomposition
# =============================================================================

"""
    historical_decomposition(chain::Chains, p, n, horizon; data, ...) -> BayesianHistoricalDecomposition

Compute Bayesian historical decomposition from MCMC chain with posterior quantiles.

# Arguments
- `chain::Chains`: MCMC chain from `estimate_bvar`
- `p::Int`: Number of lags
- `n::Int`: Number of variables
- `horizon::Int`: Maximum horizon for MA coefficients

# Keyword Arguments
- `data::AbstractMatrix`: Original data matrix (required)
- `method::Symbol=:cholesky`: Identification method
- `quantiles::Vector{<:Real}=[0.16, 0.5, 0.84]`: Posterior quantile levels
- `check_func=nothing`: Sign restriction check function
- `narrative_check=nothing`: Narrative restriction check function
- `transition_var=nothing`: Transition variable (for method=:smooth_transition)
- `regime_indicator=nothing`: Regime indicator (for method=:external_volatility)

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

# Returns
`BayesianHistoricalDecomposition` with posterior quantiles and means.

# Example
```julia
chain = estimate_bvar(Y, 2; n_samples=500)
hd = historical_decomposition(chain, 2, 3, 198; data=Y)
```
"""
function historical_decomposition(chain::Chains, p::Int, n::Int, horizon::Int;
    data::AbstractMatrix, method::Symbol=:cholesky,
    quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
    check_func=nothing, narrative_check=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
)
    _validate_narrative_data(method, data)

    samples = size(chain, 1)
    ET = eltype(data)
    T_eff = size(data, 1) - p
    horizon = min(horizon, T_eff)
    actual = ET.(data[(p + 1):end, :])

    # Storage for all posterior draws
    all_contributions = zeros(ET, samples, T_eff, n, n)
    all_initial = zeros(ET, samples, T_eff, n)
    all_shocks = zeros(ET, samples, T_eff, n)

    b_vecs, sigmas = extract_chain_parameters(chain)

    for s in 1:samples
        m = parameters_to_model(b_vecs[s, :], sigmas[s, :], p, n, data)
        Q = compute_Q(m, method, horizon, check_func, narrative_check;
                      max_draws=100, transition_var=transition_var, regime_indicator=regime_indicator)

        shocks = compute_structural_shocks(m, Q)
        Theta = _compute_structural_ma_coefficients(m, Q, horizon)
        contributions = _compute_hd_contributions(shocks, Theta)
        initial_cond = _compute_initial_conditions(actual, contributions)

        all_contributions[s, :, :, :] = contributions
        all_initial[s, :, :] = initial_cond
        all_shocks[s, :, :] = shocks
    end

    # Compute quantiles and means
    q_vec = ET.(quantiles)
    nq = length(quantiles)

    contrib_q = zeros(ET, T_eff, n, n, nq)
    contrib_m = zeros(ET, T_eff, n, n)
    initial_q = zeros(ET, T_eff, n, nq)
    initial_m = zeros(ET, T_eff, n)
    shocks_m = zeros(ET, T_eff, n)

    @inbounds for t in 1:T_eff
        for i in 1:n
            # Initial conditions
            d_init = @view all_initial[:, t, i]
            initial_q[t, i, :] = quantile(d_init, q_vec)
            initial_m[t, i] = mean(d_init)

            # Shocks
            d_shock = @view all_shocks[:, t, i]
            shocks_m[t, i] = mean(d_shock)

            for j in 1:n
                d = @view all_contributions[:, t, i, j]
                contrib_q[t, i, j, :] = quantile(d, q_vec)
                contrib_m[t, i, j] = mean(d)
            end
        end
    end

    BayesianHistoricalDecomposition{ET}(
        contrib_q, contrib_m, initial_q, initial_m, shocks_m, actual, T_eff,
        default_var_names(n), default_shock_names(n), q_vec, method
    )
end

# =============================================================================
# Arias et al. (2018) Historical Decomposition
# =============================================================================

"""
    historical_decomposition(model::VARModel, restrictions::SVARRestrictions, horizon; ...) -> BayesianHistoricalDecomposition

Compute historical decomposition using Arias et al. (2018) identification with importance weights.

# Arguments
- `model::VARModel`: Estimated VAR model
- `restrictions::SVARRestrictions`: Zero and sign restrictions
- `horizon::Int`: Maximum horizon for MA coefficients

# Keyword Arguments
- `n_draws::Int=1000`: Number of accepted draws
- `n_rotations::Int=1000`: Maximum rotation attempts per draw
- `quantiles::Vector{<:Real}=[0.16, 0.5, 0.84]`: Quantile levels for weighted quantiles

# Returns
`BayesianHistoricalDecomposition` with weighted posterior quantiles and means.

# Example
```julia
r = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
hd = historical_decomposition(model, r, 198; n_draws=500)
```
"""
function historical_decomposition(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
    n_draws::Int=1000, n_rotations::Int=1000,
    quantiles::Vector{<:Real}=[0.16, 0.5, 0.84]
) where {T<:AbstractFloat}

    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension must match model"

    T_eff = effective_nobs(model)
    horizon = min(horizon, T_eff)
    actual = model.Y[(model.p + 1):end, :]

    # Use identify_arias to get valid Q draws with weights
    arias_result = identify_arias(model, restrictions, horizon; n_draws=n_draws, n_rotations=n_rotations)

    n_acc = length(arias_result.Q_draws)
    weights = arias_result.weights

    # Compute HD for each accepted draw
    all_contributions = zeros(T, n_acc, T_eff, n, n)
    all_initial = zeros(T, n_acc, T_eff, n)
    all_shocks = zeros(T, n_acc, T_eff, n)

    for (idx, Q) in enumerate(arias_result.Q_draws)
        shocks = compute_structural_shocks(model, Q)
        Theta = _compute_structural_ma_coefficients(model, Q, horizon)
        contributions = _compute_hd_contributions(shocks, Theta)
        initial_cond = _compute_initial_conditions(actual, contributions)

        all_contributions[idx, :, :, :] = contributions
        all_initial[idx, :, :] = initial_cond
        all_shocks[idx, :, :] = shocks
    end

    # Compute weighted quantiles and means
    q_vec = T.(quantiles)
    nq = length(quantiles)

    contrib_q = zeros(T, T_eff, n, n, nq)
    contrib_m = zeros(T, T_eff, n, n)
    initial_q = zeros(T, T_eff, n, nq)
    initial_m = zeros(T, T_eff, n)
    shocks_m = zeros(T, T_eff, n)

    @inbounds for t in 1:T_eff
        for i in 1:n
            # Initial conditions (weighted)
            d_init = @view all_initial[:, t, i]
            initial_m[t, i] = sum(weights .* d_init)
            for (qi, q) in enumerate(q_vec)
                initial_q[t, i, qi] = _weighted_quantile(d_init, weights, q)
            end

            # Shocks (weighted)
            d_shock = @view all_shocks[:, t, i]
            shocks_m[t, i] = sum(weights .* d_shock)

            for j in 1:n
                d = @view all_contributions[:, t, i, j]
                contrib_m[t, i, j] = sum(weights .* d)
                for (qi, q) in enumerate(q_vec)
                    contrib_q[t, i, j, qi] = _weighted_quantile(d, weights, q)
                end
            end
        end
    end

    BayesianHistoricalDecomposition{T}(
        contrib_q, contrib_m, initial_q, initial_m, shocks_m, actual, T_eff,
        default_var_names(n), default_shock_names(n), q_vec, :arias
    )
end

# =============================================================================
# Accessor Functions
# =============================================================================

"""
    contribution(hd::HistoricalDecomposition, var, shock) -> Vector

Get contribution time series for specific variable and shock.

# Arguments
- `hd`: Historical decomposition result
- `var`: Variable index (Int) or name (String)
- `shock`: Shock index (Int) or name (String)

# Example
```julia
contrib_y1_s1 = contribution(hd, 1, 1)  # Contribution of shock 1 to variable 1
contrib_y1_s1 = contribution(hd, "Var 1", "Shock 1")
```
"""
function contribution(hd::HistoricalDecomposition{T}, var::Int, shock::Int) where {T}
    @assert 1 <= var <= length(hd.variables) "Variable index out of bounds"
    @assert 1 <= shock <= length(hd.shock_names) "Shock index out of bounds"
    hd.contributions[:, var, shock]
end

function contribution(hd::HistoricalDecomposition, var::String, shock::String)
    var_idx = findfirst(==(var), hd.variables)
    shock_idx = findfirst(==(shock), hd.shock_names)
    isnothing(var_idx) && throw(ArgumentError("Variable '$var' not found"))
    isnothing(shock_idx) && throw(ArgumentError("Shock '$shock' not found"))
    contribution(hd, var_idx, shock_idx)
end

"""
    contribution(hd::BayesianHistoricalDecomposition, var, shock; stat=:mean) -> Vector

Get contribution time series for specific variable and shock (Bayesian).

# Arguments
- `hd`: Bayesian historical decomposition result
- `var`: Variable index (Int) or name (String)
- `shock`: Shock index (Int) or name (String)
- `stat`: `:mean` or quantile index (Int)
"""
function contribution(hd::BayesianHistoricalDecomposition{T}, var::Int, shock::Int;
                      stat::Union{Symbol,Int}=:mean) where {T}
    @assert 1 <= var <= length(hd.variables) "Variable index out of bounds"
    @assert 1 <= shock <= length(hd.shock_names) "Shock index out of bounds"

    if stat == :mean
        return hd.mean[:, var, shock]
    elseif stat isa Int
        @assert 1 <= stat <= length(hd.quantile_levels) "Quantile index out of bounds"
        return hd.quantiles[:, var, shock, stat]
    else
        throw(ArgumentError("stat must be :mean or a quantile index (Int)"))
    end
end

function contribution(hd::BayesianHistoricalDecomposition, var::String, shock::String; stat=:mean)
    var_idx = findfirst(==(var), hd.variables)
    shock_idx = findfirst(==(shock), hd.shock_names)
    isnothing(var_idx) && throw(ArgumentError("Variable '$var' not found"))
    isnothing(shock_idx) && throw(ArgumentError("Shock '$shock' not found"))
    contribution(hd, var_idx, shock_idx; stat=stat)
end

"""
    total_shock_contribution(hd::AbstractHistoricalDecomposition, var) -> Vector

Get total contribution from all shocks to a variable over time.
"""
function total_shock_contribution(hd::HistoricalDecomposition{T}, var::Int) where {T}
    @assert 1 <= var <= length(hd.variables) "Variable index out of bounds"
    vec(sum(hd.contributions[:, var, :], dims=2))
end

function total_shock_contribution(hd::HistoricalDecomposition, var::String)
    var_idx = findfirst(==(var), hd.variables)
    isnothing(var_idx) && throw(ArgumentError("Variable '$var' not found"))
    total_shock_contribution(hd, var_idx)
end

function total_shock_contribution(hd::BayesianHistoricalDecomposition{T}, var::Int) where {T}
    @assert 1 <= var <= length(hd.variables) "Variable index out of bounds"
    vec(sum(hd.mean[:, var, :], dims=2))
end

function total_shock_contribution(hd::BayesianHistoricalDecomposition, var::String)
    var_idx = findfirst(==(var), hd.variables)
    isnothing(var_idx) && throw(ArgumentError("Variable '$var' not found"))
    total_shock_contribution(hd, var_idx)
end

"""
    verify_decomposition(hd::HistoricalDecomposition; tol=1e-10) -> Bool

Verify that contributions + initial_conditions ≈ actual.

# Example
```julia
hd = historical_decomposition(model, horizon)
@assert verify_decomposition(hd) "Decomposition identity failed"
```
"""
function verify_decomposition(hd::HistoricalDecomposition{T}; tol::T=T(1e-10)) where {T}
    n_vars = length(hd.variables)
    for i in 1:n_vars
        total_contrib = total_shock_contribution(hd, i)
        reconstructed = total_contrib .+ hd.initial_conditions[:, i]
        max_diff = maximum(abs.(reconstructed .- hd.actual[:, i]))
        max_diff > tol && return false
    end
    true
end

"""
    verify_decomposition(hd::BayesianHistoricalDecomposition; tol=1e-6) -> Bool

Verify that mean contributions + mean initial_conditions ≈ actual (approximately, due to averaging).
"""
function verify_decomposition(hd::BayesianHistoricalDecomposition{T}; tol::T=T(1e-6)) where {T}
    n_vars = length(hd.variables)
    for i in 1:n_vars
        total_contrib = total_shock_contribution(hd, i)
        reconstructed = total_contrib .+ hd.initial_mean[:, i]
        max_diff = maximum(abs.(reconstructed .- hd.actual[:, i]))
        max_diff > tol && return false
    end
    true
end

# =============================================================================
# Publication-Quality Show Methods
# =============================================================================

# Display helpers are defined in display_utils.jl

function Base.show(io::IO, hd::HistoricalDecomposition{T}) where {T}
    n_vars = length(hd.variables)
    n_shocks = length(hd.shock_names)

    # Specification table
    spec_data = [
        "Identification method" string(hd.method);
        "Variables" n_vars;
        "Shocks" n_shocks;
        "Time periods" hd.T_eff
    ]
    _pretty_table(io, spec_data;
        title = "Historical Decomposition",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Summary statistics for each variable
    summary_data = Matrix{Any}(undef, n_vars, n_shocks + 2)
    for i in 1:n_vars
        summary_data[i, 1] = hd.variables[i]
        for j in 1:n_shocks
            summary_data[i, j + 1] = round(mean(abs.(hd.contributions[:, i, j])), digits=4)
        end
        summary_data[i, end] = round(mean(abs.(hd.initial_conditions[:, i])), digits=4)
    end

    col_labels = vcat(["Variable"], hd.shock_names, ["Initial"])
    _pretty_table(io, summary_data;
        title = "Contribution Summary (mean absolute contribution)",
        column_labels = col_labels,
        alignment = vcat([:l], fill(:r, n_shocks + 1)),
    )

    # Verification status
    verified = verify_decomposition(hd)
    status = verified ? "Passed" : "FAILED"
    conc_data = Any["Decomposition identity" status]
    _pretty_table(io, conc_data;
        column_labels = ["", ""],
        alignment = [:l, :l],
    )
end

function Base.show(io::IO, hd::BayesianHistoricalDecomposition{T}) where {T}
    n_vars = length(hd.variables)
    n_shocks = length(hd.shock_names)
    nq = length(hd.quantile_levels)

    # Specification table
    q_str = join([string(round(q * 100, digits=0), "%") for q in hd.quantile_levels], ", ")
    spec_data = [
        "Identification method" string(hd.method);
        "Variables" n_vars;
        "Shocks" n_shocks;
        "Time periods" hd.T_eff;
        "Quantiles" q_str
    ]
    _pretty_table(io, spec_data;
        title = "Bayesian Historical Decomposition",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Summary statistics for each variable (posterior means)
    summary_data = Matrix{Any}(undef, n_vars, n_shocks + 2)
    for i in 1:n_vars
        summary_data[i, 1] = hd.variables[i]
        for j in 1:n_shocks
            summary_data[i, j + 1] = round(mean(abs.(hd.mean[:, i, j])), digits=4)
        end
        summary_data[i, end] = round(mean(abs.(hd.initial_mean[:, i])), digits=4)
    end

    col_labels = vcat(["Variable"], hd.shock_names, ["Initial"])
    _pretty_table(io, summary_data;
        title = "Posterior Mean Contribution Summary (mean absolute)",
        column_labels = col_labels,
        alignment = vcat([:l], fill(:r, n_shocks + 1)),
    )
end
