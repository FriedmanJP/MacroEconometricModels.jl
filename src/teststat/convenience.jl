# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Convenience functions: unit_root_summary(), test_all_variables().
"""

using Statistics

"""
    unit_root_summary(y; tests=[:adf, :kpss, :pp], kwargs...) -> NamedTuple

Run multiple unit root tests and return summary with PrettyTables output.

# Arguments
- `y`: Time series vector
- `tests`: Vector of test symbols to run (default: [:adf, :kpss, :pp])
- `kwargs...`: Additional arguments passed to individual tests

# Returns
NamedTuple with test results, conclusion, and summary table.

# Example
```julia
y = cumsum(randn(200))
summary = unit_root_summary(y)
summary.conclusion  # Overall conclusion
```
"""
function unit_root_summary(y::AbstractVector{T};
                           tests::Vector{Symbol}=[:adf, :kpss, :pp],
                           regression::Symbol=:constant) where {T<:AbstractFloat}

    results = Dict{Symbol,AbstractUnitRootTest}()

    for test in tests
        if test == :adf
            results[:adf] = adf_test(y; regression=regression)
        elseif test == :kpss
            reg = regression == :none ? :constant : regression
            results[:kpss] = kpss_test(y; regression=reg)
        elseif test == :pp
            results[:pp] = pp_test(y; regression=regression)
        elseif test == :za
            reg = regression == :none ? :constant : regression
            results[:za] = za_test(y; regression=reg)
        elseif test == :ngperron
            reg = regression == :none ? :constant : regression
            results[:ngperron] = ngperron_test(y; regression=reg)
        end
    end

    # Determine conclusion
    has_unit_root_adf = haskey(results, :adf) && results[:adf].pvalue > 0.05
    is_stationary_kpss = haskey(results, :kpss) && results[:kpss].pvalue > 0.05
    has_unit_root_pp = haskey(results, :pp) && results[:pp].pvalue > 0.05

    conclusion = if has_unit_root_adf && !is_stationary_kpss
        "Unit root detected (ADF fails to reject, KPSS rejects stationarity)"
    elseif !has_unit_root_adf && is_stationary_kpss
        "Series appears stationary (ADF rejects unit root, KPSS fails to reject)"
    elseif has_unit_root_adf && is_stationary_kpss
        "Inconclusive (ADF and KPSS both fail to reject)"
    else
        "Conflicting results (ADF rejects unit root, KPSS rejects stationarity)"
    end

    (results=results, conclusion=conclusion)
end

unit_root_summary(y::AbstractVector; kwargs...) = unit_root_summary(Float64.(y); kwargs...)

"""
    test_all_variables(Y; test=:adf, kwargs...) -> Vector

Apply unit root test to each column of Y.

# Arguments
- `Y`: Data matrix (T × n)
- `test`: Test to apply (:adf, :kpss, :pp, :za, :ngperron)
- `kwargs...`: Additional arguments passed to the test

# Returns
Vector of test results, one per variable.

# Example
```julia
Y = randn(200, 3)
Y[:, 1] = cumsum(Y[:, 1])  # Make first column non-stationary
results = test_all_variables(Y; test=:adf)
[r.pvalue for r in results]  # P-values for each variable
```
"""
function test_all_variables(Y::AbstractMatrix{T};
                            test::Symbol=:adf,
                            kwargs...) where {T<:AbstractFloat}

    n = size(Y, 2)
    results = Vector{AbstractUnitRootTest}(undef, n)

    test_func = if test == :adf
        adf_test
    elseif test == :kpss
        kpss_test
    elseif test == :pp
        pp_test
    elseif test == :za
        za_test
    elseif test == :ngperron
        ngperron_test
    else
        throw(ArgumentError("Unknown test: $test. Use :adf, :kpss, :pp, :za, or :ngperron"))
    end

    for i in 1:n
        results[i] = test_func(Y[:, i]; kwargs...)
    end

    results
end

test_all_variables(Y::AbstractMatrix; kwargs...) = test_all_variables(Float64.(Y); kwargs...)
