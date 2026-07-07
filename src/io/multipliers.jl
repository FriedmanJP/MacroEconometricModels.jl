# multipliers.jl — output/income/employment multipliers (Type I & II)

"Sectoral IO multipliers of a given `kind` (`:output`/`:income`/`:employment`) and `type` (`:I`/`:II`)."
struct IOMultipliers
    values::Vector{Float64}
    kind::Symbol
    type::Symbol
    sectors::Vector{String}
end

# Direct per-unit-output coefficient row for the requested kind.
function _household_coeffs(io::IOData, kind::Symbol)
    invx = _invdiag(io.x)
    if kind == :output
        return ones(length(io.x))
    elseif kind == :income
        return vec(sum(io.va, dims=1)) .* invx           # total value added / output
    elseif kind == :employment
        haskey(io.extensions, "employment") ||
            throw(ArgumentError("no 'employment' extension; add one with add_extension!"))
        F = io.extensions["employment"].F
        return vec(sum(F, dims=1)) .* invx
    else
        throw(ArgumentError("kind must be :output, :income, or :employment"))
    end
end

"""
    multipliers(io; kind=:output, type=:I) -> IOMultipliers

Sectoral multipliers. `kind` selects `:output` (column sums of `L`), `:income`
(value-added weighted), or `:employment` (jobs-weighted, needs an `employment`
extension). `type=:II` closes the model with respect to households.
"""
function multipliers(io::IOData; kind::Symbol=:output, type::Symbol=:I)
    L = leontief_inverse(io)
    h = _household_coeffs(io, kind)
    if type == :I
        vals = kind == :output ? vec(sum(L, dims=1)) : vec(L' * h)
    elseif type == :II
        n = length(io.x)
        L2 = _closed_leontief(io)            # income-closed (n+1)×(n+1) inverse
        if kind == :output
            vals = vec(sum(view(L2, 1:n, 1:n), dims=1))   # production column sums
        elseif kind == :income
            vals = collect(view(L2, n + 1, 1:n))          # induced household-income row
        else  # employment (or other): induced effects via production rows
            vals = vec(transpose(view(L2, 1:n, 1:n)) * h)
        end
    else
        throw(ArgumentError("type must be :I or :II"))
    end
    IOMultipliers(vals, kind, type, copy(io.sectors))
end

# Close the model w.r.t. households: append a household-income row (the first
# value-added category, i.e. compensation, so the model retains leakage) and a
# household-consumption column (final-demand shares) to A, then invert.
function _closed_leontief(io::IOData)
    A = technical_coefficients(io)
    n = size(A, 1)
    invx = _invdiag(io.x)
    hinc = vec(io.va[1, :]) .* invx          # compensation per unit output
    y = vec(sum(io.Y, dims=2))
    hc = y ./ max(sum(y), eps())             # household consumption column shares
    Abar = [A hc; reshape(collect(float.(hinc)), 1, n) 0.0]
    Matrix{Float64}(inv(I - Abar))
end
