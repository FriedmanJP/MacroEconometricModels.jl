# baqaee_farhi.jl — Baqaee & Farhi (2019) nonlinear IO

"""
    BaqaeeFarhiResult

Output of [`baqaee_farhi`](@ref): Domar weights, the first-order (Hulten) output
elasticities, the second-order "beyond Hulten" Hessian, and network centralities
(influence vector, upstreamness, downstreamness).
"""
struct BaqaeeFarhiResult
    domar::Vector{Float64}
    first_order::Vector{Float64}
    second_order::Matrix{Float64}
    influence::Vector{Float64}
    upstreamness::Vector{Float64}
    downstreamness::Vector{Float64}
    sectors::Vector{String}
end

"""
    domar_weights(io) -> Vector

Domar weights `λ_i = sales_i / GDP` (gross output over GDP).
"""
domar_weights(io::IOData) = io.x ./ sum(io.va)

"""
    baqaee_farhi(io; theta=nothing, sigma=nothing) -> BaqaeeFarhiResult

Baqaee & Farhi (2019) decomposition. The first-order term equals the Domar
weights (Hulten's theorem). The second-order "beyond Hulten" term is the Hessian
of log output in log productivities, parameterized by production-substitution
elasticities `theta` and consumption-substitution elasticities `sigma`; with the
Cobb-Douglas default (`theta=sigma=1`) the second-order term vanishes and Hulten
is exact.
"""
function baqaee_farhi(io::IOData; theta=nothing, sigma=nothing)
    λ = domar_weights(io)
    A = technical_coefficients(io)
    L = leontief_inverse(io)
    first_order = copy(λ)
    y = vec(sum(io.Y, dims=2)); β = y ./ sum(y)
    influence = vec(L' * β)
    upstreamness = vec(sum(L, dims=2))
    downstreamness = vec(sum(L, dims=1))
    second_order = _bf_second_order(io, A, L, λ, theta, sigma)
    BaqaeeFarhiResult(λ, first_order, second_order, influence,
                      upstreamness, downstreamness, copy(io.sectors))
end

# Second-order "beyond Hulten" Hessian. Baqaee & Farhi (2019): the macro
# second-order term aggregates input-output covariances weighted by
# microeconomic substitution elasticities. Cobb-Douglas (θ=σ=1) ⇒ zero.
function _bf_second_order(io::IOData, A, L, λ, theta, sigma)
    n = length(λ)
    θ = theta === nothing ? 1.0 : float(theta)    # production substitution
    σ = sigma === nothing ? 1.0 : float(sigma)    # consumption substitution
    y = vec(sum(io.Y, dims=2)); β = y ./ sum(y)
    Ψ = L                                          # Leontief inverse
    H = zeros(n, n)
    # producer side: weight by each sector i's input cost shares Ω_i = A[:, i]
    for i in 1:n
        ω = A[:, i]
        sω = sum(ω)
        sω == 0 && continue
        ωn = ω ./ sω
        for j in 1:n, k in 1:n
            mj = sum(ωn .* Ψ[:, j]); mk = sum(ωn .* Ψ[:, k])
            cov = sum(ωn .* (Ψ[:, j] .- mj) .* (Ψ[:, k] .- mk))
            H[j, k] += (θ - 1) * λ[i] * cov
        end
    end
    # consumer side: weight by final-demand shares β
    for j in 1:n, k in 1:n
        mj = sum(β .* Ψ[:, j]); mk = sum(β .* Ψ[:, k])
        covc = sum(β .* (Ψ[:, j] .- mj) .* (Ψ[:, k] .- mk))
        H[j, k] += (σ - 1) * covc
    end
    0.5 .* (H .+ H')                               # symmetrize numerically
end
