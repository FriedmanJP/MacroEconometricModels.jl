# MacroEconometricModels.jl — Monomial quadrature rule for DSGE numerical integration
# Judd-Maliar-Maliar (2011) monomial rule (degree-3 exact, 2n+1 points)

"""
    _monomial_nodes_weights(n_eps::Int) -> (nodes, weights)

Judd-Maliar-Maliar (2011) monomial integration rule for N(0, I).

Uses 2n+1 evaluation points: origin + ±√n along each axis.
Exact for all monomials up to degree 3.

Returns nodes (2n+1 × n_eps) and weights (2n+1,) that sum to 1.
"""
function _monomial_nodes_weights(n_eps::Int)
    n_eps >= 1 || throw(ArgumentError("n_eps must be ≥ 1"))

    n_points = 2 * n_eps + 1
    nodes = zeros(n_points, n_eps)
    weights = zeros(n_points)

    c = sqrt(Float64(n_eps))
    weights[1] = 1.0 - n_eps / c^2

    for j in 1:n_eps
        idx_pos = 1 + 2 * (j - 1) + 1
        idx_neg = 1 + 2 * (j - 1) + 2
        nodes[idx_pos, j] = c
        nodes[idx_neg, j] = -c
        weights[idx_pos] = 1.0 / (2.0 * c^2)
        weights[idx_neg] = 1.0 / (2.0 * c^2)
    end

    return nodes, weights
end
