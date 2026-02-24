# MacroEconometricModels.jl — Quadrature rules for numerical integration
# Gauss-Hermite (Golub-Welsch) and Monomial (Judd-Maliar-Maliar 2011)

"""
    _gauss_hermite_nodes_weights(n::Int) -> (nodes, weights)

Compute Gauss-Hermite quadrature nodes and weights for ∫ f(x) exp(-x²) dx.

Uses the Golub-Welsch algorithm: eigenvalues of the tridiagonal Jacobi matrix
give nodes, and first component of eigenvectors squared × √π gives weights.
"""
function _gauss_hermite_nodes_weights(n::Int)
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))

    J = zeros(n, n)
    for i in 1:(n - 1)
        beta = sqrt(i / 2.0)
        J[i, i + 1] = beta
        J[i + 1, i] = beta
    end

    F = eigen(Symmetric(J))
    nodes = F.values
    weights = F.vectors[1, :].^2 .* sqrt(π)

    perm = sortperm(nodes)
    return nodes[perm], weights[perm]
end

"""
    _gauss_hermite_scaled(n_per_dim::Int, sigma::AbstractMatrix{T}) -> (nodes, weights)

Gauss-Hermite quadrature for N(0, Σ) integration: ∫ f(x) φ(x; 0, Σ) dx.

Returns tensor-product nodes (n_points × n_dim) and weights (n_points,).
Nodes are scaled by √2 · L where Σ = L L'. Weights are normalized to sum to 1.
"""
function _gauss_hermite_scaled(n_per_dim::Int, sigma::AbstractMatrix{T}) where {T}
    n_eps = size(sigma, 1)
    nodes1d, w1d = _gauss_hermite_nodes_weights(n_per_dim)

    L = cholesky(Symmetric(sigma)).L

    n_total = n_per_dim^n_eps

    nodes_std = zeros(T, n_total, n_eps)
    weights = ones(T, n_total)

    for dim in 1:n_eps
        stride = n_per_dim^(dim - 1)
        repeat_block = n_per_dim^(n_eps - dim)
        idx = 1
        for _ in 1:repeat_block
            for j in 1:n_per_dim
                for _ in 1:stride
                    nodes_std[idx, dim] = T(nodes1d[j])
                    weights[idx] *= T(w1d[j])
                    idx += 1
                end
            end
        end
    end

    # Scale: z = √2 * L * x (change of variables from exp(-x²) to N(0,Σ))
    nodes_phys = sqrt(T(2)) * nodes_std * Matrix{T}(L')
    weights ./= T(π)^(n_eps / T(2))

    return nodes_phys, weights
end

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
