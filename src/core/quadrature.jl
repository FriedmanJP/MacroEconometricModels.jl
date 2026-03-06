# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Gauss-Hermite quadrature rules for numerical integration (Golub-Welsch algorithm)

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
    _adaptive_gauss_hermite(g, mu_hat::T, sigma_hat::T, n::Int) where {T<:AbstractFloat}

Adaptive Gauss-Hermite quadrature (Liu & Pierce 1994).

Approximates ∫ g(x) dx by centering nodes at `mu_hat` and scaling by `sigma_hat`:

    ∫ g(x) dx ≈ √2 · σ̂ · Σᵢ wᵢ · exp(xᵢ²) · g(μ̂ + √2 · σ̂ · xᵢ)

where (xᵢ, wᵢ) are standard Gauss-Hermite nodes/weights for the kernel exp(-x²).
This re-weighting removes the exp(-x²) kernel so `g` is evaluated directly.

# Arguments
- `g`: integrand function `g(x) -> scalar`
- `mu_hat::T`: mode (centering point)
- `sigma_hat::T`: scale (typically √(−1/f″(mode)) for a log-concave integrand)
- `n::Int`: number of quadrature nodes

# Returns
Scalar approximation of ∫ g(x) dx.

# Reference
Liu, Q. & Pierce, D. A. (1994). A Note on Gauss-Hermite Quadrature.
*Biometrika*, 81(3), 624--629.
"""
function _adaptive_gauss_hermite(g, mu_hat::T, sigma_hat::T, n::Int) where {T<:AbstractFloat}
    nodes, weights = _gauss_hermite_nodes_weights(n)
    result = zero(T)
    sqrt2_sigma = sqrt(T(2)) * sigma_hat
    for i in eachindex(nodes)
        xi = T(nodes[i])
        wi = T(weights[i])
        result += wi * exp(xi^2) * g(mu_hat + sqrt2_sigma * xi)
    end
    return sqrt2_sigma * result
end

# Float64 convenience method
function _adaptive_gauss_hermite(g, mu_hat::Real, sigma_hat::Real, n::Int)
    return _adaptive_gauss_hermite(g, Float64(mu_hat), Float64(sigma_hat), n)
end
