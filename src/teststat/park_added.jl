# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Park (1990) added-variables (superfluous-time-trends) test `H(p, q)` for spurious-vs-genuine
cointegration. Augments the cointegrating regression with superfluous deterministic trends
`t^{p+1}, …, t^q`; under genuine cointegration (`I(0)` errors) their coefficients are zero
and the long-run-variance-corrected Wald statistic is asymptotically `χ²(q−p)`, whereas under
a spurious regression (`I(1)` errors) it diverges. Consumes an EV-10 [`CointRegModel`](@ref).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Park (1990) H(p, q) added-variables test
# =============================================================================

"""
    park_added_test(m::CointRegModel; q_add=2, kernel=:bartlett, bandwidth=:nw)
        -> ParkAddedResult

Park (1990) `H(p, q)` test on the cointegrating regression `m`. Let `p` be the highest
deterministic-trend order already present (`0` for `:none`/`:const`, `1` for `:linear`).
The regression is re-estimated by OLS with `q_add` superfluous normalized-time trends
`(t/T)^{p+1}, …, (t/T)^{p+q_add}` appended, and the joint hypothesis that their coefficients
are zero is tested by the long-run-variance-corrected Wald statistic

    H(p, q) = γ̂' ( ω̂² · [(Z'Z)^{-1}]_{AA} )^{-1} γ̂  ~  χ²(q_add),

where `A` indexes the added trends and `ω̂²` is the long-run variance of the augmented-
regression residuals (EV-12 `lrvar`). A **large** `H` (small p-value) rejects the null of
genuine cointegration in favour of a spurious relationship.

# References
- Park, J. Y. (1990). Testing for unit roots and cointegration by adding superfluous
  regressors. CAE Working Paper, Cornell University. (See also Park 1992, *Econometrica*.)
"""
function park_added_test(m::CointRegModel{T}; q_add::Int=2,
                         kernel::Symbol=:bartlett, bandwidth=:nw) where {T<:AbstractFloat}
    q_add >= 1 || throw(ArgumentError("q_add must be ≥ 1; got $q_add"))
    regression = _coint_regression(m.trend)
    n = m.nobs
    D = _coint_deterministics(n, regression, T)
    base_order = m.trend === :linear ? 1 : 0

    # Superfluous normalized-time trends (t/T)^{base+1 .. base+q_add}.
    tau = T.(1:n) ./ T(n)
    A = Matrix{T}(undef, n, q_add)
    @inbounds for j in 1:q_add
        A[:, j] = tau .^ (base_order + j)
    end

    Z = hcat(D, m.X, A)                              # augmented design
    p_all = size(Z, 2)
    aidx = (p_all - q_add + 1):p_all                 # added-trend columns

    ZtZ = Symmetric(Z' * Z)
    ZtZinv = robust_inv(ZtZ)
    theta = ZtZinv * (Z' * m.y)
    resid_vec = m.y .- Z * theta

    bw = bandwidth === :nw ? floor(Int, 4 * (n / 100)^0.25) : bandwidth
    omega2 = lrvar(resid_vec; kernel=kernel, bandwidth=bw, demean=false)
    omega2 > zero(T) || (omega2 = max(var(resid_vec), eps(T)))

    gamma = theta[aidx]
    Vaa = omega2 .* Matrix{T}(ZtZinv[aidx, aidx])
    H = dot(gamma, robust_inv(Symmetric(Vaa)) * gamma)
    H = max(H, zero(T))

    pval = T(ccdf(Chisq(q_add), H))

    return ParkAddedResult(H, pval, q_add, base_order, regression, m.trend, m.k, n)
end
