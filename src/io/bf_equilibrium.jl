# bf_equilibrium.jl — exact nonlinear CES equilibrium (Baqaee–Farhi 2019/2020)
#
# Orientation: ROW (B&F). All linear algebra is sparse. Numéraire: nominal GDP
# E = 1. Never form Ψ densely; Domar weights via one sparse transpose-solve.
# Wedges (B&F 2020): p_i = μ_i c_i; profits rebated lump-sum; GDP = factors + π.

"""
    BFEquilibrium{T<:AbstractFloat}

Exact counterfactual equilibrium of a [`ProductionNetwork`](@ref) under
Hicks-neutral productivity shocks `dlogA`, factor-supply shocks `dlogL`, and
markup/wedge shocks `dlogmu` (Baqaee–Farhi 2019; wedges B&F 2020).

# Fields
- `dlogY` — change in log real consumption (`−log P_c`; equals welfare under
  CRS with fixed factors and lump-sum rebate of profits).
- `p` — producer prices length `M` (GDP numéraire E = 1).
- `w` — factor wages length `F`.
- `Omega` — equilibrium **cost-share** matrix (row orientation, sparse).
- `lambda` — equilibrium **revenue-based** Domar weights length `1+M+F`
  (sales / GDP; equals cost-based when `μ ≡ 1`).
- `lambda_cost` — equilibrium **cost-based** Domar weights length `1+M+F`.
- `Lambda` — equilibrium revenue-based factor income shares length `F`.
- `dlog_x` — real-sector log output changes length `n`.
- `dlog_p` — real-sector log price changes length `n` (outer nodes).
- `hulten` — first-order pure-technology benchmark `λ̃' dlogA + Λ̃' dlogL`
  (cost-based Domars; equals full Hulten when `μ ≡ 1`).
- `technology` — B&F (2020) Theorem 1 pure technology term `λ̃' dlogA`.
- `allocative` — B&F (2020) Theorem 1 allocative-efficiency term
  `−λ̃' dlogμ − Λ̃' dlogΛ` (first-order; uses base cost Domars and the change
  in revenue factor shares).
- `profit_share` — aggregate profit / GDP `Σ_i λ_i (1 − 1/μ_i)`.
- `mu` — equilibrium producer markups length `M`.
- `dlogA`, `dlogL`, `dlogmu` — shock vectors used.
- `converged`, `iterations`, `residual` — solver diagnostics.
- `sectors` — real-sector labels length `n`.
"""
struct BFEquilibrium{T<:AbstractFloat}
    dlogY::T
    p::Vector{T}
    w::Vector{T}
    Omega::SparseMatrixCSC{T,Int}
    lambda::Vector{T}
    lambda_cost::Vector{T}
    Lambda::Vector{T}
    dlog_x::Vector{T}
    dlog_p::Vector{T}
    hulten::T
    technology::T
    allocative::T
    profit_share::T
    mu::Vector{T}
    dlogA::Vector{T}
    dlogL::Vector{T}
    dlogmu::Vector{T}
    converged::Bool
    iterations::Int
    residual::T
    sectors::Vector{String}
end

"""
    bf_equilibrium(net::ProductionNetwork; dlogA, dlogL, dlogmu,
                   method=:newton, tol=1e-10, maxiter=500,
                   damping=0.5) -> BFEquilibrium

Solve the exact nested-CES equilibrium of `net` under productivity shocks
`dlogA` (length `n`, mapped to real-sector outer nodes), factor-supply shocks
`dlogL` (length `F`), and markup shocks `dlogmu` (length `n`, mapped to outer
nodes; nest nodes stay competitive).

**Numéraire**: nominal GDP `E = 1` (factor income + markup profits). Base prices
are 1. Base productivities absorb base markups so that `p = μ c / A` holds at
the calibrated point with `p = 1`. Factor supplies: `L_f = Λ_f^{base,rev} · exp(dlogL_f)`.

**Pricing with wedges** (B&F 2020): `p_i = μ_i c_i(p, w, A)`. Cost shares `Ω̃`
depend only on input prices and productivities; revenue shares are
`Ω_{ij} = Ω̃_{ij}/μ_i` for producers. Profits `π_i = (1 − 1/μ_i) p_i y_i` are
rebated lump-sum to the household.

**Algorithm** (nested loops, sparse):
1. Inner unit-cost fixed point for producer prices given wages.
   - Residual: `log μ + log c − log p = 0`.
   - `:newton` (default): solve `(I − Ω̃_PP) δ = resid`.
   - Falls back to damped Picard if Newton residual fails to decrease.
2. Rebuild equilibrium cost/revenue shares and Domar weights.
3. Outer factor-market clearing on
   `r_f = log Λ_f − log(w_f L_f)` (revenue-based `Λ_f`) by damped quasi-Newton.

Returns [`BFEquilibrium`](@ref) including the B&F (2020) Theorem 1 first-order
decomposition (`technology`, `allocative`). Unconverged solves `@warn` and set
`converged=false`.

# Orientation
Row orientation throughout (B&F).
"""
function bf_equilibrium(net::ProductionNetwork{T};
                        dlogA=nothing,
                        dlogL=nothing,
                        dlogmu=nothing,
                        method::Symbol=:newton,
                        tol::Real=1e-10,
                        maxiter::Int=500,
                        damping::Real=0.5) where {T<:AbstractFloat}
    method in (:newton, :fixedpoint) || throw(ArgumentError(
        "method must be :newton or :fixedpoint; got $method"))

    n, M, F = net.n, net.M, net.F
    N = 1 + M + F
    Ω̃ = net.Omega
    θ = net.theta
    μ_base = net.mu

    dA = dlogA === nothing ? zeros(T, n) : _bf_as_vector(T, dlogA, n, "dlogA")
    dL = dlogL === nothing ? zeros(T, F) : _bf_as_vector(T, dlogL, F, "dlogL")
    dμ_sec = dlogmu === nothing ? zeros(T, n) : _bf_as_vector(T, dlogmu, n, "dlogmu")

    # producer log-A and log-μ (length M). Base A absorbs base μ so p=1 at zero shocks:
    #   log p = log μ_cur + log c − log A,  log A = log μ_base + dlogA (outers)
    #   log μ_cur = log μ_base + dlogμ
    log_μ = log.(μ_base)
    log_A = copy(log_μ)                             # A_base = μ_base for all producers
    for (k, g) in enumerate(net.outer_nodes)
        ip = g - 1
        log_A[ip] = log_μ[ip] + dA[k]
        log_μ[ip] = log_μ[ip] + dμ_sec[k]
    end
    μ_eq = exp.(log_μ)

    # factor supplies after shock (base L_f = revenue-based factor Domar)
    L = net.factor_supplies .* exp.(dL)
    all(L .> 0) || throw(ArgumentError("factor supplies must stay strictly positive"))

    # Hulten / Theorem 1 pure technology (base cost-based Domar on outers + factors)
    λ̃ = net.lambda
    λ̃_outer = T[λ̃[g] for g in net.outer_nodes]
    Λ̃ = λ̃[M+2:N]
    hulten = dot(λ̃_outer, dA) + dot(Λ̃, dL)
    technology = dot(λ̃_outer, dA)

    # init wages: guess from revenue factor shares
    Λ_base_rev = net.lambda_rev[M+2:N]
    log_w = log.(max.(Λ_base_rev, eps(T))) .- log.(L)
    log_p = zeros(T, M)

    tolT = T(tol)
    damp = T(damping)
    converged = false
    residual = T(Inf)
    iters = 0

    efficient_single = (F == 1) && all(abs.(μ_eq .- one(T)) .<= T(1e-14)) &&
                       all(abs.(dμ_sec) .<= T(1e-14))

    if efficient_single
        # Single-factor efficient: Λ ≡ 1 ⇒ w = 1/L exactly (bit-stable path)
        log_w[1] = -log(L[1])
        log_p, ok_inner, _ = _bf_solve_prices(log_p, log_w, Ω̃, θ, log_A, log_μ, M, F;
                                              method=method, tol=tolT,
                                              maxiter=maxiter, damping=damp)
        Ω_cost, λ_rev, λ_cost, log_Pc = _bf_shares_and_domar(
            log_p, log_w, Ω̃, θ, log_A, μ_eq, M, F)
        Λ = λ_rev[M+2:N]
        residual = abs(log(max(Λ[1], eps(T))) - log_w[1] - log(L[1]))
        converged = ok_inner && residual < sqrt(tolT) * 10 + tolT
        iters = 1
    elseif F == 1
        # Single factor with wedges: still pin w from E=1 identity after prices
        # Iterate: solve prices given w, update w from factor clearing.
        log_w[1] = -log(L[1])   # initial guess as if no profits; refined below
        log_p, log_w, converged, iters, residual =
            _bf_outer_loop!(log_p, log_w, L, Ω̃, θ, log_A, log_μ, μ_eq, M, F;
                            method=method, tol=tolT, maxiter=maxiter,
                            damping=damp)
        Ω_cost, λ_rev, λ_cost, log_Pc = _bf_shares_and_domar(
            log_p, log_w, Ω̃, θ, log_A, μ_eq, M, F)
        Λ = λ_rev[M+2:N]
    else
        log_p, log_w, converged, iters, residual =
            _bf_outer_loop!(log_p, log_w, L, Ω̃, θ, log_A, log_μ, μ_eq, M, F;
                            method=method, tol=tolT, maxiter=maxiter,
                            damping=damp)
        Ω_cost, λ_rev, λ_cost, log_Pc = _bf_shares_and_domar(
            log_p, log_w, Ω̃, θ, log_A, μ_eq, M, F)
        Λ = λ_rev[M+2:N]
    end

    if !converged
        @warn "bf_equilibrium: failed to converge" residual = Float64(residual) iterations = iters tol = Float64(tolT)
    end

    p = exp.(log_p)
    w = exp.(log_w)
    dlogY = -log_Pc

    # real-sector (outer node) price and output changes
    dlog_p_real = T[log_p[g - 1] for g in net.outer_nodes]
    dlog_x = Vector{T}(undef, n)
    for (k, g) in enumerate(net.outer_nodes)
        # dlog x = dlog λ_rev − dlog p  (E = 1; nominal sales = λ · E = λ)
        dlog_λ = log(max(λ_rev[g], eps(T))) - log(max(net.lambda_rev[g], eps(T)))
        dlog_x[k] = dlog_λ - log_p[g - 1]
    end

    # Theorem 1 allocative term (first-order; base cost Domars)
    # dlogY = λ̃'dlogA − λ̃'dlogμ − Λ̃'dlogΛ
    dlog_μ_outer = T[log_μ[g - 1] - log(μ_base[g - 1]) for g in net.outer_nodes]
    dlog_Λ = log.(max.(Λ, eps(T))) .- log.(max.(Λ_base_rev, eps(T)))
    allocative = -dot(λ̃_outer, dlog_μ_outer) - dot(Λ̃, dlog_Λ)

    # profit share of GDP from revenue Domars and markups
    profit_share = zero(T)
    @inbounds for i in 1:M
        profit_share += λ_rev[i + 1] * (one(T) - one(T) / μ_eq[i])
    end

    BFEquilibrium{T}(dlogY, p, w, Ω_cost, λ_rev, λ_cost, Λ, dlog_x, dlog_p_real,
                     hulten, technology, allocative, profit_share, μ_eq,
                     dA, dL, dμ_sec, converged, iters, residual,
                     String.(net.io.sectors))
end

# ── price / share primitives ─────────────────────────────────────────────────

"""Build length-`N` log-price vector: household unused, producers, factors."""
function _bf_log_pall(log_p::Vector{T}, log_w::Vector{T}, M::Int, F::Int) where {T}
    N = 1 + M + F
    log_pall = zeros(T, N)
    @inbounds for i in 1:M
        log_pall[i + 1] = log_p[i]
    end
    @inbounds for f in 1:F
        log_pall[M + 1 + f] = log_w[f]
    end
    return log_pall
end

@inline function _bf_row_nz(Ω̃::SparseMatrixCSC{T,Int}, g::Int) where {T}
    # Extract nonzero (col, val) of row g. CSC makes this O(nnz); N is modest
    # for typical IO tables. For large N we could store CSR, but keep simple.
    cols = Int[]; vals = T[]
    N = size(Ω̃, 2)
    @inbounds for j in 1:N
        ω = Ω̃[g, j]
        if ω != 0
            push!(cols, j)
            push!(vals, ω)
        end
    end
    return cols, vals
end

function _bf_unit_log_cost_nz(g::Int, log_pall::Vector{T}, cols::Vector{Int},
                              vals::Vector{T}, θi::T, log_A_i::T;
                              θ_atol::T=T(1e-12)) where {T}
    if abs(θi - one(T)) <= θ_atol
        s = -log_A_i
        @inbounds for k in eachindex(cols)
            s += vals[k] * log_pall[cols[k]]
        end
        return s
    else
        one_m_θ = one(T) - θi
        acc = zero(T)
        @inbounds for k in eachindex(cols)
            acc += vals[k] * exp(one_m_θ * log_pall[cols[k]])
        end
        acc > 0 || return T(Inf)
        return -log_A_i + log(acc) / one_m_θ
    end
end

"""Cost-share row of node `g` given input prices (Shepard; valid off equilibrium)."""
function _bf_cost_share_row!(out_cols::Vector{Int}, out_vals::Vector{T},
                             cols::Vector{Int}, vals::Vector{T},
                             log_pall::Vector{T}, θi::T;
                             θ_atol::T=T(1e-12)) where {T}
    empty!(out_cols); empty!(out_vals)
    if abs(θi - one(T)) <= θ_atol
        append!(out_cols, cols)
        append!(out_vals, vals)
        return
    end
    one_m_θ = one(T) - θi
    acc = zero(T)
    @inbounds for k in eachindex(cols)
        acc += vals[k] * exp(one_m_θ * log_pall[cols[k]])
    end
    acc > 0 || return
    @inbounds for k in eachindex(cols)
        push!(out_cols, cols[k])
        push!(out_vals, vals[k] * exp(one_m_θ * log_pall[cols[k]]) / acc)
    end
end

"""Pre-extract nonzero pattern of each household+producer row of Ω̃."""
function _bf_preextract_rows(Ω̃::SparseMatrixCSC{T,Int}, M::Int) where {T}
    # nodes 1:M+1
    col_list = Vector{Vector{Int}}(undef, M + 1)
    val_list = Vector{Vector{T}}(undef, M + 1)
    for g in 1:(M + 1)
        col_list[g], val_list[g] = _bf_row_nz(Ω̃, g)
    end
    return col_list, val_list
end

function _bf_solve_prices(log_p0::Vector{T}, log_w::Vector{T},
                          Ω̃::SparseMatrixCSC{T,Int}, θ::Vector{T},
                          log_A::Vector{T}, log_μ::Vector{T}, M::Int, F::Int;
                          method::Symbol=:newton,
                          tol::T=T(1e-10),
                          maxiter::Int=500,
                          damping::T=T(0.5),
                          θ_atol::T=T(1e-12)) where {T}
    log_p = copy(log_p0)
    col_list, val_list = _bf_preextract_rows(Ω̃, M)
    log_c = zeros(T, M)
    prev_res = T(Inf)
    use_newton = method === :newton
    ok = false

    for it in 1:maxiter
        log_pall = _bf_log_pall(log_p, log_w, M, F)
        @inbounds for i in 1:M
            g = i + 1
            log_c[i] = _bf_unit_log_cost_nz(g, log_pall, col_list[g], val_list[g],
                                            θ[g], log_A[i]; θ_atol=θ_atol)
        end
        # residual of p = μ · c  ⇔  log μ + log c − log p = 0
        resid = log_μ .+ log_c .- log_p
        nrm = maximum(abs, resid)
        if nrm < tol
            ok = true
            return log_p, ok, it
        end

        if use_newton
            # Build Ω̃_PP (cost-share Jacobian of log c w.r.t. log p)
            if M <= 256
                Ω_PP = zeros(T, M, M)
                out_c = Int[]; out_v = T[]
                @inbounds for i in 1:M
                    g = i + 1
                    _bf_cost_share_row!(out_c, out_v, col_list[g], val_list[g],
                                        log_pall, θ[g]; θ_atol=θ_atol)
                    for k in eachindex(out_c)
                        jg = out_c[k]
                        if 2 <= jg <= M + 1
                            Ω_PP[i, jg - 1] = out_v[k]
                        end
                    end
                end
                δ = (I - Ω_PP) \ resid
            else
                Ir = Int[]; Ic = Int[]; Iv = T[]
                out_c = Int[]; out_v = T[]
                @inbounds for i in 1:M
                    g = i + 1
                    _bf_cost_share_row!(out_c, out_v, col_list[g], val_list[g],
                                        log_pall, θ[g]; θ_atol=θ_atol)
                    for k in eachindex(out_c)
                        jg = out_c[k]
                        if 2 <= jg <= M + 1
                            push!(Ir, i); push!(Ic, jg - 1); push!(Iv, out_v[k])
                        end
                    end
                end
                Ω_PP = sparse(Ir, Ic, Iv, M, M)
                δ = (sparse(T(1) * I, M, M) - Ω_PP) \ resid
            end

            log_p_trial = log_p .+ δ
            log_pall_t = _bf_log_pall(log_p_trial, log_w, M, F)
            nrm_t = zero(T)
            @inbounds for i in 1:M
                g = i + 1
                ci = _bf_unit_log_cost_nz(g, log_pall_t, col_list[g], val_list[g],
                                          θ[g], log_A[i]; θ_atol=θ_atol)
                nrm_t = max(nrm_t, abs(log_μ[i] + ci - log_p_trial[i]))
            end
            if nrm_t < nrm
                log_p = log_p_trial
                prev_res = nrm_t
                continue
            else
                use_newton = false
            end
        end

        # damped Picard: log p ← (1−d) log p + d (log μ + log c)
        log_p .= (one(T) - damping) .* log_p .+ damping .* (log_μ .+ log_c)
        prev_res = nrm
    end
    return log_p, ok, maxiter
end

"""
Equilibrium cost shares, revenue Domar, cost Domar, and log P_c.

Cost shares from CES (Shepard); revenue shares = cost / μ on producer rows.
"""
function _bf_shares_and_domar(log_p::Vector{T}, log_w::Vector{T},
                              Ω̃::SparseMatrixCSC{T,Int}, θ::Vector{T},
                              log_A::Vector{T}, μ::Vector{T}, M::Int, F::Int;
                              θ_atol::T=T(1e-12)) where {T}
    N = 1 + M + F
    log_pall = _bf_log_pall(log_p, log_w, M, F)
    col_list, val_list = _bf_preextract_rows(Ω̃, M)

    Ir = Int[]; Ic = Int[]; Iv = T[]
    out_c = Int[]; out_v = T[]
    for g in 1:(M + 1)
        θi = θ[g]
        _bf_cost_share_row!(out_c, out_v, col_list[g], val_list[g],
                            log_pall, θi; θ_atol=θ_atol)
        for k in eachindex(out_c)
            push!(Ir, g); push!(Ic, out_c[k]); push!(Iv, out_v[k])
        end
    end
    Ω_cost = sparse(Ir, Ic, Iv, N, N)

    log_Pc = _bf_unit_log_cost_nz(1, log_pall, col_list[1], val_list[1],
                                  θ[1], zero(T); θ_atol=θ_atol)

    λ_cost = _bf_domar(Ω_cost)
    Ω_rev = _bf_revenue_omega(Ω_cost, μ, M, F)
    λ_rev = _bf_domar(Ω_rev)
    return Ω_cost, λ_rev, λ_cost, log_Pc
end

function _bf_factor_residual(log_p::Vector{T}, log_w::Vector{T}, L::Vector{T},
                             Ω̃, θ, log_A, log_μ, μ, M, F;
                             method, tol, maxiter, damping) where {T}
    log_p_new, ok, _ = _bf_solve_prices(log_p, log_w, Ω̃, θ, log_A, log_μ, M, F;
                                        method=method, tol=tol, maxiter=maxiter,
                                        damping=damping)
    _, λ_rev, _, _ = _bf_shares_and_domar(log_p_new, log_w, Ω̃, θ, log_A, μ, M, F)
    Λ = λ_rev[M+2:M+1+F]
    r = log.(max.(Λ, eps(T))) .- log_w .- log.(L)
    return r, log_p_new, ok, Λ
end

function _bf_outer_loop!(log_p::Vector{T}, log_w::Vector{T}, L::Vector{T},
                         Ω̃, θ, log_A, log_μ, μ, M::Int, F::Int;
                         method::Symbol, tol::T, maxiter::Int,
                         damping::T) where {T}
    residual = T(Inf)
    # Broyden: maintain approximate inverse Jacobian H ≈ J^{-1}
    r, log_p, ok, _ = _bf_factor_residual(log_p, log_w, L, Ω̃, θ, log_A, log_μ, μ, M, F;
                                          method=method, tol=tol, maxiter=maxiter,
                                          damping=damping)
    residual = maximum(abs, r)

    # initial FD Jacobian once
    ε = T(1e-6)
    J = zeros(T, F, F)
    for f in 1:F
        log_w_p = copy(log_w)
        log_w_p[f] += ε
        r_p, _, _, _ = _bf_factor_residual(log_p, log_w_p, L, Ω̃, θ, log_A, log_μ, μ, M, F;
                                           method=method, tol=tol, maxiter=maxiter,
                                           damping=damping)
        J[:, f] .= (r_p .- r) ./ ε
    end
    # ridge-stabilized inverse for singular Walras structure
    H = Matrix{T}(robust_inv(Hermitian(J' * J + T(1e-10) * I))) * J'

    for it in 1:maxiter
        residual = maximum(abs, r)
        if residual < tol && ok
            return log_p, log_w, true, it, residual
        end
        δ = -H * r
        # backtracking on residual
        α = one(T)
        accepted = false
        r_new = r
        log_p_new = log_p
        log_w_new = log_w
        ok_new = ok
        for _bt in 1:12
            log_w_try = log_w .+ α .* δ
            r_try, log_p_try, ok_try, _ = _bf_factor_residual(
                log_p, log_w_try, L, Ω̃, θ, log_A, log_μ, μ, M, F;
                method=method, tol=tol, maxiter=maxiter, damping=damping)
            if maximum(abs, r_try) < residual * (one(T) - T(1e-4) * α) || α < T(1e-4)
                log_w_new = log_w_try
                log_p_new = log_p_try
                r_new = r_try
                ok_new = ok_try
                accepted = true
                break
            end
            α *= T(0.5)
        end
        if !accepted
            log_w_new = log_w .+ damping .* δ
            r_new, log_p_new, ok_new, _ = _bf_factor_residual(
                log_p, log_w_new, L, Ω̃, θ, log_A, log_μ, μ, M, F;
                method=method, tol=tol, maxiter=maxiter, damping=damping)
        end

        # Broyden update (good Broyden on H)
        s = log_w_new .- log_w
        y = r_new .- r
        Hs = H * y
        denom = dot(s, Hs)
        if abs(denom) > T(1e-14) * (norm(s) * norm(Hs) + one(T))
            H = H + ((s - Hs) * (s' * H)) / denom
        end

        log_w = log_w_new
        log_p = log_p_new
        r = r_new
        ok = ok_new
        residual = maximum(abs, r)
        if residual < tol && ok
            return log_p, log_w, true, it, residual
        end
    end
    return log_p, log_w, false, maxiter, residual
end
