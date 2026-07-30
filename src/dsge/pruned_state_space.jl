# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — the pruned state-space system ([T269])
#
# Pruning (Kim et al. 2008; Andreasen, Fernández-Villaverde & Rubio-Ramírez 2018) splits the
# higher-order state into components `x = xf + xs + xrd` that are propagated separately, so the
# Kronecker terms never feed back on themselves and simulated paths cannot explode.
#
# Before [T269] the recursion was written out three times — once in `simulate`, once in the
# order-2 augmented-moment routine, once in the unconditional FEVD — and the three did not
# agree. This file defines the system ONCE, as an object, and every consumer reads it from
# there.
#
# ## Timing, and the control map
#
# The package's policy functions are written over `v_t = [x_{t-1}; ε_t]`:
#
#     x_t = hx·v_t + ½·hxx·(v_t⊗v_t) + ½·hσσ + …
#     y_t = gx·v_t + ½·gxx·(v_t⊗v_t) + ½·gσσ + …
#
# so `gx_state` is the loading on the **lagged** state and `η_y` the loading on the current
# shock. Applying `gx_state` to the CURRENT state `x_t` propagates the state channel twice —
# audit S-01 / [T020]. That was fixed for order 1, but orders 2 and 3 kept evaluating the
# control on the freshly-updated components, which time-shifts every control series by one
# period: on an exactly linear model, where orders 2 and 3 must reproduce the first-order
# solution to machine precision, the control path came out shifted by a full period. The same
# error sat in the moment routine's observation map (`C_ctrl = gx_state·C_state`, i.e. the
# current-state map), biasing every covariance involving a control.
#
# Here the control is evaluated on the same `v` blocks that drive the state recursion, at every
# order, and `simulate`/`irf`/`fevd`/moments all read this one definition.
#
# References:
#   Andreasen, M. M., Fernández-Villaverde, J. & Rubio-Ramírez, J. F. (2018). The pruned
#     state-space system for non-linear DSGE models: theory and empirical applications.
#     Review of Economic Studies 85(1), 1-49.
#   Kim, J., Kim, S., Schaumburg, E. & Sims, C. A. (2008). Calculating and using second-order
#     accurate solutions of discrete time dynamic equilibrium models. JEDC 32(11), 3397-3414.

"""
    PrunedStateSpace{T}

The pruned state-space representation of a `PerturbationSolution`, built by
[`pruned_state_space`](@ref).

Holds the policy-function blocks in the split form the pruned recursion needs, together with
the variable bookkeeping (`state_indices` / `control_indices`, steady state, augmentation) that
turns pruned components back into observable variables. `simulate`, the unconditional FEVD, and
the closed-form moments all step the system through [`_pss_step`](@ref), so the three cannot
drift apart.

# Fields
- `order::Int` — 1, 2, or 3
- `nx`, `ny`, `n_eps`, `nv`, `n` — states, controls, shocks, `nv = nx + n_eps`, `n = nx + ny`
- `state_indices`, `control_indices` — positions of states/controls in the model's variable order
- `hx_state`, `eta_x` — first-order state transition and shock loading (`hx = [hx_state η_x]`)
- `gx_state`, `eta_y` — first-order control loadings on the **lagged** state and current shock
- `hxx`, `gxx`, `hss`, `gss` — second-order blocks over `v⊗v` and the `σ²` corrections
- `hxxx`, `gxxx`, `hssx`, `gssx`, `hsss`, `gsss` — third-order blocks (`nothing` below order 3)
- `steady_state`, `spec` — for mapping deviations back to levels

Empty higher-order blocks are stored as zeros rather than `nothing`, so the stepping kernel has
no branches on order beyond the component count.
"""
struct PrunedStateSpace{T<:AbstractFloat}
    order::Int
    nx::Int
    ny::Int
    n_eps::Int
    nv::Int
    n::Int
    state_indices::Vector{Int}
    control_indices::Vector{Int}
    hx_state::Matrix{T}
    eta_x::Matrix{T}
    gx_state::Matrix{T}
    eta_y::Matrix{T}
    hxx::Matrix{T}
    gxx::Matrix{T}
    hss::Vector{T}
    gss::Vector{T}
    hxxx::Matrix{T}
    gxxx::Matrix{T}
    hssx::Matrix{T}
    gssx::Matrix{T}
    hsss::Vector{T}
    gsss::Vector{T}
    steady_state::Vector{T}
    spec::DSGESpec{T}
end

_pss_mat(x, ::Type{T}, r::Int, c::Int) where {T} =
    x === nothing ? zeros(T, r, c) : Matrix{T}(x)
_pss_vec(x, ::Type{T}, r::Int) where {T} = x === nothing ? zeros(T, r) : Vector{T}(x)

"""
    pruned_state_space(sol::PerturbationSolution{T}) → PrunedStateSpace{T}

Build the pruned state-space system of a perturbation solution.

The returned object is the single definition of the pruned recursion and of the control
observation map; `simulate(sol, …)` and the closed-form moments both go through it.

```julia
sol = perturbation_solver(spec; order=3)
pss = pruned_state_space(sol)
pss.order, pss.nx, pss.ny
```
"""
function pruned_state_space(sol::PerturbationSolution{T}) where {T<:AbstractFloat}
    nx = nstates(sol)
    ny = ncontrols(sol)
    n_eps = nshocks(sol)
    nv = nx + n_eps
    n = nvars(sol)

    hx_state = nx > 0 ? Matrix{T}(sol.hx[:, 1:nx]) : zeros(T, 0, 0)
    eta_x    = nx > 0 ? Matrix{T}(sol.hx[:, nx+1:nv]) : zeros(T, 0, n_eps)
    gx_state = ny > 0 ? Matrix{T}(sol.gx[:, 1:nx]) : zeros(T, 0, nx)
    eta_y    = ny > 0 ? Matrix{T}(sol.gx[:, nx+1:nv]) : zeros(T, 0, n_eps)

    return PrunedStateSpace{T}(
        sol.order, nx, ny, n_eps, nv, n,
        collect(sol.state_indices), collect(sol.control_indices),
        hx_state, eta_x, gx_state, eta_y,
        _pss_mat(sol.hxx, T, nx, nv^2), _pss_mat(sol.gxx, T, ny, nv^2),
        _pss_vec(sol.hσσ, T, nx), _pss_vec(sol.gσσ, T, ny),
        _pss_mat(sol.hxxx, T, nx, nv^3), _pss_mat(sol.gxxx, T, ny, nv^3),
        _pss_mat(sol.hσσx, T, nx, nv), _pss_mat(sol.gσσx, T, ny, nv),
        _pss_vec(sol.hσσσ, T, nx), _pss_vec(sol.gσσσ, T, ny),
        Vector{T}(sol.steady_state), sol.spec)
end

"""
    _pss_step(pss, xf, xs, xrd, shock) → (xf_new, xs_new, xrd_new, x_obs, y_obs)

Advance the pruned system one period from the **lagged** components `(xf, xs, xrd)` under
`shock`, returning the updated components, the observed state vector `x_obs = xf' + xs' + xrd'`,
and the observed controls `y_obs`.

Writing `vᶠ = [xf; ε]`, `vˢ = [xs; 0]`, `vʳ = [xrd; 0]` — the shock enters at first order only,
which is what pruning means — the recursion is

```
xf' = hx·vᶠ
xs' = hx·vˢ + ½·hxx·(vᶠ⊗vᶠ) + ½·hσσ
xrd' = hx·vʳ + hxx·(vᶠ⊗vˢ) + ⅙·hxxx·(vᶠ⊗vᶠ⊗vᶠ) + ½·hσσx·vᶠ + ⅙·hσσσ
y  = gx·(vᶠ + vˢ + vʳ) + ½·gxx·(vᶠ⊗vᶠ + 2·vᶠ⊗vˢ) + ½·gσσ
     + ⅙·gxxx·(vᶠ⊗vᶠ⊗vᶠ) + ½·gσσx·vᶠ + ⅙·gσσσ
```

The control is evaluated on the **same** `v` blocks as the state — the lagged components and
the current shock. Evaluating it on the updated components instead applies a lagged-state
loading to a current-dated state, which shifts every control series forward by one period
(S-01 / [T020]).
"""
function _pss_step(pss::PrunedStateSpace{T}, xf::AbstractVector{T}, xs::AbstractVector{T},
                   xrd::AbstractVector{T}, shock::AbstractVector{T}) where {T}
    nx, nv, ord = pss.nx, pss.nv, pss.order

    vf = Vector{T}(undef, nv)
    nx > 0 && (vf[1:nx] = xf)
    vf[nx+1:nv] = shock

    xf_new = pss.hx_state * xf + pss.eta_x * shock
    y = pss.gx_state * xf + pss.eta_y * shock          # gx·vᶠ, lagged state (S-01)

    xs_new = zeros(T, nx)
    xrd_new = zeros(T, nx)

    if ord >= 2
        kvf = kron(vf, vf)
        xs_new = pss.hx_state * xs + T(0.5) * (pss.hxx * kvf) + T(0.5) * pss.hss
        y += pss.gx_state * xs + T(0.5) * (pss.gxx * kvf) + T(0.5) * pss.gss

        if ord >= 3
            vs = zeros(T, nv)
            nx > 0 && (vs[1:nx] = xs)
            kvfvs = kron(vf, vs)
            k3 = kron(vf, kvf)

            xrd_new = pss.hx_state * xrd + pss.hxx * kvfvs +
                      (one(T) / T(6)) * (pss.hxxx * k3) +
                      T(0.5) * (pss.hssx * vf) + (one(T) / T(6)) * pss.hsss
            y += pss.gx_state * xrd + pss.gxx * kvfvs +
                 (one(T) / T(6)) * (pss.gxxx * k3) +
                 T(0.5) * (pss.gssx * vf) + (one(T) / T(6)) * pss.gsss
        end
    end

    return (xf_new, xs_new, xrd_new, xf_new + xs_new + xrd_new, y)
end

"""
    _pss_observe!(dev, t, pss, x_obs, y_obs)

Scatter one period's pruned state/control vectors into row `t` of a `T×n` deviation matrix, in
the model's own variable ordering.
"""
@inline function _pss_observe!(dev::AbstractMatrix{T}, t::Int, pss::PrunedStateSpace{T},
                               x_obs::AbstractVector{T}, y_obs::AbstractVector{T}) where {T}
    @inbounds for (k, si) in enumerate(pss.state_indices)
        dev[t, si] = x_obs[k]
    end
    @inbounds for (k, ci) in enumerate(pss.control_indices)
        dev[t, ci] = y_obs[k]
    end
    return dev
end

"""
    _pss_simulate_dev(pss, e) → Matrix

Deviations from steady state over the shock path `e` (`T_periods × n_eps`), starting from the
steady state. All components start at zero.
"""
function _pss_simulate_dev(pss::PrunedStateSpace{T}, e::AbstractMatrix{T}) where {T}
    T_periods = size(e, 1)
    dev = zeros(T, T_periods, pss.n)
    xf = zeros(T, pss.nx)
    xs = zeros(T, pss.nx)
    xrd = zeros(T, pss.nx)
    for t in 1:T_periods
        xf, xs, xrd, x_obs, y_obs = _pss_step(pss, xf, xs, xrd, T.(view(e, t, :)))
        _pss_observe!(dev, t, pss, x_obs, y_obs)
    end
    return dev
end

function Base.show(io::IO, pss::PrunedStateSpace{T}) where {T}
    print(io, "PrunedStateSpace{$T}: order $(pss.order), ", pss.nx, " state(s), ",
          pss.ny, " control(s), ", pss.n_eps, " shock(s)")
end

"""
    report(pss::PrunedStateSpace)

Print the pruned system's dimensions, the components it propagates, and the magnitude of each
policy block that is active at its order.
"""
function report(io::IO, pss::PrunedStateSpace{T}) where {T}
    comps = pss.order == 1 ? "xf" : pss.order == 2 ? "xf, xs" : "xf, xs, xrd"
    setup = Any[
        "Perturbation order"  pss.order;
        "Pruned components"   comps;
        "States"              pss.nx;
        "Controls"            pss.ny;
        "Shocks"              pss.n_eps
    ]
    _pretty_table(io, setup; title="Pruned State-Space System", column_labels=["", ""],
                  alignment=[:l, :r])

    blocks = Tuple{String,Any}[("hx (state)", pss.hx_state), ("η_x (shock→state)", pss.eta_x),
                               ("gx (lagged state→control)", pss.gx_state),
                               ("η_y (shock→control)", pss.eta_y)]
    if pss.order >= 2
        append!(blocks, [("hxx", pss.hxx), ("gxx", pss.gxx),
                         ("hσσ", pss.hss), ("gσσ", pss.gss)])
    end
    if pss.order >= 3
        append!(blocks, [("hxxx", pss.hxxx), ("gxxx", pss.gxxx),
                         ("hσσx", pss.hssx), ("gσσx", pss.gssx),
                         ("hσσσ", pss.hsss), ("gσσσ", pss.gsss)])
    end
    bdata = Matrix{Any}(undef, length(blocks), 3)
    for (r, (nm, blk)) in enumerate(blocks)
        bdata[r, 1] = nm
        bdata[r, 2] = isempty(blk) ? "—" : join(string.(size(blk)), "×")
        bdata[r, 3] = isempty(blk) ? "—" : _fmt(maximum(abs, blk); digits=6)
    end
    _pretty_table(io, bdata; title="Policy Blocks",
                  column_labels=["Block", "Size", "max|·|"], alignment=[:l, :r, :r])
    return nothing
end

report(pss::PrunedStateSpace) = report(stdout, pss)


# =============================================================================
# Order-2 augmented linear system
# =============================================================================

"""
    _pss_ee_mean(block, nx, nv, n_eps) → Vector

`E[block·(v⊗v)]` restricted to the `ε⊗ε` sub-block, i.e. `Σ_p block[:, col(nx+p, nx+p)]`, since
`E[ε_p·ε_q] = δ_pq` for unit-variance shocks. This is the one part of the non-`x⊗x` second-order
term with a nonzero mean, so folding it in makes the closed-form mean exact rather than
approximate.
"""
function _pss_ee_mean(block::AbstractMatrix{T}, nx::Int, nv::Int, n_eps::Int) where {T}
    out = zeros(T, size(block, 1))
    size(block, 2) == nv^2 || return out
    @inbounds for p in 1:n_eps
        col = (nx + p - 1) * nv + (nx + p)
        out .+= view(block, :, col)
    end
    return out
end

"""
    _pss_obs_map_2nd(pss) → (C, noise, d)

Observation map of the order-2 pruned system over the augmented state
`z_t = [xf_t; xs_t; vec(xf_t ⊗ xf_t)]`, so that an observable at `t` is

```
obs_t = C·z_{t-1} + noise·ε_t + d
```

with `z_{t-1} ⊥ ε_t`. Rows are in the model's variable ordering.

Both halves of the map have the SAME shape — the control's is the state's with `h → g`:

```
state:   x_t = hx·xf + hx·xs + ½·hxx_xx·(xf⊗xf) + ½·hσσ + η_x·ε
control: y_t = gx·xf + gx·xs + ½·gxx_xx·(xf⊗xf) + ½·gσσ + η_y·ε
```

That symmetry is the point. The control was previously mapped as `gx·C_state`, i.e. `gx`
applied to the *current* state `x_t`, which both double-counts the state channel and drags the
state's `½·hσσ` intercept into the control (S-01 / [T020]); and the unconditional FEVD instead
loaded the control's shock channel with `gx·η_x + η_y`, which double-counts it the other way.
Both now read this one map.

## Which parts of `hxx`/`gxx` enter

This package writes the second-order term over `v⊗v` with `v = [x; ε]`, so `hxx` carries four
blocks: `x⊗x`, `x⊗ε`, `ε⊗x` and `ε⊗ε`. Andreasen et al.'s augmented-state recursion — and the
innovation variance derived from it — is stated for the `x⊗x` block alone, with the shock
channel carried separately.

- The `ε⊗ε` block has a nonzero mean, `E[ε⊗ε] = vec(I)`, so it is folded into `d` (and into the
  augmented system's constant) exactly, via [`_pss_ee_mean`](@ref).
- The `x⊗ε` and `ε⊗x` blocks are bilinear in the lagged state and the current shock. They are
  mean-zero and uncorrelated with `C·z` and with `ε`, so they leave `E` and the autocovariance
  cross-term untouched, but they **do** contribute to the variance and are **not** included
  here. The resulting variance understatement is `O(σ²)` relative to the state variance; on the
  RBC benchmark the closed form sits within Monte-Carlo error of a 2·10⁶-draw pruned simulation.

Consequently `C·z_{t-1} + noise·ε_t + d` reproduces [`_pss_step`](@ref) **exactly** when those
two blocks vanish (any model whose shocks enter linearly), and differs by the bilinear term
otherwise. The tests assert both halves of that statement rather than assuming the first.
"""
function _pss_obs_map_2nd(pss::PrunedStateSpace{T}) where {T}
    nx, ny, n, nv, n_eps = pss.nx, pss.ny, pss.n, pss.nv, pss.n_eps
    nz = 2 * nx + nx^2

    hxx_xx = nx > 0 ? _extract_xx_block(pss.hxx, nx, nv) : zeros(T, nx, 0)
    gxx_xx = (ny > 0 && nx > 0) ? _extract_xx_block(pss.gxx, nx, nv) : zeros(T, ny, nx^2)
    hxx_ee = _pss_ee_mean(pss.hxx, nx, nv, n_eps)
    gxx_ee = _pss_ee_mean(pss.gxx, nx, nv, n_eps)

    C = zeros(T, n, nz)
    noise = zeros(T, n, n_eps)
    d = zeros(T, n)

    @inbounds for (k, si) in enumerate(pss.state_indices)
        if nx > 0
            C[si, 1:nx] = pss.hx_state[k, :]
            C[si, nx+1:2*nx] = pss.hx_state[k, :]
            C[si, 2*nx+1:nz] = T(0.5) .* hxx_xx[k, :]
        end
        noise[si, :] = pss.eta_x[k, :]
        d[si] = T(0.5) * (pss.hss[k] + hxx_ee[k])
    end
    @inbounds for (k, ci) in enumerate(pss.control_indices)
        if nx > 0
            C[ci, 1:nx] = pss.gx_state[k, :]
            C[ci, nx+1:2*nx] = pss.gx_state[k, :]
            C[ci, 2*nx+1:nz] = T(0.5) .* gxx_xx[k, :]
        end
        noise[ci, :] = pss.eta_y[k, :]
        d[ci] = T(0.5) * (pss.gss[k] + gxx_ee[k])
    end
    return (C, noise, d)
end

"""
    _pss_augmented_2nd(pss; eta_x_override=nothing) → (A, c, Var_z, E_z, M)

The order-2 pruned system written as a LINEAR state space in the augmented state
`z_t = [xf_t; xs_t; vec(xf_t ⊗ xf_t)]`,

```
z_t = c + A·z_{t-1} + ξ_t
```

together with its unconditional mean `E_z = (I − A)⁻¹c`, variance `Var_z` (the augmented
Lyapunov solution), and `M = E[ξ_t·ε_t']`, which is what makes autocovariances computable
without simulation.

`eta_x_override` substitutes a single-shock loading, for the per-shock unconditional FEVD.
"""
function _pss_augmented_2nd(pss::PrunedStateSpace{T};
                            eta_x_override::Union{Nothing,Matrix{T}}=nothing) where {T}
    nx, nv, n_eps = pss.nx, pss.nv, pss.n_eps
    nz = 2 * nx + nx^2
    hx = pss.hx_state
    eta_x = eta_x_override === nothing ? pss.eta_x : eta_x_override
    hxx_xx = nx > 0 ? _extract_xx_block(pss.hxx, nx, nv) : zeros(T, nx, 0)

    A = zeros(T, nz, nz)
    if nx > 0
        A[1:nx, 1:nx] = hx
        A[nx+1:2*nx, nx+1:2*nx] = hx
        A[nx+1:2*nx, 2*nx+1:nz] = T(0.5) * hxx_xx
        A[2*nx+1:nz, 2*nx+1:nz] = kron(hx, hx)
    end

    c = zeros(T, nz)
    if nx > 0
        c[nx+1:2*nx] = T(0.5) * (pss.hss + _pss_ee_mean(pss.hxx, nx, nv, n_eps))
        c[2*nx+1:nz] = kron(eta_x, eta_x) * vec(Matrix{T}(I, n_eps, n_eps))
    end

    E_z = nz > 0 ? (Matrix{T}(I, nz, nz) - A) \ c : zeros(T, 0)
    Var_xf = nx > 0 ? _dlyap(hx, eta_x * eta_x') : zeros(T, 0, 0)
    Var_z = _dlyap(A, _innovation_variance_2nd(hx, eta_x, Var_xf, nx, n_eps))

    # E[ξ_t·ε_t']: with Gaussian (symmetric) shocks only the xf block survives, since the
    # xf⊗xf innovation is even in ε and the xs innovation is a quadratic form.
    M = zeros(T, nz, n_eps)
    nx > 0 && (M[1:nx, :] = eta_x)

    return (A, c, Var_z, E_z, M)
end


# =============================================================================
# Order-3 augmented system: exact, simulation-free innovation moments
# =============================================================================
#
# Andreasen, Fernández-Villaverde & Rubio-Ramírez (2018) give the third-order pruned system as
# a LINEAR recursion in the augmented state
#
#     z_t = [xf; xs; xf⊗xf; xrd; xf⊗xs; xf⊗xf⊗xf]        (3nx + 2nx² + nx³)
#     z_{t+1} = c + A·z_t + ξ_{t+1}
#
# and their companion code (`UnconditionalMoments_3rd_Lyap.m`) obtains `Var(ξ)` and
# `Cov(ξ_{t+1}, z_t)` from ~2000 lines of hand-derived Gaussian moment algebra, carrying shock
# moments up to order six.
#
# There is a much shorter exact route. Expanding each block of ξ shows that every term is a
# monomial in `ε_{t+1}` times a SINGLE component of `z_t` (or a constant) — the whole point of
# the augmentation is that the nonlinearity is absorbed into the state coordinates. So for a
# FIXED shock, ξ is **linear** in `z̃ = [1; z]`:
#
#     ξ_{t+1} = Ξ(ε_{t+1}) · z̃_t
#
# With `ε ⊥ z` that gives, exactly,
#
#     Var(ξ)             = E_ε[ Ξ(ε)·E[z̃z̃']·Ξ(ε)' ]
#     Cov(ξ_{t+1}, z_t)  = E_ε[Ξ(ε)] · Cov(z̃_t, z_t)
#
# and both expectations are integrals of a polynomial of degree ≤ 6 in ε, which a 4-node
# Gauss-Hermite tensor rule evaluates EXACTLY (exact through degree 2m−1 = 7). No simulation,
# no sixth-moment algebra, and the shock moments enter through the quadrature rather than
# through hand-coded `vectorMom4`/`vectorMom6` tables.
#
# `E_ε[Ξ(ε)]` is not zero: the `ε²·xf` terms inside the `xf⊗xf⊗xf` block make ξ correlated
# with `z_t`. That correlation is exactly Andreasen's `BCov_xiLeadS_z`, and dropping it — as
# this package did before — biases the third-order variance.
#
# Since `Var(ξ)` depends on `E[z̃z̃']`, which depends on `Var(z)`, the pair is a fixed point;
# `_pss_moments_3rd` iterates it to convergence (a contraction for a stable `A`).

"""
    _pss_xi_matrix(pss, u, hxx_xx) → Ξ

The matrix `Ξ(ε)` with `ξ_{t+1} = Ξ(ε)·[1; z_t]`, for the shock contribution `u = η_x·ε`.

Blocks, writing `a = hx·xf` and expanding `z_{t+1}` minus its `c + A·z_t` part:

| block | contribution |
|---|---|
| `xf` | `u` |
| `xs` | 0 — the whole update is in `A` and `c` |
| `xf⊗xf` | `[hx⊗u + u⊗hx]·xf + (u⊗u − E[u⊗u])` |
| `xrd` | 0 |
| `xf⊗xs` | `(u⊗hx)·xs + (u⊗½hxx)·(xf⊗xf) + u⊗½hσσ` |
| `xf⊗xf⊗xf` | `[hx⊗hx⊗u + hx⊗u⊗hx + u⊗hx⊗hx]·(xf⊗xf)` `+ [hx⊗u⊗u + u⊗hx⊗u + u⊗u⊗hx]·xf` `+ (u⊗u⊗u − E[u⊗u⊗u])` |

The Kronecker identities used are `(A⊗b)·x = (A·x)⊗b` for a column `b`, and
`(A⊗B⊗C)·(x⊗y⊗z) = (Ax)⊗(By)⊗(Cz)` with a scalar slot when the middle factor is a column.
"""
function _pss_xi_matrix(pss::PrunedStateSpace{T}, u::AbstractVector{T},
                        hxx_xx::AbstractMatrix{T}) where {T}
    nx = pss.nx
    nz = 3 * nx + 2 * nx^2 + nx^3
    hx = pss.hx_state
    U = reshape(u, nx, 1)
    hss2 = reshape(T(0.5) .* pss.hss, nx, 1)

    r1 = 1:nx
    r2 = nx+1:2*nx
    r3 = 2*nx+1:2*nx+nx^2
    r5 = 3*nx+nx^2+1:3*nx+2*nx^2
    r6 = 3*nx+2*nx^2+1:nz
    # columns of z̃: 1 is the constant, then 1 + (z index)
    c1 = 1
    cz(rng) = (first(rng) + 1):(last(rng) + 1)

    X = zeros(T, nz, nz + 1)

    # xf
    X[r1, c1] = u

    # xf⊗xf
    X[r3, cz(r1)] = kron(hx, U) + kron(U, hx)
    X[r3, c1] = kron(u, u) - kron(pss.eta_x, pss.eta_x) * vec(Matrix{T}(I, pss.n_eps, pss.n_eps))

    # xf⊗xs
    X[r5, cz(r2)] = kron(U, hx)
    X[r5, cz(r3)] = kron(U, T(0.5) * hxx_xx)
    X[r5, c1] = kron(u, vec(hss2))

    # xf⊗xf⊗xf
    X[r6, cz(r3)] = kron(kron(hx, hx), U) + kron(hx, kron(U, hx)) + kron(U, kron(hx, hx))
    X[r6, cz(r1)] = kron(hx, kron(U, U)) + kron(U, kron(hx, U)) + kron(U, kron(U, hx))
    X[r6, c1] = kron(u, kron(u, u))       # E[u⊗u⊗u] = 0 for symmetric shocks

    return X
end

"""
    _pss_inov_moments_3rd(pss, E_z, Var_z, hxx_xx; n_gh=4) → (Var_inov, BCov, Xbar, S1)

Exact `Var(ξ)`, `Cov(ξ_{t+1}, z_t)`, `Xbar = E_ε[Ξ(ε)]` and `S1 = Cov(z_t, ε_t)` for the
third-order augmented system, given the current augmented mean and variance. `Xbar` and `S1`
drive the autocovariance recursion.

The `ε` integral is a Gauss-Hermite tensor rule with `n_gh` nodes per shock. `ξξ'` is a
polynomial of degree ≤ 6 in `ε`, and an `m`-node rule is exact through degree `2m − 1`, so
`n_gh = 4` integrates it exactly — this is not an approximation that tightens with more nodes.
Cost is `n_gh^{n_ε}` evaluations, trivial for the shock counts DSGE models carry.
"""
function _pss_inov_moments_3rd(pss::PrunedStateSpace{T}, E_z::AbstractVector{T},
                               Var_z::AbstractMatrix{T}, hxx_xx::AbstractMatrix{T};
                               n_gh::Int=4) where {T}
    nz = length(E_z)
    nodes, weights = _gauss_hermite_scaled(n_gh, Matrix{T}(I, pss.n_eps, pss.n_eps))

    # E[z̃z̃'] and Cov(z̃, z) with z̃ = [1; z]
    Mzz = zeros(T, nz + 1, nz + 1)
    Mzz[1, 1] = one(T)
    Mzz[1, 2:end] = E_z
    Mzz[2:end, 1] = E_z
    Mzz[2:end, 2:end] = Var_z + E_z * E_z'
    Czz = vcat(zeros(T, 1, nz), Var_z)          # Cov([1; z], z)

    E_ztilde = vcat(one(T), E_z)
    Var_inov = zeros(T, nz, nz)
    Xbar = zeros(T, nz, nz + 1)
    S1 = zeros(T, nz, pss.n_eps)              # Cov(z_t, ε_t)
    for (q, w) in enumerate(weights)
        e = Vector{T}(nodes[q, :])
        X = _pss_xi_matrix(pss, pss.eta_x * e, hxx_xx)
        Var_inov .+= w .* (X * Mzz * X')
        Xbar .+= w .* X
        S1 .+= w .* ((X * E_ztilde) * e')
    end
    Var_inov = (Var_inov + Var_inov') / 2
    # ξ is centred by construction, so E[ξξ'] IS Var(ξ); Cov(ξ,z) = E_ε[Ξ]·Cov(z̃,z).
    return (Var_inov, Xbar * Czz, Xbar, S1)
end

"""
    _pss_augmented_3rd(pss, A, c, E_z, hxx_xx; tol=1e-12, maxiter=200) → (Var_z, BCov, Xbar, S1)

Unconditional variance of the third-order augmented state, simulation-free.

`Var(ξ)` depends on `E[z̃z̃']`, which depends on `Var(z)`, so the two are solved as a fixed
point: start from `Var(z) = 0`, compute the innovation moments, solve

```
Var(z) = A·Var(z)·A' + Var(ξ) + Cov(ξ,z)·A' + A·Cov(ξ,z)'
```

and repeat until `Var(z)` stops moving. The `Cov(ξ,z)` terms are what the previous
implementation omitted; they are nonzero because the `ε²·xf` terms in the `xf⊗xf⊗xf` block
correlate the innovation with the state it is added to.
"""
function _pss_augmented_3rd(pss::PrunedStateSpace{T}, A::AbstractMatrix{T},
                            c::AbstractVector{T}, E_z::AbstractVector{T},
                            hxx_xx::AbstractMatrix{T};
                            tol::Real=1e-12, maxiter::Int=200, n_gh::Int=4) where {T}
    nz = length(E_z)
    Var_z = zeros(T, nz, nz)
    BCov = zeros(T, nz, nz)
    Xbar = zeros(T, nz, nz + 1)
    S1 = zeros(T, nz, pss.n_eps)
    for iter in 1:maxiter
        Var_inov, BCov, Xbar, S1 = _pss_inov_moments_3rd(pss, E_z, Var_z, hxx_xx; n_gh=n_gh)
        const_term = Var_inov + BCov * A' + A * BCov'
        const_term = (const_term + const_term') / 2
        Var_new = _dlyap(Matrix{T}(A), Matrix{T}(const_term); warn_label="order-3 pruned moments")
        delta = maximum(abs, Var_new - Var_z) / max(one(T), maximum(abs, Var_new))
        Var_z = (Var_new + Var_new') / 2
        delta < tol && return (Var_z, BCov, Xbar, S1)
    end
    @warn "Third-order pruned variance fixed point did not converge in $maxiter iterations; " *
          "the unconditional moments may be inaccurate." maxlog = 1
    return (Var_z, BCov, Xbar, S1)
end
