# DSGE: Klein (2000) Solver — Design

**Issue**: #49
**Date**: 2026-02-25

## Goal

Add Klein (2000) generalized Schur decomposition solver as a third first-order solution method for DSGE models, accessible via `solve(spec; method=:klein)`.

## Reference

Klein, Paul. 2000. "Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model." *Journal of Economic Dynamics and Control* 24 (10): 1405–1423.

## Algorithm

Klein works on the system `A · E_t[y_{t+1}] = B · y_t + C · ε_t` using the generalized Schur (QZ) decomposition with eigenvalue reordering.

### Mapping from Sims Canonical Form

The existing `linearize()` produces `(Γ₀, Γ₁, C, Ψ, Π)` in Sims form:

```
Γ₀ · y_t = Γ₁ · y_{t-1} + C + Ψ · ε_t + Π · η_t
```

Klein's QZ decomposition operates on the pencil `(Γ₁, Γ₀)`:
- Generalized eigenvalues λ_i = T_ii / S_ii from `schur(Γ₁, Γ₀)`
- Stable: |λ| < 1 (predetermined dynamics)
- Unstable: |λ| ≥ 1 (jump variable dynamics)

### Predetermined Variable Detection

A variable is **predetermined** (state) if it has a non-zero column in Γ₁ (appears with `[t-1]` terms). Variables with zero columns in Γ₁ are **jump** (non-predetermined) variables.

```julia
function _count_predetermined(ld::LinearDSGE{T}) where {T}
    n = size(ld.Gamma1, 2)
    count(j -> any(x -> abs(x) > eps(T) * 100, ld.Gamma1[:, j]), 1:n)
end
```

### Blanchard-Kahn Condition

Unique solution exists iff: `n_stable == n_predetermined`
- `n_stable < n_predetermined` → no solution (explosive)
- `n_stable > n_predetermined` → indeterminate (multiple solutions)

### Solution Construction

1. Compute QZ: `F = schur(complex(Γ₁), complex(Γ₀))`
2. Reorder: stable eigenvalues first via `ordschur(F, |λ| < div)`
3. Partition: `Z = [Z₁₁ Z₁₂; Z₂₁ Z₂₂]` where Z₁₁ is n_pre × n_pre
4. State transition: `G1 = real(Z · (S⁻¹ · T) · Z⁻¹)` restricted to stable block
5. Impact: `impact = real(Z₁ · S₁₁⁻¹ · Q₁ · Ψ)` for shock loading

The output is the same `(G1, impact, C_sol, eu, eigenvalues)` tuple as gensys.

## Implementation

### New File: `src/dsge/klein.jl`

```julia
function klein(Gamma0::AbstractMatrix{T}, Gamma1::AbstractMatrix{T},
               Psi::AbstractMatrix{T}, n_predetermined::Int;
               div::Real=1.0) where {T<:AbstractFloat}
    → NamedTuple{(:G1, :impact, :C_sol, :eu, :eigenvalues)}
```

### Dispatcher Addition: `src/dsge/gensys.jl`

Add `:klein` case to `solve()`:

```julia
elseif method == :klein
    ld = linearize(spec)
    n_pre = _count_predetermined(ld)
    result = klein(ld.Gamma0, ld.Gamma1, ld.Psi, n_pre)
    return DSGESolution{T}(result.G1, result.impact, result.C_sol, result.eu,
                            :klein, result.eigenvalues, spec, ld)
```

### Include Order

After `blanchard_kahn.jl`, before `perfect_foresight.jl` in `MacroEconometricModels.jl`.

## Type Changes

None. `DSGESolution{T}` already has `method::Symbol` — Klein just adds `:klein` as a new value.

## Pipeline Impact

| Component | Change |
|---|---|
| `klein.jl` | New file |
| `solve()` dispatcher | Add `:klein` case |
| `MacroEconometricModels.jl` | Add `include("dsge/klein.jl")` |
| `linearize` | None |
| `simulate` / `irf` / `fevd` | None (operate on G1/impact) |
| `analytical_moments` | None |
| `estimate_dsge` | None (uses solve dispatcher) |
| `perfect_foresight` / `occbin` | None |
| `display.jl` | None (method symbol auto-displays) |

## Backward Compatibility

Fully backward compatible. `:gensys` remains the default. `:blanchard_kahn` unchanged. New `:klein` option is opt-in.

## Testing

1. **Equivalence**: AR(1), NK 3-equation — verify G1/impact match gensys and BK
2. **Predetermined detection**: `_count_predetermined` on known models
3. **BK condition**: eu flags correct for determined/indeterminate/explosive cases
4. **Downstream**: simulate/irf/fevd work with Klein solutions
5. **Augmented models**: Klein works with news shocks (issue #54 augmentation)
6. **Edge cases**: Singular Γ₀, purely forward-looking model, purely backward-looking model
