# Design: VFI Solver + Performance Optimizations (#78 + #79)

## Decisions

- VFI reuses existing Chebyshev grid/basis/quadrature infrastructure (no new grid types)
- Bellman operator constructed from existing `residual_fns` (no new `DSGESpec` fields)
- Howard improvement steps via `howard_steps::Int=0` kwarg (k=0 pure VFI, k>0 modified policy iteration)
- Anderson acceleration for VFI and PFI (`anderson_m::Int=0`), not collocation (Newton-based)
- Threading opt-in via `threaded::Bool=false` on outer grid-point loop only
- Benchmarks in standalone scripts, not in test suite
- No GPU (deferred)

## VFI Algorithm

1. Linearize model, compute state bounds (same as collocation/PFI)
2. Build Chebyshev grid (tensor or Smolyak)
3. Initialize value function coefficients from first-order perturbation
4. Bellman iteration loop:
   - At each grid point, solve Euler residuals for optimal policy (Newton on `residual_fns`)
   - Compute implied value from policy + continuation value via quadrature
   - Howard steps: if `howard_steps > 0`, do k rounds of policy evaluation (hold policy fixed, update value)
   - Refit Chebyshev coefficients via least squares
   - Sup-norm convergence on value function
   - Anderson acceleration: if `anderson_m > 0`, mix last m coefficient vectors
5. Return `ProjectionSolution{T}` with `method=:vfi`

## VFI Signature

```julia
function vfi_solver(spec::DSGESpec{T};
                    degree::Int=5,
                    grid::Symbol=:auto,
                    smolyak_mu::Int=3,
                    quadrature::Symbol=:auto,
                    n_quad::Int=5,
                    scale::Real=3.0,
                    tol::Real=1e-8,
                    max_iter::Int=1000,
                    damping::Real=1.0,
                    howard_steps::Int=0,
                    anderson_m::Int=0,
                    threaded::Bool=false,
                    verbose::Bool=false,
                    initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing
                    ) where {T<:AbstractFloat}
```

## Performance Optimizations (all three solvers)

| Optimization | Scope | Approach |
|---|---|---|
| Threading | VFI, PFI, collocation | `threaded::Bool=false` kwarg; `Threads.@threads` on grid-point loop |
| Anderson acceleration | VFI, PFI | `anderson_m::Int=0` kwarg; `_anderson_step` utility |
| Pre-allocation | All three | Allocate buffers once before loop, reuse via in-place ops |
| In-place ops | All three | `mul!`, `ldiv!` in hot paths |
| Quadrature caching | All three | Compute nodes/weights once (already mostly done) |
| Type stability | All three | `@code_warntype` audit; eliminate `Any`-typed containers |
| `@inbounds`/`@simd` | Inner loops | After correctness verification |

## Anderson Acceleration

Shared utility in `src/dsge/anderson.jl`:

```julia
function _anderson_step(history::Vector{Vector{T}}, residuals::Vector{Vector{T}}, m::Int) where T
    # min ||sum alpha_i r_i||^2  s.t. sum alpha_i = 1
    # return mixed iterate: x_new = sum alpha_i (x_i + r_i)
end
```

## File Changes

| File | Change |
|---|---|
| `src/dsge/vfi.jl` | New: VFI solver |
| `src/dsge/anderson.jl` | New: Anderson acceleration utility |
| `src/dsge/pfi.jl` | Add `anderson_m`, `threaded` kwargs; pre-allocation; in-place ops |
| `src/dsge/projection.jl` | Add `threaded` kwarg; pre-allocation; in-place ops; type stability |
| `src/dsge/gensys.jl` | Add `:vfi` dispatch in `solve()` |
| `src/MacroEconometricModels.jl` | Include `vfi.jl`, `anderson.jl`; export `vfi_solver` |
| `test/dsge/test_dsge.jl` | VFI tests |
| `benchmarks/bench_nonlinear.jl` | New: benchmark script |

## Tests

- VFI converges on linear AR(1) (recovers linear policy)
- VFI agrees with PFI and projection within tolerance
- Howard steps reduce iteration count
- Anderson acceleration reduces iteration count (VFI and PFI)
- `threaded=true` matches `threaded=false` results
- Damping works
- Smolyak grid option works
- `method=:vfi` dispatch through `solve()`

## References

- Stokey, Lucas, Prescott (1989) — contraction mapping, Bellman operator
- Howard (1960) — policy improvement steps
- Walker & Ni (2011) — Anderson acceleration
- Judd (1998) — numerical methods, Chebyshev approximation
- Santos & Rust (2003) — convergence properties of policy iteration
