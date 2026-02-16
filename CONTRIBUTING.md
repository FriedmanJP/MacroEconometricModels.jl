# Contributing to MacroEconometricModels.jl

Thank you for your interest in contributing. This guide explains how to report bugs, request features, and submit pull requests.

## Reporting Bugs

Open an [issue](https://github.com/chung9207/MacroEconometricModels.jl/issues) with:

- A clear, descriptive title
- Minimal reproducible example (MWE)
- Expected vs. actual behavior
- Julia version (`versioninfo()`) and package version (`Pkg.status("MacroEconometricModels")`)

## Requesting New Features or Models

Open an [issue](https://github.com/chung9207/MacroEconometricModels.jl/issues) with:

- A description of the feature or econometric model
- Key references (papers, textbooks) for new models
- A brief explanation of why it would be useful

## Pull Requests

Pull requests are welcome for bug fixes and new models/features. Before starting significant work, please open an issue first to discuss the approach.

### Setup

```bash
git clone https://github.com/chung9207/MacroEconometricModels.jl.git
cd MacroEconometricModels.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Workflow

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Add tests in the appropriate `test/` subdirectory (mirror the `src/` structure)
4. Run the test suite and confirm all tests pass:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
5. Submit a pull request against `main`

### Guidelines

- Follow existing code conventions: `T<:AbstractFloat` type parameters, `robust_inv` over raw `inv`, `_` prefix for internal helpers
- Use `Optim.optimize` / `Optim.LBFGS()` (qualified calls, not bare names)
- Add docstrings for all public functions
- New models should include: types in `types.jl`, estimation, analysis functions, tests, and a documentation page
- Keep PRs focused — one feature or fix per PR

### Test Architecture

The test suite contains ~7,739 tests organized into 7 parallel groups (run by default):

| Group | Contents |
|-------|----------|
| **Core & Bayesian** | core/{aqua,coverage_gaps}, var/core_var, bvar/{bayesian,samplers,utils,minnesota,bgr} |
| **IRF & FEVD + PVAR** | var/{irf,irf_ci,statsapi,fevd,hd}, core/summary, vecm/test_vecm, pvar/test_pvar |
| **Factor Models** | factor/{factormodel,dynamicfactormodel,gdfm,factor_forecast} |
| **Local Projections** | lp/{lp,lp_structural,lp_forecast,lp_fevd} |
| **ARIMA & Utilities** | teststat/unitroot, arima/{arima,arima_coverage}, core/{utils,edge_cases,examples}, gmm/gmm, core/covariance, filters/filters, teststat/{model_comparison,granger}, data/test_data |
| **Non-Gaussian & Display** | teststat/normality, nongaussian/{svar,internals}, core/{display_backends,error_paths,internal_helpers}, var/{arias2018,uhlig} |
| **Volatility** | volatility/{volatility,volatility_coverage} — bottleneck group (SV Gibbs sampler) |

For sequential execution (useful for debugging):
```bash
MACRO_SERIAL_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

### Code Architecture

Key conventions to follow when contributing:

- **Include order matters**: Source files have strict dependency ordering in `MacroEconometricModels.jl` — see `CLAUDE.md` for the full chain
- **`@float_fallback` macro**: Generates `Float64` conversion methods for `AbstractMatrix` arguments. Write manual fallbacks for `AbstractVector` inputs
- **Optim qualification**: `Optim` is `import`ed, not `using`; always qualify calls as `Optim.optimize`, `Optim.LBFGS()`, etc.
- **Numerical safety**: Use `robust_inv(A)` instead of `inv(A)`, `safe_cholesky(A)` instead of `cholesky(A)`
- **Internal helpers**: Prefix with `_` (e.g., `_unpack_arma_params`, `_numerical_hessian`)
- **Type parameters**: Use `T<:AbstractFloat` throughout; accept `AbstractMatrix`/`AbstractVector` in public APIs
- **Variable naming**: Never use `eps` (shadows `Base.eps`) — use `resid` instead

### Branch Workflow

- Development happens on the `dev` branch
- Pull requests target `main`
- Always run the full test suite before submitting a PR
- The CI pipeline runs on Ubuntu, macOS, and Windows with Julia 1.12+

### Running Documentation Locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Code of Conduct

Be respectful and constructive. We are all here to build good econometric tools.

## License

By contributing, you agree that your contributions will be licensed under the [GNU General Public License v3.0](LICENSE).
In short, when you submit a contribution, you agree to grant the project a non-exclusive, worldwide, royalty-free license to use, reproduce, modify, and distribute your contribution under the terms of the GNU General Public License v3.0 (or any later version). You retain copyright ownership of your original contributions.
