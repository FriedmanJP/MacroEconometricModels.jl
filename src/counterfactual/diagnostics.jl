# Counterfactual module — spanning + forecast-sufficiency diagnostics
# (CF-19, #399)
#
# Two diagnostics turn the module from a calculator into a credible tool:
# 1. spanning_diagnostic — CMW's title question "(when) do we need structural
#    models?": if the empirics-only counterfactual coincides with the
#    model-extrapolated one, the model choice is irrelevant for THIS
#    counterfactual. The determinant in practice is the persistence of the
#    object being offset (transitory -> spanned; persistent -> unspanned).
# 2. forecast_sufficiency — the invertibility laboratory (CMW Fig. 1):
#    second-moment/historical counterfactuals need Wold forecasts ≈
#    full-information forecasts; invertibility is sufficient, not necessary.

"""
    spanning_diagnostic(base, ce_emp, ce_full, policy; draws=:auto, tol=0.1,
                        n_sim=200, quantiles=(0.16, 0.5, 0.84),
                        rng=Random.default_rng()) -> SpanningDiagnostic

Run the same counterfactual twice — with the thin empirical container
`ce_emp` (LS projection) and with the square model menu `ce_full` (exact) —
from one `base` and `policy` (`PolicyRule` or `PolicyLoss`), and quantify the
disagreement:

- `gap`/`gap_rel`: per-outcome max-abs and relative-L2 path gaps.
- `loading_inside`: project the exact solve's required instrument-space
  perturbation `Δz = Θ_z^full·ν*` onto the column space of the empirical
  instrument responses (QR least squares — rule-free, instrument-path space);
  `≈ 1` means the counterfactual only needs empirically-spanned instrument
  movements (falls back to outcome space when no instruments are mapped).
- `spanned = all(gap_rel .< tol) && loading_inside > 1 − tol` — a reporting
  convenience; the raw numbers are the result.
- `draws = :auto` adds pairwise-independent draw bands on the gap when BOTH
  containers carry draws (expensive; `n_sim` resamples).
"""
function spanning_diagnostic(base::BaselinePath{T}, ce_emp::PolicyCausalEffects{T},
                             ce_full::PolicyCausalEffects{T},
                             policy::Union{PolicyRule{T},PolicyLoss{T}};
                             draws::Symbol=:auto, tol::Real=0.1, n_sim::Int=200,
                             quantiles::Union{Tuple,AbstractVector}=(0.16, 0.5, 0.84),
                             seed::Union{Integer,Nothing}=nothing,
                             rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    is_square(ce_full) || throw(ArgumentError(
        "ce_full must be a square (model-implied) container; got n_s = $(n_shocks(ce_full)) < H = $(ce_full.H)"))
    ce_emp.H == ce_full.H || throw(ArgumentError(
        "container horizons differ: emp H = $(ce_emp.H) vs full H = $(ce_full.H)"))
    ce_emp.outcomes == ce_full.outcomes || throw(ArgumentError(
        "containers must map the same outcomes, got $(ce_emp.outcomes) vs $(ce_full.outcomes)"))
    ce_emp.instruments == ce_full.instruments || throw(ArgumentError(
        "containers must map the same instruments, got $(ce_emp.instruments) vs $(ce_full.instruments)"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))

    solve1(ce) = _suppress_warnings() do
        _cf18_solve(base, ce, policy; draws=:off, z_wedge=nothing)
    end
    pc_e = solve1(ce_emp)
    pc_f = solve1(ce_full)

    n_x = length(ce_emp.outcomes)
    tiny = floatmin(T)^(1 // 4)
    gap = [maximum(abs.(pc_e.x_cf[i] - pc_f.x_cf[i])) for i in 1:n_x]
    # normalize by the larger of the exact counterfactual and the baseline
    # scale: strict-targeting rules drive x_cf_full to ~0, and a pure
    # ‖·‖/‖x_cf_full‖ ratio would be ill-defined exactly when the rule works
    gap_rel = [norm(pc_e.x_cf[i] - pc_f.x_cf[i]) /
               max(norm(pc_f.x_cf[i]), norm(pc_f.x_base[i]), tiny) for i in 1:n_x]

    # loading decomposition in instrument-path space (outcome space fallback)
    use_z = !isempty(ce_emp.instruments)
    S = use_z ? reduce(vcat, ce_emp.Theta_z) : reduce(vcat, ce_emp.Theta_x)
    dz = use_z ? reduce(vcat, (M * pc_f.nu for M in ce_full.Theta_z)) :
         reduce(vcat, (M * pc_f.nu for M in ce_full.Theta_x))
    loading = if norm(dz) <= tiny
        one(T)                       # nothing needs to move: trivially spanned
    else
        proj = S * (S \ dz)
        min(norm(proj) / norm(dz), one(T) + T(1e-10))
    end

    spanned = all(g -> g < T(tol), gap_rel) && loading > one(T) - T(tol)

    # draw bands on the gap (pairwise-independent resampling)
    nd_e = n_draws(ce_emp)
    nd_f = n_draws(ce_full)
    use_draws = draws == :on || (draws == :auto && nd_e > 0 && nd_f > 0)
    use_draws && (nd_e == 0 || nd_f == 0) && throw(ArgumentError(
        "draws = :on requires draws on BOTH containers"))
    bands_gap = nothing
    if use_draws
        qlev = collect(T, quantiles)
        H = ce_emp.H
        gaps = [Matrix{T}(undef, H, n_sim) for _ in 1:n_x]
        keep = falses(n_sim)
        _suppress_warnings() do
            for s in 1:n_sim
                de = rand(rng, 1:nd_e)
                df = rand(rng, 1:nd_f)
                slice(ce, d) = PolicyCausalEffects{T}(ce.outcomes, ce.instruments,
                    [Matrix{T}(ce.Theta_x_draws[i][:, :, d]) for i in eachindex(ce.outcomes)],
                    [Matrix{T}(ce.Theta_z_draws[k][:, :, d]) for k in eachindex(ce.instruments)],
                    nothing, nothing, ce.H, ce.shock_labels, ce.source)
                try
                    pe = solve1(slice(ce_emp, de))
                    pf = solve1(slice(ce_full, df))
                    for i in 1:n_x
                        gaps[i][:, s] = pe.x_cf[i] - pf.x_cf[i]
                    end
                    keep[s] = true
                catch
                    keep[s] = false
                end
            end
        end
        used = findall(keep)
        if !isempty(used)
            bands_gap = [Matrix{T}([quantile(@view(gaps[i][h, used]), q)
                                    for h in 1:ce_emp.H, q in qlev]) for i in 1:n_x]
        end
    end

    result = SpanningDiagnostic{T}(gap, gap_rel, loading, pc_e.rel_residual, spanned,
                                   copy(ce_emp.outcomes), pc_e.x_cf, pc_f.x_cf, bands_gap)
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("n_sim" => n_sim, "tol" => Float64(tol))))
end

"""
    forecast_sufficiency(sol::DSGESolution, observables; H=40) -> ForecastSufficiency

Population forecast-sufficiency check (CMW §2.3, Fig. 1): given a linear
solution `y_t = G1·y_{t−1} + impact·ε_t`, compare the `h`-step forecast-error
variances of the chosen `observables` under (i) full information (all states
observed) and (ii) their own innovations (Wold) representation, obtained from
the steady-state Kalman filter of the observables-only measurement (Riccati
fixed point by filter iteration).

- `fev_ratio[h, j] = FEV_wold(h)_j / FEV_full(h)_j ≥ 1` — near 1 means the
  observables forecast (almost) as well as the full state, which is what
  second-moment and historical counterfactuals actually require.
- `invertible` is the exact one-step condition
  `max |one_step_ratio − 1| < 1e-6`; invertibility is sufficient, NOT
  necessary (badly non-invertible sets can still pass on `fev_ratio`).

No data enters — that is the diagnostic's value (a laboratory) and its limit
(model-dependent).
"""
function forecast_sufficiency(sol::DSGESolution{T},
                              observables::AbstractVector{Symbol};
                              H::Int=40) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    spec = sol.spec
    base_vars = spec.augmented ? spec.original_endog : spec.endog
    isempty(observables) && throw(ArgumentError(
        "observables: expected at least one observable"))
    idx = Int[]
    for s in observables
        i = findfirst(==(s), spec.endog)
        i === nothing && throw(ArgumentError(
            "observable :$s not found among the solution's endogenous $(base_vars)"))
        push!(idx, i)
    end
    n = size(sol.G1, 1)
    n_obs = length(idx)
    A = sol.G1
    B = sol.impact
    C = zeros(T, n_obs, n)
    for (r, i) in enumerate(idx)
        C[r, i] = one(T)
    end
    Q = B * B'

    # full-information h-step FEV: Σ_{ℓ<h} C A^ℓ Q A^ℓ' C'
    fev_full = Matrix{T}(undef, H, n_obs)
    acc = zeros(T, n, n)
    Al = Matrix{T}(I, n, n)
    for h in 1:H
        acc .+= Al * Q * Al'
        fev_full[h, :] = diag(C * acc * C')
        Al = A * Al
    end

    # steady-state Kalman filter of the observables-only measurement:
    # P ← A P A' − A P C'(C P C')⁻¹ C P A' + Q, iterated to a fixed point
    P = copy(Q)
    converged = false
    for _ in 1:20_000
        CPC = C * P * C'
        Kg = (A * P * C') * robust_inv(Hermitian(Matrix{T}(CPC)))
        Pn = A * P * A' - Kg * (C * P * A') + Q
        Pn = (Pn + Pn') / 2
        if maximum(abs.(Pn - P)) < T(1e-12)
            P = Pn
            converged = true
            break
        end
        P = Pn
    end
    converged || @warn "forecast_sufficiency: Riccati iteration did not fully converge (tol 1e-12) — results carry that tolerance"
    Sigma_i = Matrix{T}((C * P * C' + (C * P * C')') / 2)   # innovations covariance
    Kg = (A * P * C') * robust_inv(Hermitian(Sigma_i))

    # innovations representation: y_t = C x̂ + e, x̂' = A x̂ + K e;
    # Wold h-step FEV = Σ_{ℓ<h} Ψ_ℓ Σ_i Ψ_ℓ' with Ψ_0 = I, Ψ_ℓ = C A^{ℓ−1} K
    fev_wold = Matrix{T}(undef, H, n_obs)
    accw = Matrix{T}(Sigma_i)
    fev_wold[1, :] = diag(accw)
    Psi_l = C * Kg                       # Ψ_1
    Apow = Matrix{T}(I, n, n)
    for h in 2:H
        accw = accw + Psi_l * Sigma_i * Psi_l'
        fev_wold[h, :] = diag(accw)
        Apow = A * Apow
        Psi_l = C * Apow * Kg
    end

    tinyv = floatmin(T)^(1 // 2)
    ratio = Matrix{T}(undef, H, n_obs)
    for h in 1:H, j in 1:n_obs
        r = fev_wold[h, j] / max(fev_full[h, j], tinyv)
        # clamp sub-1 numerical noise; leave genuine violations visible
        ratio[h, j] = (r < 1 && r > 1 - T(1e-8)) ? one(T) : r
    end
    one_step = ratio[1, :]
    invertible = maximum(abs.(one_step .- 1)) < T(1e-6)

    ForecastSufficiency{T}(collect(Symbol, observables), ratio, Vector{T}(one_step),
                           invertible, H)
end
