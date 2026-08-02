# sda.jl — Structural Decomposition Analysis (Dietzenbacher & Los 1998)

"Result of a structural decomposition: per-determinant `effects`, the `total`
change, and the `residual`."
struct SDAResult
    effects::Dict{Symbol,Vector{Float64}}
    total::Vector{Float64}
    residual::Vector{Float64}
    method::Symbol
end

"""
    sda(io0, io1; method=:additive) -> SDAResult

Decompose the change in gross output `x = L y` between two periods into a
technology (`ΔL`) effect and a final-demand (`Δy`) effect (two-factor `L`/`Y`
split).

- `method=:additive` uses the Dietzenbacher & Los (1998) two-polar average,
  which is exact (zero residual up to floating-point noise).
- `method=:multiplicative` uses the two-polar geometric mean of the polar
  ratio decompositions. The residual is zero by construction of the geometric
  identity `ratio = L_eff ⊙ Y_eff` when both polars are averaged this way; it
  is reported for API symmetry with the additive branch.
"""
function sda(io0::IOData, io1::IOData; method::Symbol=:additive)
    L0 = leontief_inverse(io0); L1 = leontief_inverse(io1)
    y0 = vec(sum(io0.Y, dims=2)); y1 = vec(sum(io1.Y, dims=2))
    ΔL = L1 - L0; Δy = y1 - y0
    if method == :additive
        # Two-polar average: x = L y
        #   polar 1: ΔL·y0 + L1·Δy ;  polar 2: ΔL·y1 + L0·Δy ; average the two.
        L_eff = 0.5 .* (ΔL * y0 .+ ΔL * y1)
        Y_eff = 0.5 .* (L1 * Δy .+ L0 * Δy)
        total = L1 * y1 .- L0 * y0
        resid = total .- (L_eff .+ Y_eff)
        return SDAResult(Dict(:L => L_eff, :Y => Y_eff), total, resid, :additive)
    elseif method == :multiplicative
        # Two-polar geometric mean of ratio decompositions (Dietzenbacher & Los).
        # Polar 1 holds L at period 0 for the Y effect (and L at 1 for L effect on y0);
        # polar 2 swaps the time index. Geometric mean of the two polars.
        x0 = L0 * y0; x1 = L1 * y1
        epsT = eps(eltype(x0))
        ratio = x1 ./ max.(x0, epsT)
        L_p1 = (L1 * y0) ./ max.(x0, epsT)          # L change, y fixed at 0
        Y_p1 = (L0 * y1) ./ max.(x0, epsT)          # Y change, L fixed at 0
        L_p2 = (L1 * y1) ./ max.(L0 * y1, epsT)     # L change, y fixed at 1
        Y_p2 = (L1 * y1) ./ max.(L1 * y0, epsT)     # Y change, L fixed at 1
        L_eff = sqrt.(max.(L_p1 .* L_p2, zero(eltype(x0))))
        Y_eff = sqrt.(max.(Y_p1 .* Y_p2, zero(eltype(x0))))
        return SDAResult(Dict(:L => L_eff, :Y => Y_eff), ratio,
                         ratio .- (L_eff .* Y_eff), :multiplicative)
    else
        throw(ArgumentError("method must be :additive or :multiplicative"))
    end
end
