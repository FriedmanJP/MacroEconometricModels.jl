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
    sda(io0, io1; method=:additive, factors=:LY, average=:two_polar) -> SDAResult

Decompose the change in gross output `x = L y` between two periods into a
technology (`ΔL`) effect and a final-demand (`Δy`) effect.

- `method=:additive` uses the Dietzenbacher & Los (1998) two-polar average,
  which is exact (zero residual).
- `method=:multiplicative` returns the analogous ratio decomposition.

`factors=:LY` selects the two-factor `L`/`Y` decomposition (a stable hook for
finer splits); `average=:two_polar` selects the averaging scheme.
"""
function sda(io0::IOData, io1::IOData; method::Symbol=:additive,
             factors::Symbol=:LY, average::Symbol=:two_polar)
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
        x0 = L0 * y0; x1 = L1 * y1
        ratio = x1 ./ max.(x0, eps())
        L_eff = (L1 * y0) ./ max.(x0, eps())
        Y_eff = ratio ./ max.(L_eff, eps())
        return SDAResult(Dict(:L => L_eff, :Y => Y_eff), ratio,
                         ratio .- (L_eff .* Y_eff), :multiplicative)
    else
        throw(ArgumentError("method must be :additive or :multiplicative"))
    end
end
