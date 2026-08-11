# show.jl — display methods for IO types (Stata/EViews publication style)

function Base.show(io::IO, d::IOData)
    println(io, "IOData: $(nsectors(d)) sector(s) × $(nregions(d)) region(s)")
    !isempty(d.source) && println(io, "  source: $(d.source)")
    d.year !== nothing && println(io, "  year:   $(d.year)")
    nshow = min(6, length(d.sectors))
    suffix = length(d.sectors) > nshow ? " …" : ""
    println(io, "  sectors: ", join(d.sectors[1:nshow], ", "), suffix)
    isempty(d.extensions) ||
        println(io, "  extensions: ", join(sort(collect(keys(d.extensions))), ", "))
end

function Base.show(io::IO, m::IOMultipliers)
    data = hcat(m.sectors, _fmt.(m.values))
    _pretty_table(io, data;
        title="$(m.type) $(m.kind) multipliers",
        column_labels=["Sector", "Multiplier"], alignment=[:l, :r])
end

function Base.show(io::IO, lk::LinkageResult)
    data = hcat(lk.sectors, _fmt.(lk.Ui), _fmt.(lk.Uj), string.(lk.classification))
    _pretty_table(io, data;
        title="Linkages (Rasmussen dispersion indices)",
        column_labels=["Sector", "U_i (backward)", "U_j (forward)", "Class"],
        alignment=[:l, :r, :r, :l])
end

function Base.show(io::IO, m::LeontiefModel)
    println(io, "LeontiefModel ($(length(m.x)) sectors) — Leontief inverse L:")
    show(io, "text/plain", round.(m.L; digits=4))
    println(io)
end

function Base.show(io::IO, m::GhoshModel)
    println(io, "GhoshModel ($(length(m.x)) sectors) — Ghosh inverse G:")
    show(io, "text/plain", round.(m.G; digits=4))
    println(io)
end

function Base.show(io::IO, r::ExtractionResult)
    println(io, "Hypothetical extraction of sector(s) $(r.extracted)")
    println(io, "  total output loss: $(_fmt(r.total_loss))")
end

function Base.show(io::IO, r::SDAResult)
    println(io, "Structural Decomposition Analysis ($(r.method))")
    for k in sort(collect(keys(r.effects)))
        println(io, "  $(k) effect: ", _fmt.(r.effects[k]))
    end
end

function Base.show(io::IO, bf::BaqaeeFarhiResult)
    data = hcat(bf.sectors, _fmt.(bf.domar), _fmt.(bf.influence),
                _fmt.(bf.upstreamness))
    _pretty_table(io, data;
        title="Baqaee & Farhi (2019) IO decomposition",
        column_labels=["Sector", "Domar λ", "Influence", "Upstreamness"],
        alignment=[:l, :r, :r, :r])
end

function Base.show(io::IO, net::ProductionNetwork)
    println(io, "ProductionNetwork ($(net.nests)): n=$(net.n) sectors, ",
            "M=$(net.M) producers, F=$(net.F) factors")
    nshow = min(6, net.n)
    λ_out = [net.lambda[g] for g in net.outer_nodes]
    data = hcat(net.io.sectors[1:nshow], _fmt.(λ_out[1:nshow]))
    _pretty_table(io, data;
        title="Base Domar weights (real sectors)",
        column_labels=["Sector", "λ̃"], alignment=[:l, :r])
    Λ = net.lambda[net.M+2:net.M+1+net.F]
    println(io, "  factor shares Λ̃: ", _fmt.(Λ),
            "  (sum = $(_fmt(sum(Λ))))")
    println(io, "  λ̃₁ (household) = $(_fmt(net.lambda[1]))")
end

function Base.show(io::IO, eq::BFEquilibrium)
    status = eq.converged ? "converged" : "NOT CONVERGED"
    println(io, "BFEquilibrium [$status]  dlogY = $(_fmt(eq.dlogY))  ",
            "hulten = $(_fmt(eq.hulten))")
    println(io, "  iterations=$(eq.iterations)  residual=$(_fmt(eq.residual))")
    nshow = min(6, length(eq.sectors))
    data = hcat(eq.sectors[1:nshow], _fmt.(eq.dlog_p[1:nshow]),
                _fmt.(eq.dlog_x[1:nshow]))
    _pretty_table(io, data;
        title="Real-sector price & output changes",
        column_labels=["Sector", "dlog p", "dlog x"],
        alignment=[:l, :r, :r])
    println(io, "  factor wages w: ", _fmt.(eq.w))
    println(io, "  factor shares Λ: ", _fmt.(eq.Lambda))
end

function Base.show(io::IO, fp::FootprintResult)
    println(io, "Footprint ($(fp.name)) — consumption-based account")
    data = hcat(fp.stressors, [_fmt(sum(fp.total[i, :])) for i in 1:length(fp.stressors)])
    _pretty_table(io, data;
        column_labels=["Stressor", "Total"], alignment=[:l, :r])
end
