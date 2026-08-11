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
    println(io, "Hypothetical extraction of sector(s) $(r.extracted)  ",
            "[mode=$(r.mode), share=$(_fmt(r.share))]")
    println(io, "  total output loss: $(_fmt(r.total_loss))  ",
            "($(_fmt_pct(r.loss_pct_go)) of GO, $(_fmt_pct(r.loss_pct_gdp)) of GDP)")
end

function Base.show(io::IO, r::PriceModelResult)
    data = hcat(r.sectors, _fmt.(r.dv), _fmt.(r.dp), _fmt.(r.p))
    _pretty_table(io, data;
        title="Price model ($(r.mode))",
        column_labels=["Sector", "Δv", "Δp", "p"],
        alignment=[:l, :r, :r, :r])
end

function Base.show(io::IO, r::ImpactResult)
    fix_note = isempty(r.fixed) ? "" : "  [mixed; fixed=$(r.fixed)]"
    println(io, "Impact ($(r.type) $(r.kind))$fix_note")
    println(io, "  total: $(_fmt(r.total))")
    data = hcat(r.sectors, _fmt.(r.dy), _fmt.(r.by_sector))
    _pretty_table(io, data;
        column_labels=["Sector", "Δy", "Impact"],
        alignment=[:l, :r, :r])
end

function Base.show(io::IO, r::NetworkStatsResult)
    println(io, "NetworkStats  Herfindahl(λ)=$(_fmt(r.herfindahl))  ",
            "mult. sd=$(_fmt(r.multiplier_dispersion))")
    data = hcat(r.sectors, _fmt.(r.domar), _fmt.(r.multipliers),
                _fmt.(r.upstreamness), _fmt.(r.downstreamness),
                _fmt.(r.in_degree), _fmt.(r.out_degree))
    _pretty_table(io, data;
        column_labels=["Sector", "Domar λ", "Mult.", "Upstr.", "Downstr.",
                       "In-deg", "Out-deg"],
        alignment=[:l, :r, :r, :r, :r, :r, :r])
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
    has_wedges = any(m -> m > 1 + 1e-14, net.mu)
    println(io, "ProductionNetwork ($(net.nests)): n=$(net.n) sectors, ",
            "M=$(net.M) producers, F=$(net.F) factors",
            has_wedges ? ", wedges μ" : "")
    nshow = min(6, net.n)
    λ_out = [net.lambda[g] for g in net.outer_nodes]
    λ_rev = [net.lambda_rev[g] for g in net.outer_nodes]
    μ_out = [net.mu[g - 1] for g in net.outer_nodes]
    if has_wedges
        data = hcat(net.io.sectors[1:nshow], _fmt.(λ_out[1:nshow]),
                    _fmt.(λ_rev[1:nshow]), _fmt.(μ_out[1:nshow]))
        _pretty_table(io, data;
            title="Base Domar weights (real sectors)",
            column_labels=["Sector", "λ̃ (cost)", "λ (rev)", "μ"],
            alignment=[:l, :r, :r, :r])
    else
        data = hcat(net.io.sectors[1:nshow], _fmt.(λ_out[1:nshow]))
        _pretty_table(io, data;
            title="Base Domar weights (real sectors)",
            column_labels=["Sector", "λ̃"], alignment=[:l, :r])
    end
    Λ = net.lambda[net.M+2:net.M+1+net.F]
    Λr = net.lambda_rev[net.M+2:net.M+1+net.F]
    println(io, "  factor shares Λ̃ (cost): ", _fmt.(Λ),
            "  (sum = $(_fmt(sum(Λ))))")
    if has_wedges
        println(io, "  factor shares Λ (rev):  ", _fmt.(Λr),
                "  (sum = $(_fmt(sum(Λr))))")
    end
    println(io, "  λ̃₁ (household) = $(_fmt(net.lambda[1]))")
end

function Base.show(io::IO, eq::BFEquilibrium)
    status = eq.converged ? "converged" : "NOT CONVERGED"
    println(io, "BFEquilibrium [$status]  dlogY = $(_fmt(eq.dlogY))  ",
            "hulten = $(_fmt(eq.hulten))")
    println(io, "  technology = $(_fmt(eq.technology))  ",
            "allocative = $(_fmt(eq.allocative))  ",
            "profit share = $(_fmt(eq.profit_share))")
    println(io, "  iterations=$(eq.iterations)  residual=$(_fmt(eq.residual))")
    nshow = min(6, length(eq.sectors))
    data = hcat(eq.sectors[1:nshow], _fmt.(eq.dlog_p[1:nshow]),
                _fmt.(eq.dlog_x[1:nshow]))
    _pretty_table(io, data;
        title="Real-sector price & output changes",
        column_labels=["Sector", "dlog p", "dlog x"],
        alignment=[:l, :r, :r])
    println(io, "  factor wages w: ", _fmt.(eq.w))
    println(io, "  factor shares Λ (rev): ", _fmt.(eq.Lambda))
end

function Base.show(io::IO, w::BFWedgeDecomp)
    println(io, "BFWedgeDecomp (B&F 2020 Theorem 1)")
    println(io, "  dlogY       = $(_fmt(w.dlogY))")
    println(io, "  technology  = $(_fmt(w.technology))")
    println(io, "  allocative  = $(_fmt(w.allocative))  ",
            "(μ: $(_fmt(w.allocative_mu)), factors: $(_fmt(w.allocative_factor)))")
    nshow = min(6, length(w.sectors))
    data = hcat(w.sectors[1:nshow], _fmt.(w.lambda_cost[1:nshow]),
                _fmt.(w.lambda_rev[1:nshow]), _fmt.(w.mu[1:nshow]))
    _pretty_table(io, data;
        title="Cost vs revenue Domar & markups",
        column_labels=["Sector", "λ̃ (cost)", "λ (rev)", "μ"],
        alignment=[:l, :r, :r, :r])
end

function Base.show(io::IO, bf::BFLocal)
    n = length(bf.first_order)
    has_H = !isempty(bf.second_order)
    println(io, "BFLocal ($(bf.nests)): n=$n sectors",
            has_H ? ", Hessian $(size(bf.second_order, 1))×$(size(bf.second_order, 2))" :
                    ", Hessian omitted")
    nshow = min(6, n)
    data = hcat(bf.sectors[1:nshow], _fmt.(bf.first_order[1:nshow]))
    _pretty_table(io, data;
        title="Hulten first-order (Domar λ̃)",
        column_labels=["Sector", "λ̃"], alignment=[:l, :r])
    println(io, "  factor shares Λ̃: ", _fmt.(bf.Lambda),
            "  (sum = $(_fmt(sum(bf.Lambda))))")
    if has_H
        println(io, "  tr(H) = $(_fmt(tr(bf.second_order)))  ",
                "‖H‖_F = $(_fmt(norm(bf.second_order)))")
    end
    if bf.elasticities !== nothing
        println(io, "  elasticities: dlogw/dlogA ",
                size(bf.elasticities.dlogw_dlogA), ", dlogp/dlogA ",
                size(bf.elasticities.dlogp_dlogA))
    end
end

function Base.show(io::IO, e::BFElasticities)
    println(io, "BFElasticities: F=$(size(e.dlogw_dlogA, 1)) factors × ",
            "n=$(size(e.dlogw_dlogA, 2)) sectors")
    nshow = min(6, length(e.sectors))
    # Show first factor's wage incidence across sectors
    if size(e.dlogw_dlogA, 1) >= 1
        data = hcat(e.sectors[1:nshow],
                    _fmt.(e.dlogw_dlogA[1, 1:nshow]),
                    _fmt.(e.dlogp_dlogA[1:nshow, 1]))
        fname = isempty(e.factor_names) ? "factor1" : e.factor_names[1]
        _pretty_table(io, data;
            title="Incidence (sample): ∂log $fname / ∂log A_j  &  ∂log p_j / ∂log A_1",
            column_labels=["Sector", "dlogw/dlogA", "dlogp/dlogA₁"],
            alignment=[:l, :r, :r])
    end
end

function Base.show(io::IO, sc::BFShockCurve)
    println(io, "BFShockCurve: sector=$(sc.sector) (index $(sc.sector_index)), ",
            "$(length(sc.shocks)) points")
    # Show endpoints and zero
    for (label, s, y, h, q) in (
            ("min", sc.shocks[1], sc.exact[1], sc.hulten[1], sc.second_order[1]),
            ("max", sc.shocks[end], sc.exact[end], sc.hulten[end], sc.second_order[end]))
        println(io, "  $label ΔlogA=$(_fmt(s)): exact=$(_fmt(y))  ",
                "Hulten=$(_fmt(h))  2nd=$(_fmt(q))")
    end
end

function Base.show(io::IO, fp::FootprintResult)
    println(io, "Footprint ($(fp.name)) — consumption-based account")
    data = hcat(fp.stressors, [_fmt(sum(fp.total[i, :])) for i in 1:length(fp.stressors)])
    _pretty_table(io, data;
        column_labels=["Stressor", "Total"], alignment=[:l, :r])
end

function Base.show(io::IO, fp::RegionalFootprintResult)
    println(io, "RegionalFootprint ($(fp.name)) — production vs consumption")
    n_s = length(fp.stressors)
    # One row per (stressor, region)
    rows = String[]
    prod = String[]
    cons = String[]
    for i in 1:n_s, r in 1:length(fp.regions)
        push!(rows, "$(fp.stressors[i]) / $(fp.regions[r])")
        push!(prod, _fmt(fp.production[i, r]))
        push!(cons, _fmt(fp.consumption[i, r]))
    end
    data = hcat(rows, prod, cons)
    _pretty_table(io, data;
        column_labels=["Stressor / Region", "Production", "Consumption"],
        alignment=[:l, :r, :r])
end

function Base.show(io::IO, vs::VerticalSpecialization)
    println(io, "VerticalSpecialization ($(vs.region))")
    println(io, "  gross exports:     $(_fmt(vs.gross_exports))")
    println(io, "  VS (foreign cont.): $(_fmt(vs.vs))  (share=$(_fmt(vs.vs_share)))")
    println(io, "  domestic content:  $(_fmt(vs.domestic_content))  (share=$(_fmt(vs.dc_share)))")
    println(io, "  VS1 (indirect):    $(_fmt(vs.vs1))")
end

function Base.show(io::IO, ed::ExportDecomposition)
    println(io, "ExportDecomposition KWW (2014) — $(ed.region)")
    println(io, "  gross exports: $(_fmt(ed.gross_exports))  VAX ratio: $(_fmt(ed.vax_ratio))")
    data = hcat(["DVA", "RDV", "FVA", "PDC"],
                _fmt.([ed.dva, ed.rdv, ed.fva, ed.pdc]))
    _pretty_table(io, data;
        column_labels=["Component", "Value"], alignment=[:l, :r])
    s = ed.dva + ed.rdv + ed.fva + ed.pdc
    println(io, "  sum DVA+RDV+FVA+PDC = $(_fmt(s))")
end
