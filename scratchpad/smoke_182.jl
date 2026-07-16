using MacroEconometricModels, Random, DataFrames, LinearAlgebra, Statistics
# AB-consistent DGP: dynamic panel y_it = ρ y_{i,t-1} + β x_it + α_i + ε_it, ε iid
rng = MersenneTwister(700); N=100; Tt=20; ρ=0.3; βx=0.5
ids=Int[]; ts=Int[]; ys=Float64[]; xs=Float64[]
for i in 1:N
    α=randn(rng); y=0.0
    for t in 1:Tt
        x=randn(rng); ε=0.3*randn(rng)
        y = t==1 ? α+βx*x+ε : ρ*y+βx*x+α+ε
        push!(ids,i); push!(ts,t); push!(ys,y); push!(xs,x)
    end
end
pd = xtset(DataFrame(id=ids, t=ts, y=ys, x=xs), :id, :t)
m = estimate_xtreg(pd, :y, [:x]; model=:ab)
println("method=", m.method, "  coef=", round.(coef(m),digits=3))
V = vcov(m); println("vcov size=", size(V), " symmetric? ", isapprox(V,V'), " offdiag nonzero? ", any(abs.(V.-Diagonal(diag(V))).>0))
println("diag(V)≈se².? ", isapprox(diag(V), stderror(m).^2))
d = m.dynamic_diagnostics
println("AR1 z=", round(d.ar1,digits=3), " p=", round(d.ar1_p,digits=4), " | AR2 z=", round(d.ar2,digits=3), " p=", round(d.ar2_p,digits=4))
println("Hansen J=", round(d.hansen,digits=2), " df=", d.hansen_df, " p=", round(d.hansen_p,digits=4), " ninst=", d.n_instruments)
println("AB test order=2: ", arellano_bond_ar_test(m; order=2))
io=IOBuffer(); show(io,m); s=String(take!(io)); println("show has AR(2)? ", occursin("AR(2)",s), " Hansen? ", occursin("Hansen",s))
