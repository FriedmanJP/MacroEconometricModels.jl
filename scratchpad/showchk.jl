using MacroEconometricModels, Random, DataFrames
rng=MersenneTwister(700); N=100; Tt=20
ids=Int[];ts=Int[];ys=Float64[];xs=Float64[]
for i in 1:N
  α=randn(rng); y=0.0
  for t in 1:Tt
    x=randn(rng); ε=0.3*randn(rng); y = t==1 ? α+0.5x+ε : 0.3y+0.5x+α+ε
    push!(ids,i);push!(ts,t);push!(ys,y);push!(xs,x)
  end
end
pd=xtset(DataFrame(id=ids,t=ts,y=ys,x=xs),:id,:t)
m=estimate_xtreg(pd,:y,[:x];model=:ab)
io=IOBuffer(); show(io,m); s=String(take!(io))
println(">>> nchars=",length(s)," lines=",count(==('\n'),s))
println(s)
