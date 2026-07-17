using MacroEconometricModels, DataFrames, Random
const M = MacroEconometricModels
ghnw=M._gauss_hermite_nodes_weights; agh=M._re_logit_agh_loglik; fixed=M._re_logit_loglik
rng=MersenneTwister(3072); N=150; Tp=8; n=N*Tp
ids=repeat(1:N,inner=Tp); a=repeat(randn(rng,N).*3.0,inner=Tp); x=randn(rng,n)
y=Float64.(rand(rng,n).<1.0./(1.0.+exp.(-(a.+0.5.*x))))
Xc=hcat(ones(n),x); ug=sort(unique(ids)); gobs=Dict(g=>findall(==(g),ids) for g in ug)
th=[0.0,0.5,log(3.0)]
n12,w12=ghnw(12); n60,w60=ghnw(60); n200,w200=ghnw(200)
a12=agh(th,y,Xc,ug,gobs,n12,w12); a60=agh(th,y,Xc,ug,gobs,n60,w60); a200=agh(th,y,Xc,ug,gobs,n200,w200)
f12=fixed(th,y,Xc,ids,ug,gobs,n12,w12)[1]; f200=fixed(th,y,Xc,ids,ug,gobs,n200,w200)[1]
println("AGH12=$a12  AGH60=$a60  AGH200=$a200")
println("fixed12=$f12  fixed200=$f200")
println("|AGH12-AGH60|=", abs(a12-a60), "  |fixed12-AGH200|=", abs(f12-a200))
println("|AGH12-AGH200|=", abs(a12-a200), "  AGH beats fixed? ", abs(f12-a200) > abs(a12-a200))
