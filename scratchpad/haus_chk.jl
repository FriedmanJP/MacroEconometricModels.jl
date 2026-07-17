using MacroEconometricModels, DataFrames, Random, LinearAlgebra
const M = MacroEconometricModels
for (seed, corr) in ((1234, true), (777, false))
    rng=MersenneTwister(seed); N=50; Tp=20; n=N*Tp
    ids=repeat(1:N,inner=Tp); ts=repeat(1:Tp,N); α=repeat(randn(rng,N),inner=Tp)
    x1 = corr ? 0.5α .+ randn(rng,n) : randn(rng,n); x2=randn(rng,n)
    y = α .+ 2x1 .- x2 .+ 0.3randn(rng,n)
    pd=xtset(DataFrame(id=ids,t=ts,x1=x1,x2=x2,y=y),:id,:t)
    fe=estimate_xtreg(pd,:y,[:x1,:x2];model=:fe); re=estimate_xtreg(pd,:y,[:x1,:x2];model=:re)
    ht=hausman_test(fe,re)
    println("seed=$seed corr=$corr: stat=$(round(ht.statistic,digits=3)) df=$(ht.df) p=$(round(ht.pvalue,digits=4)) desc=$(ht.description)")
end
# helper unit checks
c1 = M._hausman_quadratic_form([1.0,2.0],[2.0 0.0;0.0 3.0]); println("PSD full-rank: ", c1, " want chi2=1/2+4/3=", 1/2+4/3, " df=2")
c2 = M._hausman_quadratic_form([1.0,1.0],[1.0 0.0;0.0 -0.5]); println("indefinite: ", c2)
c3 = M._hausman_quadratic_form([1.0,1.0],[1.0 0.0;0.0 0.0]); println("rank-def: ", c3)
