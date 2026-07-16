using MacroEconometricModels, DataFrames, Random, LinearAlgebra
import ForwardDiff
rng=MersenneTwister(303); N=80; Tp=8; n=N*Tp
ids=repeat(1:N,inner=Tp); ts=repeat(1:Tp,N)
x1=randn(rng,n); α=repeat(randn(rng,N).*1.0,inner=Tp)
y=Float64.(rand(rng,n) .< 1.0 ./(1.0 .+ exp.(-(α .+ 0.7.*x1))))
pd=xtset(DataFrame(id=ids,t=ts,x1=x1,y=y),:id,:t)
mre=estimate_xtlogit(pd,:y,[:x1];model=:re)
println("RE: conv=$(mre.converged) β=$(round.(coef(mre),digits=3)) σu=$(round(mre.sigma_u,digits=3)) se=$(round.(stderror(mre),digits=3)) ll=$(round(loglikelihood(mre),digits=2))")
mcre=estimate_xtlogit(pd,:y,[:x1];model=:cre)
println("CRE: conv=$(mcre.converged) β=$(round.(coef(mcre),digits=3)) σu=$(round(mcre.sigma_u,digits=3)) se_ok=$(all(stderror(mcre).>0)) ll=$(round(loglikelihood(mcre),digits=2))")
# FOC check: gradient norm of nll at reported theta_hat
go = MacroEconometricModels._gauss_hermite_nodes_weights
nodes,weights = go(12)
X_c = hcat(ones(n), x1)
ug = sort(unique(ids)); go2=Dict(g=>findall(==(g),ids) for g in ug)
nll(th)= -MacroEconometricModels._re_logit_agh_loglik(th, y, X_c, ug, go2, nodes, weights)
th_hat = vcat(coef(mre), log(mre.sigma_u))
println("RE FOC grad norm at θ̂ = ", round(norm(ForwardDiff.gradient(nll, th_hat)),sigdigits=3))
