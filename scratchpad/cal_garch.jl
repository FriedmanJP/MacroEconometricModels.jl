using MacroEconometricModels, Random, Statistics, LinearAlgebra
const Mod = MacroEconometricModels
_std_innov(rng, innov) = innov === :t5 ?
    (randn(rng) / sqrt(sum(abs2, randn(rng,5))/5)) / sqrt(5/3) : randn(rng)
function simg(rng, n; omega=0.01, alpha1=0.05, beta1=0.90, mu=0.0, innov=:gauss)
    y=zeros(n); h=zeros(n); h[1]=omega/(1-alpha1-beta1); y[1]=mu+sqrt(h[1])*_std_innov(rng,innov)
    for t in 2:n
        h[t]=omega+alpha1*(y[t-1]-mu)^2+beta1*h[t-1]; y[t]=mu+sqrt(h[t])*_std_innov(rng,innov)
    end; y
end
# (c) correct-spec ratios
yg=simg(MersenneTwister(77),4000); mg=estimate_garch(yg,1,1)
sr=stderror(mg;cov_type=:robust); sh=stderror(mg;cov_type=:hessian)
println("(c) Gaussian n=4000 robust/hessian ratios: alpha=",round(sr[3]/sh[3],digits=3)," beta=",round(sr[4]/sh[4],digits=3))
# (d) fat tail
yt=simg(MersenneTwister(2024),3000; omega=0.02,alpha1=0.08,beta1=0.90,innov=:t5); mt=estimate_garch(yt,1,1)
srt=stderror(mt;cov_type=:robust); sht=stderror(mt;cov_type=:hessian)
println("(d) t5 n=3000: alpha SE robust=",round(srt[3],digits=4)," hess=",round(sht[3],digits=4)," rel=",round(abs(srt[3]-sht[3])/sht[3],digits=3))
println("(d) t5 n=3000: beta  SE robust=",round(srt[4],digits=4)," hess=",round(sht[4],digits=4)," rel=",round(abs(srt[4]-sht[4])/sht[4],digits=3))
println("(d) converged=",mt.converged)
# (e) MC dispersion
R=120; nrep=1200; ah=Float64[]; ser=Float64[]; seh=Float64[]; nconv=0
for r in 1:R
    yr=simg(MersenneTwister(3000+r),nrep; omega=0.02,alpha1=0.08,beta1=0.90,innov=:t5)
    mr=try estimate_garch(yr,1,1) catch; continue end
    mr.converged || continue
    a=stderror(mr;cov_type=:robust); b=stderror(mr;cov_type=:hessian)
    (all(isfinite,a)&&all(isfinite,b)) || continue
    global nconv+=1; push!(ah,mr.alpha[1]); push!(ser,a[3]); push!(seh,b[3])
end
smc=std(ah)
println("(e) nconv=",nconv," sigma_MC(alpha)=",round(smc,digits=4)," mean_SE_robust=",round(mean(ser),digits=4)," mean_SE_hess=",round(mean(seh),digits=4))
println("(e) |rob-mc|=",round(abs(mean(ser)-smc),digits=4)," |hess-mc|=",round(abs(mean(seh)-smc),digits=4)," hess<mc? ",mean(seh)<smc)
