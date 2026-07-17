using MacroEconometricModels, Random, DataFrames, LinearAlgebra, Statistics
function mkdgp(; N=50, T_total=100, m=2, rng=MersenneTwister(400))
    A1 = 0.3*I(m) + 0.05*randn(rng,m,m); F=eigvals(A1)
    while maximum(abs.(F))>=0.95; A1*=0.8; F=eigvals(A1); end
    D=zeros(N*T_total,m)
    for i in 1:N
        mu=randn(rng,m); off=(i-1)*T_total; D[off+1,:]=mu+0.1*randn(rng,m)
        for t in 2:T_total; D[off+t,:]=mu+A1*D[off+t-1,:]+0.1*randn(rng,m); end
    end
    df=DataFrame(D,["y$i" for i in 1:m]); df.id=repeat(1:N,inner=T_total); df.time=repeat(1:T_total,outer=N)
    xtset(df,:id,:time)
end
pd=mkdgp(N=50,T_total=100)
mfe=estimate_pvar_feols(pd,1)
println("FE-OLS(T=100) diag Σ = ", round.(diag(mfe.Sigma),sigdigits=3), " (want ≈0.01)")
oirf=pvar_oirf(mfe,4); println("OIRF impact diag = ", round.([oirf[1,j,j] for j in 1:2],sigdigits=3), " (want ≈0.1)")
fe=pvar_fevd(mfe,4); println("FEVD row sums = ", [round(sum(fe[end,l,:]),sigdigits=4) for l in 1:2])
# GMM on shorter panel with capped lags (avoid instrument explosion)
pdg=mkdgp(N=50,T_total=30,rng=MersenneTwister(401))
mg=estimate_pvar(pdg,1; max_lag_endo=3); mf=estimate_pvar(pdg,1;transformation=:fod,max_lag_endo=3)
println("GMM-FD(cap3)  diag Σ = ", round.(diag(mg.Sigma),sigdigits=3))
println("GMM-FOD(cap3) diag Σ = ", round.(diag(mf.Sigma),sigdigits=3))
println("FD-FOD diff norm = ", round(norm(diag(mg.Sigma).-diag(mf.Sigma)),sigdigits=3))
