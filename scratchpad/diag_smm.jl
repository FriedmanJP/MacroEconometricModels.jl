using MacroEconometricModels, Random, Statistics, LinearAlgebra
const M = MacroEconometricModels
true_rho=0.7; true_sigma=0.5; T_obs=400
cfn = d -> autocovariance_moment_contributions(d; lags=1)
# Empirical sampling covariance of the 2 moments across datasets
ms=Vector{Float64}[]
for r in 1:2000
    drng=Random.MersenneTwister(5000+r); y=zeros(T_obs)
    for t in 2:T_obs; y[t]=true_rho*y[t-1]+true_sigma*randn(drng); end
    push!(ms, autocovariance_moments(reshape(y,:,1); lags=1))
end
Mmat=reduce(hcat,ms)'  # 2000 x 2
Sigma_m_emp = cov(Mmat)  # empirical Var of the moment vector (should ≈ Ω/n)
println("Empirical Var(moments) [=Ω/n target]:"); display(Sigma_m_emp); println()
# Ω on one dataset, hac auto vs several fixed bw
drng=Random.MersenneTwister(5001); y=zeros(T_obs)
for t in 2:T_obs; y[t]=true_rho*y[t-1]+true_sigma*randn(drng); end
d1=reshape(y,:,1)
H=cfn(d1)
println("auto bw (optimal_bandwidth_nw on H): ", M.optimal_bandwidth_nw(H; kernel=:bartlett))
for bw in (0, 5, 10, 20, 40)
    O = M.smm_data_covariance(d1, cfn; hac=true, bandwidth=bw)
    println("Ω/n (hac bw=$bw):  diag=", round.(diag(O)./T_obs, sigdigits=4))
end
Onh = M.smm_data_covariance(d1, cfn; hac=false)
println("Ω/n (hac=false):   diag=", round.(diag(Onh)./T_obs, sigdigits=4))
println("target diag (emp): ", round.(diag(Sigma_m_emp), sigdigits=4))
