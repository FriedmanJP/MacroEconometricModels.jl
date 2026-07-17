using MacroEconometricModels, Random, Statistics, LinearAlgebra
const M = MacroEconometricModels
true_rho=0.7; true_sigma=0.5; T_obs=400
cfn = d -> autocovariance_moment_contributions(d; lags=1)
sim_ratio=5; T_sim=sim_ratio*T_obs
function sim_ar1(theta,T_periods,burn;rng=Random.default_rng())
    rho,sigma=theta; sim=zeros(T_periods+burn)
    for t in 2:(T_periods+burn); sim[t]=rho*sim[t-1]+abs(sigma)*randn(rng); end
    reshape(sim[(burn+1):end],:,1)
end
drng=Random.MersenneTwister(5001); y=zeros(T_obs)
for t in 2:T_obs; y[t]=true_rho*y[t-1]+true_sigma*randn(drng); end
d1=reshape(y,:,1)
rng=Random.MersenneTwister(9001)
theta_hat=[0.69,0.49]  # near truth
# Jacobian as estimate_smm computes it (CRN)
function sim_moments(theta)
    sim=sim_ar1(theta,T_sim,100;rng=copy(rng))
    autocovariance_moments(Matrix{Float64}(sim);lags=1)
end
D=M.numerical_gradient(sim_moments,theta_hat)
println("D (∂m_sim/∂θ), rows=moments cols=params:"); display(D); println()
println("D determinant=",det(D))
Omega=M.smm_data_covariance(d1,cfn;hac=true,bandwidth=0)
W=M.smm_weighting_matrix(d1,cfn;hac=true,bandwidth=0)
println("W*Omega≈I? ", round.(W*Omega,digits=4))
bread=D'*W*D
bread_inv=M.robust_inv(bread)
simc=1+1/sim_ratio
vcov_eff = simc*bread_inv/T_obs
println("vcov (efficient) diag=",diag(vcov_eff)," SE=",sqrt.(diag(vcov_eff)))
# compare: sandwich with same Ω (should equal efficient when W=Ω^{-1})
meat=D'*W*Omega*W*D
vcov_sw=simc*(bread_inv*meat*bread_inv)/T_obs
println("vcov (sandwich)  diag=",diag(vcov_sw)," SE=",sqrt.(diag(vcov_sw)))
# analytic-ish target using D^{-1} Omega D^{-T}/n
V2 = simc*(inv(D)*Omega*inv(D)')/T_obs
println("D^{-1}ΩD^{-T}/n SE=",sqrt.(diag(V2)))
