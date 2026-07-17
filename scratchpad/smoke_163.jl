using MacroEconometricModels
const MEM = MacroEconometricModels
betahat = [0.05, -0.10, 0.08, 0.15, 0.25, 0.30]
sigma = [0.01 * 0.5^abs(i-j) for i in 1:6, j in 1:6]

println("== FLCI (Δ^SD), e1 — R: M=0 [-0.0746475725, 0.3160442338]; M=0.02 [-0.1018395313, 0.3270227396]; M=0.05 [-0.1778374773, 0.3638093457]")
for M in (0.0, 0.02, 0.05)
    r = honest_did(betahat, sigma; num_pre=3, num_post=3, restriction=:sd, M=M)
    println("M=$M  [$(r.ci_lower), $(r.ci_upper)]  hl=$((r.ci_upper-r.ci_lower)/2)")
end
println("== FLCI e2 — R: M=0 [0.0073599200, 0.4847787921]; M=0.02 [-0.0655127233, 0.5344024372]")
for M in (0.0, 0.02)
    r = honest_did(betahat, sigma; num_pre=3, num_post=3, restriction=:sd, M=M, l_vec=[0.0,1.0,0.0])
    println("M=$M  [$(r.ci_lower), $(r.ci_upper)]")
end
println("== Δ^RM identified sets — R: e1 M̄=0.5 [0.06,0.24], 1 [-0.03,0.33], 2 [-0.21,0.51]; e3 M̄=0.5 [0.03,0.57], 1 [-0.24,0.84]")
for Mb in (0.5, 1.0, 2.0)
    lb, ub = MEM._deltarm_identified_set(betahat, 3, 3; Mbar=Mb)
    println("e1 M̄=$Mb  [$lb, $ub]")
end
for Mb in (0.5, 1.0)
    lb, ub = MEM._deltarm_identified_set(betahat, 3, 3; Mbar=Mb, l_vec=[0.0,0.0,1.0])
    println("e3 M̄=$Mb  [$lb, $ub]")
end
println("== Conventional e1 — R: [-0.0459963985, 0.3459963985]")
r = honest_did(betahat, sigma; num_pre=3, num_post=3, restriction=:rm, Mbar=1.0)
println("orig [$(r.original_ci_lower), $(r.original_ci_upper)]; RM robust CI M̄=1: [$(r.ci_lower), $(r.ci_upper)] (R C-LF ref: [-0.2456, 0.5865]); breakdown=$(r.breakdown)")
