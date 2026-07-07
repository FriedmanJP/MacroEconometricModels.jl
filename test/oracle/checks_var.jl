# test/oracle/checks_var.jl — VAR reduced-form cross-check vs rfvar3 / ols_reg conventions.
# Run from repo root (after ref_var.m):  $JULIA --project=. test/oracle/checks_var.jl
using MacroEconometricModels, LinearAlgebra
include(joinpath(@__DIR__, "compare.jl"))

y = load_fixture("synthetic_var")
n, p = 3, 2
m = estimate_var(y, p; check_stability=false)

# Align ordering: ours [const; lag1; ...; lagp] -> reference [lag1; ...; lagp; const]
B_ref_order = vcat(m.B[2:end, :], m.B[1:1, :])

r1 = compare("VAR coef B (reordered)", B_ref_order, read_ref("var_B"))
r2 = compare("VAR residuals U",        m.U,          read_ref("var_u"))
r3 = compare("Sigma vs ref ML  (U'U/N)",     m.Sigma, read_ref("var_Serror_ml"))   # expect PASS
r4 = compare("Sigma vs ref dof (U'U/(N-K))", m.Sigma, read_ref("var_Serror_dof"))  # expect FAIL (bug)

F  = MacroEconometricModels.companion_matrix(m.B, n, p)
ev = sort(abs.(eigvals(F)))
r5 = compare("companion eigenvalues", ev, vec(read_ref("var_eig")))

println("\nSummary:")
println("  coef/residuals/companion correct: ", all((r1.pass, r2.pass, r5.pass)))
println("  Sigma is ML (U'U/T_eff), NOT dof-adjusted: ", r3.pass && !r4.pass,
        "   (dof gap maxrel=", round(r4.maxrel, sigdigits=3), ")")
