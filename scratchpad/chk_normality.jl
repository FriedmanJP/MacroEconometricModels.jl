using MacroEconometricModels, Random
for (lbl, data) in (
        ("randn (current, borderline)", randn(MersenneTwister(94), 200, 3)),
        ("randexp (decisive skew+kurt)", randexp(MersenneTwister(94), 200, 3)),
        ("t3 heavy-tail",               randn(MersenneTwister(94), 200, 3) ./ sqrt.(rand(MersenneTwister(7), 200, 3))),
    )
    println("\n########## $lbl ##########")
    r = normality_test_suite(data)
    s = sprint(show, r)
    for ln in split(s, '\n')
        (occursin("Reject", ln) || occursin("reject", ln) || occursin("Test", ln)) && println(rstrip(ln))
    end
end
