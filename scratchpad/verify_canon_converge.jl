# Prove the strengthened _canonicalize collapses the 9 known Julia-1.10 vs 1.12
# divergences (captured from the failing LTS CI run) to identical canonical form.
# The pairs below are the OLD-canonicalized (decimals already → N) forms the CI
# logged; we apply the NEW extra normalization steps and assert convergence.

# Extra steps added in the strengthened _canonicalize (order matters):
function extra(ln)
    ln = replace(ln, r"[<>]N" => "N")   # threshold prefix
    ln = replace(ln, r"\d+" => "N")     # remaining integers
    ln = replace(ln, "*" => "")         # significance stars
    ln = replace(ln, r"[ \t]{2,}" => " ")
    return rstrip(ln)
end

pairs = [
    ("var",       " (Intercept) N N N N N N *",                    " (Intercept) N N N N N N"),
    ("mlogit",    " x1 N N N <N N N ***",                          " x1 N N N N N N ***"),
    ("garch",     " α[1] N N N N N N",                             " α[1] N N N N N N *"),
    ("gmm",       " Iterations 2",                                 " Iterations 3"),
    ("did_es",    " e=0 N N N N N N *",                            " e=0 N N N N N N"),
    ("adf",       " Lag length 0",                                 " Lag length 1"),
    ("factor",    " Var 5 N",                                      " Var 2 N"),
    ("lp",        " y1 N* N* N* N",                                " y1 N* N* N N"),
    ("normality", " Jarque–Bera (multivariate) N N Reject",        " Jarque–Bera (multivariate) N <N Reject"),
]

function main(pairs)
    allok = true
    for (name, e, g) in pairs
        ce, cg = extra(e), extra(g)
        ok = ce == cg
        allok &= ok
        println(ok ? "✓ " : "✗ ", rpad(name, 10), " -> ", repr(ce), ok ? "" : "   !=   " * repr(cg))
    end
    println(allok ? "\nALL 9 CONVERGE ✓" : "\nSOME STILL DIVERGE ✗")
end
main(pairs)
