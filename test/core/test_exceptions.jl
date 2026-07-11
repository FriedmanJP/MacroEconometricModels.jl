# T146 / #245 — typed exception hierarchy
@testset "T146: exception hierarchy (#245)" begin
    # Abstract root + subtyping
    @test isabstracttype(MacroModelError)
    @test MacroModelError <: Exception
    @test ConvergenceError    <: MacroModelError
    @test IdentificationError <: MacroModelError
    @test SingularSystemError <: MacroModelError

    # Construction + optional structured context
    ce = ConvergenceError("did not converge")
    @test ce isa MacroModelError
    @test ce.msg == "did not converge"
    @test ce.iters === nothing && ce.residual === nothing
    ce2 = ConvergenceError("diverged"; iters=7, residual=1.5e-3)
    @test ce2.iters == 7 && ce2.residual == 1.5e-3

    ie = IdentificationError("not identified")
    @test ie isa MacroModelError && ie.msg == "not identified"

    se = SingularSystemError("singular"; cond=1e12)
    @test se isa MacroModelError && se.cond == 1e12

    # A single `catch e; e isa MacroModelError` handles all three
    for E in (ConvergenceError, IdentificationError, SingularSystemError)
        caught = try
            throw(E("boom"))
        catch e
            e isa MacroModelError
        end
        @test caught
    end

    # showerror is informative
    @test occursin("ConvergenceError", sprint(showerror, ce2))
    @test occursin("iters=7", sprint(showerror, ce2))
    @test occursin("IdentificationError", sprint(showerror, ie))
    @test occursin("SingularSystemError", sprint(showerror, se))

    # Integration: an over-constrained sign/zero identification throws the typed
    # error via identify_arias (a full model integration assertion lives in
    # test/var/test_arias2018.jl, co-edited in this commit).
    @test IdentificationError("x") isa Exception
end

# T145 / #244 — recoverable-draw-error classifier (narrows bare MC/bootstrap catches)
@testset "T145: _is_recoverable_draw_error (#244)" begin
    rde = MacroEconometricModels._is_recoverable_draw_error
    # Recoverable: singular / non-PD / non-convergent / typed model errors → caught + skipped
    @test rde(LinearAlgebra.SingularException(1))
    @test rde(LinearAlgebra.PosDefException(1))
    @test rde(DomainError(-1.0, "x"))
    @test rde(ConvergenceError("nc"))
    @test rde(SingularSystemError("sing"))
    @test rde(IdentificationError("ni"))
    # NOT recoverable: genuine programming errors must propagate, not be swallowed
    @test !rde(MethodError(sin, (1,)))
    @test !rde(BoundsError([1], 5))
    @test !rde(UndefVarError(:x))
    @test !rde(ErrorException("generic"))
end
