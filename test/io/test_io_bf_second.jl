using Test, MacroEconometricModels, LinearAlgebra

@testset "Baqaee-Farhi second-order term" begin
    io = load_example(:wiot)
    bf = baqaee_farhi(io)                       # Cobb-Douglas default
    H = bf.second_order
    @test size(H) == (2, 2)
    @test issymmetric(round.(H; digits=10))     # Hessian is symmetric
    # Cobb-Douglas (θ=σ=1): no second-order reallocation, term is zero
    @test all(abs.(H) .<= 1e-8)
    # gross substitutes (θ>1) raise the diagonal (variance ≥ 0)
    bf_s = baqaee_farhi(io; theta=2.0, sigma=1.0)
    @test sum(diag(bf_s.second_order)) >= sum(diag(H)) - 1e-8
    @test all(diag(bf_s.second_order) .>= -1e-8)
end
