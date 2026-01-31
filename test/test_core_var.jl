using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using DataFrames
using Random

@testset "Core VAR & Identification" begin
    # 1. Generate Synthetic Data
    T = 200
    n = 2
    p = 1

    true_A = [0.5 0.0; 0.0 0.5] # Diagonal AR
    true_c = [0.0; 0.0]
    Sigma_true = [1.0 0.0; 0.0 1.0] # Identity

    Y = zeros(T, n)
    # Generate data
    for t in 2:T
        u = randn(2)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # 2. Estimate
    model = estimate_var(Y, p)

    # 6. Identification (Cholesky)
    L = identify_cholesky(model)
    @test istriu(L')

    # 7. Sign Restrictions
    # We want to identify a "positive shock" to variable 1
    # Restriction: Response of Var 1 to Shock 1 at h=0 is positive (> 0)

    horizon = 10
    check_func(irf) = irf[1, 1, 1] > 0

    Q_sign, irf_sign = identify_sign(model, horizon, check_func)
    @test irf_sign[1, 1, 1] > 0
    @test isapprox(Q_sign * Q_sign', I(n), atol=1e-10) # Q is orthogonal

    # 8. Narrative Restrictions
    # Assume we know that at t=5, the shock 1 was positive.
    narrative_check(shocks) = shocks[5, 1] > 0

    Q_nar, irf_nar, shocks_nar = identify_narrative(model, horizon, check_func, narrative_check)
    @test irf_nar[1, 1, 1] > 0
    @test shocks_nar[5, 1] > 0

    # 9. Long Run Identification
    # Blanchard-Quah
    Q_lr = identify_long_run(model)
    @test isapprox(Q_lr * Q_lr', I(n), atol=1e-10)

    # Check if Long Run Impact is Lower Triangular
    # LR Impact = (I - A(1))^-1 * P
    # P = L * Q_lr
    # We can approximate infinite sum IRF or compute directly.
    # IRF cumulative sum at large h should be close to LR impact.

    # Let's compute directly to verify property
    B = model.B
    A_sum = zeros(n, n)
    for i in 1:p
        start_row = 1 + (i - 1) * n + 1
        end_row = 1 + i * n
        A_sum += B[start_row:end_row, :]'
    end
    inv_lag = inv(I(n) - A_sum)
    L_chol = identify_cholesky(model)
    P = L_chol * Q_lr
    LR_Matrix = inv_lag * P

    # Check lower triangularity of LR_Matrix
    @test abs(LR_Matrix[1, 2]) < 1e-8 # Upper right element should be 0

    @testset "Higher Order Lag (VAR(12))" begin
        # Test estimation with long lags using a known DGP to verify statistical recovery
        # DGP: Y_t = A_1 Y_{t-1} + ... + A_{12} Y_{t-12} + u_t
        # A_1 = 0.4 * I
        # A_12 = 0.2 * I
        # Others = 0

        T_large = 2000
        n = 2
        p = 12

        Y = zeros(T_large, n)

        # True Parameters
        A1 = [0.4 0.0; 0.0 0.4]
        A12 = [0.2 0.0; 0.0 0.2]
        Sigma_true = [1.0 0.0; 0.0 1.0]

        # Simulation
        Random.seed!(42) # Ensure reproducibility
        for t in p+1:T_large
            u = randn(n)
            # Y_t = A1 * Y_{t-1} + A12 * Y_{t-12} + u
            Y[t, :] = A1 * Y[t-1, :] + A12 * Y[t-12, :] + u
        end

        # Estimate
        model = estimate_var(Y, p)

        # Verify Coefficients
        # B structure: [Intercept; A_1'; A_2'; ... A_p']
        # Intercept should be close to 0
        @test norm(model.B[1, :]) < 0.2

        # Check Lag 1 (Rows 2:3)
        est_A1 = model.B[2:3, :]'
        @test isapprox(est_A1, A1, atol=0.1)

        # Check Lag 12 (Rows 1+11*2+1 : 1+12*2) -> Rows 24:25
        est_A12 = model.B[end-1:end, :]'
        @test isapprox(est_A12, A12, atol=0.1)

        # Check a middle lag (e.g., Lag 6) is close to zero
        # Rows for Lag 6: 1 + 5*2 + 1 = 12 to 13?
        # Lag k starts at 2 + (k-1)*n
        # Lag 6: 2 + 5*2 = 12. So rows 12:13.
        est_A6 = model.B[12:13, :]'
        @test norm(est_A6) < 0.1

        # Verify Residuals Covariance
        @test isapprox(model.Sigma, Sigma_true, atol=0.1)
    end
end
