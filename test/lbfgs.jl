using ParallelParticleSwarms, Optimization, SciMLBase, StaticArrays, KernelAbstractions, Test
using BlackBoxOptimizationBenchmarking, Random
const PPS = ParallelParticleSwarms

@testset "SimpleLBFGS hybrid local polish" begin
    function _solve_hybrid(
            f, x0, p = nothing; lb = nothing, ub = nothing,
            maxiters = 50, local_maxiters = 200, adtype = Optimization.AutoForwardDiff()
        )
        n = length(x0)
        optf = OptimizationFunction{false}(f, adtype)
        u0 = SVector{n, Float64}(x0)
        kwargs = NamedTuple()
        if lb !== nothing
            kwargs = (;
                lb = SVector{n, Float64}(lb),
                ub = SVector{n, Float64}(ub),
            )
        end
        return solve(
            OptimizationProblem(optf, u0, p; kwargs...),
            HybridPSO(;
                pso = ParallelSyncPSOKernel(64; backend = CPU()),
                backend = CPU(),
                local_opt = SimpleLBFGS(),
            );
            maxiters, local_maxiters, abstol = 1.0e-10
        )
    end

    beale(x, p) = (1.5 - x[1] + x[1] * x[2])^2 +
        (2.25 - x[1] + x[1] * x[2]^2)^2 +
        (2.625 - x[1] + x[1] * x[2]^3)^2
    booth(x, p) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2
    himmelblau(x, p) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    quadratic(x, p) = p[1] * (x[1] - 1)^2 + (x[2] + 2)^2
    boundary_quadratic(x, p) = (x[1] + one(eltype(x)))^2
    rosen2(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

    sol = _solve_hybrid(rosen2, [0.0, 0.0], [1.0, 100.0])
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(rosen2, [-1.2, 1.0], [1.0, 100.0])
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(rosen2, [0.0, 0.0], [1.0, 100.0]; lb = [-2.0, -2.0], ub = [2.0, 2.0])
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8
    @test all(-2 .≤ sol.u) && all(sol.u .≤ 2)

    adtypes = (Optimization.AutoEnzyme(), SciMLBase.NoAD())
    @testset "adtype = $(nameof(typeof(adtype)))" for adtype in adtypes
        sol = _solve_hybrid(rosen2, [-1.2, 1.0], [1.0, 100.0]; adtype)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test sol.objective < 1.0e-8

        sol = _solve_hybrid(
            rosen2, [0.0, 0.0], [1.0, 100.0]; adtype, lb = [-2.0, -2.0], ub = [2.0, 2.0]
        )
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test sol.objective < 1.0e-8
    end

    sol = _solve_hybrid(beale, [1.0, 1.0])
    @test sol.u ≈ [3.0, 0.5] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(booth, [1.0, 1.0])
    @test sol.u ≈ [1.0, 3.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(himmelblau, [1.0, 1.0])
    @test sol.objective < 1.0e-8

    sol = _solve_hybrid(quadratic, [4.0, 4.0], [1000.0]; lb = [-5.0, -5.0], ub = [5.0, 5.0])
    @test sol.u ≈ [1.0, -2.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8
    @test all(-5 .≤ sol.u) && all(sol.u .≤ 5)

    sol = _solve_hybrid(
        boundary_quadratic, [1.5]; lb = [0.0], ub = [2.0],
        maxiters = 20, local_maxiters = 50
    )
    @test sol.u[1] ≈ 0.0 atol = 1.0e-6
    @test sol.objective ≈ 1.0 atol = 1.0e-6
end

@testset "NelderMeadPolish" begin
    # Unrotated sharp ridge: the gradient keeps norm ≥ 100 next to the minimum at xopt.
    xopt = SVector(1.0, -2.0, 0.5, 1.5, -1.0)
    mask = SVector(0.0, 1.0, 1.0, 1.0, 1.0)
    ridge(x, p) = (x[1] - xopt[1])^2 + 100 * sqrt(sum(abs2, (x - xopt) .* mask))

    @test isbits(NelderMeadPolish())
    x0 = SVector(1.3, -1.8, 0.6, 1.2, -0.7)
    lb, ub = fill(-5.0, SVector{5}), fill(5.0, SVector{5})
    # A single run can collapse short of the ridge; restarts finish it.
    x, fx = PPS.nelder_mead(ridge, nothing, x0, lb, ub, 10_000, 1.0e-12, 5)
    @test fx < 1.0e-6
    @test fx == ridge(x, nothing)
    allocs(x0, lb, ub) = @allocated PPS.nelder_mead(ridge, nothing, x0, lb, ub, 10_000, 1.0e-12, 5)
    allocs(x0, lb, ub)
    @test allocs(x0, lb, ub) == 0

    # 1D: the shrink step must not collapse the simplex onto one vertex. With a shrink
    # factor of 0 this start stalls at f ≈ 0.016.
    wavy(x, p) = (x[1] - 0.3)^2 + 0.1 * (1 - cos(30 * (x[1] - 0.3)))
    _, fx = PPS.nelder_mead(wavy, nothing, SVector(-2.7), nothing, nothing, 10_000, 1.0e-12)
    @test fx < 1.0e-8

    # With `all_particles`, the host still polishes a best point that came from the swarm.
    nm_all = NelderMeadPolish(; all_particles = true)
    @test PPS.polish_best(nm_all, ridge, nothing, x0, ridge(x0, nothing), lb, ub, true)[2] ==
        ridge(x0, nothing)
    @test PPS.polish_best(nm_all, ridge, nothing, x0, ridge(x0, nothing), lb, ub, false)[2] <
        1.0e-6

    # Bounds hold even when the minimum is outside the box.
    x, _ = PPS.nelder_mead(ridge, nothing, x0, lb, zero(ub), 10_000, 1.0e-12, 5)
    @test all(lb .≤ x .≤ 0)

    function hybrid(f, D; polish, local_opt = SimpleLBFGS())
        optf = OptimizationFunction{false}((x, p) -> f(x), Optimization.AutoForwardDiff())
        lb = SVector{D, Float64}(ntuple(_ -> -5.0, Val(D)))
        prob = OptimizationProblem{false}(optf, zero(lb), nothing; lb, ub = -lb)
        alg = HybridPSO(;
            pso = ParallelSyncPSOKernel(1000; backend = CPU()),
            backend = CPU(), local_opt, polish,
        )
        return solve(prob, alg; maxiters = 100, local_maxiters = 100, abstol = 1.0e-8, reltol = 1.0e-8)
    end

    # BBOB F13 (rotated sharp ridge): L-BFGS stalls short of the 1e-6 target, the polish
    # finishes it.
    Random.seed!(42)
    f13 = bbob_suite(Val(5); seed = 1)[13]
    sol = hybrid(f13, 5; polish = NelderMeadPolish())
    @test sol.objective - f13.f_opt < 1.0e-6

    f13_3 = bbob_suite(Val(3); seed = 1)[13]
    sol = hybrid(f13_3, 3; polish = NelderMeadPolish(; all_particles = true))
    @test sol.objective - f13_3.f_opt < 1.0e-6

    sol = hybrid(f13_3, 3; polish = NelderMeadPolish(), local_opt = PPS.BFGS())
    @test sol.objective - f13_3.f_opt < 1.0e-6
end
