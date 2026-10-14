using ParallelParticleSwarms, Optimization, SciMLBase, StaticArrays, KernelAbstractions, Test

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

@testset "HybridPSO default lockstep L-BFGS on bounded Rosenbrock" begin
    rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    prob = OptimizationProblem(
        OptimizationFunction{false}(rosenbrock, Optimization.AutoForwardDiff()),
        SVector(0.0, 0.0); lb = SVector(-2.0, -2.0), ub = SVector(2.0, 2.0)
    )
    sol = solve(prob, HybridPSO(; backend = CPU()); local_maxiters = 200)
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    @test sol.objective < 1.0e-8
end

@testset "HybridPSO lockstep L-BFGS stops at maxtime" begin
    rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    prob = OptimizationProblem(
        OptimizationFunction{false}(rosenbrock, Optimization.AutoForwardDiff()),
        SVector(0.0, 0.0); lb = SVector(-2.0, -2.0), ub = SVector(2.0, 2.0)
    )
    opt = HybridPSO(; backend = CPU())
    solve(prob, opt; maxiters = 2, local_maxiters = 2)   # compile
    # No abstol and far more iterations than can run, so only maxtime ends the local phase.
    t = @elapsed sol = solve(prob, opt; maxiters = 2, local_maxiters = 10^9, maxtime = 0.5)
    @test t < 10
    @test isfinite(sol.objective)
end

# The lockstep L-BFGS must visit the same points as SimpleOptimization's `_lbfgs` run in
# one thread per particle (master's kernel, kept here for this comparison). StaticArrays'
# `dot` is an `@simd` loop, so its rounding depends on what it is inlined into and even
# `_lbfgs` in a kernel and on the host differ in the last bits; for a bitwise comparison,
# the same left-to-right sum without `@simd` stands in for it during this testset.
using BlackBoxOptimizationBenchmarking, LinearAlgebra, Random
import ParallelParticleSwarms: SimpleOptimization, StrongWolfeLineSearch

@inline function LinearAlgebra.dot(a::SVector{D, T}, b::SVector{D, T}) where {D, T <: AbstractFloat}
    r = zero(T) * zero(T) + zero(T) * zero(T)
    for j in 1:D
        r += a[j] * b[j]
    end
    return r
end
const simd_free_dot = which(LinearAlgebra.dot, Tuple{SVector{2, Float64}, SVector{2, Float64}})

@kernel function lbfgs_run!(
        grad_f, f, p, particles, result, result_fx, lb, ub,
        maxiters, abstol, reltol, linesearch, history_size
    )
    i = @index(Global, Linear)
    @inbounds x0 = ParallelParticleSwarms.as_svector(particles[i].best_position)
    u, v, _, _ = SimpleOptimization._lbfgs(
        grad_f, f, p, x0, lb, ub, maxiters, abstol, reltol,
        linesearch, history_size
    )
    @inbounds result[i] = u
    @inbounds result_fx[i] = isfinite(v) ? v : eltype(u)(Inf)
end

@testset "Lockstep L-BFGS matches per-thread _lbfgs" begin
    function compare(f, xs::Vector{SVector{D, T}}, lb, ub, adtype, maxiters, tol) where {D, T}
        particles = [(; best_position = x) for x in xs]
        n = length(xs)
        grad_f = ParallelParticleSwarms.as_svector_grad(
            ParallelParticleSwarms.instantiate_gradient(f, adtype)
        )
        ls = SimpleLBFGS().linesearch
        ls = StrongWolfeLineSearch(;
            autodiff = ls.autodiff, c1 = T(ls.c1), c2 = T(ls.c2),
            α_init = T(ls.α_init), α_max = T(ls.α_max),
            maxiters = ls.maxiters, zoom_maxiters = ls.zoom_maxiters,
        )
        u1, f1 = similar(xs), fill(T(Inf), n)
        lbfgs_run!(CPU())(
            grad_f, f, nothing, particles, u1, f1, lb, ub,
            maxiters, tol, tol, ls, Val(10); ndrange = n
        )
        u2, f2 = similar(xs), fill(T(Inf), n)
        ParallelParticleSwarms.lockstep_lbfgs!(
            u2, f2, grad_f, f, nothing, adtype, particles, lb, ub,
            maxiters, tol, tol, ls, Val(10)
        )
        @test u2 == u1
        @test f2 == f1
    end

    rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    f8 = bbob_suite(Val(10); seed = 1)[8]
    bbob_f8(x, p) = f8(x)
    adtypes = (Optimization.AutoForwardDiff(), Optimization.AutoEnzyme(), SciMLBase.NoAD())
    @testset "$T $(nameof(typeof(adtype)))" for T in (Float64, Float32), adtype in adtypes
        Random.seed!(1)
        xs = [SVector{2, T}(4 .* rand(T, 2) .- 2) for _ in 1:32]
        for (lb, ub) in (
                    (nothing, nothing),
                    (SVector{2, T}(-2, -2), SVector{2, T}(2, 2)),
                    (SVector{2, T}(-2, -2), SVector{2, T}(0.5, 2)),   # active upper bound
                ), tol in (nothing, T(1.0e-8))
            compare(rosenbrock, xs, lb, ub, adtype, 100, tol)
        end

        lb, ub = fill(T(-5), SVector{10, T}), fill(T(5), SVector{10, T})
        xs = [SVector{10, T}(10 .* rand(T, 10) .- 5) for _ in 1:32]
        compare(bbob_f8, xs, nothing, nothing, adtype, 50, nothing)
        compare(bbob_f8, xs, lb, ub, adtype, 50, nothing)
    end
end

Base.delete_method(simd_free_dot)
