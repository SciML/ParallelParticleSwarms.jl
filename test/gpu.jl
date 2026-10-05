using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, LinearAlgebra, Random,
    BlackBoxOptimizationBenchmarking

include("./utils.jl")

@testset "Rosenbrock GPU tests $(N)" for N in 2:4
    Random.seed!(1234)

    ## Solving the rosenbrock problem
    lb = @SArray ones(Float32, N)
    lb = -1 * lb
    ub = @SArray fill(Float32(10.0), N)

    function rosenbrock(x, p)
        res = zero(eltype(x))
        for i in 1:(length(x) - 1)
            res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
        end
        res
    end

    x0 = @SArray zeros(Float32, N)
    p = @SArray Float32[1.0, 100.0]

    # Use out-of-place form {false} since SVector is immutable
    opt_f = OptimizationFunction{false}(rosenbrock)
    prob = OptimizationProblem(opt_f, x0, p; lb = lb, ub = ub)

    n_particles = 5000

    sol = solve(prob, ParallelPSOKernel(n_particles; backend), maxiters = 500)

    @test prob.f(prob.u0, prob.p) > sol.objective

    @test sol.objective < 2.0e-3

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelPSOKernel(n_particles; backend, global_update = false),
        maxiters = 1000
    )

    @test prob.f(prob.u0, prob.p) > sol.objective

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelSyncPSOKernel(n_particles; backend),
        maxiters = 500
    )

    @test prob.f(prob.u0, prob.p) > sol.objective

    @test sol.objective < 6.0e-4
end

@testset "reduction is dimension-independent" begin
    Random.seed!(1234)
    n_particles = 512
    for D in (3, 10, 30)
        lb = SVector{D, Float64}(ntuple(_ -> -5.0, Val(D)))
        ub = SVector{D, Float64}(ntuple(_ -> 5.0, Val(D)))
        x0 = zero(lb)
        optf = OptimizationFunction{false}((x, p) -> sum(abs2, x), SciMLBase.NoAD())
        prob = OptimizationProblem{false}(optf, x0, nothing; lb, ub)
        for opt in (
                ParallelSyncPSOKernel(n_particles; backend),
                ParallelPSOKernel(n_particles; backend, global_update = true),
            )
            sol = solve(prob, opt; maxiters = 100)
            @test isfinite(sol.objective)
        end
        for opt in (
                ParallelSyncPSOKernel(1024; backend, workgroupsize = 1024),
                ParallelPSOKernel(
                    1024; backend, global_update = true, workgroupsize = 1024
                ),
            )
            sol = solve(prob, opt; maxiters = 20)
            @test isfinite(sol.objective)
        end
    end
end

@testset "block argmin matches brute force" begin
    Random.seed!(1234)
    D = 10
    lb = SVector{D, Float64}(ntuple(_ -> -5.0, Val(D)))
    ub = SVector{D, Float64}(ntuple(_ -> 5.0, Val(D)))
    x0 = zero(lb)
    optf = OptimizationFunction{false}((x, p) -> sum(abs2, x), SciMLBase.NoAD())
    prob = OptimizationProblem{false}(optf, x0, nothing; lb, ub)
    for n in (10, 100, 5000),
            opt_f in (
                n -> ParallelSyncPSOKernel(n; backend),
                n -> ParallelPSOKernel(n; backend, global_update = true),
            )
        cache = init(prob, opt_f(n))
        sol = solve!(cache; maxiters = 5)
        best = minimum(p.best_cost for p in Array(cache.particles))
        @test sol.objective == best
    end
end

@testset "local_best sub-swarms" begin
    Random.seed!(1234)
    D = 2
    lb = SVector{D, Float64}(ntuple(_ -> -5.0, Val(D)))
    ub = SVector{D, Float64}(ntuple(_ -> 5.0, Val(D)))
    a = SVector{D, Float64}(ntuple(_ -> 3.0, Val(D)))
    # Two equally deep minima at ±a: a global-best swarm settles in one, sub-swarms in both.
    twowells(x, p) = min(sum(abs2, x - a), sum(abs2, x + a))
    optf = OptimizationFunction{false}(twowells, SciMLBase.NoAD())
    prob = OptimizationProblem{false}(optf, zero(lb), nothing; lb, ub)

    n, ws = 1024, 64
    function block_bests(cache)
        particles = Array(cache.particles)
        return [
            minimum(particles[((b - 1) * ws + 1):min(b * ws, n)]).best_position
                for b in 1:cld(n, ws)
        ]
    end
    nwells(xs) = count(s -> any(x -> norm(x - s * a) < 0.1, xs), (1, -1))

    cache = init(prob, ParallelSyncPSOKernel(n; backend, workgroupsize = ws, local_best = true))
    sol = solve!(cache; maxiters = 200)
    @test sol.objective == minimum(p.best_cost for p in Array(cache.particles))
    @test sol.objective < 1.0e-6
    @test nwells(block_bests(cache)) == 2

    cache = init(prob, ParallelSyncPSOKernel(n; backend, workgroupsize = ws))
    solve!(cache; maxiters = 200)
    @test nwells(block_bests(cache)) == 1

    # Chunked solves continue each block from its own best.
    cache = init(prob, ParallelSyncPSOKernel(n; backend, workgroupsize = ws, local_best = true))
    for _ in 1:10
        solve!(cache; maxiters = 20)
    end
    @test nwells(block_bests(cache)) == 2

    sol = solve(
        prob,
        ParallelParticleSwarms.HybridPSO(;
            pso = ParallelSyncPSOKernel(n; backend, workgroupsize = ws, local_best = true),
            backend,
        );
        maxiters = 50, local_maxiters = 50
    )
    @test sol.objective < 1.0e-8
end

if GROUP == "CUDA"
    @testset "HybridPSO L-BFGS BBOB F8 CUDA" begin
        Random.seed!(42)
        D = 10
        f = bbob_suite(Val(D); seed = 1)[8]

        lb = SVector{D, Float32}(ntuple(_ -> -5.0f0, Val(D)))
        ub = SVector{D, Float32}(ntuple(_ -> 5.0f0, Val(D)))
        x0 = SVector{D, Float32}(ntuple(_ -> -5.0f0 + rand(Float32) * 10.0f0, Val(D)))
        optf = OptimizationFunction{false}((x, p) -> f(x), SciMLBase.NoAD())
        prob = OptimizationProblem{false}(optf, x0, nothing; lb, ub)

        sol = solve(
            prob,
            ParallelParticleSwarms.HybridPSO(;
                pso = ParallelSyncPSOKernel(5_000; backend),
                backend,
            );
            maxiters = 150,
            local_maxiters = 50,
            abstol = 1.0f-8,
            reltol = 1.0f-8,
        )

        @test isfinite(sol.objective)
        @test all(isfinite, sol.u)
        @test all(lb .≤ sol.u) && all(sol.u .≤ ub)
        @test Float32(sol.objective) - f.f_opt ≤ 1.0f-3
    end
end
