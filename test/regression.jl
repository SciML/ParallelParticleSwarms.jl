using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, LinearAlgebra, Random,
    KernelAbstractions
using QuasiMonteCarlo

@testset "Rosenbrock test dimension = $(N)" for N in 2:3

    ## Solving the rosenbrock problem
    Random.seed!(123)
    lb = @SArray fill(Float32(-1.0), N)
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

    array_prob = OptimizationProblem(
        rosenbrock,
        zeros(Float32, N),
        Float32[1.0, 100.0];
        lb = lb,
        ub = ub
    )

    prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

    n_particles = 2000

    sol = solve(
        array_prob,
        ParallelPSOArray(n_particles),
        maxiters = 500
    )

    @test sol.objective < 3.0e-4

    sol = solve(
        prob,
        SerialPSO(n_particles),
        maxiters = 600
    )

    @test sol.objective < 1.0e-4

    sol = solve!(
        init(
            prob, ParallelPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 500
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelPSOKernel(n_particles; backend = CPU()),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve!(
        init(
            prob, ParallelSyncPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 500
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelSyncPSOKernel(n_particles; backend = CPU()),
        maxiters = 500
    )

    @test sol.objective < 3.0e-3

    lb = @SVector fill(Float32(-Inf), N)
    ub = @SVector fill(Float32(Inf), N)

    array_prob = remake(array_prob; lb = lb, ub = ub)
    prob = remake(prob; lb = lb, ub = ub)

    sol = solve(
        array_prob,
        ParallelPSOArray(n_particles),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve(
        prob,
        SerialPSO(n_particles),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve!(
        init(
            prob, ParallelPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 500
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelPSOKernel(n_particles; backend = CPU()),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve!(
        init(
            prob, ParallelSyncPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 500
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelSyncPSOKernel(n_particles; backend = CPU()),
        maxiters = 500
    )

    @test sol.objective < 2.0e-4

    array_prob = remake(array_prob; lb = nothing, ub = nothing)
    prob = remake(prob; lb = nothing, ub = nothing)

    sol = solve(
        array_prob,
        ParallelPSOArray(n_particles),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve(
        prob,
        SerialPSO(n_particles),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve!(
        init(
            prob, ParallelPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 500
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelPSOKernel(n_particles; backend = CPU()),
        maxiters = 500
    )

    @test sol.objective < 1.0e-4

    sol = solve!(
        init(
            prob, ParallelSyncPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 500
    )

    sol = solve(
        prob,
        ParallelSyncPSOKernel(n_particles; backend = CPU()),
        maxiters = 500
    )

    @test sol.objective < 2.0e-2
end

## Separate tests for N = 4 as the problem becomes non-convex and requires more iterations to converge
@testset "Rosenbrock test dimension N = 4" begin

    ## Solving the rosenbrock problem
    N = 4
    Random.seed!(123)
    lb = @SArray fill(Float32(-1.0), N)
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

    array_prob = OptimizationProblem(
        rosenbrock,
        zeros(Float32, N),
        Float32[1.0, 100.0];
        lb = lb,
        ub = ub
    )

    prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

    n_particles = 2000

    sol = solve(
        prob,
        SerialPSO(n_particles),
        maxiters = 1000
    )

    @test sol.objective < 2.0e-3

    sol = solve!(
        init(
            prob, ParallelPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 2000
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelPSOKernel(n_particles; backend = CPU()),
        maxiters = 2000
    )

    @test sol.objective < 2.0e-2

    sol = solve!(
        init(
            prob, ParallelSyncPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 2000
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelSyncPSOKernel(n_particles; backend = CPU()),
        maxiters = 2000
    )

    @test sol.objective < 3.0e-2

    lb = @SVector fill(Float32(-Inf), N)
    ub = @SVector fill(Float32(Inf), N)

    array_prob = remake(array_prob; lb = lb, ub = ub)
    prob = remake(prob; lb = lb, ub = ub)

    sol = solve(
        prob,
        SerialPSO(n_particles),
        maxiters = 1000
    )

    @test sol.objective < 2.0e-3

    array_prob = remake(array_prob; lb = nothing, ub = nothing)
    prob = remake(prob; lb = nothing, ub = nothing)

    sol = solve(
        prob,
        SerialPSO(n_particles),
        maxiters = 1000
    )

    @test sol.objective < 2.0e-3

    sol = solve!(
        init(
            prob, ParallelPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 1000
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelPSOKernel(n_particles; backend = CPU()),
        maxiters = 1000
    )

    @test sol.objective < 2.0e-3

    sol = solve!(
        init(
            prob, ParallelSyncPSOKernel(n_particles; backend = CPU());
            sampler = LatinHypercubeSample()
        ),
        maxiters = 2000
    )

    @test sol.retcode == ReturnCode.Default

    sol = solve(
        prob,
        ParallelSyncPSOKernel(n_particles; backend = CPU()),
        maxiters = 2000
    )

    @test sol.objective < 4.0e-1
end
