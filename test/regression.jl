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

@testset "GPU-PSO loss reduction (CPU)" begin
    Random.seed!(123)
    m = 5
    n = 7
    gpu_data = [@SVector Float32[1, 2] for _ in 1:m]
    us = reshape([@SVector Float32[0, 0] for _ in 1:(m * n)], m, n)
    losses = zeros(Float32, n)

    ParallelParticleSwarms._reduce_losses!(losses, gpu_data, us)

    loss_mat = map(x -> sum(x .^ 2), gpu_data .- us)
    @test losses == vec(sum(loss_mat, dims = 1))
end

@testset "NeuralODE tuple params prob_func" begin
    u0 = @SVector Float32[1.0, -1.0]
    tspan = (0.0f0, 1.0f0)
    nn = (u, p) -> u .+ p[1]
    p_static = @SVector Float32[2.0, 3.0]
    prob = ODEProblem{false}((u, p, t) -> p[1](u, p[2]), u0, tspan, (nn, p_static))

    new_pos = @SVector Float32[5.0, 6.0]
    particle = ParallelParticleSwarms.SPSOParticle(new_pos, new_pos, 0.0f0, new_pos, 0.0f0)

    function prob_func(prob_local, gpu_particle)
        return remake(prob_local, p = (prob_local.p[1], gpu_particle.position))
    end

    updated = prob_func(prob, particle)
    @test updated.p[1] === prob.p[1]
    @test updated.p[2] == new_pos
end

@testset "_reduce_losses! with 2D losses array" begin
    Random.seed!(123)
    m, n = 5, 7
    gpu_data = [@SVector Float32[1, 2] for _ in 1:m]
    us = reshape([@SVector Float32[0, 0] for _ in 1:(m * n)], m, n)
    losses_2d = zeros(Float32, 1, n)

    ParallelParticleSwarms._reduce_losses!(losses_2d, gpu_data, us)

    expected = vec(sum(map(x -> sum(x .^ 2), gpu_data .- us), dims = 1))
    @test vec(losses_2d) == expected
end
