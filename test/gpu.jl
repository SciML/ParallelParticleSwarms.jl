using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, LinearAlgebra, Random

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

    prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

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
