using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, KernelAbstractions
using JLArrays, ArrayInterface

# Test function
function rosenbrock(x, p)
    res = zero(eltype(x))
    for i in 1:(length(x) - 1)
        res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
    end
    res
end

@testset "Interface Compatibility" begin
    N = 2
    n_particles = 50

    @testset "Regular Array with SerialPSO" begin
        x0 = zeros(Float32, N)
        p = Float32[1.0, 100.0]
        lb = fill(Float32(-1.0), N)
        ub = fill(Float32(10.0), N)

        prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

        sol = solve(prob, SerialPSO(n_particles), maxiters = 100)
        @test sol.retcode == ReturnCode.Default
        @test eltype(sol.u) == Float32
        @test sol.objective < 0.1
    end

    @testset "Regular Array with ParallelPSOArray" begin
        x0 = zeros(Float32, N)
        p = Float32[1.0, 100.0]
        lb = fill(Float32(-1.0), N)
        ub = fill(Float32(10.0), N)

        prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

        sol = solve(prob, ParallelPSOArray(n_particles), maxiters = 100)
        @test sol.retcode == ReturnCode.Default
        @test eltype(sol.u) == Float32
        @test sol.objective < 0.1
    end

    @testset "BigFloat support with SerialPSO" begin
        x0 = zeros(BigFloat, N)
        p = BigFloat[1.0, 100.0]
        lb = fill(BigFloat(-1.0), N)
        ub = fill(BigFloat(10.0), N)

        prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

        sol = solve(prob, SerialPSO(n_particles), maxiters = 100)
        @test sol.retcode == ReturnCode.Default
        @test eltype(sol.u) == BigFloat
        @test sol.objective < 0.1
    end

    @testset "BigFloat support with ParallelPSOArray" begin
        x0 = zeros(BigFloat, N)
        p = BigFloat[1.0, 100.0]
        lb = fill(BigFloat(-1.0), N)
        ub = fill(BigFloat(10.0), N)

        prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

        sol = solve(prob, ParallelPSOArray(n_particles), maxiters = 100)
        @test sol.retcode == ReturnCode.Default
        @test eltype(sol.u) == BigFloat
        @test sol.objective < 0.1
    end

    @testset "JLBackend support with ParallelPSOKernel" begin
        x0_static = @SArray zeros(Float32, N)
        p_static = @SArray Float32[1.0, 100.0]
        lb_static = @SArray fill(Float32(-1.0), N)
        ub_static = @SArray fill(Float32(10.0), N)

        prob = OptimizationProblem(rosenbrock, x0_static, p_static; lb = lb_static, ub = ub_static)

        jl_backend = JLArrays.JLBackend()
        sol = solve(prob, ParallelPSOKernel(n_particles; backend = jl_backend), maxiters = 100)
        @test sol.retcode == ReturnCode.Default
        @test sol.objective < 0.1
    end

    @testset "ArrayInterface compatibility" begin
        x_array = [1.0f0, 2.0f0, 3.0f0]
        x_static = @SArray [1.0f0, 2.0f0, 3.0f0]

        @test ArrayInterface.can_setindex(x_array) == true
        @test ArrayInterface.fast_scalar_indexing(x_array) == true

        @test ArrayInterface.can_setindex(x_static) == false
        @test ArrayInterface.fast_scalar_indexing(x_static) == true
    end
end
