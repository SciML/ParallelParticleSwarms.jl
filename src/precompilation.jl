using PrecompileTools

@setup_workload begin
    using StaticArrays

    # Simple test function that doesn't require external packages
    function _rosenbrock_precompile(x, p)
        res = zero(eltype(x))
        for i in 1:(length(x) - 1)
            res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
        end
        res
    end

    @compile_workload begin
        # Setup common problem types
        N = 2
        lb = @SArray fill(Float32(-1.0), N)
        ub = @SArray fill(Float32(10.0), N)
        x0 = @SArray zeros(Float32, N)
        p = @SArray Float32[1.0, 100.0]

        # Create optimization problem with StaticArrays
        prob = OptimizationProblem(_rosenbrock_precompile, x0, p; lb = lb, ub = ub)

        # Precompile SerialPSO - most commonly used CPU algorithm
        sol = solve(prob, SerialPSO(10), maxiters = 2)

        # Precompile ParallelSyncPSOKernel with CPU backend
        sol = solve(prob, ParallelSyncPSOKernel(10; backend = CPU()), maxiters = 2)

        # Precompile ParallelPSOKernel with CPU backend (global_update=true)
        sol = solve(prob, ParallelPSOKernel(10; backend = CPU(), global_update = true), maxiters = 2)

        # Precompile ParallelPSOKernel with CPU backend (global_update=false)
        sol = solve(prob, ParallelPSOKernel(10; backend = CPU(), global_update = false), maxiters = 2)

        # Precompile ParallelPSOArray with regular arrays
        array_prob = OptimizationProblem(_rosenbrock_precompile, zeros(Float32, N), Float32[1.0, 100.0]; lb = lb, ub = ub)
        sol = solve(array_prob, ParallelPSOArray(10), maxiters = 2)

        # Precompile init/solve! pattern
        cache = init(prob, ParallelPSOKernel(10; backend = CPU()))
        sol = solve!(cache, maxiters = 2)
        reinit!(cache)

        cache = init(prob, ParallelSyncPSOKernel(10; backend = CPU()))
        sol = solve!(cache, maxiters = 2)
        reinit!(cache)
    end
end
