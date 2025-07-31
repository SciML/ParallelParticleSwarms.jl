using Pkg

Pkg.activate(@__DIR__)

using ParallelParticleSwarms, StaticArrays, KernelAbstractions, Optimization
using CUDA

device!(2)

N = 10
function rosenbrock(x, p)
    res = zero(eltype(x))
    for i in 1:(length(x) - 1)
        res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
    end
    res
end
# x0 = @SArray zeros(Float32, N)

x0 = @SArray fill(5.0f0, N)

p = @SArray Float32[1.0, 100.0]

lb = @SArray fill(Float32(-1.0), N)
ub = @SArray fill(Float32(10.0), N)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb = lb, ub = ub)

n_particles = 10_000

sol = solve(prob, ParallelSyncPSOKernel(n_particles; backend = CPU()), maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelSyncPSOKernel(n_particles; backend = CUDABackend()),
    maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = false),
    maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
    maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
        pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles;
            global_update = false,
            backend = CUDABackend()),
        local_opt = ParallelParticleSwarms.LBFGS()), maxiters = 500,
    local_maxiters = 30)

@show sol.objective
@show sol.stats.time

cpu_times = Float64[]
gpu_sync_times = Float64[]
gpu_async_times = Float64[]
gpu_queue_lock_times = Float64[]

cpu_loss = Float32[]
gpu_sync_loss = Float32[]
gpu_async_loss = Float32[]
gpu_queue_lock_loss = Float32[]

gpu_hybrid_times = Float64[]
gpu_hybrid_loss = Float32[]

Ns = [2^i for i in 3:2:16]

using Random

rng = Random.default_rng()
# Random.seed!(rng, 0)

function solve_run(prob, alg, maxiters; runs = 10, kwargs...)
    losses = Float32[]
    times = Float32[]
    # 4 was a good candidate
    Random.seed!(rng, 1)
    for run in 1:runs
        sol = if alg isa ParallelParticleSwarms.HybridPSO
            solve(prob, alg; maxiters, local_maxiters = 30)
        else
            solve(prob, alg; maxiters, kwargs...)
        end
        push!(losses, sol.objective)
        push!(times, sol.stats.time)
    end
    minimum(losses), minimum(times)
end

for n_particles in Ns
    @info n_particles

    obj,
    sol_time = solve_run(prob,
        ParallelSyncPSOKernel(n_particles; backend = CPU()),
        500)

    push!(cpu_loss, obj)
    push!(cpu_times, sol_time)

    obj,
    sol_time = solve_run(prob,
        ParallelSyncPSOKernel(n_particles; backend = CUDABackend()),
        500)

    push!(gpu_sync_loss, obj)
    push!(gpu_sync_times, sol_time)

    obj,
    sol_time = solve_run(prob,
        ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = false),
        500)

    push!(gpu_async_loss, obj)
    push!(gpu_async_times, sol_time)

    obj,
    sol_time = solve_run(prob,
        ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
        500;
        runs = 2)

    push!(gpu_queue_lock_loss, obj)
    push!(gpu_queue_lock_times, sol_time)

    obj,
    solve_time = solve_run(prob,
        ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
            pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles;
                global_update = false,
                backend = CUDABackend()),
            local_opt = ParallelParticleSwarms.LBFGS()), 500)

    push!(gpu_hybrid_loss, obj)
    push!(gpu_hybrid_times, solve_time)
end

@show cpu_times
@show gpu_sync_times
@show gpu_async_times
@show gpu_queue_lock_times

using Plots

gr(size = (720, 480))

yticks = 10 .^ round.(range(-6, 5, length = 12), digits = 2)

xticks = 10 .^ round.(range(1, -3, length = 9), digits = 2)

plt = plot(gpu_sync_times,
    gpu_sync_loss,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: GPU",
    xlabel = "Time (s)",
    ylabel = "Loss",
    title = "Loss curves for the 10D Rosenbrock Problem",
    legend = :topright,
    xticks = xticks,
    yticks = yticks,
    marker = :circle,
    dpi = 600    # color = :Green
)

plt = plot!(cpu_times,
    cpu_loss,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: CPU",
    marker = :circle,
    # color = :Orange
    ls = :dash)

plt = plot!(gpu_async_times,
    gpu_async_loss,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelPSOKernel (Async): GPU",
    marker = :circle    # color = :Green
)

plt = plot!(gpu_queue_lock_times,
    gpu_queue_lock_loss,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelPSOKernel (Queue-lock): GPU",
    marker = :circle    # color = :Green
)

plt = plot!(gpu_hybrid_times,
    gpu_hybrid_loss,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "HybridPSO-LBFGS: GPU",
    marker = :circle    # color = :Green
)

savefig("wp_pso.svg")

function rosenbrock(x, p)
    loss = zero(eltype(x))
    @inbounds for i in 1:(length(x) - 1)
        loss += p[2] * sin(1.5f0 * x[i])^2 * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
    end
    loss
    # sum(p[2] * exp(-x[i]^2) * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end
# x0 = @SArray zeros(Float32, N)

x0 = @SArray fill(5.0f0, N)

p = @SArray Float32[1.0, 100.0]

lb = @SArray fill(Float32(-1.0), N)
ub = @SArray fill(Float32(5.0), N)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb = lb, ub = ub)

using Statistics

@show mean(cpu_times ./ gpu_sync_times)

@show mean(cpu_times ./ gpu_async_times)

@show mean(cpu_times ./ gpu_queue_lock_times)

@show mean(gpu_sync_times ./ gpu_queue_lock_times)

using OptimizationFlux, OptimizationBBO, OptimizationOptimJL

## Test with Adam, BlackboxOptim

n_particles = 16384

sol = solve(prob,
    ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
    maxiters = 1000)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
        pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles;
            global_update = false,
            backend = CUDABackend()),
        local_opt = ParallelParticleSwarms.LBFGS()), maxiters = 500,
    local_maxiters = 30)

@show sol.objective
@show sol.stats.time

## Adam does not work with StaticArrays

# uncons_prob = remake(prob; lb = nothing, ub = nothing)
x0 = @SArray fill(5.0f0, N)
uncons_prob = OptimizationProblem(optf, Array(x0), Array(p))
arr_prob = OptimizationProblem(rosenbrock,
    Array(Float64.(x0)),
    Array(Float64.(p));
    lb = Array(Float64.(lb)),
    ub = Array(Float64.(ub)))

## Adam is similar in speed and not GPU accelerated. The loss is much better.
## Might not be good enough for a comparison here.
sol = solve(uncons_prob, ADAM(), maxiters = 30)

@show sol.objective
@show sol.stats.time

sol = solve(uncons_prob,
    LBFGS(),
    reltol = -Inf,
    x_tol = -Inf,
    f_abstol = -Inf,
    x_reltol = -Inf)

@show sol.objective
@show sol.stats.time

function solve_run(prob, alg, maxiters; runs = 10, kwargs...)
    losses = Float32[]
    times = Float32[]
    # 4 was a good candidate
    Random.seed!(rng, 9)
    for run in 1:runs
        sol = if alg isa ParallelParticleSwarms.HybridPSO
            solve(prob, alg; maxiters, local_maxiters = maxiters)
        else
            solve(prob, alg; maxiters, kwargs...)
        end
        push!(losses, sol.objective)
        push!(times, sol.stats.time)
    end
    minimum(losses), minimum(times)
end

begin
    lbfgs_losses = Float32[]
    lbfgs_time = Float64[]
    adam_losses = Float32[]
    adam_time = Float64[]
    queue_lock_losses = Float32[]
    queue_lock_time = Float64[]
    hybrid_losses = Float32[]
    hybrid_time = Float64[]
    bbo_losses = Float32[]
    bbo_time = Float64[]
    pso_cpu_losses = Float32[]
    pso_cpu_time = Float64[]

    tot_maxiters = [10, 30, 40]

    for iters in tot_maxiters
        @info iters

        obj,
        sol_time = solve_run(uncons_prob,
            LBFGS(),
            iters;
            reltol = -Inf,
            x_tol = -Inf,
            f_abstol = -Inf,
            x_reltol = -Inf)

        # sol = solve(uncons_prob, LBFGS(), maxiters = iters)

        # sol = solve(uncons_prob, LBFGS(), maxiters = iters)

        push!(lbfgs_losses, obj)
        push!(lbfgs_time, sol_time)

        obj, sol_time = solve_run(uncons_prob, ADAM(), iters)

        # sol = solve(uncons_prob, LBFGS(), maxiters = iters)

        # sol = solve(uncons_prob, LBFGS(), maxiters = iters)

        push!(adam_losses, obj)
        push!(adam_time, sol_time)

        # sol = solve(prob,
        # ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
        # maxiters = iters)

        # sol = solve(prob,
        # ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
        # maxiters = iters)

        obj,
        sol_time = solve_run(prob,
            ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
            iters)

        push!(queue_lock_losses, obj)
        push!(queue_lock_time, sol_time)

        obj,
        sol_time = solve_run(prob,
            ParallelSyncPSOKernel(n_particles; backend = CPU()),
            iters)

        push!(pso_cpu_losses, obj)
        push!(pso_cpu_time, sol_time)

        # sol = solve(prob,
        #     ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
        #     pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles; global_update = false, backend = CUDABackend()),
        #     local_opt = ParallelParticleSwarms.LBFGS()), maxiters = iters,
        #     local_maxiters = iters)

        # sol = solve(prob,
        #         ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
        #         pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles; global_update = false, backend = CUDABackend()),
        #         local_opt = ParallelParticleSwarms.LBFGS()), maxiters = iters,
        #         local_maxiters = iters)

        obj,
        solve_time = solve_run(prob,
            ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
                pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles;
                    global_update = false,
                    backend = CUDABackend()),
                local_opt = ParallelParticleSwarms.LBFGS()), iters)

        push!(hybrid_losses, obj)
        push!(hybrid_time, solve_time)

        # sol = solve(arr_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = iters)

        # sol = solve(arr_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = iters)

        obj,
        sol_time = solve_run(arr_prob,
            BBO_adaptive_de_rand_1_bin_radiuslimited(),
            iters * 100)

        push!(bbo_losses, obj)
        push!(bbo_time, sol_time)
    end
end

using Plots

gr(size = (720, 480))

yticks = 10 .^ round.(range(-6, 5, length = 12), digits = 2)

xticks = 10 .^ round.(range(1, -4, length = 11), digits = 2)

@. lbfgs_losses += 1.0f-8

plt = plot(lbfgs_time,
    lbfgs_losses,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "LBFGS",
    xlabel = "Time (s)",
    ylabel = "Loss",
    title = "Loss curves for the 10D Modified Rosenbrock Problem",
    legend = :bottomleft,
    xticks = xticks,
    yticks = yticks,
    marker = :circle,
    dpi = 600    # color = :Green
)

plt = plot!(adam_time,
    adam_losses,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ADAM",
    marker = :circle    # color = :Orange
)

plt = plot!(pso_cpu_time,
    pso_cpu_losses,
    # xaxis = :log,
    # yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: CPU",
    marker = :circle    # color = :Green
)

plt = plot!(queue_lock_time,
    queue_lock_losses,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelPSOKernel (queue-lock): GPU",
    marker = :circle    # color = :Orange
)

plt = plot!(hybrid_time,
    hybrid_losses,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "HybridPSO-LBFGS: GPU",
    marker = :circle    # color = :Orange
)

plt = plot!(bbo_time,
    bbo_losses,
    # xaxis = :log,
    # yaxis = :log,
    linewidth = 2,
    label = "BlackboxOptim",
    marker = :circle    # color = :Green
)

savefig("wp_algs.svg")
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(uncons_prob, ADAM(0.01), maxiters = 500)

sol = solve(uncons_prob, BFGS(), maxiters = 500)
