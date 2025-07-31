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
x0 = @SArray zeros(Float32, N)
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

using Random
rng = Random.default_rng()

Random.seed!(rng, 0)

Ns = [2^i for i in 3:2:20]
for n_particles in Ns
    @info n_particles
    ## CPU solve
    backend = CPU()
    opt = ParallelSyncPSOKernel(n_particles; backend)
    init_gbest,
    particles = ParallelParticleSwarms.init_particles(
        prob, opt, typeof(prob.u0))

    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)

    backend_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))

    copyto!(backend_particles, particles)

    ParallelParticleSwarms.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    el_time = @elapsed ParallelParticleSwarms.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    push!(cpu_times, el_time)
    ## GPU Solve

    backend = CUDABackend()

    opt = ParallelSyncPSOKernel(n_particles; backend)

    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)

    backend_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))

    copyto!(backend_particles, particles)

    ParallelParticleSwarms.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    el_time = @elapsed ParallelParticleSwarms.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    push!(gpu_sync_times, el_time)

    opt = ParallelPSOKernel(n_particles; backend, global_update = false)

    gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
    copyto!(gpu_init_gbest, [init_gbest])

    ParallelParticleSwarms.vectorized_solve!(prob,
        gpu_init_gbest,
        backend_particles,
        opt, Val(opt.global_update); maxiters = 500)

    el_time = @elapsed ParallelParticleSwarms.vectorized_solve!(prob,
        gpu_init_gbest,
        backend_particles,
        opt, Val(opt.global_update); maxiters = 500)

    push!(gpu_async_times, el_time)

    opt = ParallelPSOKernel(n_particles; backend, global_update = true)

    gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
    copyto!(gpu_init_gbest, [init_gbest])

    ParallelParticleSwarms.vectorized_solve!(prob,
        gpu_init_gbest,
        backend_particles,
        opt, Val(opt.global_update); maxiters = 500)

    el_time = @elapsed ParallelParticleSwarms.vectorized_solve!(prob,
        gpu_init_gbest,
        backend_particles,
        opt, Val(opt.global_update); maxiters = 500)

    push!(gpu_queue_lock_times, el_time)
end

gpu_hybrid_times = Float64[]

Random.seed!(rng, 0)

for n_particles in Ns
    @info n_particles

    sol = solve(prob,
        ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
            pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles;
                global_update = false,
                backend = CUDABackend()),
            local_opt = ParallelParticleSwarms.LBFGS()), maxiters = 500,
        local_maxiters = 30)

    sol = solve(prob,
        ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
            pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles;
                global_update = false,
                backend = CUDABackend()),
            local_opt = ParallelParticleSwarms.LBFGS()), maxiters = 500,
        local_maxiters = 30)

    push!(gpu_hybrid_times, sol.stats.time)
end

@show cpu_times
@show gpu_sync_times
@show gpu_async_times
@show gpu_queue_lock_times

using Plots

xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)

yticks = 10 .^ round.(range(1, -3, length = 9), digits = 2)

plt = plot(Ns,
    gpu_sync_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: GPU",
    ylabel = "Time (s)",
    xlabel = "No. of Particles",
    title = "Benchmarking the 10D Rosenbrock Problem",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
    marker = :circle,
    dpi = 600    # color = :Green
)

plt = plot!(Ns,
    cpu_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: CPU",
    marker = :circle,
    ls = :dash    # color = :Orange
)

plt = plot!(Ns,
    gpu_async_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelPSOKernel (Async): GPU",
    marker = :circle    # color = :Green
)

plt = plot!(Ns,
    gpu_queue_lock_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelPSOKernel (Queue-lock): GPU",
    marker = :circle    # color = :Green
)

plt = plot!(Ns,
    gpu_hybrid_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "HybridPSO-LBFGS: GPU",
    marker = :circle    # color = :Green
)

savefig("benchmark_hybrid.svg")

using Statistics

@show mean(cpu_times ./ gpu_sync_times)

@show mean(cpu_times ./ gpu_async_times)

@show mean(cpu_times ./ gpu_queue_lock_times)

@show mean(cpu_times ./ gpu_hybrid_times)

@show mean(gpu_sync_times ./ gpu_queue_lock_times)

# cpu_times_total = Float64[]
# gpu_sync_times_total = Float64[]
# gpu_async_times_total = Float64[]
# gpu_queue_lock_times_total = Float64[]
# gpu_hybrid_times_total = Float64[]

# for n_particles in Ns
#     @info n_particles

#     sol = solve(prob, ParallelSyncPSOKernel(n_particles; backend = CPU()), maxiters = 500)

#     el_time = @elapsed solve(prob, ParallelSyncPSOKernel(n_particles; backend = CPU()), maxiters = 500)

#     push!(cpu_times_total, el_time)

#     sol = solve(prob,
#         ParallelSyncPSOKernel(n_particles; backend = CUDABackend()),
#         maxiters = 500)

#     el_time = @elapsed solve(prob,
#                              ParallelSyncPSOKernel(n_particles; backend = CUDABackend()),
#                               maxiters = 500)

#     push!(gpu_sync_times_total, el_time)

#     sol = solve(prob,
#         ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = false),
#         maxiters = 500)

#     el_time = @elapsed solve(prob,
#                 ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = false),
#                 maxiters = 500)

#     push!(gpu_async_times_total, el_time)

#     sol = solve(prob,
#         ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
#         maxiters = 500)

#     el_time = @elapsed solve(prob,
#     ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = true),
#     maxiters = 500)

#     push!(gpu_queue_lock_times_total, el_time)

#     sol = solve(prob,
#         ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
#         pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles; global_update = false, backend = CUDABackend()),
#         local_opt = ParallelParticleSwarms.LBFGS()), maxiters = 500,
#         local_maxiters = 30)

#     el_time = @elapsed solve(prob,
#     ParallelParticleSwarms.HybridPSO(; backend = CUDABackend(),
#     pso = ParallelParticleSwarms.ParallelPSOKernel(n_particles; global_update = false, backend = CUDABackend()),
#     local_opt = ParallelParticleSwarms.LBFGS()), maxiters = 500,
#     local_maxiters = 30)

#     push!(gpu_hybrid_times_total, el_time)

# end

# plt = plot(Ns,
#     gpu_sync_times_total,
#     xaxis = :log,
#     yaxis = :log,
#     linewidth = 2,
#     label = "ParallelSyncPSOKernel: GPU",
#     ylabel = "Time (s)",
#     xlabel = "No. of Particles",
#     title = "Bechmarking the 10D Rosenbrock Problem",
#     legend = :topleft,
#     xticks = xticks,
#     yticks = yticks,
#     marker = :circle,
#     dpi = 600,
#     # color = :Green
#     )

# plt = plot!(Ns,
#     cpu_times_total,
#     xaxis = :log,
#     yaxis = :log,
#     linewidth = 2,
#     label = "ParallelSyncPSOKernel: CPU",
#     marker = :circle,
#     # color = :Orange
#     )

# plt = plot!(Ns,
#     gpu_async_times_total,
#     xaxis = :log,
#     yaxis = :log,
#     linewidth = 2,
#     label = "ParallelPSOKernel (Async): GPU",
#     marker = :circle,
#     # color = :Green
#     )

# plt = plot!(Ns,
#     gpu_queue_lock_times_total,
#     xaxis = :log,
#     yaxis = :log,
#     linewidth = 2,
#     label = "ParallelPSOKernel (Queue-lock): GPU",
#     marker = :circle,
#     # color = :Green
#     )

# plt = plot!(Ns,
#     gpu_hybrid_times_total,
#     xaxis = :log,
#     yaxis = :log,
#     linewidth = 2,
#     label = "HybridPSO-LBFGS: GPU",
#     marker = :circle,
#     # color = :Green
#     )
