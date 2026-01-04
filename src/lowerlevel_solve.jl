function vectorized_solve!(
        prob,
        gbest,
        gpu_particles, opt::ParallelSyncPSOKernel;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false
    )
    backend = get_backend(gpu_particles)

    ## TODO: Get dynamic workgroupsize
    workgroupsize = (min(length(gpu_particles), 1024),)

    update_particle_kernel = update_particle_states!(backend, workgroupsize)

    block_particles = KernelAbstractions.allocate(
        backend,
        typeof(gbest),
        cld(length(gpu_particles), workgroupsize[1])
    )
    for i in 1:maxiters
        update_particle_kernel(
            prob,
            gpu_particles, block_particles,
            gbest,
            w, opt;
            ndrange = length(gpu_particles), workgroupsize
        )
        gbest = minimum(block_particles)
        w = w * wdamp
    end

    return gbest, gpu_particles
end

function vectorized_solve!(
        prob,
        gbest,
        gpu_particles, opt::ParallelSyncPSOKernel{Backend, T, G, H};
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false
    ) where {Backend <: CPU, T, G, H}
    backend = get_backend(gpu_particles)

    update_particle_kernel = update_particle_states!(backend)

    for i in 1:maxiters
        update_particle_kernel(
            prob,
            gpu_particles,
            gbest,
            w, opt;
            ndrange = length(gpu_particles)
        )
        best_particle = minimum(gpu_particles)
        gbest = SPSOGBest(best_particle.position, best_particle.best_cost)
        w = w * wdamp
    end

    return gbest, gpu_particles
end

function vectorized_solve!(
        prob,
        gbest,
        gpu_particles, opt::ParallelPSOKernel, ::Val{true};
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false
    )

    ## Initialize stuff

    backend = get_backend(gpu_particles)

    kernel = update_particle_states!(backend, 1024)

    lock = KernelAbstractions.allocate(backend, UInt32, 1)
    fill!(lock, UInt32(0))
    for i in 1:maxiters
        ## Invoke GPU Kernel here
        kernel(prob, gpu_particles, gbest, w, opt, lock; ndrange = length(gpu_particles))
        w = w * wdamp
    end

    return Array(gbest)[1], gpu_particles
end

function vectorized_solve!(
        prob,
        gbest,
        gpu_particles, opt::ParallelPSOKernel, ::Val{false};
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false
    )
    backend = get_backend(gpu_particles)

    kernel = update_particle_states_async!(backend)
    kernel(
        prob,
        gpu_particles,
        gbest,
        w,
        wdamp,
        maxiters,
        opt;
        ndrange = length(gpu_particles)
    )

    best_particle = minimum(gpu_particles)
    return SPSOGBest(best_particle.best_position, best_particle.best_cost), gpu_particles
end

function vectorized_solve!(
        prob, gbest,
        particles, opt::ParallelPSOArray;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        c1 = 1.4962f0,
        c2 = 1.4962f0,
        verbose = false
    )
    cost_func = prob.f
    num_particles = length(particles)
    rand_eltype = eltype(particles[1].velocity)
    # main loop

    for iter in 1:maxiters
        Threads.@threads for i in 1:num_particles
            particles[i].velocity .= w .* particles[i].velocity .+
                c1 .* rand.(rand_eltype) .*
                (
                particles[i].best_position .-
                    particles[i].position
            ) .+
                c2 .* rand.(rand_eltype) .*
                (gbest.position .- particles[i].position)

            particles[i].position .= particles[i].position .+ particles[i].velocity
            particles[i].position .= max.(particles[i].position, prob.lb)
            particles[i].position .= min.(particles[i].position, prob.ub)

            if !isnothing(prob.f.cons)
                penalty = calc_penalty(
                    particles[i].position,
                    prob,
                    iter + 1,
                    opt.θ,
                    opt.γ,
                    opt.h
                )
                particles[i].cost = prob.f(particles[i].position, prob.p) + penalty
            else
                particles[i].cost = prob.f(particles[i].position, prob.p)
            end

            if particles[i].cost < particles[i].best_cost
                copy!(particles[i].best_position, particles[i].position)
                particles[i].best_cost = particles[i].cost

                ## Possible race condition here
                if particles[i].best_cost < gbest.cost
                    copy!(gbest.position, particles[i].best_position)
                    gbest.cost = particles[i].best_cost
                end
            end
        end
        w = w * wdamp
    end
    return gbest, particles
end

function update_particle_states_cpu!(
        prob, particles, gbest_ref, w, iter, opt;
        c1 = 1.4962f0,
        c2 = 1.4962f0
    )
    gbest = gbest_ref[]

    for i in eachindex(particles)
        @inbounds particle = particles[i]
        particle = update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)

        if particle.best_cost < gbest.cost
            @set! gbest.position = particle.best_position
            @set! gbest.cost = particle.best_cost
        end

        particles[i] = particle
    end
    gbest_ref[] = gbest
    return nothing
end

function vectorized_solve!(
        prob,
        gbest,
        particles, opt::SerialPSO;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false
    )
    sol_ref = Ref(gbest)
    for i in 1:maxiters
        update_particle_states_cpu!(prob, particles, sol_ref, w, i, opt)
        w = w * wdamp
    end
    return sol_ref[], particles
end
