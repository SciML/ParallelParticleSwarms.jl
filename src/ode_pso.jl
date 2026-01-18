@kernel function ode_update_particle_states!(
        gpu_particles, lb, ub, gbest, w; c1 = 1.4962f0,
        c2 = 1.4962f0
    )
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        @inbounds particle = gpu_particles[i]

        updated_velocity = w .* particle.velocity .+
            c1 .* rand(typeof(particle.velocity)) .*
            (particle.best_position - particle.position) .+
            c2 .* rand(typeof(particle.velocity)) .*
            (gbest.position - particle.position)

        @set! particle.velocity = updated_velocity
        @set! particle.position = particle.position + particle.velocity

        update_pos = max(particle.position, lb)
        update_pos = min(update_pos, ub)

        @set! particle.position = update_pos
        @inbounds gpu_particles[i] = particle
    end
end

@kernel function ode_update_particle_costs!(losses, gpu_particles)
    i = @index(Global, Linear)
    if i <= length(losses)
        @inbounds particle = gpu_particles[i]
        @inbounds loss = losses[i]

        @set! particle.cost = loss

        if particle.cost < particle.best_cost
            @set! particle.best_position = particle.position
            @set! particle.best_cost = particle.cost
        end

        @inbounds gpu_particles[i] = particle
    end
end

function default_prob_func(prob, gpu_particle)
    return remake(prob, p = gpu_particle.position)
end

function parameter_estim_ode!(
        prob::SciMLBase.AbstractODEProblem, cache,
        lb,
        ub, ::Val{true};
        ode_alg = GPUTsit5(),
        prob_func = default_prob_func,
        w = 0.7298f0,
        wdamp = 1.0f0,
        maxiters = 100, kwargs...
    )
    (losses, gpu_particles, gpu_data, gbest, probs) = cache
    backend = get_backend(gpu_particles)
    update_states! = ParallelParticleSwarms.ode_update_particle_states!(backend)
    update_costs! = ParallelParticleSwarms.ode_update_particle_costs!(backend)

    for i in 1:maxiters
        update_states!(
            gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange = length(gpu_particles)
        )

        KernelAbstractions.synchronize(backend)

        probs = prob_func.(probs, gpu_particles)

        KernelAbstractions.synchronize(backend)
        ###TODO: Somehow vectorized_asolve hangs and does not here :(

        ts, us = vectorized_asolve(
            probs,
            prob,
            ode_alg; kwargs...
        )

        KernelAbstractions.synchronize(backend)

        loss_mat = map(x -> sum(x .^ 2), gpu_data .- us)
        loss_view = ndims(losses) == 1 ? reshape(losses, 1, :) : losses
        sum!(loss_view, loss_mat)

        update_costs!(losses, gpu_particles; ndrange = length(losses))

        KernelAbstractions.synchronize(backend)

        best_particle = minimum(gpu_particles)

        KernelAbstractions.synchronize(backend)

        gbest = ParallelParticleSwarms.SPSOGBest(
            best_particle.best_position, best_particle.best_cost
        )
        w = w * wdamp
    end
    return gbest
end

function parameter_estim_ode!(
        prob::SciMLBase.AbstractODEProblem, cache,
        lb,
        ub, ::Val{false};
        ode_alg = GPUTsit5(),
        prob_func = default_prob_func,
        w = 0.7298f0,
        wdamp = 1.0f0,
        maxiters = 100, kwargs...
    )
    (losses, gpu_particles, gpu_data, gbest, probs) = cache
    backend = get_backend(gpu_particles)
    update_states! = ParallelParticleSwarms.ode_update_particle_states!(backend)
    update_costs! = ParallelParticleSwarms.ode_update_particle_costs!(backend)

    for i in 1:maxiters
        update_states!(
            gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange = length(gpu_particles)
        )

        KernelAbstractions.synchronize(backend)

        probs = prob_func.(probs, gpu_particles)

        KernelAbstractions.synchronize(backend)
        ###TODO: Somehow vectorized_asolve hangs and does not here :(

        ts, us = vectorized_solve(
            probs,
            prob,
            ode_alg; kwargs...
        )

        KernelAbstractions.synchronize(backend)

        loss_mat = map(x -> sum(x .^ 2), gpu_data .- us)
        loss_view = ndims(losses) == 1 ? reshape(losses, 1, :) : losses
        sum!(loss_view, loss_mat)

        update_costs!(losses, gpu_particles; ndrange = length(losses))

        KernelAbstractions.synchronize(backend)

        best_particle = minimum(gpu_particles)

        KernelAbstractions.synchronize(backend)

        gbest = ParallelParticleSwarms.SPSOGBest(
            best_particle.best_position, best_particle.best_cost
        )
        w = w * wdamp
    end
    return gbest
end
