@inline function update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)
    updated_velocity = w .* particle.velocity .+
        c1 .* rand(typeof(particle.velocity)) .*
        (
        particle.best_position -
            particle.position
    ) .+
        c2 .* rand(typeof(particle.velocity)) .*
        (gbest.position - particle.position)

    @set! particle.velocity = updated_velocity

    @set! particle.position = particle.position + particle.velocity

    update_pos = max.(particle.position, prob.lb)
    update_pos = min.(update_pos, prob.ub)
    @set! particle.position = update_pos

    particle = handle_constraints(particle, prob, iter, opt)

    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end
    return particle
end

@inline function handle_constraints(particle, prob, iter, opt)
    if !isnothing(prob.f.cons)
        penalty = calc_penalty(particle.position, prob, iter + 1, opt.θ, opt.γ, opt.h)
        @set! particle.cost = prob.f(particle.position, prob.p) + penalty
    else
        @set! particle.cost = prob.f(particle.position, prob.p)
    end
    return particle
end

@kernel function update_particle_states!(
        prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, gbest_ref, w,
        opt::ParallelPSOKernel, lock::AbstractArray{UInt32}; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {T1, T2}
    i = @index(Global, Linear)
    tidx = @index(Local, Linear)

    @uniform gs = @groupsize()[1]
    @uniform n = length(gpu_particles)

    queue_cost = @localmem T2 (gs)
    queue_idx = @localmem Int32 (gs)
    queue_num = @localmem UInt32 1

    particle = @private SPSOParticle{T1, T2} 1

    if i <= n
        @inbounds particle[1] = gpu_particles[i]
    end
    if tidx == 1
        queue_num[1] = UInt32(0)
    end

    @synchronize

    if i <= n
        @inbounds particle[1] = update_particle_state(
            particle[1],
            prob,
            gbest_ref[1],
            w,
            c1,
            c2,
            i,
            opt
        )
        @inbounds gpu_particles[i] = particle[1]
        @inbounds if particle[1].best_cost < gbest_ref[1].cost
            q = @atomic queue_num[1] += UInt32(1)
            @inbounds queue_cost[q] = particle[1].best_cost
            @inbounds queue_idx[q] = Int32(i)
        end
    end

    @synchronize

    if tidx == 1 && queue_num[1] > 0
        best = 1
        for j in 2:queue_num[1]
            @inbounds if queue_cost[j] < queue_cost[best]
                best = j
            end
        end
        @inbounds p = gpu_particles[queue_idx[best]]
        cand = SPSOGBest(p.best_position, p.best_cost)

        while true
            res = @atomicreplace lock[1] UInt32(0) => UInt32(1)
            if res.success
                break
            end
        end

        @inbounds if cand.cost < gbest_ref[1].cost
            gbest_ref[1] = cand
        end

        @atomicreplace lock[1] UInt32(1) => UInt32(0)
    end
end

@kernel function update_particle_states!(
        prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, block_particles, gbest, w,
        opt::ParallelSyncPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {T1, T2}
    i = @index(Global, Linear)
    tidx = @index(Local, Linear)
    gidx = @index(Group, Linear)

    @uniform gs = @groupsize()[1]
    @uniform n = length(gpu_particles)

    costs = @localmem T2 (gs)
    idxs = @localmem Int32 (gs)

    @inbounds costs[tidx] = convert(T2, Inf)
    @inbounds idxs[tidx] = Int32(0)

    if i <= n
        @inbounds particle = gpu_particles[i]
        particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)
        @inbounds gpu_particles[i] = particle
        @inbounds costs[tidx] = particle.best_cost
        @inbounds idxs[tidx] = Int32(tidx)
    end

    nactive = gs
    while nactive > 1
        half = cld(nactive, 2)
        @synchronize
        if tidx <= nactive - half
            @inbounds if costs[tidx + half] < costs[tidx]
                costs[tidx] = costs[tidx + half]
                idxs[tidx] = idxs[tidx + half]
            end
        end
        nactive = half
    end

    @synchronize

    if tidx == 1
        @inbounds win = idxs[1]
        if win == 0
            @inbounds block_particles[gidx] = SPSOGBest(
                gbest.position, convert(T2, Inf)
            )
        else
            @inbounds p = gpu_particles[i - tidx + win]
            @inbounds block_particles[gidx] = SPSOGBest(p.best_position, p.best_cost)
        end
    end
end

# Why you say we need a different code for CPUs for sync version? Turns out
# that you cannot do reduction within a kernel due to some bugs in KA.jl
# https://github.com/JuliaGPU/KernelAbstractions.jl/issues/330
@kernel function update_particle_states!(
        prob, gpu_particles, gbest, w,
        opt::ParallelSyncPSOKernel{Backend, T, G, H}; c1 = 1.4962f0,
        c2 = 1.4962f0
    ) where {Backend <: CPU, T, G, H}
    i = @index(Global, Linear)

    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)

    @inbounds gpu_particles[i] = particle
end

@kernel function update_particle_states_async!(
        prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, maxiters, opt;
        c1 = 1.4962f0,
        c2 = 1.4962f0
    )
    i = @index(Global, Linear)

    if i <= length(gpu_particles)
        gbest = gbest_ref[1]

        ## Access the particle
        @inbounds particle = gpu_particles[i]

        ## Run all generations
        for iter in 1:maxiters
            particle = update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)
            if particle.best_cost < gbest.cost
                @set! gbest.position = particle.best_position
                @set! gbest.cost = particle.best_cost
            end
            w = w * wdamp
        end

        @inbounds gpu_particles[i] = particle
        @inbounds gbest_ref[1] = gbest
    end
end
