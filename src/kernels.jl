# PSO acceleration coefficients. Kept as constants rather than `@kernel` keyword arguments:
# a keyword argument splits the kernel into a separate, non-inlined body function, and
# passing `prob` to it makes every GPU thread copy `prob` from parameter space onto its stack.
const PSO_C1 = 1.4962f0
const PSO_C2 = 1.4962f0

# StaticArrays' `rand(::Type{<:SArray})` is not inlined on the GPU, so each call returns its
# result through a stack slot.
@inline rand_static(::Type{S}) where {S <: SArray} =
    S(ntuple(_ -> rand(eltype(S)), Val(length(S))))

@inline function update_particle_state(
        particle, prob, gbest, w, c1, c2, iter, opt, pbest = particle.best_position
    )
    updated_velocity = w .* particle.velocity .+
        c1 .* rand_static(typeof(particle.velocity)) .*
        (
        pbest -
            particle.position
    ) .+
        c2 .* rand_static(typeof(particle.velocity)) .*
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

# Competitive learning (Cheng & Jin, 2015): in each random pair the loser learns from the
# winner and the swarm mean. `prev` is the generation's snapshot, so both see the same costs.
const CSO_PHI = 0.05f0

@inline function competitive_update(particle, i, (prev, partner, x̄, refine, g), prob, opt)
    particle = rescore(particle, prob, g, opt, prob.f.cons)
    @inbounds j = Int(partner[i])
    j == 0 && return particle
    @inbounds rival = score(prev[j], prob, g, opt, prob.f.cons)
    if isless((rival.cost, j), (particle.cost, i))
        S = typeof(particle.velocity)
        particle = update_particle_state(
            particle, prob, rival, rand_static(S), CSO_PHI, one(eltype(S)), g, opt, x̄
        )
        return record_best(particle)
    end
    return refine ? refine_winner(particle, prev, prob, g, opt) : particle
end

# Refinement runs: the winner tries a differential point and keeps it if no worse.
@inline function refine_winner(particle, prev, prob, g, opt)
    S = typeof(particle.position)
    T = eltype(S)
    n = length(prev)
    pick() = @inbounds prev[min(n, unsafe_trunc(Int, rand(T) * n) + 1)].position
    y = pick() .+ T(0.5) .* (pick() .- pick())
    jr = min(length(S), unsafe_trunc(Int, rand(T) * length(S)) + 1)
    take = SVector(ntuple(d -> d == jr || rand(T) < T(0.9), Val(length(S))))
    trial = particle
    @set! trial.position = min.(max.(S(ifelse.(take, y, particle.position)), prob.lb), prob.ub)
    trial = handle_constraints(trial, prob, g, opt)
    isless(particle.cost, trial.cost) && return particle
    return record_best(trial)
end

# `isless` orders NaN last, so a finite cost replaces a NaN best.
@inline function record_best(particle)
    if isless(particle.cost, particle.best_cost)
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end
    return particle
end

# Cost under generation `g`'s penalty weight (no-op without constraints). The weight can
# reorder a particle's two points, so `rescore` also records the cheaper one as its best.
@inline score(particle, prob, g, opt, ::Nothing) = particle
@inline score(particle, prob, g, opt, cons) = handle_constraints(particle, prob, g, opt)
@inline rescore(particle, prob, g, opt, ::Nothing) = particle
@inline function rescore(particle, prob, g, opt, cons)
    particle = handle_constraints(particle, prob, g, opt)
    best = particle
    @set! best.position = particle.best_position
    @set! particle.best_cost = handle_constraints(best, prob, g, opt).cost
    return record_best(particle)
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
        opt::ParallelPSOKernel, lock::AbstractArray{UInt32}
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
            PSO_C1,
            PSO_C2,
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
        opt::ParallelSyncPSOKernel, competition
    ) where {T1, T2}
    i = @index(Global, Linear)
    tidx = @index(Local, Linear)
    gidx = @index(Group, Linear)

    @uniform gs = @groupsize()[1]
    @uniform n = length(gpu_particles)

    costs = @localmem T2 (gs)
    idxs = @localmem Int32 (gs)

    # Rank with `isless` so NaN sorts after every number. Padding (i > n) fills the
    # block's highest slots with NaN, and a slot only takes a higher slot's candidate
    # when it is strictly less, so padding never displaces a real particle.
    @inbounds costs[tidx] = convert(T2, NaN)
    @inbounds idxs[tidx] = Int32(0)

    if i <= n
        @inbounds particle = gpu_particles[i]
        particle = competition === nothing ?
            update_particle_state(particle, prob, gbest, w, PSO_C1, PSO_C2, i, opt) :
            competitive_update(particle, i, competition, prob, opt)
        @inbounds gpu_particles[i] = particle
        @inbounds costs[tidx] = particle.best_cost
        @inbounds idxs[tidx] = Int32(tidx)
    end

    nactive = @private Int 1
    half = @private Int 1
    nactive[1] = gs
    while nactive[1] > 1
        half[1] = cld(nactive[1], 2)
        @synchronize
        if tidx <= nactive[1] - half[1]
            @inbounds if isless(costs[tidx + half[1]], costs[tidx])
                costs[tidx] = costs[tidx + half[1]]
                idxs[tidx] = idxs[tidx + half[1]]
            end
        end
        nactive[1] = half[1]
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
        opt::ParallelSyncPSOKernel{Backend, T, G, H}, competition
    ) where {Backend <: CPU, T, G, H}
    i = @index(Global, Linear)

    @inbounds particle = gpu_particles[i]

    particle = competition === nothing ?
        update_particle_state(particle, prob, gbest, w, PSO_C1, PSO_C2, i, opt) :
        competitive_update(particle, i, competition, prob, opt)

    @inbounds gpu_particles[i] = particle
end

@kernel function update_particle_states_async!(
        prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, maxiters, opt
    )
    i = @index(Global, Linear)

    if i <= length(gpu_particles)
        gbest = gbest_ref[1]

        ## Access the particle
        @inbounds particle = gpu_particles[i]

        ## Run all generations
        for iter in 1:maxiters
            particle = update_particle_state(particle, prob, gbest, w, PSO_C1, PSO_C2, iter, opt)
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
