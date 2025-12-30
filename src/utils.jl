import SciMLBase: @add_kwonly, AbstractNonlinearProblem, AbstractNonlinearFunction,
                  AbstractODEFunction, AbstractODEProblem, warn_paramtype, ConstructionBase,
                  NullParameters, StandardNonlinearProblem, @reset, updated_u0_p,
                  remake_initialization_data, maybe_eager_initialize_problem

@inbounds function uniform_itr(
        dim::Int, lb::AbstractArray{T}, ub::AbstractArray{T}) where {T}
    (rand(T) * (ub[i] - lb[i]) + lb[i] for i in 1:dim)
end

function uniform(dim::Int, lb::AbstractArray{T}, ub::AbstractArray{T}) where {T}
    arr = rand(T, dim)
    @inbounds for i in 1:dim
        arr[i] = arr[i] * (ub[i] - lb[i]) + lb[i]
    end
    return arr
end

function _gen_sampling_kernel(prob, num_particles::Int, sampling::QuasiMonteCarlo.SamplingAlgorithm)
    lb = prob.lb
    ub = prob.ub

    positions = QuasiMonteCarlo.sample(num_particles, lb, ub, sampling)
    return positions
end

@kernel function gpu_init_particles!(
        particles, prob, opt, ::Type{T}, ::Type{GPUUnboundedSampler}) where {T <: SArray}
    i = @index(Global, Linear)

    dim = length(prob.u0)
    cost_func = prob.f
    p = prob.p

    position = StaticArrays.sacollect(T,
        ifelse(abs(prob.u0[i]) > 0,
            prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i]),
            rand(eltype(prob.u0))) for i in 1:dim)

    velocity = zero(T)

    if !isnothing(prob.f.cons)
        penalty = calc_penalty(position, prob, 1, opt.θ, opt.γ, opt.h)
        cost = cost_func(position, p) + penalty
    else
        cost = cost_func(position, p)
    end

    best_position = position
    best_cost = cost
    @inbounds particles[i] = SPSOParticle(
        position, velocity, cost, best_position, best_cost)
end

@kernel function gpu_init_particles!(
        particles, prob, opt, ::Type{T}, ::Type{GPUUniformSampler}) where {T <: SArray}
    i = @index(Global, Linear)

    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p

    position = StaticArrays.sacollect(T, uniform_itr(dim, lb, ub))

    velocity = zero(T)

    if !isnothing(prob.f.cons)
        penalty = calc_penalty(position, prob, 1, opt.θ, opt.γ, opt.h)
        cost = cost_func(position, p) + penalty
    else
        cost = cost_func(position, p)
    end

    best_position = position
    best_cost = cost
    @inbounds particles[i] = SPSOParticle(
        position, velocity, cost, best_position, best_cost)
end

@kernel function gpu_init_particles!(particles, qmc_particles, prob, opt, ::Type{T},
        ::Type{T1}) where {T <: SArray, T1 <: QuasiMonteCarlo.SamplingAlgorithm}
    i = @index(Global, Linear)

    dim = length(prob.u0)
    cost_func = prob.f
    p = prob.p

    @inbounds position = StaticArrays.sacollect(T, qmc_particles[j, i] for j in 1:dim)
    velocity = zero(T)

    if !isnothing(prob.f.cons)
        penalty = calc_penalty(position, prob, 1, opt.θ, opt.γ, opt.h)
        cost = cost_func(position, p) + penalty
    else
        cost = cost_func(position, p)
    end

    best_position = position
    best_cost = cost

    @inbounds particles[i] = SPSOParticle(
        position, velocity, cost, best_position, best_cost)
end

function init_particles!(particles, prob, opt, ::Type{T}) where {T <: SArray}
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p
    num_particles = opt.num_particles

    if lb === nothing || (all(isinf, lb) && all(isinf, ub))
        gbest_position = StaticArrays.sacollect(T,
            ifelse(
                abs(prob.u0[i]) > 0, prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i]),
                rand(eltype(prob.u0))) for i in 1:dim)
    else
        gbest_position = StaticArrays.sacollect(T, uniform_itr(dim, lb, ub))
    end

    gbest_position = convert(T, gbest_position)
    gbest_cost = cost_func(gbest_position, p)
    if !isnothing(prob.f.cons)
        penalty = calc_penalty(gbest_position, prob, 1, opt.θ, opt.γ, opt.h)
        gbest_cost = cost_func(gbest_position, p) + penalty
    else
        gbest_cost = cost_func(gbest_position, p)
    end
    gbest_cost = cost_func(gbest_position, p)
    # particles = SPSOParticle[]

    if !(lb === nothing || (all(isinf, lb) && all(isinf, ub)))
        positions = QuasiMonteCarlo.sample(num_particles, lb, ub, LatinHypercubeSample())
    end

    for i in 1:num_particles
        if lb === nothing || (all(isinf, lb) && all(isinf, ub))
            position = StaticArrays.sacollect(T,
                ifelse(abs(prob.u0[i]) > 0,
                    prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i]),
                    rand(eltype(prob.u0))) for i in 1:dim)
        else
            @inbounds position = StaticArrays.sacollect(T, positions[j, i] for j in 1:dim)
        end

        velocity = zero(T)

        if !isnothing(prob.f.cons)
            penalty = calc_penalty(position, prob, 1, opt.θ, opt.γ, opt.h)
            cost = cost_func(position, p) + penalty
        else
            cost = cost_func(position, p)
        end

        best_position = position
        best_cost = cost
        @inbounds particles[i] = SPSOParticle(
            position, velocity, cost, best_position, best_cost)

        if best_cost < gbest_cost
            gbest_position = best_position
            gbest_cost = best_cost
        end
    end
    gbest = SPSOGBest(gbest_position, gbest_cost)
    return gbest, particles
end

function init_particles(prob, opt, ::Type{T}) where {T <: SArray}
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p
    num_particles = opt.num_particles

    unbounded = lb === nothing || (all(isinf, lb) && all(isinf, ub))

    if unbounded
        gbest_position = StaticArrays.sacollect(T,
            ifelse(
                abs(prob.u0[i]) > 0, prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i]),
                rand(eltype(prob.u0))) for i in 1:dim)
    else
        gbest_position = StaticArrays.sacollect(T, uniform_itr(dim, lb, ub))
    end

    gbest_cost = cost_func(gbest_position, p)
    if !isnothing(prob.f.cons)
        penalty = calc_penalty(gbest_position, prob, 1, opt.θ, opt.γ, opt.h)
        gbest_cost = cost_func(gbest_position, p) + penalty
    else
        gbest_cost = cost_func(gbest_position, p)
    end
    particles = SPSOParticle{T, eltype(T)}[]

    positions = unbounded ? nothing : QuasiMonteCarlo.sample(num_particles, lb, ub, LatinHypercubeSample())

    for i in 1:num_particles
        if unbounded
            @inbounds position = StaticArrays.sacollect(T,
                ifelse(abs(prob.u0[j]) > 0,
                    prob.u0[j] + rand(eltype(prob.u0)) * abs(prob.u0[j]),
                    rand(eltype(prob.u0))) for j in 1:dim)
        else
            @inbounds position = StaticArrays.sacollect(T, positions[j, i] for j in 1:dim)
        end
        velocity = zero(T)

        if !isnothing(prob.f.cons)
            penalty = calc_penalty(position, prob, 1, opt.θ, opt.γ, opt.h)
            cost = cost_func(position, p) + penalty
        else
            cost = cost_func(position, p)
        end

        best_position = position
        best_cost = cost
        push!(particles, SPSOParticle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest_cost
            gbest_position = best_position
            gbest_cost = best_cost
        end
    end
    gbest = SPSOGBest(gbest_position, gbest_cost)
    return gbest, particles
end

function init_particles(prob, opt, ::Type{T}) where {T <: AbstractArray}
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p
    num_particles = opt.num_particles

    unbounded = lb === nothing || (all(isinf, lb) && all(isinf, ub))

    if unbounded
        gbest_position = Array{eltype(prob.u0), 1}(undef, dim)
        for i in 1:dim
            if abs(prob.u0[i]) > 0
                gbest_position[i] = prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i])
            else
                gbest_position[i] = rand(eltype(prob.u0))
            end
        end
    else
        gbest_position = uniform(dim, lb, ub)
    end
    gbest = MPSOGBest(gbest_position, cost_func(gbest_position, prob.p))

    particles = MPSOParticle[]

    positions = unbounded ? nothing : QuasiMonteCarlo.sample(num_particles, lb, ub, LatinHypercubeSample())

    for i in 1:num_particles
        if unbounded
            position = Array{eltype(prob.u0), 1}(undef, dim)
            for j in 1:dim
                if abs(prob.u0[j]) > 0
                    position[j] = prob.u0[j] + rand(eltype(prob.u0)) * abs(prob.u0[j])
                else
                    position[j] = rand(eltype(prob.u0))
                end
            end
        else
            position = @view positions[:, i]
        end
        velocity = zeros(eltype(position), dim)
        cost = cost_func(position, prob.p)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, MPSOParticle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest.cost
            gbest.position = copy(best_position)
            gbest.cost = copy(best_cost)
        end
    end
    return gbest, convert(Vector{typeof(particles[1])}, particles)
end

function check_init_bounds(prob)
    lb = prob.lb === nothing ? fill(eltype(prob.u0)(-Inf), length(prob.u0)) : prob.lb
    ub = prob.ub === nothing ? fill(eltype(prob.u0)(Inf), length(prob.u0)) : prob.ub
    if prob.u0 isa SArray
        lb = SVector{length(lb), eltype(lb)}(lb)
        ub = SVector{length(ub), eltype(ub)}(ub)
    end
    lb, ub
end

@inline function θ_default(x::T) where {T <: Number}
    if x < 0.001
        return T(10.0)
    elseif x <= 0.1
        return T(20.0)
    elseif x <= 1.0
        return T(100.0)
    else
        return T(300.0)
    end
end

@inline function γ_default(x::T) where {T <: Number}
    if x < 1
        return T(1)
    else
        return T(2)
    end
end

"""
Based on the paper: Particle swarm optimization method for constrained optimization problems

@article{parsopoulos2002particle,
  title={Particle swarm optimization method for constrained optimization problems},
  author={Parsopoulos, Konstantinos E and Vrahatis, Michael N and others},
  journal={Intelligent technologies--theory and application: New trends in intelligent technologies},
  volume={76},
  number={1},
  pages={214--220},
  year={2002}
}
"""
@inline function calc_penalty(u,
        prob::OptimizationProblem,
        iter, θ, γ, h)
    T = eltype(u)
    cons_ret = prob.f.cons(u, prob.p)
    q = max.(cons_ret, T(0))
    thetaq = θ.(q)
    gammaq = γ.(q)
    penalty = T(0)
    penalty = sum(thetaq)

    for i in eachindex(thetaq)
        @fastmath pen_pow = q[i]^gammaq[i]
        penalty += thetaq[i] * pen_pow
    end
    penalty = h(T(iter)) * penalty
    penalty
end

#TODO: Possible migration to DifferentiationInterface.jl,
# however I cannot compile GPU-compatible gradients with Enzyme as Mar 2025
@inline function instantiate_gradient(f, adtype::AutoForwardDiff)
    (θ, p) -> ForwardDiff.gradient(x -> f(x, p), θ)
end

@inline function instantiate_gradient(f, adtype::AutoEnzyme)
    (θ, p) -> autodiff_deferred(Reverse, Const(x -> f(x, p)), Active, Active(θ))[1][1]
end
