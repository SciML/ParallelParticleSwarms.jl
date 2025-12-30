mutable struct PSOCache{TP, TAlg, TPart, TGbest, TSampler}
    prob::TP
    alg::TAlg
    particles::TPart
    gbest::TGbest
    sampler::TSampler
end

struct HybridPSOCache{TPc, TSp, TAlg}
    pso_cache::TPc
    start_points::TSp
    alg::TAlg
end

function __init!(particles, prob::OptimizationProblem,
        opt::Union{ParallelPSOKernel, ParallelSyncPSOKernel}, sampler::T,
        args...; kwargs...) where {T <: QuasiMonteCarlo.SamplingAlgorithm}
    backend = opt.backend

    qmc_samples = QuasiMonteCarlo.sample(opt.num_particles, prob.lb, prob.ub, sampler)

    qmc_samples = adapt(backend, qmc_samples)

    kernel! = gpu_init_particles!(backend)

    kernel!(
        particles, qmc_samples, prob, opt, typeof(prob.u0), T; ndrange = opt.num_particles)

    best_particle = minimum(particles)
    init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    return particles, init_gbest
end

function __init!(particles, prob::OptimizationProblem,
        opt::Union{ParallelPSOKernel, ParallelSyncPSOKernel}, sampler::T,
        args...; kwargs...) where {T <: GPUSamplingAlgorithm}
    backend = opt.backend

    kernel! = gpu_init_particles!(backend)

    kernel!(particles, prob, opt, typeof(prob.u0), T; ndrange = opt.num_particles)

    best_particle = minimum(particles)

    init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    particles, init_gbest
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::ParallelPSOKernel, args...; sampler = GPUUniformSampler(), kwargs...)
    @assert prob.u0 isa SArray

    ## Bounds check
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    particles = KernelAbstractions.allocate(
        opt.backend, SPSOParticle{typeof(prob.u0), eltype(typeof(prob.u0))}, opt.num_particles)

    _sampler = if lb === nothing || ub === nothing || (all(isinf, lb) && all(isinf, ub))
        GPUUnboundedSampler()
    else
        sampler
    end

    particles, _init_gbest = __init!(particles, prob, opt, _sampler, args...; kwargs...)

    init_gbest = KernelAbstractions.allocate(opt.backend, typeof(_init_gbest), (1,))
    copyto!(init_gbest, [_init_gbest])

    return PSOCache{
        typeof(prob), typeof(opt), typeof(particles), typeof(init_gbest), typeof(_sampler)}(
        prob, opt, particles, init_gbest, _sampler)
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::ParallelSyncPSOKernel, args...; sampler = GPUUniformSampler(), kwargs...)
    @assert prob.u0 isa SArray

    ## Bounds check
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    particles = KernelAbstractions.allocate(
        opt.backend, SPSOParticle{typeof(prob.u0), eltype(typeof(prob.u0))}, opt.num_particles)

    _sampler = if lb === nothing || ub === nothing || (all(isinf, lb) && all(isinf, ub))
        GPUUnboundedSampler()
    else
        sampler
    end

    particles, init_gbest = __init!(particles, prob, opt, _sampler, args...; kwargs...)

    return PSOCache{
        typeof(prob), typeof(opt), typeof(particles), typeof(init_gbest), typeof(_sampler)}(
        prob, opt, particles, init_gbest, _sampler)
end

function SciMLBase.reinit!(cache::Union{PSOCache, HybridPSOCache})
    reinit_cache!(cache, cache.alg)
end

function reinit_cache!(cache::PSOCache, opt::ParallelPSOKernel)
    prob = cache.prob
    particles = cache.particles

    particles, _init_gbest = __init!(particles, prob, opt, cache.sampler)

    copyto!(cache.gbest, [_init_gbest])

    return nothing
end

function reinit_cache!(cache::PSOCache, opt::ParallelSyncPSOKernel)
    prob = cache.prob
    particles = cache.particles

    particles, init_gbest = __init!(particles, prob, opt, cache.sampler)

    cache.gbest = init_gbest

    return nothing
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::HybridPSO{Backend, LocalOpt}, args...;
        kwargs...) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}
    psoalg = opt.pso
    backend = opt.backend

    pso_cache = init(prob, psoalg, args...; kwargs...)

    start_points = KernelAbstractions.allocate(
        backend, typeof(prob.u0), opt.pso.num_particles)

    return HybridPSOCache{
        typeof(pso_cache), typeof(start_points), typeof(opt)}(pso_cache, start_points, opt)
end

function reinit_cache!(cache::HybridPSOCache,
        opt::HybridPSO{Backend, LocalOpt}) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}
    reinit!(cache.pso_cache)
    fill!(cache.start_points, zero(eltype(cache.start_points)))
    return nothing
end

function Base.getproperty(cache::HybridPSOCache, name::Symbol)
    if name ∈ (:start_points, :pso_cache, :alg)
        return getfield(cache, name)
    else
        return getproperty(cache.pso_cache, name)
    end
end

function Base.setproperty!(cache::HybridPSOCache, name::Symbol, val)
    if name ∈ (:start_points, :pso_cache, :alg)
        error("Cannot set property $name on immutable HybridPSOCache")
    else
        return setproperty!(getfield(cache, :pso_cache), name, val)
    end
end
