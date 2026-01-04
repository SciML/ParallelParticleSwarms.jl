function get_pos(particle)
    return particle.position
end

function SciMLBase.solve!(
        cache::Union{PSOCache, HybridPSOCache}, args...; maxiters = 100, kwargs...
    )
    return solve!(cache, cache.alg, args...; maxiters, kwargs...)
end

function SciMLBase.solve!(
        cache::PSOCache, opt::ParallelPSOKernel, args...; maxiters = 100, kwargs...
    )
    prob = cache.prob
    t0 = time()
    gbest,
        particles = vectorized_solve!(
        cache.prob,
        cache.gbest,
        cache.particles,
        opt,
        Val(opt.global_update),
        args...;
        maxiters, kwargs...
    )
    t1 = time()

    particles_positions = get_pos.(particles)
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, prob.f(gbest.position, prob.p), original = particles_positions,
        stats = Optimization.OptimizationStats(; time = t1 - t0)
    )
end

function SciMLBase.solve!(
        cache::PSOCache, opt::ParallelSyncPSOKernel, args...; maxiters = 100, kwargs...
    )
    prob = cache.prob
    t0 = time()
    gbest,
        particles = vectorized_solve!(
        prob,
        cache.gbest,
        cache.particles,
        opt,
        args...;
        maxiters,
        kwargs...
    )
    t1 = time()

    particles_positions = get_pos.(particles)
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, prob.f(gbest.position, prob.p), original = particles_positions,
        stats = Optimization.OptimizationStats(; time = t1 - t0)
    )
end

function SciMLBase.solve(
        prob::OptimizationProblem,
        opt::Union{ParallelPSOKernel, ParallelSyncPSOKernel, HybridPSO},
        args...; maxiters = 100, kwargs...
    )
    return solve!(init(prob, opt, args...; kwargs...), opt, args...; maxiters, kwargs...)
end

function SciMLBase.__solve(
        prob::OptimizationProblem,
        opt::PSOAlgorithm,
        args...;
        maxiters = 100,
        kwargs...
    )
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    gbest, particles, solve_time = pso_solve(prob, opt, args...; maxiters, kwargs...)
    particles_positions = get_pos.(particles)
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, prob.f(gbest.position, prob.p), original = particles_positions,
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end

function pso_solve(
        prob::OptimizationProblem,
        opt::ParallelPSOArray,
        args...;
        kwargs...
    )
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    t0 = time()
    gbest,
        particles = vectorized_solve!(
        prob,
        init_gbest,
        particles,
        opt,
        args...;
        kwargs...
    )
    t1 = time()
    return gbest, particles, t1 - t0
end

function pso_solve(prob::OptimizationProblem, opt::SerialPSO, args...; kwargs...)
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    t0 = time()
    gbest, particles = vectorized_solve!(prob, init_gbest, particles, opt; kwargs...)
    t1 = time()
    return gbest, particles, t1 - t0
end
