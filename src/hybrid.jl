@kernel function simplebfgs_run!(nlprob, x0s, result, opt, maxiters, abstol, reltol)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    sol = solve(nlcache, opt; maxiters, abstol, reltol)
    @inbounds result[i] = sol.u
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100, local_maxiters = 10, kwargs...
    ) where {
        Backend, LocalOpt <: Union{LBFGS, BFGS},
    }
    pso_cache = cache.pso_cache

    sol_pso = solve!(pso_cache)
    x0s = sol_pso.original

    backend = opt.backend

    prob = remake(cache.prob, lb = nothing, ub = nothing)

    result = cache.start_points
    copyto!(result, x0s)

    ∇f = instantiate_gradient(prob.f.f, prob.f.adtype)

    kernel = simplebfgs_run!(backend)
    nlprob = SimpleNonlinearSolve.ImmutableNonlinearProblem{false}(∇f, prob.u0, prob.p)

    nlalg = opt.local_opt isa LBFGS ?
        SimpleLimitedMemoryBroyden(;
            threshold = opt.local_opt.threshold,
            linesearch = Val(true)
        ) : SimpleBroyden(; linesearch = Val(true))

    t0 = time()
    kernel(
        nlprob,
        x0s,
        result,
        nlalg,
        local_maxiters,
        abstol,
        reltol;
        ndrange = length(x0s)
    )

    sol_bfgs = (x -> prob.f(x, prob.p)).(result)
    sol_bfgs = (x -> isnan(x) ? convert(eltype(prob.u0), Inf) : x).(sol_bfgs)

    minobj, ind = findmin(sol_bfgs)
    sol_u,
        sol_obj = minobj > sol_pso.objective ? (sol_pso.u, sol_pso.objective) :
        (view(result, ind), minobj)
    t1 = time()

    # @show sol_pso.stats.time

    solve_time = (t1 - t0) + sol_pso.stats.time

    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        sol_u, sol_obj,
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end
