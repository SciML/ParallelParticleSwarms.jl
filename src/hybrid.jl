# The local solve runs inside a GPU kernel, so it must stay allocation-free and
# isbits: `ImmutableNonlinearProblem` plus these unwrapping overloads keep
# SimpleNonlinearSolve off the mutable `NonlinearProblem`/`resid_prototype` path.
@inline (f::NonlinearFunction{false, G})(u, p) where {G} = f.f(u, p)

@inline NLBUtils.evaluate_f(prob::ImmutableNonlinearProblem, u) =
    prob.f.f(u, prob.p)

@inline NLBUtils.evaluate_f!!(prob::ImmutableNonlinearProblem, fu, u) =
    prob.f.f(u, prob.p)

@inline _unwrap_scalar(x::Real) = x
@inline _unwrap_scalar(x) = x[]

struct BoundedGrad{G, LB, UB}
    raw::G
    lb::LB
    ub::UB
end

@inline (bg::BoundedGrad{G, Nothing, Nothing})(θ, p) where {G} = as_svector(bg.raw(θ, p))

@inline function (bg::BoundedGrad)(θ, p)
    T = eltype(θ)
    w = bg.ub .- bg.lb
    in_box = all(isfinite, θ) &&
        all(θ .>= bg.lb .- T(2) .* w) &&
        all(θ .<= bg.ub .+ T(2) .* w)
    g = in_box ? bg.raw(θ, p) : map(_ -> T(1.0e15), θ)
    return as_svector(g)
end

@kernel function simplebfgs_run!(
        grad_f, f_raw, p, x0s, result, result_fx, nlalg, maxiters, abstol, reltol
    )
    i = @index(Global, Linear)
    @inbounds x0 = as_svector(x0s[i])
    nlprob = ImmutableNonlinearProblem{false}(NonlinearFunction{false}(grad_f), x0, p)
    sol = SciMLBase.solve(nlprob, nlalg; maxiters, abstol, reltol, grad_f = grad_f)
    u = as_svector(sol.u)
    T = eltype(u)
    v = f_raw(u, p)
    @inbounds result[i] = u
    @inbounds result_fx[i] = (isnan(v) | !isfinite(v)) ? T(Inf) : convert(T, v)
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, <:BFGS}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 50,
        linesearch = StrongWolfeLineSearch(),
        kwargs...
    ) where {Backend}

    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters)
    best_u = sol_pso.u
    best_obj = _unwrap_scalar(sol_pso.objective)

    prob = cache.prob
    f_raw = prob.f.f
    p = prob.p
    T = eltype(prob.u0)
    d = length(prob.u0)
    lb, ub = _static_bounds(prob, Val(d), T)

    grad_f = as_svector_grad(BoundedGrad(ForwardDiffGradient(f_raw), lb, ub))
    nlalg = SimpleBroyden(; linesearch)

    x0s = sol_pso.original
    n = length(x0s)
    result = similar(x0s)
    result_fx = KernelAbstractions.allocate(opt.backend, T, n)

    t0 = time()
    simplebfgs_run!(opt.backend)(
        grad_f, f_raw, p,
        x0s, result, result_fx,
        nlalg, local_maxiters, abstol, reltol;
        ndrange = n,
    )
    KernelAbstractions.synchronize(opt.backend)

    fx_host = Array(result_fx)
    minobj, ind = findmin(fx_host)
    if minobj < best_obj
        best_obj = minobj
        best_u = Array(result)[ind]
    end

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        best_u, best_obj;
        stats = OptimizationStats(; time = solve_time),
    )
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, <:SimpleLBFGS}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 50,
        linesearch = opt.local_opt.linesearch,
        kwargs...
    ) where {Backend}

    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters)
    best_u = sol_pso.u
    best_obj = _unwrap_scalar(sol_pso.objective)

    prob = cache.prob
    f_raw = prob.f.f
    p = prob.p
    T = eltype(prob.u0)
    d = length(prob.u0)
    lb, ub = _static_bounds(prob, Val(d), T)

    result = similar(sol_pso.original)
    particles = cache.pso_cache.particles
    n = length(particles)
    length(result) == n || throw(DimensionMismatch("particle and result counts must match"))
    result_fx = KernelAbstractions.allocate(opt.backend, T, n)

    t0 = time()
    grad_f = as_svector_grad(ForwardDiffGradient(f_raw))
    linesearch isa StrongWolfeLineSearch ||
        throw(ArgumentError("HybridPSO with SimpleLBFGS requires a StrongWolfeLineSearch"))
    typed_linesearch = StrongWolfeLineSearch(;
        autodiff = linesearch.autodiff,
        c1 = T(linesearch.c1), c2 = T(linesearch.c2),
        α_init = T(linesearch.α_init), α_max = T(linesearch.α_max),
        maxiters = linesearch.maxiters, zoom_maxiters = linesearch.zoom_maxiters,
    )
    lbfgs_run!(opt.backend)(
        grad_f, f_raw, p, particles, result, result_fx, lb, ub,
        local_maxiters, abstol, reltol, typed_linesearch,
        SimpleOptimization.__get_threshold(opt.local_opt);
        ndrange = n,
    )
    KernelAbstractions.synchronize(opt.backend)

    fx_host = Array(result_fx)
    minobj, ind = findmin(fx_host)
    if minobj < best_obj
        best_obj = minobj
        best_u = Array(@view result[ind:ind])[1]
    end

    solve_time = (time() - t0) + sol_pso.stats.time
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        best_u, best_obj;
        stats = OptimizationStats(; time = solve_time),
    )
end
