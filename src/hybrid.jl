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

# Largest step α with lb ≤ x + α d ≤ ub.
@inline _max_step(x, d, ::Nothing, ::Nothing) = eltype(x)(Inf)
@inline function _max_step(x, d, lb, ub)
    α = ifelse.(d .> 0, (ub .- x) ./ d, ifelse.(d .< 0, (lb .- x) ./ d, eltype(x)(Inf)))
    return max(zero(eltype(x)), minimum(α))
end

# BFGS with a weak-Wolfe line search (Lewis & Overton, 2013): a step only needs sufficient
# decrease and ∇f(x + α d)'d ≥ c2 ∇f(x)'d, found by doubling then bisection. Unlike a
# strong-Wolfe search it also terminates at kinks, so BFGS keeps converging on nonsmooth
# problems such as sharp ridges. Steps are capped at the box.
@inline function bfgs_weak_wolfe(
        grad_f, f, p, x::SVector{D, T}, lb, ub, maxiters, gtol; c1 = T(1.0e-4), c2 = T(0.9)
    ) where {D, T}
    eye = SMatrix{D, D, T}(ntuple(k -> T((k - 1) % D == (k - 1) ÷ D), Val(D * D)))
    H, fx, g = eye, T(f(x, p)), grad_f(x, p)
    for _ in 1:maxiters
        (isfinite(fx) && all(isfinite, g) && maximum(abs, g) > gtol) || break
        d = -(H * g)
        sum(g .* d) < 0 || (H = eye; d = -g)
        gd, α_max = sum(g .* d), _max_step(x, d, lb, ub)
        α_max > 0 || break
        lo, hi, α, ok = zero(T), T(Inf), min(one(T), α_max), false
        xn, fn, gn = x, fx, g
        for _ in 1:60
            xn = x .+ α .* d
            fn = T(f(xn, p))
            if fn <= fx + c1 * α * gd
                gn = grad_f(xn, p)
                (ok = sum(gn .* d) >= c2 * gd || α >= α_max) && break
                lo = α
            else
                hi = α
            end
            α = isfinite(hi) ? (lo + hi) / 2 : min(2α, α_max)
        end
        ok || break
        s, y = xn .- x, gn .- g
        sy = sum(s .* y)
        if sy > 0
            Hy = H * y
            H = H - (s * Hy' + Hy * s') / sy + (sum(y .* Hy) / sy + 1) / sy * (s * s')
        end
        x, fx, g = xn, fn, gn
    end
    return x, fx
end

@kernel function simplebfgs_run!(
        grad_f, f_raw, p, x0s, result, result_fx, lb, ub, maxiters, gtol
    )
    i = @index(Global, Linear)
    @inbounds u, v = bfgs_weak_wolfe(grad_f, f_raw, p, as_svector(x0s[i]), lb, ub, maxiters, gtol)
    @inbounds result[i] = u
    @inbounds result_fx[i] = isfinite(v) ? v : eltype(u)(Inf)
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, <:BFGS}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 50,
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

    grad_f = as_svector_grad(BoundedGrad(instantiate_gradient(f_raw, prob.f.adtype), lb, ub))

    x0s = sol_pso.original
    n = length(x0s)
    result = similar(x0s)
    # Entries a thread never writes (e.g. a kernel cut short by a display
    # watchdog) must not win the findmin below.
    result_fx = KernelAbstractions.allocate(opt.backend, T, n)
    fill!(result_fx, T(Inf))

    t0 = time()
    simplebfgs_run!(opt.backend)(
        grad_f, f_raw, p,
        x0s, result, result_fx,
        lb, ub, local_maxiters, abstol === nothing ? zero(T) : T(abstol);
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
    # Entries a thread never writes (e.g. a kernel cut short by a display
    # watchdog) must not win the findmin below.
    result_fx = KernelAbstractions.allocate(opt.backend, T, n)
    fill!(result_fx, T(Inf))

    t0 = time()
    grad_f = as_svector_grad(instantiate_gradient(f_raw, prob.f.adtype))
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
