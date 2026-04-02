using KernelAbstractions
using SciMLBase
using Optimization
using LineSearch
using SimpleNonlinearSolve
using NonlinearSolveBase: ImmutableNonlinearProblem
import SciMLBase: NonlinearFunction
import NonlinearSolveBase.Utils as NLBUtils

@inline (f::NonlinearFunction{false, G})(u, p) where {G} = f.f(u, p)

@inline NLBUtils.evaluate_f(prob::ImmutableNonlinearProblem, u) =
    prob.f.f(u, prob.p)

@inline NLBUtils.evaluate_f!!(prob::ImmutableNonlinearProblem, fu, u) =
    prob.f.f(u, prob.p)

@inline _unwrap_scalar(x::Real) = x
@inline _unwrap_scalar(x) = x[]

function _hybrid_bounds(prob, ::Val{d}, ::Type{T}) where {d, T}
    lb = prob.lb === nothing ? nothing : SVector{d, T}(prob.lb)
    ub = prob.ub === nothing ? nothing : SVector{d, T}(prob.ub)
    return lb, ub
end

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

@inline function _nlalg(local_opt::LBFGS, linesearch)
    return SimpleLimitedMemoryBroyden(; threshold = local_opt.threshold, linesearch)
end
@inline _nlalg(::BFGS, linesearch) = SimpleBroyden(; linesearch)

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
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 50,
        linesearch = StrongWolfeLineSearch(),
        kwargs...
    ) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}

    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters)
    best_u = sol_pso.u
    best_obj = _unwrap_scalar(sol_pso.objective)

    prob = cache.prob
    f_raw = prob.f.f
    p = prob.p
    T = eltype(prob.u0)
    d = length(prob.u0)
    lb, ub = _hybrid_bounds(prob, Val(d), T)

    grad_f = as_svector_grad(BoundedGrad(ForwardDiffGradient(f_raw), lb, ub))

    nlalg = _nlalg(opt.local_opt, linesearch)

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
        stats = Optimization.OptimizationStats(; time = solve_time),
    )
end
