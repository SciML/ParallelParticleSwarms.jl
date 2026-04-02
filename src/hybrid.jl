using LinearAlgebra: norm, dot
using KernelAbstractions
using SciMLBase
using Optimization

# Clamp x to interior of bounds and evaluate objective
@inline function _safe_eval(f, p, x, lb, ub)
    T = eltype(x)
    ε = T(1.0e-6)
    xc = clamp.(x, lb .+ ε, ub .- ε)
    xc = map(xi -> abs(xi) < ε ? ε : xi, xc)
    v = f(xc, p)
    return ifelse(isfinite(v), v, T(Inf))
end

# Clamp x to interior of bounds and evaluate gradient
@inline function _safe_grad(grad_f, p, x, lb, ub)
    T = eltype(x)
    ε = T(1.0e-6)
    xc = clamp.(x, lb .+ ε, ub .- ε)
    xc = map(xi -> abs(xi) < ε ? ε : xi, xc)
    g = grad_f(xc, p)
    return map(gi -> ifelse(isfinite(gi), gi, zero(gi)), g)
end

# Cubic interpolation for line search; falls back to bisection if non-convex
@inline function _interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo, dϕ_hi)
    d1 = dϕ_lo + dϕ_hi - 3 * (ϕ_lo - ϕ_hi) / (a_lo - a_hi)
    desc = d1 * d1 - dϕ_lo * dϕ_hi
    # Use max to ensure non-negative argument to sqrt (avoids DomainError with ForwardDiff)
    d2 = sqrt(max(desc, zero(desc)))
    return ifelse(desc < 0, (a_lo + a_hi) / 2, a_hi - (a_hi - a_lo) * ((dϕ_hi + d2 - d1) / (dϕ_hi - dϕ_lo + 2 * d2)))
end

# Zoom phase of Strong Wolfe line search (Nocedal & Wright Algorithm 3.6)
@inline function _zoom(f, grad_f, p, x, dir, lb, ub, a_lo, a_hi, ϕ_0, dϕ_0, ϕ_lo, c1, c2)
    T = eltype(x)
    xn_out, ϕ_out, gn_out, ok_out = x, ϕ_0, _safe_grad(grad_f, p, x, lb, ub), false
    done = false
    for _ in 1:10
        if !done
            a_j = _interpolate(
                a_lo, a_hi, ϕ_lo, _safe_eval(f, p, clamp.(x + a_hi * dir, lb, ub), lb, ub),
                dot(_safe_grad(grad_f, p, clamp.(x + a_lo * dir, lb, ub), lb, ub), dir),
                dot(_safe_grad(grad_f, p, clamp.(x + a_hi * dir, lb, ub), lb, ub), dir)
            )
            a_j = clamp(a_j, min(a_lo, a_hi) + T(1.0e-3), max(a_lo, a_hi) - T(1.0e-3))
            xn_j = clamp.(x + a_j * dir, lb, ub)
            ϕ_j = _safe_eval(f, p, xn_j, lb, ub)
            if (ϕ_j > ϕ_0 + c1 * a_j * dϕ_0) || (ϕ_j >= ϕ_lo)
                a_hi = a_j
            else
                gn_j = _safe_grad(grad_f, p, xn_j, lb, ub)
                dϕ_j = dot(gn_j, dir)
                if abs(dϕ_j) <= -c2 * dϕ_0
                    xn_out, ϕ_out, gn_out, ok_out, done = xn_j, ϕ_j, gn_j, true, true
                else
                    if dϕ_j * (a_hi - a_lo) >= zero(T)
                        a_hi = a_lo
                    end
                    a_lo, ϕ_lo = a_j, ϕ_j
                end
            end
        end
    end
    if !done
        xn_lo = clamp.(x + a_lo * dir, lb, ub)
        xn_out, ϕ_out, gn_out = xn_lo, _safe_eval(f, p, xn_lo, lb, ub), _safe_grad(grad_f, p, xn_lo, lb, ub)
    end
    return (xn_out, ϕ_out, gn_out, ok_out)
end

# Strong Wolfe line search (Nocedal & Wright Algorithm 3.5)
@inline function _strong_wolfe(f, grad_f, p, x, fx, g, dir, lb, ub)
    T = eltype(x)
    c1, c2 = T(1.0e-4), T(0.9)
    dϕ_0 = dot(g, dir)
    xn_out, ϕ_out, gn_out, ok_out = x, fx, g, false
    if dϕ_0 < zero(T)
        a_prev, a_i, ϕ_0, ϕ_prev, done = zero(T), one(T), fx, fx, false
        for i in 1:10
            if !done
                xn = clamp.(x + a_i * dir, lb, ub)
                ϕ_i = _safe_eval(f, p, xn, lb, ub)
                if (ϕ_i > ϕ_0 + c1 * a_i * dϕ_0) || (ϕ_i >= ϕ_prev && i > 1)
                    xn_out, ϕ_out, gn_out, ok_out, done = _zoom(f, grad_f, p, x, dir, lb, ub, a_prev, a_i, ϕ_0, dϕ_0, ϕ_prev, c1, c2)..., true
                else
                    gn_i = _safe_grad(grad_f, p, xn, lb, ub)
                    dϕ_i = dot(gn_i, dir)
                    if abs(dϕ_i) <= -c2 * dϕ_0
                        xn_out, ϕ_out, gn_out, ok_out, done = xn, ϕ_i, gn_i, true, true
                    elseif dϕ_i >= zero(T)
                        xn_out, ϕ_out, gn_out, ok_out, done = _zoom(f, grad_f, p, x, dir, lb, ub, a_i, a_prev, ϕ_0, dϕ_0, ϕ_i, c1, c2)..., true
                    else
                        a_prev, ϕ_prev, a_i = a_i, ϕ_i, a_i * T(2.0)
                    end
                end
            end
        end
        if !done
            xn = clamp.(x + a_prev * dir, lb, ub)
            xn_out, ϕ_out, gn_out, ok_out = xn, _safe_eval(f, p, xn, lb, ub), _safe_grad(grad_f, p, xn, lb, ub), true
        end
    end
    return (xn_out, ϕ_out, gn_out, ok_out)
end

# L-BFGS two-loop recursion (Nocedal & Wright Algorithm 7.4)
@inline function _lbfgs_dir(g, S, Y, Rho, ::Val{M}, k) where {M}
    T = eltype(g)
    q, a = g, ntuple(_ -> zero(T), Val(M))
    for j in 0:(M - 1)
        idx = k - j
        if idx >= 1
            ii = mod1(idx, M)
            a = Base.setindex(a, Rho[ii] * dot(S[ii], q), ii)
            q = q - a[ii] * Y[ii]
        end
    end
    kk = mod1(k, M)
    yy = sum(abs2, Y[kk])
    γ = ifelse(k >= 1 && yy > T(1.0e-30), dot(S[kk], Y[kk]) / yy, one(T))
    γ = ifelse(isfinite(γ) && γ > zero(T), γ, one(T))
    r = γ * q
    for j in (M - 1):-1:0
        idx = k - j
        if idx >= 1
            ii = mod1(idx, M)
            r = r + (a[ii] - Rho[ii] * dot(Y[ii], r)) * S[ii]
        end
    end
    return -r
end

# Run L-BFGS independently on each starting point
@kernel function lbfgs_kernel!(f, grad_f, p, x0s, result, lb, ub, maxiters, ::Val{M}) where {M}
    i = @index(Global, Linear)
    x = clamp.(x0s[i], lb, ub)
    T = eltype(x)
    z = zero(typeof(x))
    S, Y = ntuple(_ -> z, Val(M)), ntuple(_ -> z, Val(M))
    Rho = ntuple(_ -> zero(T), Val(M))
    fx = _safe_eval(f, p, x, lb, ub)
    g = _safe_grad(grad_f, p, x, lb, ub)
    k, active = 0, isfinite(fx) && all(isfinite, g)
    for _ in 1:maxiters
        if active && norm(g) >= T(1.0e-7)
            dir = _lbfgs_dir(g, S, Y, Rho, Val(M), k)
            if dot(g, dir) >= zero(T)
                dir, k = -g, 0
            end
            xn, fn, gn, ok = _strong_wolfe(f, grad_f, p, x, fx, g, dir, lb, ub)
            if !ok
                xn, fn, gn, ok = _strong_wolfe(f, grad_f, p, x, fx, g, -g, lb, ub); k = 0
            end
            if ok && isfinite(fn) && all(isfinite, gn)
                s, y = xn - x, gn - g
                sy = dot(s, y)
                if isfinite(one(T) / sy) && sy > T(1.0e-10)
                    k += 1; ii = mod1(k, M)
                    S, Y, Rho = Base.setindex(S, s, ii), Base.setindex(Y, y, ii), Base.setindex(Rho, one(T) / sy, ii)
                else
                    k = 0
                end
                x, g, fx = xn, gn, fn
            else
                active = false
            end
        end
    end
    @inbounds result[i] = x
end

# Main solve: runs PSO for global exploration, then L-BFGS for local refinement
function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing, reltol = nothing, maxiters = 100,
        local_maxiters = 50, n_starts = 20, kwargs...
    ) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}

    # Phase 1: Global search with PSO
    sol_pso = SciMLBase.solve!(cache.pso_cache; maxiters = maxiters)

    prob = cache.prob
    f_raw, p = prob.f.f, prob.p
    lb = prob.lb === nothing ? convert.(eltype(prob.u0), -Inf) : prob.lb
    ub = prob.ub === nothing ? convert.(eltype(prob.u0), Inf) : prob.ub

    best_u = sol_pso.u
    best_obj = sol_pso.objective isa Real ? sol_pso.objective : sol_pso.objective[]

    _obj(x) = let v = prob.f(clamp.(x, lb, ub), p)
        (isnan(v) || isinf(v)) ? convert(eltype(best_obj), Inf) : v
    end

    # Build candidate pool: inject global best into particle positions
    x0s = copy(sol_pso.original)
    x0s[1] = best_u

    costs = map(_obj, x0s)
    n = min(n_starts, length(x0s))
    pool = [x0s[j] for j in partialsortperm(Vector(costs), 1:n)]

    D = length(first(pool))
    m_val = D > 20 ? Val(5) : Val(10)

    grad_f = instantiate_gradient(f_raw, prob.f.adtype)
    t0 = time()

    result = similar(pool)
    copyto!(result, pool)

    # Phase 2: Local refinement with L-BFGS on top candidates
    kernel = lbfgs_kernel!(opt.backend)
    kernel(f_raw, grad_f, p, pool, result, lb, ub, local_maxiters, m_val; ndrange = n)
    KernelAbstractions.synchronize(opt.backend)

    # Select best result across all refined candidates
    for j in 1:n
        r = clamp.(result[j], lb, ub)
        fval = _obj(r)
        if fval < best_obj
            best_obj = fval
            best_u = r
        end
    end

    solve_time = (time() - t0) + sol_pso.stats.time

    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, best_u, best_obj;
        stats = Optimization.OptimizationStats(; time = solve_time)
    )
end
