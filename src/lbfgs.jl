function _static_bounds(prob, ::Val{N}, ::Type{T}) where {N, T}
    lb = prob.lb === nothing ? nothing : SVector{N, T}(prob.lb)
    ub = prob.ub === nothing ? nothing : SVector{N, T}(prob.ub)
    return lb, ub
end

# Lockstep projected L-BFGS for HybridPSO's local phase.
#
# All particles take each step together: the loop runs on the host, and each step is a few
# short broadcasts over all particles (search direction, a line search that tries every step
# size in `_STEPS` at once, and a gradient with one directional derivative per thread),
# instead of one long, divergent thread per particle. Coordinates with a finite box are
# optimized in y = (x - lb) / (ub - lb), so the step sizes mean the same on every problem;
# unbounded coordinates are left as they are.

const _STEPS = ntuple(i -> 2.0^(3 - i), Val(16))   # 4, 2, 1, ..., 2^-13
const _CURVATURE_TOL = 1.0e-12                      # skip (s, y) pairs with s'y below this

struct LBFGSState{D, T, M}
    x::SVector{D, T}
    f::T
    g::SVector{D, T}
    S::NTuple{M, SVector{D, T}}   # newest pair first; only the first `k` are valid
    Y::NTuple{M, SVector{D, T}}
    k::Int
end

@inline _dot(a, b) = sum(a .* b)

# Two-loop recursion, unrolled over the memory so the history tuples are only ever
# indexed with constants. Falls back to the normalized steepest descent direction
# when there is no history or the quasi-Newton direction is not a descent direction.
@generated function _lbfgs_direction(st::LBFGSState{D, T, M}) where {D, T, M}
    ex = quote
        g = st.g
        k = st.k
        q = g
    end
    for i in 1:M
        a = Symbol(:a, i)
        push!(
            ex.args, quote
                $a = $i <= k ? _dot(st.S[$i], q) / _dot(st.S[$i], st.Y[$i]) : zero(T)
                q = q - $a * st.Y[$i]
            end
        )
    end
    push!(ex.args, :(r = k > 0 ? (_dot(st.S[1], st.Y[1]) / _dot(st.Y[1], st.Y[1])) * q : q))
    for i in M:-1:1
        a = Symbol(:a, i)
        push!(
            ex.args, quote
                if $i <= k
                    r = r + ($a - _dot(st.Y[$i], r) / _dot(st.S[$i], st.Y[$i])) * st.S[$i]
                end
            end
        )
    end
    push!(
        ex.args, quote
            d = -r
            if k == 0 || !(_dot(d, g) < zero(T))
                gn = sqrt(_dot(g, g))
                d = gn > zero(T) ? -g / gn : zero(g)
            end
            return d
        end
    )
    return ex
end

# Accept the step if it lowered the cost, then store (s, y) if it has positive curvature.
# A rejected step keeps the point and clears the history.
function _lbfgs_update(st::LBFGSState{D, T, M}, x, f, g) where {D, T, M}
    f < st.f || return LBFGSState(st.x, st.f, st.g, st.S, st.Y, 0)
    s, y = x - st.x, g - st.g
    _dot(s, y) > _CURVATURE_TOL || return LBFGSState(x, f, g, st.S, st.Y, st.k)
    return LBFGSState(x, f, g, (s, Base.front(st.S)...), (y, Base.front(st.Y)...), min(st.k + 1, M))
end

function lockstep_lbfgs(
        f, p, starts::AbstractArray{SVector{D, T}}, lb, ub, ::Val{M}, local_maxiters, abstol;
        maxtime = nothing
    ) where {D, T, M}
    t_start = time()
    backend = get_backend(starts)

    # Box transform x = lo + y .* w; a missing bound is stored as ±Inf.
    lb = something(lb, fill(T(-Inf), SVector{D, T}))
    ub = something(ub, fill(T(Inf), SVector{D, T}))
    boxed = isfinite.(ub - lb)
    lo = ifelse.(boxed, lb, zero(lb))
    w = ifelse.(boxed, ub - lb, one.(lb))
    ylo, yhi = (lb - lo) ./ w, (ub - lo) ./ w
    project(y) = clamp.(y, ylo, yhi)

    # Objective in y: a cost that never returns NaN, and one directional derivative.
    obj(y) = f(lo + y .* w, p)
    cost(y) = (c = obj(y); isfinite(c) ? T(c) : T(Inf))
    function partial(j, y)
        e = SVector{D, T}(ntuple(i -> T(i == j), Val(D)))
        dj = ForwardDiff.derivative(t -> obj(y + t * e), zero(T))
        return isfinite(dj) ? T(dj) : zero(T)
    end
    js = reshape(adapt(backend, collect(1:D)), D, 1)
    grad(ys) = vec(reinterpret(SVector{D, T}, partial.(js, reshape(ys, 1, :))))   # N × D threads

    # Line search: every step size for every particle at once (N × length(_STEPS) threads).
    steps = reshape(adapt(backend, collect(T.(_STEPS))), 1, :)
    trial_cost(st, α, d) = cost(project(st.x + α * d))
    take_step(st, I, d, fb) = fb < st.f ? project(st.x + T(_STEPS[I[2]]) * d) : st.x

    # Converged: projected gradient, in x units, within `tol` (as in SimpleOptimization's
    # L-BFGS). The gradient is dropped where it pushes against an active bound.
    tol = abstol === nothing ? sqrt(eps(T)) : max(T(abstol), sqrt(eps(T)))
    blocked(y, g, l, h) = (y <= l && g > zero(T)) || (y >= h && g < zero(T))
    converged(st) = maximum(abs.(ifelse.(blocked.(st.x, st.g, ylo, yhi), zero(T), st.g) ./ w)) <= tol

    ys = map(x -> project((x - lo) ./ w), starts)
    zs = ntuple(_ -> zero(SVector{D, T}), Val(M))
    states = map((y, g) -> LBFGSState(y, cost(y), g, zs, zs, 0), ys, grad(ys))

    for _ in 1:local_maxiters
        dirs = _lbfgs_direction.(states)
        fbest, idx = findmin(trial_cost.(states, steps, dirs); dims = 2)
        fbest = vec(fbest)
        ys = take_step.(states, vec(idx), dirs, fbest)
        states = _lbfgs_update.(states, ys, fbest, grad(ys))
        mapreduce(converged, &, states; init = true) && break
        maxtime !== nothing && time() - t_start > maxtime && break
    end

    best_f, ind = findmin(Array(map(st -> st.f, states)))
    best = Array(@view states[ind:ind])[1]
    return lo + best.x .* w, best_f
end
