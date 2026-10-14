function _static_bounds(prob, ::Val{N}, ::Type{T}) where {N, T}
    lb = prob.lb === nothing ? nothing : SVector{N, T}(prob.lb)
    ub = prob.ub === nothing ? nothing : SVector{N, T}(prob.ub)
    return lb, ub
end

# Lockstep projected L-BFGS for HybridPSO's local phase. All particles take each step
# together: the iteration loop runs on the host and every step is a few short
# broadcasts over all particles, instead of one long divergent thread per particle.
# Internally a coordinate with a finite box works in y = (x - lb) / (ub - lb) clamped to
# [0, 1]; any other coordinate (`init` stores a missing bound as ±Inf) is left as is and
# only clamped to the bound it has.

struct LBFGSState{D, T, M}
    x::SVector{D, T}
    f::T
    g::SVector{D, T}
    # Newest pair first; only the first `k` are valid.
    S::NTuple{M, SVector{D, T}}
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

function lockstep_lbfgs(
        f, p, starts::AbstractArray{SVector{D, T}}, lb, ub, ::Val{M}, local_maxiters, abstol;
        maxtime = nothing
    ) where {D, T, M}
    t_start = time()
    backend = get_backend(starts)

    lb = something(lb, fill(T(-Inf), SVector{D, T}))
    ub = something(ub, fill(T(Inf), SVector{D, T}))
    boxed = isfinite.(ub - lb)
    lo = ifelse.(boxed, lb, zero(lb))
    w = ifelse.(boxed, ub - lb, one.(lb))
    ylo, yhi = (lb - lo) ./ w, (ub - lo) ./ w
    project(y) = clamp.(y, ylo, yhi)

    obj = y -> f(lo + y .* w, p)
    cost(y) = (c = obj(y); isfinite(c) ? T(c) : T(Inf))
    function partial(j, y)
        e = SVector{D, T}(ntuple(i -> T(i == j), Val(D)))
        dj = ForwardDiff.derivative(t -> obj(y + t * e), zero(T))
        return isfinite(dj) ? T(dj) : zero(T)
    end
    # N × D threads, one directional derivative each, assembled back into SVectors.
    js = reshape(adapt(backend, collect(1:D)), D, 1)
    grad(X) = vec(reinterpret(SVector{D, T}, partial.(js, reshape(X, 1, :))))

    # Parallel line search over 16 step sizes 4, 2, 1, ..., 2^-13 per particle.
    αs = ntuple(i -> T(2.0^(3 - i)), Val(16))
    alphas = reshape(adapt(backend, collect(αs)), 1, :)
    cand(st, α, d) = cost(project(st.x + α * d))
    stepx(st, I, d, fb) = fb < st.f ? project(st.x + αs[I[2]] * d) : st.x
    function finish(st, x, fb, g)
        fb < st.f || return LBFGSState(st.x, st.f, st.g, st.S, st.Y, 0)
        s = x - st.x
        y = g - st.g
        _dot(s, y) > 1.0e-12 || return LBFGSState(x, fb, g, st.S, st.Y, st.k)
        return LBFGSState(
            x, fb, g, (s, Base.front(st.S)...), (y, Base.front(st.Y)...), min(st.k + 1, M)
        )
    end

    X = map(x -> project((x - lo) ./ w), starts)
    G = grad(X)
    zs = ntuple(_ -> zero(SVector{D, T}), Val(M))
    states = map((x, g) -> LBFGSState(x, cost(x), g, zs, zs, 0), X, G)

    for _ in 1:local_maxiters
        dirs = _lbfgs_direction.(states)
        fmin, idx = findmin(cand.(states, alphas, dirs); dims = 2)
        fmin = vec(fmin)
        X = stepx.(states, vec(idx), dirs, fmin)
        states = finish.(states, X, fmin, grad(X))
        abstol !== nothing && minimum(st -> st.f, states) < abstol && break
        maxtime !== nothing && time() - t_start > maxtime && break
    end

    best_f, ind = findmin(Array(map(st -> st.f, states)))
    best = Array(@view states[ind:ind])[1]
    return lo + best.x .* w, best_f
end
