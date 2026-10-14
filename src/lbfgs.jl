function _static_bounds(prob, ::Val{N}, ::Type{T}) where {N, T}
    lb = prob.lb === nothing ? nothing : SVector{N, T}(prob.lb)
    ub = prob.ub === nothing ? nothing : SVector{N, T}(prob.ub)
    return lb, ub
end

# Local polish: SimpleOptimization's `_lbfgs` (projected L-BFGS with LineSearch's
# Strong-Wolfe search), re-executed in lockstep across particles. Each particle's run is
# cut at its objective/gradient evaluations: every host step evaluates f and the gradient
# at each particle's pending point, then `_lbfgs_advance` runs `_lbfgs`'s own control
# flow up to that particle's next evaluation. Each particle visits the same points as
# `_lbfgs` would on its own; finished particles pass through unchanged.

const _START, _LS_ZERO, _LS_BRACKET, _LS_ZOOM, _LS_ACCEPT, _DONE = 1, 2, 3, 4, 5, 6

struct LBFGSState{D, T, M}
    phase::Int
    xe::SVector{D, T}            # point whose f and gradient the phase is waiting for
    x::SVector{D, T}
    fx::T
    g::SVector{D, T}
    s_hist::NTuple{M, SVector{D, T}}
    y_hist::NTuple{M, SVector{D, T}}
    pseudo_iteration::Int
    iter::Int
    tol::T
    direction::SVector{D, T}
    # Strong-Wolfe search (`_sw_search`, `_sw_zoom`)
    α_max::T
    α_init::T
    ϕ_0::T
    dϕ_0::T
    α::T                         # trial step: α_i while bracketing, α_j while zooming
    k::Int                       # bracketing or zoom iteration
    α_prev::T
    ϕ_prev::T
    dϕ_prev::T
    a_lo::T
    ϕ_lo::T
    dϕ_lo::T
    a_hi::T
    ϕ_hi::T
    dϕ_hi::T
end

function _lbfgs_start(x0::SVector{D, T}, lb, ub, ::Val{M}) where {D, T, M}
    x = SimpleOptimization._project(x0, lb, ub)
    z = zero(T)
    zs = ntuple(_ -> zero(x), Val(M))
    return LBFGSState(
        _START, x, x, z, zero(x), zs, zs, 0, 0, z, zero(x),
        z, z, z, z, z, 0, z, z, z, z, z, z, z, z, z
    )
end

_isdone(st) = st.phase == _DONE

# Top of `_lbfgs`'s loop, through the line search's evaluation of φ(0).
function _lbfgs_iterate(st::LBFGSState{D, T, M}, lb, ub, maxiters, ls) where {D, T, M}
    (; x, fx, g) = st
    st.iter < maxiters || return @set st.phase = _DONE
    (!isfinite(fx) || !all(isfinite, g)) && return @set st.phase = _DONE
    maximum(abs, SimpleOptimization._projected_gradient(x, g, lb, ub)) ≤ st.tol &&
        return @set st.phase = _DONE

    pseudo_iteration = st.pseudo_iteration + 1
    direction = SimpleOptimization._feasible_direction(
        x, SimpleOptimization._lbfgs_direction(
            g, st.s_hist, st.y_hist, pseudo_iteration, Val(M)
        ), lb, ub
    )
    if dot(direction, g) ≥ zero(T)
        pseudo_iteration = 1
        direction = SimpleOptimization._feasible_direction(x, -g, lb, ub)
    end
    @set! st.iter = st.iter + 1
    @set! st.pseudo_iteration = pseudo_iteration
    @set! st.direction = direction
    maximum(abs, direction) ≤ eps(T) && return @set st.phase = _DONE

    # `_objective_line_search`, then the static `solve!` of StrongWolfeLineSearch.
    α_max = SimpleOptimization._max_feasible_step(x, direction, lb, ub, T(ls.α_max))
    α_max ≤ zero(T) && return @set st.phase = _DONE
    α_max_val = max(zero(T), min(T(ls.α_max), T(α_max)))
    α_max_val > zero(T) || return @set st.phase = _DONE
    @set! st.α_max = α_max_val
    @set! st.α_init = min(T(ls.α_init), α_max_val)
    @set! st.phase = _LS_ZERO
    @set! st.xe = x .+ zero(T) .* direction
    return st
end

function _ls_trial(st, phase, α)
    @set! st.phase = phase
    @set! st.α = α
    @set! st.xe = st.x .+ α .* st.direction
    return st
end

function _ls_accept(st::LBFGSState{D, T}, α, lb, ub) where {D, T}
    @set! st.phase = _LS_ACCEPT
    @set! st.xe = SimpleOptimization._project(st.x + T(α) * st.direction, lb, ub)
    return st
end

# `_sw_zoom` from its start, or its next iteration, through the next trial evaluation.
function _ls_zoom_trial(st::LBFGSState{D, T}, ls) where {D, T}
    st.k > ls.zoom_maxiters && return @set st.phase = _DONE
    (; a_lo, a_hi) = st
    α_j = LineSearch._sw_interpolate(a_lo, a_hi, st.ϕ_lo, st.ϕ_hi, st.dϕ_lo, st.dϕ_hi)
    bracket = T(0.01) * abs(a_hi - a_lo)
    α_j = clamp(α_j, min(a_lo, a_hi) + bracket, max(a_lo, a_hi) - bracket)
    return _ls_trial(st, _LS_ZOOM, α_j)
end

function _ls_zoom_start(st, a_lo, a_hi, ϕ_lo, dϕ_lo, ϕ_hi, dϕ_hi, ls)
    @set! st.a_lo = a_lo
    @set! st.a_hi = a_hi
    @set! st.ϕ_lo = ϕ_lo
    @set! st.dϕ_lo = dϕ_lo
    @set! st.ϕ_hi = ϕ_hi
    @set! st.dϕ_hi = dϕ_hi
    @set! st.k = 1
    return _ls_zoom_trial(st, ls)
end

# Consume f and the gradient at `st.xe`, then run to the particle's next evaluation.
function _lbfgs_advance(
        st::LBFGSState{D, T, M}, fe, ge, lb, ub, maxiters, abstol, reltol, ls
    ) where {D, T, M}
    phase = st.phase
    if phase == _START
        x, fx, g = st.xe, T(fe), SimpleOptimization._as_u(st.xe, ge)
        default_tol = sqrt(eps(T))
        abs_tol = abstol === nothing ? default_tol : max(T(abstol), default_tol)
        rel_tol = reltol === nothing ? zero(T) : max(T(reltol), zero(T))
        initial_residual = maximum(abs, SimpleOptimization._projected_gradient(x, g, lb, ub))
        @set! st.x = x
        @set! st.fx = fx
        @set! st.g = g
        @set! st.tol = max(abs_tol, rel_tol * initial_residual)
        return _lbfgs_iterate(st, lb, ub, maxiters, ls)
    elseif phase == _LS_ZERO
        ϕ_0, dϕ_0 = fe, dot(ge, st.direction)
        (isfinite(ϕ_0) && isfinite(dϕ_0) && dϕ_0 < zero(T)) || return @set st.phase = _DONE
        @set! st.ϕ_0 = ϕ_0
        @set! st.dϕ_0 = dϕ_0
        @set! st.α_prev = zero(T)
        @set! st.ϕ_prev = ϕ_0
        @set! st.dϕ_prev = dϕ_0
        @set! st.k = 1
        ls.maxiters ≥ 1 || return @set st.phase = _DONE
        return _ls_trial(st, _LS_BRACKET, st.α_init)
    elseif phase == _LS_BRACKET
        ϕ_i, dϕ_i = fe, dot(ge, st.direction)
        (; ϕ_0, dϕ_0, α_prev, ϕ_prev, dϕ_prev) = st
        α_i, c1, c2 = st.α, ls.c1, ls.c2
        if (ϕ_i > ϕ_0 + c1 * α_i * dϕ_0) || (ϕ_i >= ϕ_prev && st.k > 1)
            return _ls_zoom_start(st, α_prev, α_i, ϕ_prev, dϕ_prev, ϕ_i, dϕ_i, ls)
        elseif abs(dϕ_i) <= -c2 * dϕ_0
            return _ls_accept(st, α_i, lb, ub)
        elseif dϕ_i >= zero(T)
            return _ls_zoom_start(st, α_i, α_prev, ϕ_i, dϕ_i, ϕ_prev, dϕ_prev, ls)
        elseif α_i >= st.α_max
            return _ls_accept(st, α_i, lb, ub)
        end
        @set! st.α_prev = α_i
        @set! st.ϕ_prev = ϕ_i
        @set! st.dϕ_prev = dϕ_i
        @set! st.k = st.k + 1
        st.k > ls.maxiters && return @set st.phase = _DONE
        return _ls_trial(st, _LS_BRACKET, min(α_i * T(2), st.α_max))
    elseif phase == _LS_ZOOM
        ϕ_j, dϕ_j = fe, dot(ge, st.direction)
        α_j, c1, c2 = st.α, ls.c1, ls.c2
        if (ϕ_j > st.ϕ_0 + c1 * α_j * st.dϕ_0) || (ϕ_j >= st.ϕ_lo)
            @set! st.a_hi = α_j
            @set! st.ϕ_hi = ϕ_j
            @set! st.dϕ_hi = dϕ_j
        elseif abs(dϕ_j) <= -c2 * st.dϕ_0
            return _ls_accept(st, α_j, lb, ub)
        else
            if dϕ_j * (st.a_hi - st.a_lo) >= zero(T)
                @set! st.a_hi = st.a_lo
                @set! st.ϕ_hi = st.ϕ_lo
                @set! st.dϕ_hi = st.dϕ_lo
            end
            @set! st.a_lo = α_j
            @set! st.ϕ_lo = ϕ_j
            @set! st.dϕ_lo = dϕ_j
        end
        return _ls_zoom_trial((@set st.k = st.k + 1), ls)
    elseif phase == _LS_ACCEPT
        x_next, fx_next, g_next = st.xe, T(fe), SimpleOptimization._as_u(st.x, ge)
        s, y = x_next - st.x, g_next - st.g
        curvature = dot(s, y)
        curvature_floor = sqrt(eps(T)) * max(norm(s) * norm(y), eps(T))
        if !isfinite(curvature) || curvature ≤ curvature_floor
            @set! st.pseudo_iteration = 0
        else
            s_hist, y_hist = SimpleOptimization._store_history(
                st.s_hist, st.y_hist, st.pseudo_iteration, s, y, Val(M)
            )
            @set! st.s_hist = s_hist
            @set! st.y_hist = y_hist
        end
        @set! st.x = x_next
        @set! st.fx = fx_next
        @set! st.g = g_next
        return _lbfgs_iterate(st, lb, ub, maxiters, ls)
    end
    return st
end

# Gradients at the pending points. With ForwardDiff, one thread per (particle, direction):
# a single-partial dual seeded exactly as `ForwardDiff.gradient` seeds that direction, so
# the values match it. Other backends call `grad_f` once per particle.
function _pending_partial(f, p, st::LBFGSState{D, T}, j) where {D, T}
    _isdone(st) && return zero(T)
    x = st.xe
    return T(
        ForwardDiff.derivative(
            t -> f(SVector{D}(ntuple(i -> i == j ? t : oftype(t, x[i]), Val(D))), p), x[j]
        )
    )
end

function _gradients(
        ::AutoForwardDiff, grad_f, f, p, states::AbstractArray{<:LBFGSState{D, T}}
    ) where {D, T}
    js = reshape(adapt(get_backend(states), collect(1:D)), D, 1)
    partials = _pending_partial.(Ref(f), Ref(p), reshape(states, 1, :), js)
    return vec(reinterpret(SVector{D, T}, partials))
end
function _gradients(adtype, grad_f, f, p, states)
    return map(st -> _isdone(st) ? zero(st.g) : grad_f(st.xe, p), states)
end

function lockstep_lbfgs!(
        result::AbstractArray{SVector{D, T}}, result_fx, grad_f, f, p, adtype, particles,
        lb, ub, maxiters, abstol, reltol, linesearch, ::Val{M}; maxtime = nothing
    ) where {D, T, M}
    t_start = time()
    states = map(pt -> _lbfgs_start(as_svector(pt.best_position), lb, ub, Val(M)), particles)
    advance(st, fe, ge) = _lbfgs_advance(st, fe, ge, lb, ub, maxiters, abstol, reltol, linesearch)
    while !mapreduce(_isdone, &, states; init = true)
        fes = map(st -> _isdone(st) ? zero(T) : T(f(st.xe, p)), states)
        states = advance.(states, fes, _gradients(adtype, grad_f, f, p, states))
        maxtime !== nothing && time() - t_start > maxtime && break
    end
    map!(st -> st.x, result, states)
    map!(st -> isfinite(st.fx) ? st.fx : T(Inf), result_fx, states)
    return nothing
end
