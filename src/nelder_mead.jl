"""
    NelderMeadPolish(; maxiters = 10_000, reltol = 1.0e-12, restarts = 5,
        all_particles = false)

Derivative-free Nelder-Mead polish for [`HybridPSO`](@ref), run after its gradient-based
local stage. It finishes minima that gradient methods stall short of, such as a minimum on
a kink (BBOB's sharp ridge) or at the end of a badly conditioned valley.

The implementation works on static arrays and does not allocate, so it also runs inside
GPU kernels.

# Keywords

- `maxiters`: Maximum Nelder-Mead iterations per run.
- `reltol`: A run stops once the objective values on the simplex span less than
  `reltol * max(1, |f_best|)`.
- `restarts`: Up to this many extra runs, each from a fresh simplex around the best point
  so far, while they keep improving. Nelder-Mead's simplex can collapse short of a kinked
  minimum; a fresh simplex gets it moving again.
- `all_particles`: If `false`, polish only the best point after the local stage, on the
  host. If `true`, polish every particle's local result inside the local-stage kernel.
"""
struct NelderMeadPolish{T}
    maxiters::Int
    reltol::T
    restarts::Int
    all_particles::Bool
end

function NelderMeadPolish(;
        maxiters = 10_000, reltol = 1.0e-12, restarts = 5, all_particles = false
    )
    return NelderMeadPolish(maxiters, reltol, restarts, all_particles)
end

# Convert on the host so kernels never touch the Float64 default.
_typed_polish(::Nothing, ::Type) = nothing
_typed_polish(nm::NelderMeadPolish, ::Type{T}) where {T} =
    NelderMeadPolish(nm.maxiters, T(nm.reltol), nm.restarts, nm.all_particles)

# Only an `all_particles` polish is passed into the local-stage kernel; otherwise the
# kernel gets `nothing` and Nelder-Mead is not compiled into it.
_kernel_polish(polish) = polish isa NelderMeadPolish && polish.all_particles ? polish : nothing

@inline _nm_lower(x, ::Nothing) = x
@inline _nm_lower(x, lb) = max.(x, lb)
@inline _nm_upper(x, ::Nothing) = x
@inline _nm_upper(x, ub) = min.(x, ub)
@inline _nm_project(x, lb, ub) = _nm_upper(_nm_lower(x, lb), ub)

# Copy of `x` with entry `j` replaced (`Base.setindex` is not public API).
@inline _nm_setindex(x::SVector{N}, v, j) where {N} =
    SVector{N}(ntuple(k -> k == j ? v : x[k], Val(N)))

@inline function _nm_value(f, x, p)
    T = eltype(x)
    v = f(x, p)
    return isfinite(v) ? convert(T, v) : T(Inf)
end

# Optim.jl's default affine simplex: vertex j moves coordinate j by 0.5 x0[j] + 0.025,
# flipped inward when the bound would clamp it back onto x0.
@inline function _nm_vertex(x0, j, lb, ub)
    T = eltype(x0)
    h = T(0.5) * x0[j] + T(0.025)
    h = iszero(h) ? T(0.025) : h
    v = _nm_project(_nm_setindex(x0, x0[j] + h, j), lb, ub)
    return v[j] == x0[j] ? _nm_project(_nm_setindex(x0, x0[j] - h, j), lb, ub) : v
end

# Indices of the best, worst and second-worst vertices.
@inline function _nm_order(fs)
    ib = 1
    iw = 1
    for i in 2:length(fs)
        fs[i] < fs[ib] && (ib = i)
        fs[i] > fs[iw] && (iw = i)
    end
    is = iw == 1 ? 2 : 1
    for i in 1:length(fs)
        i != iw && fs[i] > fs[is] && (is = i)
    end
    return ib, iw, is
end

"""
    nelder_mead(f, p, x0, lb, ub, maxiters, reltol, restarts = 0)

Bound-projected Nelder-Mead from `x0` with the adaptive coefficients of Gao and Han (2012),
restarted from the best point while that keeps improving. Returns the best point and its
objective value.
"""
@inline function nelder_mead(f, p, x0, lb, ub, maxiters, reltol, restarts = 0)
    x, fx = _nelder_mead_run(f, p, x0, lb, ub, maxiters, reltol)
    for _ in 1:restarts
        xn, fn = _nelder_mead_run(f, p, x, lb, ub, maxiters, reltol)
        fn < fx || break
        x, fx = xn, fn
    end
    return x, fx
end

@inline function _nelder_mead_run(
        f, p, x0::SVector{N, T}, lb, ub, maxiters, reltol
    ) where {N, T}
    α = one(T)
    β = one(T) + T(2) / N
    γ = T(0.75) - one(T) / (2N)
    δ = one(T) - one(T) / N

    x0 = _nm_project(x0, lb, ub)
    simplex = SVector{N + 1}(
        ntuple(j -> j == 1 ? x0 : _nm_vertex(x0, j - 1, lb, ub), Val(N + 1))
    )
    fs = map(x -> _nm_value(f, x, p), simplex)

    for _ in 1:maxiters
        ib, iw, is = _nm_order(fs)
        fb, fw = fs[ib], fs[iw]
        (isfinite(fb) && fw - fb > reltol * max(one(T), abs(fb))) || break

        xw = simplex[iw]
        c = (sum(simplex) - xw) / N
        xr = _nm_project(c + α * (c - xw), lb, ub)
        fr = _nm_value(f, xr, p)

        if fr < fb
            xe = _nm_project(c + β * (xr - c), lb, ub)
            fe = _nm_value(f, xe, p)
            xn, fn = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < fs[is]
            xn, fn = xr, fr
        else
            outside = fr < fw
            xc = _nm_project(c + γ * ((outside ? xr : xw) - c), lb, ub)
            fc = _nm_value(f, xc, p)
            if outside ? fc <= fr : fc < fw
                xn, fn = xc, fc
            else
                xb = simplex[ib]
                simplex = map(x -> _nm_project(xb + δ * (x - xb), lb, ub), simplex)
                fs = map(x -> _nm_value(f, x, p), simplex)
                continue
            end
        end
        simplex = _nm_setindex(simplex, xn, iw)
        fs = _nm_setindex(fs, fn, iw)
    end

    ib, _, _ = _nm_order(fs)
    return simplex[ib], fs[ib]
end

# Polish one local-stage result; keeps the original unless Nelder-Mead improves on it.
@inline polish_point(::Nothing, f, p, u, fu, lb, ub) = (u, fu)
@inline function polish_point(nm::NelderMeadPolish, f, p, u, fu, lb, ub)
    x, fx = nelder_mead(f, p, u, lb, ub, nm.maxiters, nm.reltol, nm.restarts)
    return isfinite(fx) && !(fx >= fu) ? (x, fx) : (u, fu)
end

# Host-side polish of the final best point, unless every particle was already polished.
polish_best(::Nothing, f, p, u, fu, lb, ub) = (u, fu)
function polish_best(nm::NelderMeadPolish, f, p, u, fu, lb, ub)
    nm.all_particles && return (u, fu)
    return polish_point(nm, f, p, as_svector(u), fu, lb, ub)
end
