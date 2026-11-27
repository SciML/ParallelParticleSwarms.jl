function _static_bounds(prob, ::Val{N}, ::Type{T}) where {N, T}
    lb = prob.lb === nothing ? nothing : SVector{N, T}(prob.lb)
    ub = prob.ub === nothing ? nothing : SVector{N, T}(prob.ub)
    return lb, ub
end

# Local polish runs inside a KernelAbstractions kernel, so call the static
# SimpleOptimization L-BFGS core directly rather than host-side `solve`.
@kernel function lbfgs_run!(
        grad_f, f, p, particles, result, result_fx, lb, ub,
        maxiters, abstol, reltol, linesearch, history_size, polish
    )
    i = @index(Global, Linear)
    @inbounds x0 = as_svector(particles[i].best_position)
    u, v, _, _ = SimpleOptimization._lbfgs(
        grad_f, f, p, x0, lb, ub, maxiters, abstol, reltol,
        linesearch, history_size
    )
    u, v = polish_point(polish, f, p, u, v, lb, ub)
    @inbounds result[i] = u
    @inbounds result_fx[i] = isfinite(v) ? v : eltype(u)(Inf)
end
