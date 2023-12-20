function uniform(dim::Int, lb::AbstractArray{T}, ub::AbstractArray{T}) where {T}
    arr = rand(T, dim)
    @inbounds for i in 1:dim
        arr[i] = arr[i] * (ub[i] - lb[i]) + lb[i]
    end
    return arr
end

function init_particles(prob, n_particles, ::Type{T}) where {T <: SArray}
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p

    if lb === nothing || (all(isinf, lb) && all(isinf, ub))
        gbest_position = Array{eltype(prob.u0), 1}(undef, dim)
        for i in 1:dim
            if abs(prob.u0[i]) > 0
                gbest_position[i] = prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i])
            else
                gbest_position[i] = rand(eltype(prob.u0))
            end
        end
    else
        gbest_position = uniform(dim, lb, ub)
    end

    gbest_position = SVector{length(gbest_position), eltype(gbest_position)}(gbest_position)
    gbest_cost = cost_func(gbest_position, p)
    particles = SPSOParticle[]
    for i in 1:n_particles
        if lb === nothing || (all(isinf, lb) && all(isinf, ub))
            position = Array{eltype(prob.u0), 1}(undef, dim)
            for i in 1:dim
                if abs(prob.u0[i]) > 0
                    position[i] = prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i])
                else
                    position[i] = rand(eltype(prob.u0))
                end
            end
        else
            position = uniform(dim, lb, ub)
        end
        position = SVector{length(position), eltype(position)}(position)
        velocity = @SArray zeros(eltype(position), dim)
        cost = cost_func(position, p)
        best_position = position
        best_cost = cost
        push!(particles, SPSOParticle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest_cost
            gbest_position = best_position
            gbest_cost = best_cost
        end
    end
    gbest = SPSOGBest(gbest_position, gbest_cost)
    return gbest, convert(Vector{typeof(particles[1])}, particles)
end

function init_particles(prob, n_particles, ::Type{T}) where {T <: AbstractArray}
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f

    if lb === nothing || (all(isinf, lb) && all(isinf, ub))
        gbest_position = Array{eltype(prob.u0), 1}(undef, dim)
        for i in 1:dim
            if abs(prob.u0[i]) > 0
                gbest_position[i] = prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i])
            else
                gbest_position[i] = rand(eltype(prob.u0))
            end
        end
    else
        gbest_position = uniform(dim, lb, ub)
    end
    gbest = MPSOGBest(gbest_position, cost_func(gbest_position, prob.p))

    particles = MPSOParticle[]
    for i in 1:n_particles
        if lb === nothing || (all(isinf, lb) && all(isinf, ub))
            position = Array{eltype(prob.u0), 1}(undef, dim)
            for i in 1:dim
                if abs(prob.u0[i]) > 0
                    position[i] = prob.u0[i] + rand(eltype(prob.u0)) * abs(prob.u0[i])
                else
                    position[i] = rand(eltype(prob.u0))
                end
            end
        else
            position = uniform(dim, lb, ub)
        end
        velocity = zeros(eltype(position), dim)
        cost = cost_func(position, prob.p)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, MPSOParticle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest.cost
            gbest.position = copy(best_position)
            gbest.cost = copy(best_cost)
        end
    end
    return gbest, convert(Vector{typeof(particles[1])}, particles)
end

function check_init_bounds(prob)
    lb = prob.lb === nothing ? fill(eltype(prob.u0)(-Inf), length(prob.u0)) : prob.lb
    ub = prob.ub === nothing ? fill(eltype(prob.u0)(Inf), length(prob.u0)) : prob.ub
    if prob.u0 isa SArray
        lb = SVector{length(lb), eltype(lb)}(lb)
        ub = SVector{length(ub), eltype(ub)}(ub)
    end
    lb, ub
end
