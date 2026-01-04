# https://stackoverflow.com/questions/65342388/why-my-code-in-julia-is-getting-slower-for-higher-iteration
using Statistics
using Dates
using Base.Threads
using Suppressor

function uniform(dim::Int, lb::Array{Float64, 1}, ub::Array{Float64, 1})
    arr = rand(Float64, dim)
    @inbounds for i in 1:dim
        arr[i] = arr[i] * (ub[i] - lb[i]) + lb[i]
    end
    return arr
end

mutable struct Problem
    cost_func::Any
    dim::Int
    lb::Array{Float64, 1}
    ub::Array{Float64, 1}
end

mutable struct Particle
    position::Array{Float64, 1}
    velocity::Array{Float64, 1}
    cost::Float64
    best_position::Array{Float64, 1}
    best_cost::Float64
end

mutable struct Gbest
    position::Array{Float64, 1}
    cost::Float64
end

function PSO(
        problem,
        data_dict;
        max_iter = 100,
        population = 100,
        c1 = 1.4962,
        c2 = 1.4962,
        w = 0.7298,
        wdamp = 1.0
    )
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    cost_func = problem.cost_func

    gbest, particles = initialize_particles(problem, population, data_dict)

    # main loop
    for iter in 1:max_iter
        @threads for i in 1:population
            particles[i].velocity .= w .* particles[i].velocity .+
                c1 .* rand(dim) .* (
                particles[i].best_position .-
                    particles[i].position
            ) .+
                c2 .* rand(dim) .*
                (gbest.position .- particles[i].position)

            particles[i].position .= particles[i].position .+ particles[i].velocity
            particles[i].position .= max.(particles[i].position, lb)
            particles[i].position .= min.(particles[i].position, ub)

            particles[i].cost = cost_func(particles[i].position, data_dict)

            if particles[i].cost < particles[i].best_cost
                particles[i].best_position = copy(particles[i].position)
                particles[i].best_cost = copy(particles[i].cost)

                if particles[i].best_cost < gbest.cost
                    gbest.position = copy(particles[i].best_position)
                    gbest.cost = copy(particles[i].best_cost)
                end
            end
        end
        w = w * wdamp
        if iter % 50 == 1
            println("Iteration " * string(iter) * ": Best Cost = " * string(gbest.cost))
            println("Best Position = " * string(gbest.position))
            println()
        end
    end
    return gbest, particles
end

function serial_PSO(
        problem,
        data_dict;
        max_iter = 100,
        population = 100,
        c1 = 1.4962,
        c2 = 1.4962,
        w = 0.7298,
        wdamp = 1.0
    )
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    cost_func = problem.cost_func

    gbest, particles = initialize_particles(problem, population, data_dict)

    # main loop
    for iter in 1:max_iter
        for i in 1:population
            particles[i].velocity .= w .* particles[i].velocity .+
                c1 .* rand(dim) .* (
                particles[i].best_position .-
                    particles[i].position
            ) .+
                c2 .* rand(dim) .*
                (gbest.position .- particles[i].position)

            particles[i].position .= particles[i].position .+ particles[i].velocity
            particles[i].position .= max.(particles[i].position, lb)
            particles[i].position .= min.(particles[i].position, ub)

            particles[i].cost = cost_func(particles[i].position, data_dict)

            if particles[i].cost < particles[i].best_cost
                particles[i].best_position = copy(particles[i].position)
                particles[i].best_cost = copy(particles[i].cost)

                if particles[i].best_cost < gbest.cost
                    gbest.position = copy(particles[i].best_position)
                    gbest.cost = copy(particles[i].best_cost)
                end
            end
        end
        w = w * wdamp
        if iter % 50 == 1
            println("Iteration " * string(iter) * ": Best Cost = " * string(gbest.cost))
            println("Best Position = " * string(gbest.position))
            println()
        end
    end
    return gbest, particles
end

function initialize_particles(problem, population, data_dict)
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    cost_func = problem.cost_func

    gbest_position = uniform(dim, lb, ub)
    gbest = Gbest(gbest_position, cost_func(gbest_position, data_dict))

    particles = Particle[]
    for i in 1:population
        position = uniform(dim, lb, ub)
        velocity = zeros(dim)
        cost = cost_func(position, data_dict)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, Particle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest.cost
            gbest.position = copy(best_position)
            gbest.cost = copy(best_cost)
        end
    end
    return gbest, particles
end

### Example

dimension = 2
lb = [-1.0, 1.0]
ub = [1.0, 1.0]

cost_func(x, args...; kwargs...) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
p = [1.0, 100.0]

problem = Problem(cost_func, dimension, lb, ub)

population = 100
max_iter = 1001

data_dict = Dict()

gbest, particles = PSO(problem, data_dict, max_iter = max_iter, population = population)
