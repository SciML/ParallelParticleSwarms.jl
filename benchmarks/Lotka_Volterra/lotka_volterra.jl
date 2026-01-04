using Pkg

Pkg.activate(@__DIR__)

using StaticArrays, SciMLBase, OrdinaryDiffEq

using Optimization, OptimizationFlux, SciMLSensitivity

function f(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2]
    dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    return SVector{2}(dx, dy)
end

u0 = @SArray [1.0; 1.0]
p = @SArray [1.5, 1.0, 3.0, 1.0]

tspan = (0.0, 30.0)                  # sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(f, u0, tspan, p)
tspan2 = (0.0, 3.0)                     # sample of 3000 observations over the (0,30) timespan
prob_short = ODEProblem(f, u0, tspan2, p)

dt = 30.0 / 3000
tf = 30.0
tinterval = 0:dt:tf
t = collect(tinterval)

h = 0.01
M = 300
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)

#Generate Data
data_sol_short = solve(prob_short, Tsit5(), saveat = t_short, reltol = 1.0e-9, abstol = 1.0e-9)
data_short = convert(Array, data_sol_short)
data_sol = solve(prob, Tsit5(), saveat = t, reltol = 1.0e-9, abstol = 1.0e-9)
data = convert(Array, data_sol)

function loss(u, p)
    odeprob, t = p
    prob = remake(odeprob; p = u)
    pred = Array(solve(prob, Tsit5(), saveat = t))
    return sum(abs2, data_short .- pred)
end

lb = @SArray fill(0.0, 4)
ub = @SArray fill(10.0, 4)

opt_u0 = rand(4)
optprob = OptimizationProblem(loss, opt_u0, (prob_short, t_short); lb = lb, ub = ub)

u_guess = @MArray zeros(Float64, 4)

optprob = OptimizationProblem(
    Optimization.OptimizationFunction(
        loss,
        Optimization.AutoForwardDiff()
    ),
    u_guess,
    (prob_short, t_short)
)

# optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x),
#     Optimization.AutoZygote())
# optprob = Optimization.OptimizationProblem(optf, p_nn)

using OptimizationFlux

@time res_adam = Optimization.solve(optprob, ADAM(0.05), maxiters = 100)
@show res_adam.objective

using BenchmarkTools

@benchmark Optimization.solve(optprob, ADAM(0.05), maxiters = 100)

## Evaluate the perf of LBFGS

using OptimizationOptimJL

@time res_lbfgs = Optimization.solve(optprob, LBFGS(), maxiters = 100)
@show res_lbfgs.objective

@benchmark Optimization.solve(optprob, LBFGS(), maxiters = 100)

optprob = OptimizationProblem(loss, prob.p, (prob, t_short); lb = lb, ub = ub)

using ParallelParticleSwarms
using CUDA

CUDA.allowscalar(false)

n_particles = 10_000

opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(optprob, opt, typeof(prob.u0))

@show gbest

using Adapt

# using KernelAbstractions
# backend = CPU()

backend = CUDABackend()

gpu_data = adapt(
    backend,
    [
        SVector{length(prob.u0), eltype(prob.u0)}(@view data_short[:, i])
            for i in 1:length(t_short)
    ]
)

gpu_particles = adapt(backend, particles)

losses = adapt(backend, ones(eltype(prob.u0), (1, n_particles)))

solver_cache = (; losses, gpu_particles, gpu_data, gbest)

adaptive = false

@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob, solver_cache,
    lb,
    ub, Val(adaptive); saveat = t_short, dt = 0.1, maxiters = 100
)

using BenchmarkTools

@benchmark ParallelParticleSwarms.parameter_estim_ode!(
    $prob, $(deepcopy(solver_cache)),
    $lb,
    $ub, Val(adaptive); saveat = t_short, dt = 0.1, maxiters = 100
)

@show gsol
