using Pkg
Pkg.activate(@__DIR__)
using ParallelParticleSwarms, OrdinaryDiffEq, StaticArrays
using SciMLSensitivity, Optimization

function f(u, p, t)
    a, b, τinv, l = p
    v, w = u
    dv = v - v^3 / 3 - w + l
    dw = τinv * (v + a - b * w)
    return SVector{2}(dv, dw)
end

p = @SArray [0.7f0, 0.8f0, 0.08f0, 0.5f0]              # Parameters used to construct the dataset
r0 = @SArray [1.0f0; 1.0f0]                     # initial value
tspan = (0.0f0, 30.0f0)                 # sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(f, r0, tspan, p)

tspan2 = (0.0f0, 3.0f0)                 # sample of 300 observations with a timestep of 0.01
prob_short = ODEProblem(f, r0, tspan2, p)

dt = 30.0f0 / 3000.0f0
tf = 30.0f0
tinterval = 0.0f0:dt:tf
t = collect(tinterval)

h = 0.01f0
M = 300
tstart = 0.0f0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)

data_sol_short = solve(
    prob_short,
    Vern9(),
    saveat = t_short,
    reltol = 1.0f-6,
    abstol = 1.0f-6
)
data_short = convert(Array, data_sol_short) # This operation produces column major dataset obs as columns, equations as rows
data_sol = solve(prob, Vern9(), saveat = t, reltol = 1.0f-6, abstol = 1.0f-6)
data = convert(Array, data_sol)

# using Plots

# plot(data_sol_short)

# plot(data_sol)

n_particles = 10_000

# obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),tstops=t_short)
function loss(u, p)
    odeprob, t = p
    prob = remake(odeprob; p = u)
    pred = Array(solve(prob, Tsit5(), saveat = t))
    return sum(abs2, data_short .- pred)
end

lb = @SArray fill(0.0f0, 4)
ub = @SArray fill(5.0f0, 4)

u_guess = @MArray zeros(Float32, 4)

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

using Random

rng = Random.default_rng()

Random.seed!(rng, 0)

opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(optprob, opt, typeof(prob.u0))

gpu_data = cu(
    [
        SVector{length(prob.u0), eltype(prob.u0)}(@view data_short[:, i])
            for i in 1:length(t_short)
    ]
)

gpu_particles = cu(particles)

CUDA.allowscalar(false)

using Adapt

losses = adapt(CUDABackend(), ones(eltype(prob.u0), (1, n_particles)))

solver_cache = (; losses, gpu_particles, gpu_data, gbest)

adaptive = false

@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob,
    solver_cache,
    lb,
    ub, Val(adaptive);
    saveat = t_short,
    dt = 0.1f0,
    maxiters = 100
)

using BenchmarkTools

@benchmark ParallelParticleSwarms.parameter_estim_ode!(
    $prob,
    $(deepcopy(solver_cache)),
    $lb,
    $ub, $Val(adaptive);
    saveat = t_short,
    dt = 0.1f0,
    maxiters = 100
)

@show gbest.cost, gsol
