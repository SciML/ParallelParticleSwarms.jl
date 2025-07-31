using SimpleChains,
      StaticArrays, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationFlux, Plots

#device!(2)
# Get Tesla V100S
u0 = @SArray Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(static(2),
    Activation(x -> x .^ 3),
    TurboDense{true}(tanh, static(2)),
    TurboDense{true}(identity, static(2)))

p_nn = SimpleChains.init_params(sc)

f(u, p, t) = sc(u, p)

sprob_nn = ODEProblem(f, u0, tspan)

function predict_neuralode(p)
    Array(solve(sprob_nn,
        Tsit5();
        p = p,
        saveat = tsteps,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = true)
    display(l)
    plt = scatter(tsteps, data[1, :], label = "data")
    scatter!(plt, tsteps, pred[1, :], label = "prediction")
    if doplot
        display(plot(plt))
    end
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x),
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_nn)

@time res = Optimization.solve(optprob, ADAM(0.05), maxiters = 100)
@show res.objective

@benchmark Optimization.solve(optprob, ADAM(0.05), maxiters = 100)

## ParallelParticleSwarms stuff

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps)
end

nn = (u, p, t) -> sc(u, p)

p_static = SArray{Tuple{size(p_nn)...}}(p_nn...)

prob_nn = ODEProblem(nn_fn, u0, tspan, (sc, p_static))

n_particles = 10_000

function loss(u, p)
    odeprob, t = p
    prob = remake(odeprob; p = (odeprob.p[1], u))
    pred = Array(solve(prob, Tsit5(), saveat = t))
    sum(abs2, data .- pred)
end

# lb = SVector{length(p_static), eltype(p_static)}(fill(eltype(p_static)(-10),
#     length(p_static))...)
# ub = SVector{length(p_static), eltype(p_static)}(fill(eltype(p_static)(10),
#     length(p_static))...)

lb = @SArray fill(Float32(-Inf), length(p_static))
ub = @SArray fill(Float32(Inf), length(p_static))

soptprob = OptimizationProblem(loss, prob_nn.p[2], (prob_nn, tsteps); lb = lb, ub = ub)

using ParallelParticleSwarms
using CUDA
using KernelAbstractions
using Adapt

backend = CUDABackend()

## Initialize Particles
gbest,
particles = ParallelParticleSwarms.init_particles(soptprob,
    ParallelSyncPSOKernel(n_particles; backend = CUDABackend()),
    typeof(prob.u0))

gpu_data = adapt(backend,
    [SVector{length(prob_nn.u0), eltype(prob_nn.u0)}(@view data[:, i])
     for i in 1:length(tsteps)])

CUDA.allowscalar(false)

function prob_func(prob, gpu_particle)
    return remake(prob, p = (prob.p[1], gpu_particle.position))
end

gpu_particles = adapt(backend, particles)

losses = adapt(backend, ones(eltype(prob.u0), (1, n_particles)))

solver_cache = (; losses, gpu_particles, gpu_data, gbest)

@time gsol = ParallelParticleSwarms.parameter_estim_ode!(prob_nn,
    solver_cache,
    lb,
    ub;
    saveat = tsteps,
    dt = 0.1f0,
    prob_func = prob_func,
    maxiters = 100)

@benchmark ParallelParticleSwarms.parameter_estim_ode!($prob_nn,
    $(deepcopy(solver_cache)),
    $lb,
    $ub;
    saveat = tsteps,
    dt = 0.1f0,
    prob_func = prob_func,
    maxiters = 100)

@show gsol.best

using Plots

function predict_neuralode(p)
    Array(solve(prob_nn, Tsit5(); p = p, saveat = tsteps))
end

using Plots

plt = scatter(tsteps, data[1, :], label = "data")

pred_pso = predict_neuralode((sc, gsol.position))
scatter!(plt, tsteps, pred[1, :], label = "PSO prediction")

pred_adam = predict_neuralode((sc, res.u))
scatter!(plt, tsteps, pred_adam[1, :], label = "Adam prediction")
