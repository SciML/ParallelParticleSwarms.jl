using Pkg
Pkg.activate(@__DIR__)

using SimpleChains
using StaticArrays, OrdinaryDiffEq, SciMLSensitivity, Optimization
using OptimizationOptimisers
using Optimisers: Adam
using OptimizationOptimJL
using BenchmarkTools
using Plots

using ParallelParticleSwarms
using DiffEqGPU
using CUDA
using KernelAbstractions
using Adapt
using Random

device!(0)

u0 = @SVector Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    return ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(
    static(2),
    Activation(x -> x .^ 3),
    TurboDense{true}(tanh, static(2)),
    TurboDense{true}(identity, static(2))
)

rng = Random.default_rng()
Random.seed!(rng, 0)

p_nn = SimpleChains.init_params(sc; rng)

f(u, p, t) = sc(u, p)

sprob_nn = ODEProblem(f, u0, tspan)

function predict_neuralode(p)
    return Array(
        solve(
            sprob_nn, Tsit5(); p = p, saveat = tsteps,
            sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())
        )
    )
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss, pred
end

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_nn)

@time res_adam = Optimization.solve(optprob, Adam(0.05), maxiters = 100)
@show res_adam.objective

@benchmark Optimization.solve($optprob, Adam(0.05), maxiters = 100)

## LBFGS

moptprob = OptimizationProblem(optf, MArray{Tuple{size(p_nn)...}}(p_nn...))

@time res_lbfgs = Optimization.solve(moptprob, LBFGS(), maxiters = 100)
@show res_lbfgs.objective

@benchmark Optimization.solve($moptprob, LBFGS(), maxiters = 100)

## GPU-PSO

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps)
end

p_static = SVector{12, Float32}(p_nn...)

prob_nn = ODEProblem{false}(nn_fn, u0, tspan, (sc, p_static))

n_particles = 10_000

lb = @SVector fill(Float32(-10.0), 12)
ub = @SVector fill(Float32(10.0), 12)

loss_pso(u, p) = eltype(u)(Inf)
soptprob = OptimizationProblem(loss_pso, p_static, nothing; lb = lb, ub = ub)

backend = CUDABackend()

Random.seed!(rng, 0)

opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(soptprob, opt, typeof(p_static))

gpu_data = adapt(backend, [SVector{2, Float32}(@view data[:, i]) for i in 1:datasize])

CUDA.allowscalar(false)

function prob_func(prob, gpu_particle)
    return remake(prob, p = (prob.p[1], gpu_particle.position))
end

gpu_particles = adapt(backend, particles)
losses = adapt(backend, ones(Float32, n_particles))
improb = DiffEqGPU.make_prob_compatible(prob_nn)
probs = adapt(backend, fill(improb, n_particles))

solver_cache = (; losses, gpu_particles, gpu_data, gbest, probs)

@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob_nn, solver_cache, lb, ub, Val(true);
    saveat = tsteps, dt = 0.1f0, maxiters = 100, prob_func = prob_func
)

@benchmark ParallelParticleSwarms.parameter_estim_ode!(
    $prob_nn, $(deepcopy(solver_cache)), $lb, $ub, Val(true);
    saveat = tsteps, dt = 0.1f0, maxiters = 100, prob_func = prob_func
)

@show gsol.cost

## Plot

function predict_plot(p)
    return Array(solve(prob_nn, Tsit5(); p = p, saveat = tsteps))
end

plt = scatter(
    tsteps, data[1, :],
    label = "data",
    ylabel = "u(t)",
    xlabel = "t",
    linewidth = 4,
    title = "Optimizers performance after 100 iterations"
)

pred_pso = predict_plot((sc, gsol.position))
scatter!(plt, tsteps, pred_pso[1, :], label = "PSO prediction", markershape = :star5)

pred_adam = predict_plot((sc, SVector{12, Float32}(res_adam.u...)))
scatter!(plt, tsteps, pred_adam[1, :], label = "ADAM prediction", markershape = :xcross)

pred_lbfgs = predict_plot((sc, SVector{12, Float32}(res_lbfgs.u...)))
scatter!(plt, tsteps, pred_lbfgs[1, :], label = "LBFGS prediction", markershape = :cross)

savefig("neural_ode.svg")
