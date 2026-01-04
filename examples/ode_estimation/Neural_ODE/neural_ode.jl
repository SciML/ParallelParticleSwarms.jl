using SimpleChains, StaticArrays, OrdinaryDiffEq

u0 = @SArray Float32[2.0, 0.0]
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

p_nn = SimpleChains.init_params(sc)

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
    return sum(abs2, data .- pred)
end

lb = SVector{length(p_static), eltype(p_static)}(
    fill(
        eltype(p_static)(-10),
        length(p_static)
    )...
)
ub = SVector{length(p_static), eltype(p_static)}(
    fill(
        eltype(p_static)(10),
        length(p_static)
    )...
)

optprob = OptimizationProblem(loss, prob_nn.p[2], (prob_nn, tsteps); lb = lb, ub = ub)

using ParallelParticleSwarms
using CUDA

opt = ParallelPSOKernel(n_particles)
gbest, particles = ParallelParticleSwarms.init_particles(optprob, opt, typeof(prob.u0))

gpu_data = cu(
    [
        SVector{length(prob_nn.u0), eltype(prob_nn.u0)}(@view data[:, i])
            for i in 1:length(tsteps)
    ]
)

gpu_particles = cu(particles)

CUDA.allowscalar(false)

function prob_func(prob, gpu_particle)
    return remake(prob, p = (prob.p[1], gpu_particle.position))
end

@time gsol = ParallelParticleSwarms.parameter_estim_ode!(
    prob_nn,
    gpu_particles,
    gbest,
    gpu_data,
    lb,
    ub; saveat = tsteps, dt = 0.1f0, prob_func = prob_func, maxiters = 300
)
