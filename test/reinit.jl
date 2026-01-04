using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, LinearAlgebra, Random,
    KernelAbstractions
using QuasiMonteCarlo

## Solving the rosenbrock problem
Random.seed!(123)
lb = @SArray fill(Float32(-1.0), 3)
ub = @SArray fill(Float32(10.0), 3)

function rosenbrock(x, p)
    res = zero(eltype(x))
    for i in 1:(length(x) - 1)
        res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
    end
    return res
end
x0 = @SArray zeros(Float32, 3)
p = @SArray Float32[1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n_particles = 2000

cache = init(prob, ParallelPSOKernel(n_particles; backend = CPU()))

reinit!(cache)

cache = init(prob, ParallelSyncPSOKernel(n_particles; backend = CPU()))

reinit!(cache)

cache = init(
    prob,
    ParallelParticleSwarms.HybridPSO(;
        local_opt = ParallelParticleSwarms.BFGS(), backend = CPU()
    )
)

reinit!(cache)

cache = init(
    prob,
    ParallelParticleSwarms.HybridPSO(;
        local_opt = ParallelParticleSwarms.LBFGS(), backend = CPU()
    )
)

reinit!(cache)
