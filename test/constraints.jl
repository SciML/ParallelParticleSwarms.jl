using ParallelParticleSwarms, StaticArrays, SciMLBase, Test, LinearAlgebra, Random

include("./utils.jl")

Random.seed!(1234)

objective(x, p) = (x[1] - p[1])^2 + (x[2] - p[2])^2

p = @SVector [2.0f0, 1.0f0]

function conss(x, p)
    return SVector{3}(-x[1] + 2 * x[2] - 1, +x[1] - 2 * x[2] + 1, (x[1]^2) / 4 + x[2]^2 - 1)
end

opt_f = OptimizationFunction(objective, cons = conss)

x0 = @SVector [1.0f0, 1.0f0]
lb = @SVector [0.0f0, 0.0f0]
ub = @SVector [2.0f0, 2.0f0]
lcons = @SVector [-Inf32, -Inf32]
ucons = @SVector [0.0f0, 0.0f0]
prob = OptimizationProblem(opt_f, x0, p, lcons = lcons, ucons = ucons, lb = lb, ub = ub)

n_particles = 1000

sol = solve(prob, ParallelSyncPSOKernel(n_particles; backend), maxiters = 500)
@test sol.retcode == ReturnCode.Default
@test abs(1 - 2 * sol.u[2] + sol.u[1]) < 1.0e-1

sol = solve(prob, ParallelPSOKernel(n_particles; backend), maxiters = 500)
@test sol.retcode == ReturnCode.Default
@test abs(1 - 2 * sol.u[2] + sol.u[1]) < 1.0e-1

sol = solve(prob, SerialPSO(n_particles), maxiters = 500)
@test sol.retcode == ReturnCode.Default
