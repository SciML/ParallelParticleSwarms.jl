using ParallelParticleSwarms, BenchmarkTools
using StaticArrays, SciMLBase, KernelAbstractions

const SUITE = BenchmarkGroup()

backend = CPU()

# Out-of-place objective on SVector states
function rosenbrock(x, p)
    return (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
end

opt_f = OptimizationFunction{false}(rosenbrock)
x0 = @SVector [0.5f0, 0.5f0]
lb = @SVector [-5.0f0, -5.0f0]
ub = @SVector [5.0f0, 5.0f0]
p = @SVector [1.0f0, 100.0f0]
prob = OptimizationProblem(opt_f, x0, p; lb = lb, ub = ub)

n_particles = 200

# =============================================================================
# PSO solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["parallel_pso"] = @benchmarkable solve(
    $prob, ParallelPSOKernel($n_particles; backend = $backend);
    maxiters = 100
)
SUITE["solve"]["sync_pso"] = @benchmarkable solve(
    $prob, ParallelSyncPSOKernel($n_particles; backend = $backend);
    maxiters = 100
)
SUITE["solve"]["serial_pso"] = @benchmarkable solve(
    $prob, SerialPSO($n_particles); maxiters = 100
)

# Constrained variant
function conss(x, p)
    return SVector{2}(-x[1] + 2 * x[2] - 1, x[1]^2 / 4 + x[2]^2 - 1)
end
opt_fc = OptimizationFunction{false}(rosenbrock, cons = conss)
lcons = @SVector [-Inf32, -Inf32]
ucons = @SVector [0.0f0, 0.0f0]
prob_c = OptimizationProblem(
    opt_fc, x0, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
)

SUITE["solve"]["constrained"] = @benchmarkable solve(
    $prob_c, ParallelSyncPSOKernel($n_particles; backend = $backend);
    maxiters = 100
)
