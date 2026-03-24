using ParallelParticleSwarms, Optimization, StaticArrays

include("./utils.jl")

# Reclaim GPU memory from previous test files to avoid OOM
if GROUP == "CUDA"
    CUDA.reclaim()
end

function objf(x, p)
    return 1 - x[1]^2 - x[2]^2
end

# Use out-of-place form {false} since SVector is immutable
optprob = OptimizationFunction{false}(objf, Optimization.AutoEnzyme())
x0 = rand(2)
x0 = SVector{2}(x0)
prob = OptimizationProblem(optprob, x0)
l1 = objf(x0, nothing)
sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.LBFGS(),
    maxiters = 10
)

N = 10
function rosenbrock(x, p)
    res = zero(eltype(x))
    for i in 1:(length(x) - 1)
        res += p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2
    end
    return res
end
x0 = @SArray rand(Float32, N)
p = @SArray Float32[1.0, 100.0]
# Use out-of-place form {false} since SArray is immutable
optf = OptimizationFunction{false}(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p)
l0 = rosenbrock(x0, p)

@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.LBFGS(; threshold = 7),
    maxiters = 20
)
@show sol.objective
@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.ParallelPSOKernel(100; backend),
    maxiters = 100
)
@show sol.objective

@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.HybridPSO(; backend),
    maxiters = 30
)
@show sol.objective

@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.HybridPSO(;
        local_opt = ParallelParticleSwarms.BFGS(), backend = backend
    ),
    maxiters = 30
)
@show sol.objective

# Use out-of-place form {false} since SArray is immutable
optf = OptimizationFunction{false}(rosenbrock, Optimization.AutoEnzyme())
prob = OptimizationProblem(optf, x0, p)
l0 = rosenbrock(x0, p)

@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.LBFGS(; threshold = 7),
    maxiters = 20
)
@show sol.objective
@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.ParallelPSOKernel(100, backend = backend),
    maxiters = 100
)
@show sol.objective

@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.HybridPSO(; backend = backend),
    local_maxiters = 30
)
@show sol.objective

@time sol = Optimization.solve(
    prob,
    ParallelParticleSwarms.HybridPSO(;
        local_opt = ParallelParticleSwarms.BFGS(), backend = backend
    ),
    local_maxiters = 30
)
@show sol.objective
