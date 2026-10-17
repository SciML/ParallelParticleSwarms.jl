# ParallelParticleSwarms.jl

## Algorithms

```@docs
ParallelPSOKernel
ParallelSyncPSOKernel
ParallelPSOArray
SerialPSO
HybridPSO
BFGS
```

Local L-BFGS polish for [`HybridPSO`](@ref) uses
[`SimpleLBFGS`](https://docs.sciml.ai/Optimization/stable/optimization_packages/simpleoptimization/)
from SimpleOptimization.jl.

## Extension Interface

```@docs
PSOAlgorithm
pso_solve
```

## Reexported SciML common interface

`using ParallelParticleSwarms` also brings in the parts of the SciML common
optimization interface needed to build a problem, solve it and inspect the result, so
they do not have to be imported separately. These names are owned and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/) -- ParallelParticleSwarms only
re-exports them:

  - Problems and functions:
    [`OptimizationProblem`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/),
    [`OptimizationFunction`](https://docs.sciml.ai/SciMLBase/stable/interfaces/SciMLFunctions/),
    `NullParameters`
  - Solutions: [`OptimizationSolution`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/),
    `OptimizationStats`
  - Solving: [`solve`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Common_Keywords/),
    `init`, `solve!`, `reinit!`, `remake`
  - Return status: `ReturnCode`, `successful_retcode`

`SimpleLBFGS`, the local polish algorithm [`HybridPSO`](@ref) accepts, is re-exported
from [SimpleOptimization.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/simpleoptimization/).

The usual workflow is either a single `solve` call:

```julia
prob = OptimizationProblem(OptimizationFunction{false}(f), u0, p; lb = lb, ub = ub)
sol = solve(prob, ParallelSyncPSOKernel(1000; backend = CPU()); maxiters = 500)
```

or the cache workflow, which lets the same swarm be re-run against updated bounds or
parameters:

```julia
cache = init(prob, ParallelPSOKernel(1000; backend = CPU()))
sol = solve!(cache; maxiters = 500)
reinit!(cache)
```

Anything else from SciMLBase -- the ODE/SDE/DAE/nonlinear problem classes, the
integrator interface, the SciML operators -- is not re-exported here, because
ParallelParticleSwarms does not solve those problems; import it from SciMLBase
directly.
