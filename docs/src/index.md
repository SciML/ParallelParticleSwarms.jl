# ParallelParticleSwarms.jl

## Algorithms

```@docs
ParallelPSOKernel
ParallelSyncPSOKernel
ParallelPSOArray
SerialPSO
HybridPSO
LBFGS
BFGS
```

## Extension Interface

```@docs
PSOAlgorithm
pso_solve
```

## SciML Interface

`ParallelParticleSwarms` provides the documented generic optimization entry points
through its solver interface. Construct an `OptimizationProblem` and pass a
`ParallelParticleSwarms` algorithm to `solve`.

### `OptimizationProblem`

Construct an optimization problem with an objective, initial point, and optional
parameters or bounds. The full field and constructor documentation is maintained by
SciMLBase in its [optimization interface documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/SciMLFunctions/).

### `solve`

Pass the problem and one of the algorithms documented above to `solve`. The generic
solver contract and keyword forwarding rules are defined by SciMLBase/CommonSolve.
