# ParallelParticleSwarms.jl

[![CI](https://github.com/SciML/ParallelParticleSwarms.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SciML/ParallelParticleSwarms.jl/actions/workflows/CI.yml)
[![Build status](https://badge.buildkite.com/caf5d6f9d5129b5796049b085df39fd8fab055826b513d361e.svg)](https://buildkite.com/julialang/parallelparticleswarms-dot-jl)
[![codecov](https://codecov.io/gh/utkarsh530/ParallelParticleSwarms.jl/graph/badge.svg?token=H5U5UAIRXX)](https://codecov.io/gh/utkarsh530/ParallelParticleSwarms.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Accelerating convex/non-convex optimization with GPUs using Particle-Swarm based methods.

Supports Julia's generic SciML interface.

```julia

using ParallelParticleSwarms, StaticArrays, CUDA

lb = @SArray [-1.0f0, -1.0f0, -1.0f0]
ub = @SArray [10.0f0, 10.0f0, 10.0f0]

function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end

x0 = @SArray zeros(Float32, 3)
p = @SArray Float32[1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

sol = solve(prob,
    ParallelSyncPSOKernel(1000, backend = CUDA.CUDABackend()),
    maxiters = 500)
```
