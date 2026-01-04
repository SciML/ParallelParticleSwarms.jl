abstract type PSOAlgorithm end
abstract type HybridPSOAlgorithm{LocalOpt} end
abstract type GPUSamplingAlgorithm end

struct GPUUniformSampler <: GPUSamplingAlgorithm
end

struct GPUUnboundedSampler <: GPUSamplingAlgorithm
end

"""
```julia
ParallelPSOKernel(num_particles; global_update = true, backend = CPU())
```

Particle Swarm Optimization on a GPU. Creates and launches a kernel which updates the particle states in parallel
on a GPU. Static Arrays for parameters in the `OptimizationProblem` are required for successful GPU compilation.

## Arguments

- `num_particles`: Number of particles in the simulation (positional argument)

## Keyword Arguments

- `global_update`: defaults to `true`. Setting it to `false` allows particles to evolve completely on their own,
  i.e. no information is sent about the global best position.
- `backend`: defaults to `CPU()`. The KernelAbstractions backend for performing the computation
  (e.g., `CUDA.CUDABackend()` for NVIDIA GPUs).

## Limitations

Running the optimization with `global_update=true` updates the global best positions with possible thread races.
This is the price to be paid to fuse all the updates into a single kernel. Techniques such as queue lock and atomic
updates can be used to fix this.

"""
struct ParallelPSOKernel{Backend, T, G, H} <: PSOAlgorithm
    num_particles::Int
    global_update::Bool
    backend::Backend
    θ::T
    γ::G
    h::H
end

"""
```julia
ParallelSyncPSOKernel(num_particles; backend = CPU())
```

Particle Swarm Optimization on a GPU. Creates and launches a kernel which updates the particle states in parallel
on a GPU. However, it requires a synchronization after each generation to calculate the global best position of the particles.

## Arguments

- `num_particles`: Number of particles in the simulation (positional argument)

## Keyword Arguments

- `backend`: defaults to `CPU()`. The KernelAbstractions backend for performing the computation
  (e.g., `CUDA.CUDABackend()` for NVIDIA GPUs).

"""
struct ParallelSyncPSOKernel{Backend, T, G, H} <: PSOAlgorithm
    num_particles::Int
    backend::Backend
    θ::T
    γ::G
    h::H
end

"""
```julia
ParallelPSOArray(num_particles)
```

Particle Swarm Optimization on a CPU. It keeps the arrays used in particle data structure
to be Julia's `Array`, which may be better for high-dimensional problems.

## Arguments

- `num_particles`: Number of particles in the simulation (positional argument)

## Limitations

Running the optimization updates the global best positions with possible thread races.
This is the price to be paid to fuse all the updates into a single kernel. Techniques such as queue lock and atomic
updates can be used to fix this.

"""
struct ParallelPSOArray{T, G, H} <: PSOAlgorithm
    num_particles::Int
    θ::T
    γ::G
    h::H
end

"""
```julia
SerialPSO(num_particles)
```

Serial Particle Swarm Optimization on a CPU.

## Arguments

- `num_particles`: Number of particles in the simulation (positional argument)

"""
struct SerialPSO{T, G, H} <: PSOAlgorithm
    num_particles::Int
    θ::T
    γ::G
    h::H
end

function ParallelPSOKernel(
        num_particles::Int;
        global_update = true, backend = CPU(), θ = θ_default, γ = γ_default, h = sqrt
    )
    return ParallelPSOKernel(num_particles, global_update, backend, θ, γ, h)
end

function ParallelSyncPSOKernel(
        num_particles::Int;
        backend = CPU(), θ = θ_default, γ = γ_default, h = sqrt
    )
    return ParallelSyncPSOKernel(num_particles, backend, θ, γ, h)
end

function ParallelPSOArray(num_particles::Int; θ = θ_default, γ = γ_default, h = sqrt)
    return ParallelPSOArray(num_particles, θ, γ, h)
end

function SerialPSO(num_particles::Int; θ = θ_default, γ = γ_default, h = sqrt)
    return SerialPSO(num_particles, θ, γ, h)
end

SciMLBase.allowsbounds(::PSOAlgorithm) = true
SciMLBase.allowsconstraints(::PSOAlgorithm) = true

struct LBFGS
    threshold::Int
end

function LBFGS(; threshold = 10)
    return LBFGS(threshold)
end

struct BFGS end

struct HybridPSO{Backend, LocalOpt} <: HybridPSOAlgorithm{LocalOpt}
    pso::PSOAlgorithm
    local_opt::LocalOpt
    backend::Backend
end

function HybridPSO(;
        backend = CPU(),
        pso = ParallelParticleSwarms.ParallelPSOKernel(100; global_update = false, backend),
        local_opt = LBFGS()
    )
    return HybridPSO(pso, local_opt, backend)
end

SciMLBase.allowsbounds(::HybridPSOAlgorithm{LocalOpt}) where {LocalOpt} = true
