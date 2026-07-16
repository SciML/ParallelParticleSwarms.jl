abstract type PSOAlgorithm end
abstract type HybridPSOAlgorithm{LocalOpt} end
abstract type GPUSamplingAlgorithm end

struct GPUUniformSampler <: GPUSamplingAlgorithm
end

struct GPUUnboundedSampler <: GPUSamplingAlgorithm
end

"""
    ParallelPSOKernel(num_particles; global_update = true, backend = CPU(),
        θ = θ_default, γ = γ_default, h = sqrt)

Particle Swarm Optimization that launches a KernelAbstractions kernel for
parallel particle updates.

Static arrays for parameters in the `OptimizationProblem` are required for
successful GPU compilation.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `global_update`: Whether particles share the global best position during each
    kernel update. Set to `false` to evolve particles independently.
- `backend`: KernelAbstractions backend used for computation.
- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.

# Examples

```julia
using KernelAbstractions
using ParallelParticleSwarms

alg = ParallelPSOKernel(100; backend = CPU(), global_update = false)
```

# Limitations

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
    ParallelSyncPSOKernel(num_particles; backend = CPU(), θ = θ_default,
        γ = γ_default, h = sqrt)

Particle Swarm Optimization that updates particles in parallel and synchronizes
after each generation to compute the global best position.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `backend`: KernelAbstractions backend used for computation.
- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.

# Examples

```julia
using KernelAbstractions
using ParallelParticleSwarms

alg = ParallelSyncPSOKernel(100; backend = CPU())
```

"""
struct ParallelSyncPSOKernel{Backend, T, G, H} <: PSOAlgorithm
    num_particles::Int
    backend::Backend
    θ::T
    γ::G
    h::H
end

"""
    ParallelPSOArray(num_particles; θ = θ_default, γ = γ_default, h = sqrt)

Particle Swarm Optimization on a CPU. It keeps the arrays used in particle data structure
to be Julia's `Array`, which may be better for high-dimensional problems.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.

# Examples

```julia
using ParallelParticleSwarms

alg = ParallelPSOArray(100)
```

# Limitations

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
    SerialPSO(num_particles; θ = θ_default, γ = γ_default, h = sqrt)

Serial Particle Swarm Optimization on a CPU.

# Arguments

- `num_particles`: Number of particles in the swarm.

# Keywords

- `θ`: PSO coefficient schedule.
- `γ`: PSO velocity update coefficient schedule.
- `h`: Transformation applied in the update rule.

# Examples

```julia
using ParallelParticleSwarms

alg = SerialPSO(100)
```

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
