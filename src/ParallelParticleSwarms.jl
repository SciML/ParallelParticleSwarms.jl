module ParallelParticleSwarms

using SciMLBase, StaticArrays, Setfield, KernelAbstractions
using QuasiMonteCarlo, Optimization, SimpleNonlinearSolve, ForwardDiff
import Adapt
import Adapt: adapt
import Enzyme: autodiff_deferred, Active, Reverse, Const
import KernelAbstractions: @atomic, @atomicreplace, @atomicswap
using QuasiMonteCarlo
import DiffEqGPU: GPUTsit5, make_prob_compatible, vectorized_solve, vectorized_asolve

using Reexport
@reexport using SciMLBase

## Use lb and ub either as StaticArray or pass them separately as CuArrays
## Passing as CuArrays makes more sense, or maybe SArray? The based on no. of dimension
struct SPSOParticle{T1, T2 <: eltype(T1)}
    position::T1
    velocity::T1
    cost::T2
    best_position::T1
    best_cost::T2
end
struct SPSOGBest{T1, T2 <: eltype(T1)}
    position::T1
    cost::T2
end

mutable struct MPSOParticle{T}
    position::AbstractArray{T}
    velocity::AbstractArray{T}
    cost::T
    best_position::AbstractArray{T}
    best_cost::T
end
mutable struct MPSOGBest{T}
    position::AbstractArray{T}
    cost::T
end

## required overloads for min or max computation on particles
function Base.isless(
        a::ParallelParticleSwarms.SPSOParticle{T1, T2},
        b::ParallelParticleSwarms.SPSOParticle{T1, T2}
    ) where {T1, T2}
    return a.best_cost < b.best_cost
end

function Base.isless(
        a::ParallelParticleSwarms.SPSOGBest{T1, T2},
        b::ParallelParticleSwarms.SPSOGBest{T1, T2}
    ) where {T1, T2}
    return a.cost < b.cost
end

function Base.typemax(::Type{ParallelParticleSwarms.SPSOParticle{T1, T2}}) where {T1, T2}
    return ParallelParticleSwarms.SPSOParticle{T1, T2}(
        similar(T1),
        similar(T1),
        typemax(T2),
        similar(T1),
        typemax(T2)
    )
end

function Base.typemax(::Type{ParallelParticleSwarms.SPSOGBest{T1, T2}}) where {T1, T2}
    return ParallelParticleSwarms.SPSOGBest{T1, T2}(
        similar(T1),
        typemax(T2)
    )
end

include("./algorithms.jl")
include("./utils.jl")
include("./ode_pso.jl")
include("./kernels.jl")
include("./lowerlevel_solve.jl")
include("init.jl")
include("./solve.jl")
include("./bfgs.jl")
include("./hybrid.jl")
include("./precompilation.jl")

export ParallelPSOKernel,
    ParallelSyncPSOKernel, ParallelPSOArray, SerialPSO
end
