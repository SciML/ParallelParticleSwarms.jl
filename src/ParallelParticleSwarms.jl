module ParallelParticleSwarms

import Adapt
import Adapt: adapt
import ADTypes: AutoEnzyme, AutoForwardDiff
import Atomix: @atomic, @atomicreplace
import Enzyme: autodiff, Active, Reverse, Const, Duplicated, make_zero!
import DiffEqGPU: GPUTsit5
import KernelAbstractions
import KernelAbstractions: CPU, @groupsize, @index, @kernel, @localmem, @private,
    @synchronize, @uniform, get_backend
import LineSearch: StrongWolfeLineSearch
import PrecompileTools: @compile_workload, @setup_workload
import QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm
import SciMLBase
import SciMLBase: OptimizationFunction, OptimizationProblem, NonlinearProblem, init, remake,
    reinit!, solve, solve!
# The SciML common interface that ParallelParticleSwarms reexports (see the second
# `export` below), so that `using ParallelParticleSwarms` keeps giving access to the
# problem, function, solution and solve API it exposed when it reexported SciMLBase
# wholesale. Every name stays owned and documented upstream in SciMLBase.
using SciMLBase: AbstractAnalyticalProblem, AddVector, AffineOperator, AllObserved,
    AnalyticalProblem, BVPFunction, BVProblem, BatchIntegralFunction, BlockDiagonalOperator,
    CallbackSet, CheckInit, Clocks, ContinuousCallback, ConvexOptimizationProblem,
    DAEFunction, DAEProblem, DAESolution, DDEFunction, DDEProblem, DiagonalOperator,
    DiscreteCallback, DiscreteFunction, DiscreteProblem, DynamicalBVPFunction,
    DynamicalDDEFunction, DynamicalDDEProblem, DynamicalODEFunction, DynamicalODEProblem,
    DynamicalSDEFunction, DynamicalSDEProblem, EigenvalueProblem, EigenvalueSolution,
    EigenvalueTarget, EnsembleAnalysis, EnsembleContext, EnsembleDistributed,
    EnsembleProblem, EnsembleSerial, EnsembleSolution, EnsembleSplitThreads,
    EnsembleSummary, EnsembleTestSolution, EnsembleThreads, FunctionOperator,
    HomotopyNonlinearFunction, HomotopyProblem, IdentityOperator, ImplicitDiscreteFunction,
    ImplicitDiscreteProblem, IncrementingODEFunction, IncrementingODEProblem,
    IntegralFunction, IntegralProblem, IntegralSolution, IntervalNonlinearFunction,
    IntervalNonlinearProblem, InvertibleOperator, LinearAliasSpecifier, LinearProblem,
    LinearSolution, MatrixOperator, MultiObjectiveOptimizationFunction, NoiseProblem,
    NonlinearFunction, NonlinearLeastSquaresProblem, NonlinearSolution, NullOperator,
    ODEAliasSpecifier, ODEFunction, ODEInputFunction, ODEProblem, ODESolution,
    OptimizationSolution, PDENoTimeSolution, PDEProblem, PDETimeSeriesSolution,
    RODEFunction, RODEProblem, RODESolution, ReturnCode, SCCNonlinearProblem, SDDEFunction,
    SDDEProblem, SDEFunction, SDEProblem, SampledIntegralProblem, ScalarOperator,
    SciMLOperators, SecondOrderBVProblem, SecondOrderDDEProblem, SecondOrderODEProblem,
    SplitFunction, SplitODEProblem, SplitSDEFunction, SplitSDEProblem, StaticWOperator,
    SteadyStateProblem, SteadyStateSolution, TensorProductOperator, TensorSumOperator,
    TimeDomain, TwoPointBVPFunction, TwoPointBVProblem, TwoPointDynamicalBVPFunction,
    TwoPointSecondOrderBVProblem, VectorContinuousCallback, WOperator, add_saveat!,
    add_tstop!, addat!, addat_non_user_cache!, addsteps!, auto_dt_reset!, cache_operator,
    change_t_via_interpolation!, check_error, check_keywords, concretize, deleteat!,
    deleteat_non_user_cache!, derivative_discontinuity!, discretize, du_cache, first_tstop,
    full_cache, get_dt, get_du, get_du!, get_proposed_dt, get_rng, get_tmp_cache,
    has_adjoint, has_concretization, has_exp, has_expmv, has_expmv!, has_ldiv, has_ldiv!,
    has_mul, has_mul!, has_rng, has_tstop, is_discrete_time_domain, iscached, isclock,
    isconstant, iscontinuous, isconvertible, isdiscrete, isinplace, islinear,
    issolverstepclock, issquare, kronsum, pop_tstop!, rand_cache, ratenoise_cache,
    reeval_internals_due_to_modification!, resize_non_user_cache!, savevalues!, set_abstol!,
    set_proposed_dt!, set_reltol!, set_rng!, set_t!, set_u!, step!, supports_solve_rng,
    symbolic_discretize, terminate!, u_cache, u_modified!, update_coefficients,
    update_coefficients!, user_cache, warn_compat
import Setfield: @set!
import SimpleNonlinearSolve: SimpleBroyden, SimpleLimitedMemoryBroyden
import StaticArrays: @SArray, MVector, SArray, SVector

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

struct PPSOptimizationCache{F, P} <: SciMLBase.AbstractOptimizationCache
    f::F
    p::P
end

_optimization_cache(prob) = PPSOptimizationCache(prob.f, prob.p)

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
    ParallelSyncPSOKernel, ParallelPSOArray, SerialPSO, PSOAlgorithm, HybridPSO, LBFGS, BFGS,
    pso_solve

# Reexported SciML common interface; approved via `reexports_allow` in test/qa/qa.jl.
export AbstractAnalyticalProblem, AddVector, AffineOperator, AllObserved, AnalyticalProblem,
    BVPFunction, BVProblem, BatchIntegralFunction, BlockDiagonalOperator, CallbackSet,
    CheckInit, Clocks, ContinuousCallback, ConvexOptimizationProblem, DAEFunction,
    DAEProblem, DAESolution, DDEFunction, DDEProblem, DiagonalOperator, DiscreteCallback,
    DiscreteFunction, DiscreteProblem, DynamicalBVPFunction, DynamicalDDEFunction,
    DynamicalDDEProblem, DynamicalODEFunction, DynamicalODEProblem, DynamicalSDEFunction,
    DynamicalSDEProblem, EigenvalueProblem, EigenvalueSolution, EigenvalueTarget,
    EnsembleAnalysis, EnsembleContext, EnsembleDistributed, EnsembleProblem, EnsembleSerial,
    EnsembleSolution, EnsembleSplitThreads, EnsembleSummary, EnsembleTestSolution,
    EnsembleThreads, FunctionOperator, HomotopyNonlinearFunction, HomotopyProblem,
    IdentityOperator, ImplicitDiscreteFunction, ImplicitDiscreteProblem,
    IncrementingODEFunction, IncrementingODEProblem, IntegralFunction, IntegralProblem,
    IntegralSolution, IntervalNonlinearFunction, IntervalNonlinearProblem,
    InvertibleOperator, LinearAliasSpecifier, LinearProblem, LinearSolution, MatrixOperator,
    MultiObjectiveOptimizationFunction, NoiseProblem, NonlinearFunction,
    NonlinearLeastSquaresProblem, NonlinearProblem, NonlinearSolution, NullOperator,
    ODEAliasSpecifier, ODEFunction, ODEInputFunction, ODEProblem, ODESolution,
    OptimizationFunction, OptimizationProblem, OptimizationSolution, PDENoTimeSolution,
    PDEProblem, PDETimeSeriesSolution, RODEFunction, RODEProblem, RODESolution, ReturnCode,
    SCCNonlinearProblem, SDDEFunction, SDDEProblem, SDEFunction, SDEProblem,
    SampledIntegralProblem, ScalarOperator, SciMLBase, SciMLOperators, SecondOrderBVProblem,
    SecondOrderDDEProblem, SecondOrderODEProblem, SplitFunction, SplitODEProblem,
    SplitSDEFunction, SplitSDEProblem, StaticWOperator, SteadyStateProblem,
    SteadyStateSolution, TensorProductOperator, TensorSumOperator, TimeDomain,
    TwoPointBVPFunction, TwoPointBVProblem, TwoPointDynamicalBVPFunction,
    TwoPointSecondOrderBVProblem, VectorContinuousCallback, WOperator, add_saveat!,
    add_tstop!, addat!, addat_non_user_cache!, addsteps!, auto_dt_reset!, cache_operator,
    change_t_via_interpolation!, check_error, check_keywords, concretize, deleteat!,
    deleteat_non_user_cache!, derivative_discontinuity!, discretize, du_cache, first_tstop,
    full_cache, get_dt, get_du, get_du!, get_proposed_dt, get_rng, get_tmp_cache,
    has_adjoint, has_concretization, has_exp, has_expmv, has_expmv!, has_ldiv, has_ldiv!,
    has_mul, has_mul!, has_rng, has_tstop, init, is_discrete_time_domain, iscached, isclock,
    isconstant, iscontinuous, isconvertible, isdiscrete, isinplace, islinear,
    issolverstepclock, issquare, kronsum, pop_tstop!, rand_cache, ratenoise_cache,
    reeval_internals_due_to_modification!, reinit!, remake, resize_non_user_cache!,
    savevalues!, set_abstol!, set_proposed_dt!, set_reltol!, set_rng!, set_t!, set_u!,
    solve, solve!, step!, supports_solve_rng, symbolic_discretize, terminate!, u_cache,
    u_modified!, update_coefficients, update_coefficients!, user_cache, warn_compat
end
