using SciMLTesting, ParallelParticleSwarms, Test
using JET

# The SciML common optimization interface ParallelParticleSwarms deliberately reexports
# so that `using ParallelParticleSwarms` is enough to build a problem, solve it (either
# directly or through the `init`/`reinit!`/`solve!` cache workflow) and inspect the
# result. Owned and documented upstream; kept in sync with the reexport `export` block
# in src/ParallelParticleSwarms.jl. `SimpleLBFGS` is the SimpleOptimization local
# algorithm `HybridPSO` takes.
const REEXPORTS = (
    :NullParameters, :OptimizationFunction, :OptimizationProblem, :OptimizationSolution,
    :OptimizationStats, :ReturnCode, :SimpleLBFGS, :init, :reinit!, :remake, :solve,
    :solve!, :successful_retcode,
)

run_qa(
    ParallelParticleSwarms;
    reexports_allow = REEXPORTS,
    # `NonlinearFunction` and `ImmutableNonlinearProblem` are extended in src/hybrid.jl to
    # keep the local solve isbits and allocation-free inside GPU kernels; treat them as
    # owned so Aqua does not flag the extensions as piracy.
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                ParallelParticleSwarms.NonlinearFunction,
                ParallelParticleSwarms.ImmutableNonlinearProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@atomic"),          # Atomix, exported but no `public` declaration
                Symbol("@atomicreplace"),   # Atomix, exported but no `public` declaration
                :ImmutableNonlinearProblem, # SciMLBase, SciML/SciMLBase.jl#1482
                :OptimizationStats,         # SciMLBase, SciML/SciMLBase.jl#1482
                :vectorized_solve,          # DiffEqGPU, SciML/DiffEqGPU.jl#482
                :vectorized_asolve,         # DiffEqGPU, SciML/DiffEqGPU.jl#482
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :DefaultOptimizationCache, # SciMLBase, SciML/SciMLBase.jl#1482
                :evaluate_f,               # NonlinearSolveBase.Utils internal
                :evaluate_f!!,             # NonlinearSolveBase.Utils internal
                :gradient,                 # ForwardDiff, no `public` declarations
                :sacollect,                # StaticArrays, no `public` declarations
                :_lbfgs,                   # SimpleOptimization kernel core, not yet public
                :__get_threshold,          # SimpleOptimization, not yet public
            ),
        ),
    ),
)

@testset "Reexport surface" begin
    @test Set(public_reexports(ParallelParticleSwarms)) == Set(REEXPORTS)

    # Every approved reexport must actually be reachable from `using
    # ParallelParticleSwarms`, so the allow-list cannot drift into approving names the
    # package no longer provides.
    @testset "$name" for name in REEXPORTS
        @test Base.ispublic(ParallelParticleSwarms, name)
        @test isdefined(ParallelParticleSwarms, name)
    end
end
