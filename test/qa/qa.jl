using SciMLTesting, ParallelParticleSwarms, Test
using JET

run_qa(
    ParallelParticleSwarms;
    explicit_imports = true,
    # `NonlinearFunction` (SciMLBase) and `ImmutableNonlinearProblem`
    # (NonlinearSolveBase re-export of a SciMLBase type) are deliberately
    # extended by the hybrid solver in src/hybrid.jl; treat them as owned so
    # Aqua does not flag the extensions as piracy.
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                ParallelParticleSwarms.SciMLBase.NonlinearFunction,
                ParallelParticleSwarms.ImmutableNonlinearProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@atomic"),          # KernelAbstractions (re-exports Atomix), still non-public
                Symbol("@atomicreplace"),   # KernelAbstractions (re-exports Atomix), still non-public
                :ImmutableNonlinearProblem, # SciMLBase type re-exported by NonlinearSolveBase, still non-public there
                :vectorized_solve,          # DiffEqGPU internal
                :vectorized_asolve,         # DiffEqGPU internal
            ),
        ),
        all_explicit_imports_via_owners = (;
            ignore = (
                Symbol("@atomic"),          # owner Atomix, re-exported by KernelAbstractions
                Symbol("@atomicreplace"),   # owner Atomix, re-exported by KernelAbstractions
                :ImmutableNonlinearProblem, # owner SciMLBase, imported via NonlinearSolveBase
            ),
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (:OptimizationStats,), # owner SciMLBase, accessed via Optimization
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :DefaultOptimizationCache, # SciMLBase internal
                :OptimizationStats,        # Optimization internal (owner SciMLBase)
                :__solve,                  # SciMLBase internal
                :evaluate_f,               # NonlinearSolveBase.Utils internal
                :evaluate_f!!,             # NonlinearSolveBase.Utils internal
                :gradient,                 # ForwardDiff internal
                :sacollect,                # StaticArrays internal
                :sample,                   # QuasiMonteCarlo internal
            ),
        ),
    ),
    # The module relies on many implicit imports from heavy `using` deps
    # (SciMLBase, KernelAbstractions, StaticArrays, Optimization, ...); making
    # them all explicit is a large refactor tracked in
    # SciML/ParallelParticleSwarms.jl#106.
    ei_broken = (:no_implicit_imports,),
)
