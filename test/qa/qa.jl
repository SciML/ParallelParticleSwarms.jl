using SciMLTesting, ParallelParticleSwarms, Test
using JET

# The SciMLBase common interface ParallelParticleSwarms deliberately reexports, so that
# `using ParallelParticleSwarms` is enough to build an `OptimizationProblem` and `solve`
# it. Owned and documented upstream by SciMLBase; kept in sync with the reexport
# `export` block in src/ParallelParticleSwarms.jl.
const REEXPORTS = (
    :AbstractAnalyticalProblem, :AddVector, :AffineOperator, :AllObserved,
    :AnalyticalProblem, :BVPFunction, :BVProblem, :BatchIntegralFunction,
    :BlockDiagonalOperator, :CallbackSet, :CheckInit, :Clocks, :ContinuousCallback,
    :ConvexOptimizationProblem, :DAEFunction, :DAEProblem, :DAESolution, :DDEFunction,
    :DDEProblem, :DiagonalOperator, :DiscreteCallback, :DiscreteFunction, :DiscreteProblem,
    :DynamicalBVPFunction, :DynamicalDDEFunction, :DynamicalDDEProblem,
    :DynamicalODEFunction, :DynamicalODEProblem, :DynamicalSDEFunction,
    :DynamicalSDEProblem, :EigenvalueProblem, :EigenvalueSolution, :EigenvalueTarget,
    :EnsembleAnalysis, :EnsembleContext, :EnsembleDistributed, :EnsembleProblem,
    :EnsembleSerial, :EnsembleSolution, :EnsembleSplitThreads, :EnsembleSummary,
    :EnsembleTestSolution, :EnsembleThreads, :FunctionOperator, :HomotopyNonlinearFunction,
    :HomotopyProblem, :IdentityOperator, :ImplicitDiscreteFunction,
    :ImplicitDiscreteProblem, :IncrementingODEFunction, :IncrementingODEProblem,
    :IntegralFunction, :IntegralProblem, :IntegralSolution, :IntervalNonlinearFunction,
    :IntervalNonlinearProblem, :InvertibleOperator, :LinearAliasSpecifier, :LinearProblem,
    :LinearSolution, :MatrixOperator, :MultiObjectiveOptimizationFunction, :NoiseProblem,
    :NonlinearFunction, :NonlinearLeastSquaresProblem, :NonlinearProblem,
    :NonlinearSolution, :NullOperator, :ODEAliasSpecifier, :ODEFunction, :ODEInputFunction,
    :ODEProblem, :ODESolution, :OptimizationFunction, :OptimizationProblem,
    :OptimizationSolution, :PDENoTimeSolution, :PDEProblem, :PDETimeSeriesSolution,
    :RODEFunction, :RODEProblem, :RODESolution, :ReturnCode, :SCCNonlinearProblem,
    :SDDEFunction, :SDDEProblem, :SDEFunction, :SDEProblem, :SampledIntegralProblem,
    :ScalarOperator, :SciMLBase, :SciMLOperators, :SecondOrderBVProblem,
    :SecondOrderDDEProblem, :SecondOrderODEProblem, :SplitFunction, :SplitODEProblem,
    :SplitSDEFunction, :SplitSDEProblem, :StaticWOperator, :SteadyStateProblem,
    :SteadyStateSolution, :TensorProductOperator, :TensorSumOperator, :TimeDomain,
    :TwoPointBVPFunction, :TwoPointBVProblem, :TwoPointDynamicalBVPFunction,
    :TwoPointSecondOrderBVProblem, :VectorContinuousCallback, :WOperator, :add_saveat!,
    :add_tstop!, :addat!, :addat_non_user_cache!, :addsteps!, :auto_dt_reset!,
    :cache_operator, :change_t_via_interpolation!, :check_error, :check_keywords,
    :concretize, :deleteat!, :deleteat_non_user_cache!, :derivative_discontinuity!,
    :discretize, :du_cache, :first_tstop, :full_cache, :get_dt, :get_du, :get_du!,
    :get_proposed_dt, :get_rng, :get_tmp_cache, :has_adjoint, :has_concretization, :has_exp,
    :has_expmv, :has_expmv!, :has_ldiv, :has_ldiv!, :has_mul, :has_mul!, :has_rng,
    :has_tstop, :init, :is_discrete_time_domain, :iscached, :isclock, :isconstant,
    :iscontinuous, :isconvertible, :isdiscrete, :isinplace, :islinear, :issolverstepclock,
    :issquare, :kronsum, :pop_tstop!, :rand_cache, :ratenoise_cache,
    :reeval_internals_due_to_modification!, :reinit!, :remake, :resize_non_user_cache!,
    :savevalues!, :set_abstol!, :set_proposed_dt!, :set_reltol!, :set_rng!, :set_t!,
    :set_u!, :solve, :solve!, :step!, :supports_solve_rng, :symbolic_discretize,
    :terminate!, :u_cache, :u_modified!, :update_coefficients, :update_coefficients!,
    :user_cache, :warn_compat,
)

run_qa(
    ParallelParticleSwarms;
    reexports_allow = REEXPORTS,
    # The reexported names are documented by SciMLBase; they are not rendered in this
    # package's docs, which cover the ParallelParticleSwarms API only.
    api_docs_kwargs = (; rendered_ignore = REEXPORTS),
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            # `@atomic`/`@atomicreplace` are exported by Atomix but not declared
            # `public` there; the ignore drops once Atomix marks them public.
            ignore = (Symbol("@atomic"), Symbol("@atomicreplace")),
        ),
    ),
)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using
    # ParallelParticleSwarms`, so the allow-list cannot drift into approving names the
    # package no longer provides.
    @testset "$name" for name in REEXPORTS
        @test Base.ispublic(ParallelParticleSwarms, name)
        @test isdefined(ParallelParticleSwarms, name)
    end
end
