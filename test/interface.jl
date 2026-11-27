using ParallelParticleSwarms
import ParallelParticleSwarms: PSOAlgorithm, pso_solve
using KernelAbstractions: CPU
using Test

struct MockPSO <: PSOAlgorithm end

function pso_solve(prob, ::MockPSO; kwargs...)
    position = copy(prob.u0)
    global_best = (; position, cost = prob.f(position, prob.p))
    return global_best, [(; position)], 0.0
end

@testset "PSOAlgorithm interface" begin
    prob = OptimizationProblem((u, _) -> sum(abs2, u), [1.0, -2.0], nothing)
    sol = solve(prob, MockPSO())

    @test sol.u == prob.u0
    @test sol.objective == 5.0
    @test sol.original == [prob.u0]
end

@testset "HybridPSO constructors" begin
    pso = ParallelSyncPSOKernel(8; backend = CPU())
    lo = SimpleLBFGS()
    alg = HybridPSO(pso, lo, CPU())
    @test alg isa HybridPSO{CPU, typeof(lo)}
    @test isconcretetype(HybridPSO{CPU, typeof(lo)})
    @test alg.polish === nothing
    @test HybridPSO{CPU, typeof(lo)}(pso, lo, CPU()) === alg
    @test HybridPSO(; pso, local_opt = lo, backend = CPU()) === alg

    nm = NelderMeadPolish(; all_particles = true)
    polished = HybridPSO(; pso, local_opt = lo, backend = CPU(), polish = nm)
    @test polished.polish === nm
    @test typeof(polished) === typeof(alg)
end
