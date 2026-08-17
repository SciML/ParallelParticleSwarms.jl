using ParallelParticleSwarms
import ParallelParticleSwarms: PSOAlgorithm, pso_solve
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
