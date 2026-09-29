using ParallelParticleSwarms
import ParallelParticleSwarms: PSOAlgorithm, pso_solve, SoAParticles, SPSOParticle, SVector
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

@testset "SoAParticles getindex/setindex!/minimum" begin
    ps = SoAParticles{SVector{2, Float64}}(zeros(3, 8))
    q = SPSOParticle(SVector(1.0, 2.0), SVector(3.0, 4.0), 5.0, SVector(6.0, 7.0), -1.0)
    ps[2] = q
    @test ps[2] == q
    @test minimum(ps) == q
end
