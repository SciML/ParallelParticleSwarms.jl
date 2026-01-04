using JET
using ParallelParticleSwarms
using StaticArrays
using Test

@testset "JET static analysis" begin
    @testset "Utility functions type stability" begin
        @test_opt target_modules = (ParallelParticleSwarms,) ParallelParticleSwarms.θ_default(
            0.5f0
        )
        @test_opt target_modules = (ParallelParticleSwarms,) ParallelParticleSwarms.γ_default(
            0.5f0
        )

        @test_opt target_modules = (ParallelParticleSwarms,) ParallelParticleSwarms.uniform(
            2, Float32[-5.0, -5.0], Float32[5.0, 5.0]
        )
    end

    @testset "Particle struct construction" begin
        T = SVector{2, Float32}
        position = @SVector zeros(Float32, 2)
        velocity = @SVector zeros(Float32, 2)
        cost = 0.0f0

        @test_opt target_modules = (ParallelParticleSwarms,) ParallelParticleSwarms.SPSOParticle(
            position, velocity, cost, position, cost
        )

        @test_opt target_modules = (ParallelParticleSwarms,) ParallelParticleSwarms.SPSOGBest(
            position, cost
        )
    end
end
