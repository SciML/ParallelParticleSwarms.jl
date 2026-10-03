using ParallelParticleSwarms, KernelAbstractions, StaticArrays, Test

@testset "parameter_estim_ode! particle update clamps each coordinate to the box" begin
    lb = SVector(-10.0f0, -10.0f0)
    ub = SVector(10.0f0, 10.0f0)
    positions = [SVector(5.0f0, -20.0f0), SVector(-20.0f0, 5.0f0), SVector(20.0f0, -20.0f0)]
    backend = CPU()
    update! = ParallelParticleSwarms.ode_update_particle_states!(backend)
    for x in positions
        single = [ParallelParticleSwarms.SPSOParticle(x, zero(x), Inf32, x, Inf32)]
        update!(
            single, lb, ub, ParallelParticleSwarms.SPSOGBest(x, Inf32), 0.0f0;
            ndrange = 1
        )
        KernelAbstractions.synchronize(backend)
        @test single[1].position == clamp.(x, lb, ub)
    end
end
