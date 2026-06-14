using SafeTestsets
using Test
using SciMLTesting

run_tests(;
    env = "PPS_TEST_GROUP",
    default = "Core",
    core = function ()
        global CI_GROUP = get(ENV, "GROUP", "CPU")

        @safetestset "Regression tests" include("./regression.jl")
        @safetestset "Reinitialization tests" include("./reinit.jl")

        #TODO: Current throws warning for redefinition with the use of @testset multiple times. Migrate to TestItemRunners.jl
        return @testset for BACKEND in unique(("CPU", CI_GROUP))
            global GROUP = BACKEND
            @testset "$(BACKEND) optimizers tests" include("./gpu.jl")
            GC.gc(true)
            @testset "$(BACKEND) optimizers with constraints tests" include("./constraints.jl")
            GC.gc(true)
            @testset "$(BACKEND) hybrid optimizers" include("./lbfgs.jl")
        end
    end,
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
)
