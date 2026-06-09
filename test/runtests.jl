using Pkg

const TEST_GROUP = get(ENV, "PPS_TEST_GROUP", "Core")

if TEST_GROUP == "QA"
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
    include("qa.jl")
else
    using SafeTestsets
    using Test

    global CI_GROUP = get(ENV, "GROUP", "CPU")

    @safetestset "Regression tests" include("./regression.jl")
    @safetestset "Reinitialization tests" include("./reinit.jl")

    #TODO: Current throws warning for redefinition with the use of @testset multiple times. Migrate to TestItemRunners.jl
    @testset for BACKEND in unique(("CPU", CI_GROUP))
        global GROUP = BACKEND
        @testset "$(BACKEND) optimizers tests" include("./gpu.jl")
        GC.gc(true)
        @testset "$(BACKEND) optimizers with constraints tests" include("./constraints.jl")
        GC.gc(true)
        @testset "$(BACKEND) hybrid optimizers" include("./lbfgs.jl")
    end
end
