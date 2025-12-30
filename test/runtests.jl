using SafeTestsets
using Test

global CI_GROUP = get(ENV, "GROUP", "CPU")

@safetestset "Regression tests" include("./regression.jl")
@safetestset "Reinitialization tests" include("./reinit.jl")
@safetestset "JET static analysis" include("./jet.jl")

#TODO: Curent throws warning for redefinition with the use of @testset multiple times. Migrate to TestItemRunners.jl
@testset for BACKEND in unique(("CPU", CI_GROUP))
    global GROUP = BACKEND
    @testset "$(BACKEND) optimizers tests" include("./gpu.jl")
    @testset "$(BACKEND) optimizers with constraints tests" include("./constraints.jl")
    @testset "$(BACKEND) hybrid optimizers" include("./lbfgs.jl")
end
