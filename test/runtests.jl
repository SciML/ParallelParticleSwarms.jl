using SafeTestsets
using Test

global CI_GROUP = get(ENV, "GROUP", "CPU")

@safetestset "Regression tests" include("./regression.jl")
@safetestset "Reinitialization tests" include("./reinit.jl")
@safetestset "JET static analysis" include("./jet.jl")

#TODO: Curent throws warning for redefinition with the use of @testset multiple times. Migrate to TestItemRunners.jl
@testset for BACKEND in unique(("CPU", CI_GROUP))
    global GROUP = BACKEND
    # Run hybrid optimizers first on GPU — the HybridPSO kernel is the most
    # complex and needs the most GPU memory for JIT compilation.
    @testset "$(BACKEND) hybrid optimizers" include("./lbfgs.jl")
    GC.gc(true)
    @testset "$(BACKEND) optimizers tests" include("./gpu.jl")
    GC.gc(true)
    @testset "$(BACKEND) optimizers with constraints tests" include("./constraints.jl")
end
