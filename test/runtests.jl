# I101696 GPU runner diagnostics (non-workflow probe)
try
    println("=== I101696 GPU PROBE START ===")
    println("I101696 hostname: ", gethostname())
    println("I101696 nvidia-smi: ", Sys.which("nvidia-smi"))
    run(`nvidia-smi -L`)
    run(`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`)
    println("=== I101696 GPU PROBE END ===")
catch e
    println("I101696 GPU PROBE FAILED: ", e)
end

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
        @safetestset "SimpleLBFGS hybrid polish" include("./lbfgs.jl")
        @safetestset "Algorithm interface tests" include("./interface.jl")

        #TODO: Current throws warning for redefinition with the use of @testset multiple times. Migrate to TestItemRunners.jl
        return @testset for BACKEND in unique(("CPU", CI_GROUP))
            global GROUP = BACKEND
            @testset "$(BACKEND) ODE parameter estimation particle update" include("./ode_pso.jl")
            @testset "$(BACKEND) optimizers tests" include("./gpu.jl")
            GC.gc(true)
            @testset "$(BACKEND) optimizers with constraints tests" include("./constraints.jl")
        end
    end,
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
)
