using SafeTestsets

@safetestset "JET static analysis" begin
    include("jet_tests.jl")
end
