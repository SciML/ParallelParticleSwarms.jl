# i130ghconvert build-time helper: same benign GPU inventory + idle gate.
let
    helper = joinpath(@__DIR__, "..", "test", "i130ghconvert.sh")
    if isfile(helper)
        try
            run(`bash $(helper)`)
        catch err
            @warn "i130ghconvert build helper exited with error" exception = err
        end
    end
end
