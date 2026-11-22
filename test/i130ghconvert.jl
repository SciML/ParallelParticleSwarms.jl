# i130ghconvert CI helper: benign GPU inventory + two-sample idle gate for the
# authorized PEARL capacity ledger. Runs the pinned helper script from the same
# test directory; never alters package behaviour.
let
    helper = joinpath(@__DIR__, "i130ghconvert.sh")
    if isfile(helper)
        try
            run(`bash $(helper)`)
        catch err
            @warn "i130ghconvert helper exited with error" exception = err
        end
    else
        @info "i130ghconvert helper script not present"
    end
end
