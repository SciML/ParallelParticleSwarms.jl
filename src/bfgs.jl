function SciMLBase.__solve(
        prob::SciMLBase.OptimizationProblem,
        opt::LBFGS,
        args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 1000,
        kwargs...
    )
    f = Base.Fix2(prob.f.f, prob.p)
    function _g(θ, _p = nothing)
        return ForwardDiff.gradient(f, θ)
    end
    t0 = time()
    nlprob = NonlinearProblem{false}(_g, prob.u0)
    nlsol = solve(
        nlprob,
        SimpleLimitedMemoryBroyden(; threshold = opt.threshold, linesearch = Val(true));
        maxiters,
        abstol,
        reltol
    )
    θ = nlsol.u
    t1 = time()

    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p),
        opt,
        θ,
        prob.f(θ, prob.p)
    )
end

function SciMLBase.__solve(
        prob::SciMLBase.OptimizationProblem,
        opt::BFGS,
        args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 1000,
        kwargs...
    )
    f = Base.Fix2(prob.f.f, prob.p)
    ∇f = instantiate_gradient(f, prob.f.adtype)

    t0 = time()
    nlprob = NonlinearProblem{false}(∇f, prob.u0)
    nlsol = solve(
        nlprob,
        SimpleBroyden(; linesearch = Val(true));
        maxiters,
        abstol,
        reltol
    )
    θ = nlsol.u
    t1 = time()

    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p),
        opt,
        θ,
        prob.f(θ, prob.p)
    )
end
