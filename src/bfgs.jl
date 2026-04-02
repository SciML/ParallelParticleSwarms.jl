function SciMLBase.__solve(
        prob::SciMLBase.OptimizationProblem,
        opt::LBFGS,
        args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 1000,
        linesearch = StrongWolfeLineSearch(),
        kwargs...
    )
    u0 = as_svector(prob.u0)
    ∇f = as_svector_grad(instantiate_gradient(prob.f.f, prob.f.adtype))
    t0 = time()
    nlprob = ImmutableNonlinearProblem{false}(∇f, u0, prob.p)
    nlsol = solve(
        nlprob,
        SimpleLimitedMemoryBroyden(; threshold = opt.threshold, linesearch);
        maxiters,
        abstol,
        reltol,
        grad_f = ∇f,
        kwargs...
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

# `BFGS` here solves ∇f = 0 via `SimpleBroyden` (secant/quasi-Newton on the gradient).
function SciMLBase.__solve(
        prob::SciMLBase.OptimizationProblem,
        opt::BFGS,
        args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 1000,
        linesearch = StrongWolfeLineSearch(),
        kwargs...
    )
    u0 = as_svector(prob.u0)
    ∇f = as_svector_grad(instantiate_gradient(prob.f.f, prob.f.adtype))

    t0 = time()
    nlprob = ImmutableNonlinearProblem{false}(∇f, u0, prob.p)
    nlsol = solve(
        nlprob,
        SimpleBroyden(; linesearch);
        maxiters,
        abstol,
        reltol,
        grad_f = ∇f,
        kwargs...
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
