using NeuralPDE, Lux, Optimization, OptimizationOptimJL, ParallelParticleSwarms
import ModelingToolkit: Interval

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C = 1
eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))

# Initial and boundary conditions
bcs = [
    u(t, 0) ~ 0.0, # for all t > 0
    u(t, 1) ~ 0.0, # for all t > 0
    u(0, x) ~ x * (1.0 - x), #for all 0 < x < 1
    Dt(u(0, x)) ~ 0.0,
] #for all  0 < x < 1]

# Space and time domains
domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0),
]
# Discretization
dx = 0.1

# Neural network
chain = Lux.Chain(Dense(2, 10, Lux.σ), Dense(10, 5, Lux.σ), Dense(5, 1))
discretization = PhysicsInformedNN(chain, GridTraining(dx))

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = OptimizationOptimJL.BFGS()
@time res = Optimization.solve(prob, opt; callback = callback, maxiters = 1200)
phi = discretization.phi
