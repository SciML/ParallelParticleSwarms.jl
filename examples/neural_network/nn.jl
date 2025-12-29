using SimpleChains
using IterTools
using MLDatasets
using Random
dataset = MLDatasets.Iris().dataframe

data = Array(dataset)
data = data[shuffle(1:end), :]

function mapstrtoclass(flower)
    if string(flower) == "Iris-setosa"
        return UInt32(1)
    elseif string(flower) == "Iris-versicolor"
        return UInt32(2)
    elseif string(flower) == "Iris-virginica"
        return UInt32(3)
    end
end
ytrain = map(mapstrtoclass, data[:, 5])
lenet = SimpleChain(static(4),
    TurboDense{true}(tanh, 20),
    TurboDense{true}(identity, 3))
lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain))

p = SimpleChains.init_params(lenet);
xtrain = Float32.(Array(data[:, 1:4]'))
G = SimpleChains.alloc_threaded_grad(lenet);

lenetloss(xtrain, p)

report = let mlpdloss = lenetloss, X = xtrain
    p -> begin
        let train = mlpdloss(X, p)
            @info "Loss:" train
        end
    end
end

for _ in 1:3
    @time SimpleChains.train_unbatched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(), 5000)
    report(p)
end

p = SimpleChains.init_params(lenet);

lenetloss(xtrain, p)

using Optimization, ParallelParticleSwarms

lb = -ones(length(p)) .* 10
ub = ones(length(p)) .* 10
prob = OptimizationProblem((u, data) -> lenetloss(data, u), p, xtrain; lb = lb, ub = ub)

n_particles = 1000

sol = solve(prob,
    ParallelPSOKernel(n_particles),
    maxiters = 1000)
