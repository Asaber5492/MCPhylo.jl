################################################################################
## Linear Regression
##   y ~ MvNormal(X * (beta0 .* gamma), 1)
##   gamma ~ DiscreteUniform(0, 1)
################################################################################

using MCPhylo

## Data
n, p = 25, 10
X = randn(n, p)
beta0 = randn(p)
gamma0 = rand(0:1, p)
y = X * (beta0 .* gamma0) + randn(n)

## Log-transformed Posterior(gamma) + Constant
logf = function(gamma::DenseVector)
  logpdf(MvNormal(X * (beta0 .* gamma), 1.0), y)
end

## MCMC Simulation with Binary Metropolised Gibbs
t = 10000
sim1 = Chains(t, p, names = map(i -> "gamma[$i]", 1:p))
sim2 = Chains(t, p, names = map(i -> "gamma[$i]", 1:p))
gamma1 = BMGVariate(zeros(p), logf)
gamma2 = BMGVariate(zeros(p), logf, k=Vector{Int}[[i] for i in 1:p])
for i in 1:t
  sample!(gamma1)
  sample!(gamma2)
  sim1[i, :, 1] = gamma1
  sim2[i, :, 1] = gamma2
end
describe(sim1)
describe(sim2)

#mark that we got to the end of the test file succesfully
@test true
