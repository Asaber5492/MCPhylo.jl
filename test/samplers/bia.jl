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

## MCMC Simulation with Binary Individual Adaptation Sampler
t = 10000
sim = Chains(t, p, names = map(i -> "gamma[$i]", 1:p))
gamma = BIAVariate(zeros(p), logf)
for i in 1:t
  sample!(gamma)
  sim[i, :, 1] = gamma
end
describe(sim)

#mark that we got to the end of the test file succesfully
@test true
