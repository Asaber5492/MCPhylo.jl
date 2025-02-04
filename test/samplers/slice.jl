################################################################################
## Linear Regression
##   y ~ N(b0 + b1 * x, s2)
##   b0, b1 ~ N(0, 1000)
##   s2 ~ invgamma(0.001, 0.001)
################################################################################

using MCPhylo

## Data
data = Dict(
  :x => [1, 2, 3, 4, 5],
  :y => [1, 3, 3, 3, 5]
)

## Log-transformed Posterior(b0, b1, log(s2)) + Constant
logf = function(x::DenseVector)
   b0 = x[1]
   b1 = x[2]
   logs2 = x[3]
   r = data[:y] .- b0 .- b1 .* data[:x]
   (-0.5 * length(data[:y]) - 0.001) * logs2 -
     (0.5 * dot(r, r) + 0.001) / exp(logs2) -
     0.5 * b0^2 / 1000 - 0.5 * b1^2 / 1000
end

## MCMC Simulation with Slice Sampling
## With multivariate (1) and univariate (2) updating
n = 5000
sim1 = Chains(n, 3, names = ["b0", "b1", "s2"])
sim2 = Chains(n, 3, names = ["b0", "b1", "s2"])
width = [1.0, 1.0, 2.0]
theta1 = SliceUnivariate([0.0, 0.0, 0.0], width, logf)
theta2 = SliceMultivariate([0.0, 0.0, 0.0], width, logf)
for i in 1:n
  sample!(theta1)
  sample!(theta2)
  sim1[i, :, 1] = [theta1[1:2]; exp(theta1[3])]
  sim2[i, :, 1] = [theta2[1:2]; exp(theta2[3])]
end
describe(sim1)
describe(sim2)

#mark that we got to the end of the test file succesfully
@test true
