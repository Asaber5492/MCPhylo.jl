#################### Hamiltonian Monte Carlo ####################

#################### Types and Constructors ####################

mutable struct HMCTune <: SamplerTune
  logfgrad::Union{Function, Missing}
  epsilon::Float64
  L::Int
  SigmaL::Union{UniformScaling{Bool}, LowerTriangular{Float64}}

  HMCTune() = new()

  HMCTune(x, epsilon::Real, L::Integer) =
    new(missing, epsilon, L, I)

  HMCTune(x, epsilon::Real, L::Integer, logfgrad::Function) =
    new(logfgrad, epsilon, L, I)

  function HMCTune(x, epsilon::Real, L::Integer,
                   Sigma::Matrix{T}) where {T<:Real}
    new(missing, epsilon, L, cholesky(Sigma).L)
  end

  function HMCTune(x, epsilon::Real, L::Integer,
                   Sigma::Matrix{T}, logfgrad::Function) where {T<:Real}
    new(logfgrad, epsilon, L, cholesky(Sigma).L)
  end
end


const HMCVariate = SamplerVariate{HMCTune}

validate(v::HMCVariate) = validate(v, v.tune.SigmaL)

validate(v::HMCVariate, SigmaL::UniformScaling) = v

function validate(v::HMCVariate, SigmaL::LowerTriangular)
  n = length(v)
  size(SigmaL, 1) == n ||
    throw(ArgumentError("Sigma dimension differs from variate length $n"))
  v
end


#################### Sampler Constructor ####################
"""
    HMC(params::ElementOrVector{Symbol}, epsilon::Real, L::Integer; args...)

Construct a `Sampler` object for HMC sampling. Parameters are assumed to be
continuous, but may be constrained or unconstrained.

Returns a `Sampler{HMCTune}` type object.

* `params`: stochastic node(s) to be updated with the sampler. Constrained parameters are mapped to unconstrained space according to transformations defined by the Stochastic `unlist()` function.

* `epsilon`: step size.

* `L`: number of steps to take in the Leapfrog algorithm.

* `Sigma`: covariance matrix for the multivariate normal proposal distribution. The covariance matrix is relative to the unconstrained parameter space, where candidate draws are generated. If omitted, the identity matrix is assumed.

* `dtype` : differentiation for gradient calculations. Options are
    * `:central` : central differencing
    * `:forward` : forward differencing.
"""
function HMC(params::ElementOrVector{Symbol}, epsilon::Real, L::Integer;
             args...)
  HMCSampler(params, epsilon, L; args...)
end
"""
    HMC(params::ElementOrVector{Symbol}, epsilon::Real, L::Integer, Sigma::Matrix{T}; args...)

Construct a `Sampler` object for HMC sampling. Parameters are assumed to be
continuous, but may be constrained or unconstrained.

Returns a `Sampler{HMCTune}` type object.
"""
function HMC(params::ElementOrVector{Symbol}, epsilon::Real,
              L::Integer, Sigma::Matrix{T}; args...) where {T<:Real}
  HMCSampler(params, epsilon, L, Sigma; args...)
end

function HMCSampler(params, pargs...; dtype::Symbol=:forward)
  samplerfx = function(model::Model, block::Integer)
    block = SamplingBlock(model, block, true)
    v = SamplerVariate(block, pargs...)
    sample!(v, x -> logpdfgrad!(block, x, dtype))
    relist(block, v)
  end
  Sampler(params, samplerfx, HMCTune())
end


#################### Sampling Functions ####################

sample!(v::HMCVariate) = sample!(v, v.tune.logfgrad)
"""
    sample!(v::HMCVariate, logfgrad::Function)

Draw one sample from a target distribution using the HMC sampler. Parameters
are assumed to be continuous and unconstrained.

Returns `v` updated with simulated values and associated tuning parameters.
"""
function sample!(v::HMCVariate, logfgrad::Function)
  tune = v.tune

  x1 = v[:]
  logf0, grad0 = logf1, grad1 = logfgrad(x1)

  ## Momentum variables
  p0 = p1 = tune.SigmaL * randn(length(v))

  ## Make a half step for a momentum at the beginning
  p1 += 0.5 * tune.epsilon * grad0

  ## Alternate full steps for position and momentum
  for i in 1:tune.L
    ## Make a full step for the position
    x1 += tune.epsilon * p1

    logf1, grad1 = logfgrad(x1)

    ## Make a full step for the momentum
    p1 += tune.epsilon * grad1
  end

  ## Make a half step for momentum at the end
  p1 -= 0.5 * tune.epsilon * grad1

  ## Negate momentum at end of trajectory to make the proposal symmetric
  p1 *= -1.0

  ## Evaluate potential and kinetic energies at start and end of trajectory
  SigmaLinv = inv(tune.SigmaL)
  Kp0 = 0.5 * sum(abs2, SigmaLinv * p0)
  Kp1 = 0.5 * sum(abs2, SigmaLinv * p1)

  if rand() < exp((logf1 - Kp1) - (logf0 - Kp0))
    v[:] = x1
  end

  v
end
