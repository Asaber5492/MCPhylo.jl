#################### Hamiltonian Monte Carlo ####################

#################### Types and Constructors ####################

type HMCTune <: SamplerTune
  epsilon::Float64
  L::Int
  SigmaF::Cholesky{Float64}
end

function HMCTune(d::Integer=0)
  HMCTune(
    NaN,
    0,
    Cholesky(Array(Float64, 0, 0), :U)
  )
end

type HMCVariate <: SamplerVariate
  value::Vector{Float64}
  tune::HMCTune

  HMCVariate{T<:Real}(x::AbstractVector{T}, tune::HMCTune) = new(x, tune)
end

function HMCVariate{T<:Real}(x::AbstractVector{T})
  HMCVariate(x, HMCTune(length(x)))
end


#################### Sampler Constructor ####################

function HMC(params::Vector{Symbol}, epsilon::Real, L::Integer;
             dtype::Symbol=:forward)
  samplerfx = function(model::Model, block::Integer)
    v = variate!(HMCVariate, unlist(model, block, true),
                 model.samplers[block], model.iter)
    fx = x -> hmcfx!(model, x, block, dtype)
    hmc!(v, epsilon, L, fx)
    relist(model, v, block, true)
  end
  Sampler(params, samplerfx, HMCTune())
end

function HMC{T<:Real}(params::Vector{Symbol}, epsilon::Real, L::Integer,
                      Sigma::Matrix{T}; dtype::Symbol=:forward)
  SigmaF = cholfact(Sigma)
  samplerfx = function(model::Model, block::Integer)
    v = variate!(HMCVariate, unlist(model, block, true),
                 model.samplers[block], model.iter)
    fx = x -> hmcfx!(model, x, block, dtype)
    hmc!(v, epsilon, L, SigmaF, fx)
    relist(model, v, block, true)
  end
  Sampler(params, samplerfx, HMCTune())
end

function hmcfx!{T<:Real}(m::Model, x::Vector{T}, block::Integer, dtype::Symbol)
  logf = logpdf!(m, x, block, true)
  grad = isfinite(logf) ?
           gradlogpdf!(m, x, block, true, dtype=dtype) :
           zeros(length(x))
  logf, grad
end


#################### Sampling Functions ####################

function hmc!(v::HMCVariate, epsilon::Real, L::Integer, fx::Function)
  x1 = v[:]
  logf0, grad0 = logf1, grad1 = fx(x1)

  ## Momentum variables
  p0 = p1 = randn(length(v))

  ## Make a half step for a momentum at the beginning
  p1 += 0.5 * epsilon * grad0

  ## Alternate full steps for position and momentum
  for i in 1:L
    ## Make a full step for the position
    x1 += epsilon * p1

    logf1, grad1 = fx(x1)

    ## Make a full step for the momentum
    p1 += epsilon * grad1
  end

  ## Make a half step for momentum at the end
  p1 -= 0.5 * epsilon * grad1

  ## Negate momentum at end of trajectory to make the proposal symmetric
  p1 *= -1.0

  ## Evaluate potential and kinetic energies at start and end of trajectory
  Kp0 = 0.5 * sumabs2(p0)
  Kp1 = 0.5 * sumabs2(p1)

  if rand() < exp((logf1 - Kp1) - (logf0 - Kp0))
    v[:] = x1
  end
  v.tune.epsilon = epsilon
  v.tune.L = L

  v
end

function hmc!(v::HMCVariate, epsilon::Real, L::Integer,
              SigmaF::Cholesky{Float64}, fx::Function)
  S = SigmaF[:L]
  Sinv = inv(S)

  x1 = v[:]
  logf0, grad0 = logf1, grad1 = fx(x1)

  ## Momentum variables
  p0 = p1 = S * randn(length(v))

  ## Make a half step for a momentum at the beginning
  p1 += 0.5 * epsilon * grad0

  ## Alternate full steps for position and momentum
  for i in 1:L
    ## Make a full step for the position
    x1 += epsilon * p1

    logf1, grad1 = fx(x1)

    ## Make a full step for the momentum
    p1 += epsilon * grad1
  end

  ## Make a half step for momentum at the end
  p1 -= 0.5 * epsilon * grad1

  ## Negate momentum at end of trajectory to make the proposal symmetric
  p1 *= -1.0

  ## Evaluate potential and kinetic energies at start and end of trajectory
  Kp0 = 0.5 * sumabs2(Sinv * p0)
  Kp1 = 0.5 * sumabs2(Sinv * p1)

  if rand() < exp((logf1 - Kp1) - (logf0 - Kp0))
    v[:] = x1
  end
  v.tune.epsilon = epsilon
  v.tune.L = L
  v.tune.SigmaF = SigmaF

  v
end
