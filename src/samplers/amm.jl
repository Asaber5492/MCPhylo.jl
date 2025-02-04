#################### Adaptive Mixture Metropolis ####################

#################### Types and Constructors ####################

mutable struct AMMTune <: SamplerTune
  logf::Union{Function, Missing}
  adapt::Bool
  beta::Float64
  m::Int
  Mv::Vector{Float64}
  Mvv::Matrix{Float64}
  scale::Float64
  SigmaL::LowerTriangular{Float64}
  SigmaLm::Matrix{Float64}

  AMMTune() = new()

  function AMMTune(x::Vector, Sigma::Matrix{T},
                  logf::Union{Function, Missing}; beta::Real=0.05,
                  scale::Real=2.38) where {T<:Real}
    new(logf, false, beta, 0, Vector{Float64}(undef, 0), Matrix{Float64}(undef, 0, 0), scale, cholesky(Sigma).L, Matrix{Float64}(undef, 0, 0))
  end
end

AMMTune(x::Vector, Sigma::Matrix{T}; args...) where {T<:Real} = AMMTune(x, Sigma, missing; args...)

const AMMVariate = SamplerVariate{AMMTune}

function validate(v::AMMVariate)
  n = length(v)
  size(v.tune.SigmaL, 1) == n ||
    throw(ArgumentError("Sigma dimension differs from variate length $n"))
  v
end


#################### Sampler Constructor ####################
"""
    AMM(params::ElementOrVector{Symbol}, Sigma::Matrix{T};
        adapt::Symbol=:all, args...) where {T<:Real}

Construct a `Sampler` object for AMM sampling. Parameters are assumed to be
 continuous, but may be constrained or unconstrained.

Returns a `Sampler{AMMTune}` type object.

* `params` : stochastic node(s) to be updated with the sampler.  Constrained parameters are mapped to unconstrained space according to transformations defined by the Stochastic `unlist()` function.

* `Sigma` : covariance matrix for the non-adaptive multivariate normal proposal distribution.  The covariance matrix is relative to the unconstrained parameter space, where candidate draws are generated.

* `adapt` : type of adaptation phase.  Options are
    * `:all` : adapt proposal during all iterations.
    * `:burnin` : adapt proposal during burn-in iterations.
    * `:none` : no adaptation (multivariate Metropolis sampling with fixed proposal).

* `args...` : additional keyword arguments to be passed to the `AMMVariate` constructor.
"""
function AMM(params::ElementOrVector{Symbol}, Sigma::Matrix{T};
              adapt::Symbol=:all, args...) where {T<:Real}
  adapt in [:all, :burnin, :none] ||
    throw(ArgumentError("adapt must be one of :all, :burnin, or :none"))

  samplerfx = function(model::Model, block::Integer)
    block = SamplingBlock(model, block, true)
    v = SamplerVariate(block, Sigma; args...)
    isadapt = adapt == :burnin ? model.iter <= model.burnin :
              adapt == :all ? true : false
    sample!(v, x -> logpdf!(block, x), adapt=isadapt)
    relist(block, v)
  end
  Sampler(params, samplerfx, AMMTune())
end


#################### Sampling Functions ####################

sample!(v::AMMVariate; args...) = sample!(v, v.tune.logf; args...)

"""
    sample!(v::AMMVariate, logf::Function; adapt::Bool=true)

  Draw one sample from a target distribution using the AMM sampler. Parameters
   are assumed to be continuous and unconstrained.
Returns ``v`` updated with simulated values and associated tuning parameters.
"""
function sample!(v::AMMVariate, logf::Function; adapt::Bool=true)
  tune = v.tune
  n = length(v)

  setadapt!(v, adapt)

  x = tune.SigmaL * randn(n)
  if tune.m > 2 * n
    x = tune.beta * x + (1.0 - tune.beta) * (tune.SigmaLm * randn(n))
  end
  x += v
  if rand() < exp(logf(x) - logf(v.value))
    v[:] = x
  end

  if tune.adapt
    tune.m += 1
    p = tune.m / (tune.m + 1.0)
    tune.Mv = p * tune.Mv + (1.0 - p) * v
    tune.Mvv = p * tune.Mvv + (1.0 - p) * v * v'
    Sigma = (tune.scale^2 / n / p) * (tune.Mvv - tune.Mv * tune.Mv')
    F = cholesky(Hermitian(Sigma), Val{true}(), check=false)
    if rank(F.L, sqrt(eps(Float64))) == n
      tune.SigmaLm = F.P * F.L
    end
  end

  v
end


function setadapt!(v::AMMVariate, adapt::Bool)
  tune = v.tune
  if adapt && !tune.adapt
    n = length(v)
    tune.m = 0
    tune.Mv = v
    tune.Mvv = v * v'
    tune.SigmaLm = zeros(n, n)
  end
  tune.adapt = adapt
  v
end
