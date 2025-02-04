#################### Adaptive Metropolis within Gibbs ####################

#################### Types and Constructors ####################

mutable struct AMWGTune <: SamplerTune
  logf::Union{Function, Missing}
  adapt::Bool
  accept::Vector{Int}
  batchsize::Int
  m::Int
  sigma::Vector{Float64}
  target::Float64

  AMWGTune() = new()

  function AMWGTune(x::Vector, sigma::ElementOrVector{T},
                    logf::Union{Function, Missing}; batchsize::Integer=50,
                    target::Real=0.44) where {T<:Real}
    new(logf, false, zeros(Int, length(x)), batchsize, 0, copy(sigma), target)
  end
end

AMWGTune(x::Vector, sigma::ElementOrVector{T}; args...) where {T<:Real} =
  AMWGTune(x, sigma, missing; args...)

AMWGTune(x::Vector, sigma::Real, logf::Union{Function, Missing}; args...) =
  AMWGTune(x, fill(sigma, length(x)), logf; args...)


const AMWGVariate = SamplerVariate{AMWGTune}

function validate(v::AMWGVariate)
  n = length(v)
  length(v.tune.sigma) == n ||
    throw(ArgumentError("length(sigma) differs from variate length $n"))
  v
end


#################### Sampler Constructor ####################
"""
    AMWG(params::ElementOrVector{Symbol},

    sigma::ElementOrVector{T};

    adapt::Symbol=:all,

    args...) where {T<:Real}

Construct a `Sampler` object for AMWG sampling. Parameters are assumed to be continuous, but may be constrained or unconstrained.

Returns a `Sampler{ABCTune}` type object.

* `params`:  stochastic node(s) to be updated with the sampler. Constrained parameters are mapped to unconstrained space according to transformations defined by the Stochastic `unlist()` function.

* `sigma`: scaling value or vector of the same length as the combined elements of nodes

* `params`, defining initial standard deviations for univariate normal proposal distributions. Standard deviations are relative to the unconstrained parameter space, where candidate draws are generated.

* `adapt` : type of adaptation phase.  Options are
    * `:all` : adapt proposal during all iterations.
    * `:burnin` : adapt proposal during burn-in iterations.
    * `:none` : no adaptation (Metropolis-within-Gibbs sampling with fixed proposal).

* `args...`: additional keyword arguments to be passed to the `AMWGVariate` constructor.
"""
function AMWG(params::ElementOrVector{Symbol},
              sigma::ElementOrVector{T}; adapt::Symbol=:all, args...) where {T<:Real}
  adapt in [:all, :burnin, :none] ||
    throw(ArgumentError("adapt must be one of :all, :burnin, or :none"))

  samplerfx = function(model::Model, block::Integer)
    block = SamplingBlock(model, block, true)
    v = SamplerVariate(block, sigma; args...)
    isadapt = adapt == :burnin ? model.iter <= model.burnin :
              adapt == :all ? true : false
    sample!(v, x -> logpdf!(block, x), adapt=isadapt)
    relist(block, v)
  end
  Sampler(params, samplerfx, AMWGTune())
end


#################### Sampling Functions ####################

sample!(v::AMWGVariate; args...) = sample!(v, v.tune.logf; args...)
"""
    sample!(v::AMWGVariate, logf::Function; adapt::Bool=true)

Draw one sample from a target distribution using the AMWG sampler.
Parameters are assumed to be continuous and unconstrained.
Returns `v` updated with simulated values and associated tuning parameters.
"""
function sample!(v::AMWGVariate, logf::Function; adapt::Bool=true)
  tune = v.tune
  setadapt!(v, adapt)
  if tune.adapt
    tune.m += 1
    amwg_sub!(v, logf)
    if tune.m % tune.batchsize == 0
      delta = min(0.01, (tune.m / tune.batchsize)^-0.5)
      for i in 1:length(tune.sigma)
        epsilon = tune.accept[i] / tune.m < tune.target ? -delta : delta
        tune.sigma[i] *= exp(epsilon)
      end
    end
  else
    amwg_sub!(v, logf)
  end
  v
end


function setadapt!(v::AMWGVariate, adapt::Bool)
  tune = v.tune
  if adapt && !tune.adapt
    tune.accept[:] .= 0
    tune.m = 0
  end
  tune.adapt = adapt
  v
end


function amwg_sub!(v::AMWGVariate, logf::Function)
  logf0 = logf(v.value)
  n = length(v)
  z = v.tune.sigma .* randn(n)
  for i in 1:n
    x = v[i]
    v[i] += z[i]
    logfprime = logf(v.value)
    if rand() < exp(logfprime - logf0)
      logf0 = logfprime
      v.tune.accept[i] += v.tune.adapt
    else
      v[i] = x
    end
  end
  v
end
