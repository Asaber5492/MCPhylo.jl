################## Binary Hamiltonian Monte Carlo  ##################

#################### Types ####################

type BHMCTune <: SamplerTune
  traveltime::Float64
  position::Vector{Float64}
  velocity::Vector{Float64}
  wallhits::Int
  wallcrosses::Int
end

function BHMCTune(d::Integer=0)
  BHMCTune(
    NaN,
    rand(Normal(0, 1), d),
    rand(Normal(0, 1), d),
    0,
    0
  )
end

type BHMCVariate <: SamplerVariate
  value::Vector{Float64}
  tune::BHMCTune

  function BHMCVariate{T<:Real}(x::AbstractVector{T}, tune::BHMCTune)
    all(insupport(Bernoulli, x)) ||
      throw(ArgumentError("x is not a binary vector"))
    new(x, tune)
  end
end

function BHMCVariate{T<:Real}(x::AbstractVector{T})
  BHMCVariate(x, BHMCTune(length(x)))
end


#################### Sampler Constructor ####################

function BHMC(params::Vector{Symbol}, traveltime::Real)
  samplerfx = function(model::Model, block::Integer)
    v = variate!(BHMCVariate, unlist(model, block),
                 model.samplers[block], model.iter)
    f = x -> logpdf!(model, x, block)
    bhmc!(v, traveltime, f)
    relist(model, v, block)
  end
  Sampler(params, samplerfx, BHMCTune())
end


#################### Sampling Functions ####################

function bhmc!(v::BHMCVariate, traveltime::Real, logf::Function)
  tune = v.tune
  flag = false
  nearzero = 1e4 * eps()
  j = 0
  totaltime = 0.0                     ## time the particle already moved

  d = length(v)                       ## length of binary vector
  S = sign(tune.position)

  while true
    a = tune.velocity[:]
    b = tune.position[:]
    phi = atan2(b, a)

    ## time to hit or cross wall
    walltime = -phi
    idx = find(x-> x > 0.0, phi)
    walltime[idx] = pi - phi[idx]

    ## if there was a previous reflection (j > 0) and there is a potential
    ## reflection at the sample plane make sure that a new reflection at j
    ## is not found because of numerical error
    if j > 0
      if abs(walltime[j]) < nearzero || abs(walltime[j] - 2.0 * pi) < nearzero
        walltime[j] = Inf
      end
    end

    ## time till particle j hits or crosses wall
    movetime, j = findmin(walltime)
    if movetime == 0.0
      error("walking length zero!")
    elseif movetime == Inf
      movetime = pi
    end

    totaltime += movetime
    if totaltime >= traveltime
      movetime -= totaltime - traveltime
      flag = true
    else
      tune.wallhits += 1
    end

    ## move the particle a time mt
    tune.velocity[:] = a * cos(movetime) - b * sin(movetime)
    tune.position[:] = a * sin(movetime) + b * cos(movetime)

    flag && break

    tune.position[j] = 0

    S1 = (S + ones(d)) / 2.0
    S1[j] = 0.0
    S2 = (S + ones(d)) / 2.0
    S2[j] = 1.0

    v2_new = tune.velocity[j]^2 +
             sign(tune.velocity[j]) * 2.0 * (logf(S2) - logf(S1))
    if v2_new > 0.0
      tune.velocity[j] = sqrt(v2_new) * sign(tune.velocity[j])
      S[j] *= -1.0
      tune.wallcrosses += 1
    else
      tune.velocity[j] *= -1.0
    end
  end

  ## convert from (-/+1) to (0/1)
  v[:] = (sign(tune.position) + ones(d)) / 2.0
  v
end
