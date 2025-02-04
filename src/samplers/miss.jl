#################### Missing Values Sampler ####################

#################### Types and Constructors ####################

struct MISSTune
  dims::Tuple
  valueinds::Array
  distrinds::Array
end

function MISSTune(s::AbstractStochastic)
  MISSTune(s.distr, s.value)
end

function MISSTune(d::Distribution, v)
  MISSTune((), findall(isnan.(v)), Int[])
end

function MISSTune(D::Array{UnivariateDistribution}, v::Array)
  inds = findall(isnan.(v))
  MISSTune(dims(D), inds, inds)
end

function MISSTune(D::Array{MultivariateDistribution}, v::Array)
  isvalueinds = falses(size(v))
  isdistrinds = falses(size(D))
  for sub in CartesianIndices(size(D))
    n = length(D[sub])
    for i in 1:n
      if isnan(v[sub, i])
        isvalueinds[sub, i] = isdistrinds[sub] = true
      end
    end
  end
  MISSTune(dims(D), findall(isvalueinds), findall(isdistrinds))
end


#################### Sampler Constructor ####################
"""
    MISS(params::ElementOrVector{Symbol})

Construct a `Sampler` object to sampling missing output values. The constructor
should only be used to sample stochastic nodes upon which no other stochastic
node depends. So-called ‘output nodes’ can be identified with the `keys()`
function. Moreover, when the `MISS` constructor is included in a vector of
`Sampler` objects to define a sampling scheme, it should be positioned at the
beginning of the vector. This ensures that missing output values are updated
before any other samplers are executed.

Returns a `Sampler{Dict{Symbol, MISSTune}}` type object.

* `params`: stochastic node(s) that contain missing values (`NaN`) to be updated with the sampler.
"""
function MISS(params::ElementOrVector{Symbol})
  params = asvec(params)
  samplerfx = function(model::Model, block::Integer)
    tune = gettune(model, block)
    if model.iter == 1
      for key in params
        miss = MISSTune(model[key])
        if !isempty(miss.valueinds)
          tune[key] = miss
        end
      end
      params = intersect(keys(model, :dependent), keys(tune))
    end
    for key in params
      node = model[key]
      miss = tune[key]
      node[miss.valueinds] = rand(node, miss)
      update!(model, node.targets)
    end
    nothing
  end
  Sampler(params, samplerfx, Dict{Symbol, MISSTune}())
end


#################### Sampling Functions ####################

rand(s::AbstractStochastic, miss::MISSTune) = rand_sub(s.distr, miss)

function rand_sub(d::Distribution, miss::MISSTune)
  x = rand(d)
  Float64[x[i] for i in miss.valueinds]
end

function rand_sub(D::Array{UnivariateDistribution}, miss::MISSTune)
  Float64[rand(d) for d in D[miss.distrinds]]
end

function rand_sub(D::Array{MultivariateDistribution}, miss::MISSTune)
  X = Array{Float64}(undef, miss.dims)
  for i in miss.distrinds
    d = D[i]
    X[ind2sub(D, i)..., 1:length(d)] = rand(d)
  end
  X[miss.valueinds]
end
