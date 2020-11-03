mutable struct BrownianDistr <: DiscreteMatrixDistribution
    μ::Array{Float64,1}
    σ::Array{Float64,1}
    Σ::Array{Float64,2}
    dim::Tuple{Int64,Int64}

    function BrownianDistr(μ::Array{Float64,1}, σ::Array{Float64,1}, Σ::Array{Float64})
        new(μ, σ, Σ, (size(Σ,1), size(μ,1)))
    end
end

function BrownianDistr(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate)
    BrownianDistr(μ.value, σ.value, Σ.value)
end

minimum(d::BrownianDistr) = -Inf
maximum(d::BrownianDistr) = Inf

Base.size(d::BrownianDistr) = d.dim

sampler(d::BrownianDistr) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianDistr, x::Array{Float64,2})
    return 0
end

function _rand!(r::A, d::BrownianDistr, x::AbstractMatrix) where A <: AbstractRNG
    n_leaves, n_concs = d.dim
    μ_arr::Array{Float64,2} = reshape(repeat(d.μ, outer=n_leaves), n_concs, n_leaves)
    @inbounds for i in 1:n_concs
        x[:,i] .= Int.(invlogit.(rand(MvNormal(μ_arr[i,:], d.σ[i].*d.Σ))) .> 0.5)
    end
    return x
end

mutable struct BrownianDistrMulti <: DiscreteMatrixDistribution
    μ::Array{Float64,1}
    σ::Array{Float64,1}
    Σ::Array{Float64,2}
    dim::Tuple{Int64,Int64}

    function BrownianDistrMulti(μ::Array{Float64,1}, σ::Array{Float64,1}, Σ::Array{Float64})
        new(μ, σ, Σ, (size(Σ,1), size(μ,1)))
    end
end

function BrownianDistrMulti(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate)
    BrownianDistrMulti(μ.value, σ.value, Σ.value)
end

minimum(d::BrownianDistrMulti) = -Inf
maximum(d::BrownianDistrMulti) = Inf

Base.size(d::BrownianDistrMulti) = d.dim

sampler(d::BrownianDistrMulti) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianDistrMulti, x::Array{Float64,2})
    return 0
end

function _rand!(r::A, d::BrownianDistrMulti, x::AbstractMatrix) where A <: AbstractRNG
    n_leaves, n_concs = d.dim
    μ_arr::Array{Float64,2} = reshape(repeat(d.μ, outer=n_leaves), n_concs, n_leaves)
    @inbounds for i in 1:n_concs
        x[:,i] .= Int.(invlogit.(rand(MvNormal(μ_arr[i,:], d.σ[i].*d.Σ))) .> 0.5)
    end
    return x
end
