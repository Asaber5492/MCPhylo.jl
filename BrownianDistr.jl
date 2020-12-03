mutable struct BrownianDistr <: DiscreteMatrixDistribution
    μ_Arr::Array{Float64,2}
    Σ_L_Arr::Array{Float64,3}
    latent::Array{Float64,2}
    dim::Tuple{Int64,Int64}
    covmats::Array{Float64,3}

    function BrownianDistr(μ_Arr::Array{Float64,2}, Σ_L_Arr::Array{Float64,3}, latent::Array{Float64,2})
        new(μ_Arr, Σ_L_Arr, latent, (size(Σ_L_Arr,1), size(Σ_L_Arr,3)), ones(3,3,3))
    end

    function BrownianDistr(μ::Array{Float64,1}, σ::Array{Float64,1}, Σ::Array{Float64}, latent::Array{Float64,2})
        n_concs = size(μ,1)
        n_leaves = size(Σ,1)

        Σ_L_Arr = Array{Float64,3}(undef, n_leaves, n_leaves, n_concs)
        μ_Arr::Array{Float64,2} = reshape(repeat(μ, outer=n_leaves), n_concs, n_leaves)
        @inbounds @simd for i in 1:n_concs
            covmats[:, :, i] .= σ[i] .* Σ
            Σ_L_Arr[:, :, i] .= cholesky(covmats[:, :, i]).L
        end
        new(μ_Arr, Σ_L_Arr, latent, (size(Σ,1), n_concs), covmats)
    end
end

function BrownianDistr(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate, latent::ArrayVariate)
    BrownianDistr(μ.value, σ.value, Σ.value, latent.value)
end

function BrownianDistr(μ_Arr::ArrayVariate, Σ_Arr::ArrayVariate, latent::ArrayVariate)
    BrownianDistr(μ_Arr.value, Σ_Arr.value, latent.value)
end

minimum(d::BrownianDistr) = -Inf
maximum(d::BrownianDistr) = Inf

Base.size(d::BrownianDistr) = d.dim

sampler(d::BrownianDistr) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianDistr, x::Array{Float64,2})
    langs, concs = size(x)
    res = 0
    println(size(d.Σ_L_Arr))
    for i in 1:concs
        res += sum(logpdf.(Bernoulli.(invlogit.(d.latent[:, i])), x[:,i]))
        mv_c0 = -(langs*Distributions.log2π+2*sum(log.(diag(d.Σ_L_Arr[:, :, i]))))/2


        invq = sum(abs2, d.Σ_L_Arr[:, :, i] \ Vector(x[:, i]))/2
        res += (mv_c0 -invq)
    end
    return res
end


function _rand!(r::A, d::BrownianDistr, x::AbstractMatrix) where A <: AbstractRNG
    n_leaves, n_concs = d.dim
    @inbounds @simd for i in 1:n_concs
        x[:,i] .= Int.(rand.(Bernoulli.(invlogit.(d.μ_Arr[i,:] + BLAS.gemv('N',1.0,d.Σ_L_Arr[:, :, i], d.latent[i, :])))))
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
