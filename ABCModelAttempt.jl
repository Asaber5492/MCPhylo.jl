using Revise
using MCPhylo
using Distributions

mutable struct ContactDist{T<:GeneralNode, D<:Distribution, F<:Distribution}
    tree::T
    opening_params::Array{Float64}
    opening_dist::D

    strength_params::Array{Float64}
    strength_dist::F
    accessible::Dict
    alphabetsize::Int64
end # mutable struct


mutable struct ReticluationDist
    tree
    opening_prob
end # mutable struct




function generate(d::ContactDist)
    lo = level_order(d.tree)
    for node in lo
        if !node.root
            # if true open a reticulation
            # d.opening_params[node.num,:]... to account for distributions with
            # multiple paramteres
            if rand(d.opening_dist(d.opening_params[node.num,:]...))
                target = rand(d.accessible[node.num])
                strength = d.strength_dist(d.strength_params[node.num, :]...)
                transfer_inds = sample(1:d.alphabetsize, strength, replace=false, ordered=true)
                target.data[transfer_inds] .= node.data[transfer_inds]
            end
        end
    end
end



end
