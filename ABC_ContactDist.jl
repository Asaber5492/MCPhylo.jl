include("./src/MCPhylo.jl")
using .MCPhylo
using Random
import Random: GLOBAL_RNG, rand!

Random.seed!(42)

using Distributions
import Distributions: cdf, dim, gradlogpdf, insupport, isprobvec, logpdf, logpdf!,
                      maximum, minimum, pdf, quantile, rand, sample!, support, length, _rand!, DiscreteMatrixDistribution

"""

* Backbone Tree with time depth
*
Reticulation_Matrix:
    size:
        nlang x nlang
    Interpretation:
        Reticulation_Matrix[2, 5] == 1 -> Reticulation from lang 2 to lang 5
        Reticulation_Matrix[5, 2] == 1 -> Reticulation from lang 5 to lang 2
        Reticulation_Matrix[2, 6] == 0 -> No Reticulation from lang 2 to lang 6

swadesh_list ~ ContactDist(Reticulation_Matrix, ρ, τ, β)
Reticulation_Matrix ~ ReticulationDist(tree, n_ret)
ρ ~
τ ~
β ~
"""

"""
Reticulation How To
po = post_order(tree)

for l in po
    lifespan_l = height(m(l)) - height(l)
end
For 2 languages if lifespan overlaps: Reticulation Possible
"""




"""
    Random Process of Contact Dist

Ideal Process (also for REAL DATA)
1) Follow tree in level_order
    2) Evolve Words acording to branchlength between mother node and current node
    3) Check if reticulations exist with current node as origin
        yes: transfer words according to probabilities
        no: continue
4) return to 1)

For Real Data:
    - Dated Language Tree -> Igor, Gerhard, Notes Marisa
"""
mutable struct ContactDist{T<:GeneralNode} <: DiscreteMatrixDistribution
    tree::T
    # tmax::Int64 --> Only relevant for tree generation not for contact
    ρ::Float64 # New word is born threshold, constant replacement rate (only relevant if evolution==true)
    # δ::Float64 # extinction rate of languages --> Only relevant for tree generation not for contact
    # σ::Float64 # Splitting Threshold, split rate --> Only relevant for tree generation not for contact
    # α::Float64 # probability of opening a channel --> Reticulation Dist --> Only relevant if n_ret is inferred
    # ω::Float64 # contact breakoff probability --> Reticulation Dist --> Only relevant if n_ret is inferred
    β::Float64 # relation of channel strength and transfer rate
    τ::Float64 # channel strength
    #gs::Int64  # gridsize
    reticulation_mat::Array{Float64,2} # Matrix indicating if reticulations are present
end # mutable struct

minimum(d::ContactDist) = -Inf
maximum(d::ContactDist) = Inf

Base.size(d::ContactDist) = d.dim

sampler(d::ContactDist) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::ContactDist, x::Array{Float64,2})
    return 0
end

"""
    _rand!(r::A, d::ContactDist, x::AbstractMatrix) where A <: AbstractRNG

r:: Abstract Random number generator
d:: Instance of contact dist with parameters
x:: pre allocated output (has correct size) needs to be filled

* Reticulations ≠ Entlehnung
* Reticulations = Contact Events
* Layerübergreifende Reticulations -> Currently implemented
* Intensität ≠ Hohe Kontaktdauer
* Hohe Anzahl von Entlehnungen entweder durch langen Kontakt
    oder hohe Intensität (oder beides)
* β*τ_dict[t_key]

Layerübergreifend: Eine Reticulation Matrix, die sagt ob Kontakt oder nicht, solange beide lebendig sind
Nicht Layerübergreifend: Pro Layer eine Reticulation Matrix, die Kontakt nur für dieses Layer beschreibt

IMPORTANT: Tree height flip!!!
"""
function _rand!(r::A, d::ContactDist, x::AbstractMatrix) where A <: AbstractRNG
    # Here Comes our function
    nsteps = 500 # number of layers

    height = tree_height(d.tree)

    po = post_order(d.tree)
    sort!(po, by=x->x.num)
    height_list = node_height.(po) # ordered by node num

    n_langs, n_words = size(x)
    upper = lower = 0.0
    for i in 1:nsteps
        upper = i/nsteps*height # time range of current layer
        # find living languages
        living_inds = lower .< height_list .< upper
        # indices of all the langauges alive in current layer
        living_languages = findall(x-> x==true, living_inds)
        for alive in living_languages
            # evolve words
            for c_ind in 1:n_words       # [r, r, a, b]    # [r, b, a, b]
                if rand() < d.ρ          # [a, r, a, b]    # [a, c, a, b]
                    x[alive, c_ind] = alive
                end
            end
        end

        for alive in living_languages
            if any(0 .!= d.reticulation_mat[alive, :])
                recipient_langs = findall(x-> x != 0.0, d.reticulation_mat[alive, :])
                for recipient in recipient_langs
                    if recipient in living_languages
                        for c_ind in 1:n_words
                            if rand() < β*d.reitculation_mat[alive, recipient]
                                x[recipient, c_ind] = x[alive,c_ind]
                            end # if rand() < β*d.reitculation_mat[alive, recipient]
                        end # for c_ind in 1:n_words
                    end # for recipient in living_languages
                end # if recipient in recipient_langs
            end # if any()
        end
        lower = upper # move layer
    end



    for (ind, node) in enumerate(lo)
        # check if reticulation is present
        if any(1 .== d.reticulation_mat[node.num, :])
            # do something here
        else
            for i in 1:Int(node.inc_length)
                for c_ind in 1:n_words
                    if rand() < d.ρ
                        x[node.mother.num, c_ind] = node.num
                    end
                end
            end



end
