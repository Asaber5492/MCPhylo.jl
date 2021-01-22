using Revise
using MCPhylo
using Distributions


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
end # mutable struct
