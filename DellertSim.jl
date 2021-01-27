# get necessary Packages
#include("./src/MCPhylo.jl")
#using .MCPhylo # use this for tree structures
using Random
Random.seed!(42)
using LinearAlgebra
include("./src/MCPhylo.jl")
using .MCPhylo
using StatsBase
"""
Nice 2 Know

Create empty array with 2 dimensions
Array{Int64, 2}(undef, 4,1)

-> Broadcasting
x = rand(4,4)
x .= -1
"""

# Problems:
# Reticulations are not supported in the tree structures: -> Separate data structure

"""

Simulate Network Function from Dellert(2017) p.116

    k: initial number of languages
    tmax: maximum time
    n: number of concepts
    ρ: New word is born threshold, constant replacement rate
    δ: extinction rate
    σ: Splitting Threshold, split rate
    α: probability of opening a channel
    ω: contact breakoff probability
    β: relation of channel strength and transfer rate
    τ: channel strength
    gs: gridsize
"""
function SimulateNetwork(k::Int64, tmax::Int64, n::Int64, ρ::Float64,
                         δ::Float64, σ::Float64, α::Float64, ω::Float64,
                         β::Float64, τ::Float64, gs::Int64)
    """
    1. Initialize square grid and randomly place languages in the grid
    2. Initialize Language families with new words
    3. Start simulation according to Pseudocode


    ToDo: - Include Timedepth in trees
          - Keep Track of Loanwords in a Languages Lexicon --> Disable replacement of loanwords
    """
    # check that the grid is big enough to hold proto languages
    @assert gs^2 > k

    # initialize Grid, where languages are placed
    # uninhabited cells are identified by -1
    Landscape = Array{Int64, 2}(undef, gs, gs)
    Landscape .= -1
    # initialize Lexicon
    Lexica = Dict{Int64, Vector{Int64}}()
    # initialize Phylogenies
    Phylogenies = [Node(string(i)) for i in 1:k]
    indexer = Dict{Int64, Int64}()
    for i in 1:k
        indexer[i] = i
    end
    # Store reticulations in dict
    Reticulation_dict = Dict{Int64, Set{Int64}}()
    # initialize τ_dict
    τ_dict = Dict{Tuple{Int64, Int64}, Float64}()
    for i in 1:k, j in 1:k
        τ_dict[(i, j)] = 0.0
    end
    # keep track of living languages
    Living = Array{Bool,1}(undef, k)
    Living .= true

    # place languages in grid and fill lexica
    for l_ind in 1:k
        while true
            x = rand(1:gs)
            y = rand(1:gs)
            if Landscape[x, y] == -1
                Landscape[x, y] = l_ind
                break
            end
        end
        # a word: 1_adsu8
        # concept identifier not necessary, since no semantic shift present
        Lexica[l_ind] = zeros(Int64, n) .= l_ind#[string(l_ind)*"_"*randstring(5) for i in 1:n]
    end

    # start simulating history
    t = 1
    while t < tmax
        # 1. create new children
        curr_k = k
        for l_ind in 1:curr_k
            # language branches of
            if Living[l_ind] # equivalent to Living[l_ind] == true
                # get correct Phylogeny
                phylo_ind = indexer[l_ind]
                tree = Phylogenies[phylo_ind]
                mother_node = find_by_name(tree, string(l_ind))
                if rand() < σ
                    L1 = deepcopy(Lexica[l_ind])
                    L2 = deepcopy(Lexica[l_ind])
                    Living[l_ind] = false
                    push!(Living, true)
                    push!(Living, true)
                    k += 1
                    Lexica[k] = L1

                    #add new language to tree
                    l1_node = Node(string(k))
                    l1_node.inc_length = 1.0
                    add_child!(mother_node, l1_node)
                    indexer[k] = phylo_ind
                    # find position of mother language
                    pos_L = findfirst(x-> x==l_ind, Landscape)
                    Landscape[pos_L] = k
                    k += 1
                    Lexica[k] = L2
                    # add new language to tree
                    indexer[k] = phylo_ind
                    l2_node = Node(string(k))
                    l2_node.inc_length = 1.0
                    add_child!(mother_node, l2_node)
                    pos_L2 = find_new_home(Landscape, pos_L)
                    if Landscape[pos_L2] != -1
                        # wipe out language living in new home
                        Living[Landscape[pos_L2]] = false
                    end
                    Landscape[pos_L2] = k
                else
                    mother_node.inc_length += 1
                end # end if rand() < σ
                Phylogenies[phylo_ind] = tree
            end # end if Living[l_ind]
        end # end for l_ind in 1:k

        # 2. Simulate evolution of words
        """
        Fallunterscheidung in den Experimenten machen:
            1. Fall: keine Evolution of Loanwords
                Begründung: Wenn ein Wort entlehnt wurde gab es eine Lücke. Das
                            entlehnte Wort durch ein neues eigenes Wort zu ersetzen
                            ist unökonomisch. Sprachinterne Evolution ist für die
                            Summarystatisitcs und unsere Fragestellung nicht relevant.
            2. Fall: Evolution of Loanwords
                Begründung: Es besteht die Möglichkeit, das Sprachkontakt verwässert
                            über die Zeit und ein intensiver Kontakt nicht mehr so
                            sichtbar ist, da entlehnte Wörter durch Sprachwandelprozesse
                            ersetzt wurden, z.B. Semantischer Wandel, Register, Frequenz.
                            Fragestellung: Kann verwässerter Kontakt noch gefunden werden?
        Need vs. Prestige
        Beispiele:
            Geldbörse vs. Portmonaie
            Gehweg vs. Trottoir
        Followup diskussion: Kommen diese Beispiele in Swadeshlisten vor?
        Interaktion von Evolution und Kontaktintensität
        """

        for (l_ind, alive) in enumerate(Living)
            if alive
                for c_ind in 1:n
                    if rand() < ρ
                        Lexica[l_ind][c_ind] = l_ind#string(l_ind)*"_"*randstring(5)
                        """
                        1. Fall
                        Language in Question: 1
                        a) 2 is not an ancestral Language
                            CONTACT
                            1: [1, 1, 2, 3, 1]              [1, 1, 2, 3, 2]
                            ==> intraceable contact            less traceable
                               [1, 1, 1, 3, 1]              [1, 1, 1, 3, 2]
                       b) 2 is an ancestral Language
                           NO CONTACT
                           1: [1, 1, 2, 3, 1]
                                Replace ancestral Form
                              [1, 1, 1, 3, 1]

                        Irrelevant:
                        2. Fall
                        1: [1, 1, 2, 3, 1]
                        ==>
                           [1, 1', 2, 3, 1]
                        """

                    end
                end
            end
        end

        # 3. open channels and close channels
        for (l_ind_1, alive_1) in enumerate(Living), (l_ind_2, alive_2) in enumerate(Living)
            if alive_2 && alive_1
                t_key = (l_ind_1, l_ind_2)
                pos1 = findfirst(x-> x==l_ind_1, Landscape)
                pos2 = findfirst(x-> x==l_ind_2, Landscape)
                # scale euclidean_distance to get a distance between 0 and 1
                d = euclidean_distance(pos1[1]/gs, pos1[2]/gs, pos2[1]/gs, pos2[2]/gs)
                if !(t_key in keys(τ_dict))
                    τ_dict[t_key] = 0
                end
                if !(l_ind_1 in keys(Reticulation_dict))
                    Reticulation_dict[l_ind_1] = Set{Int64}()
                end
                if τ_dict[t_key] > 0 && rand() < ω*d
                    τ_dict[t_key] = 0
                elseif τ_dict[t_key] == 0 && rand() < α*d
                    τ_dict[t_key] = rand() * (1-d)
                end
                if τ_dict[t_key] > 0
                    for c_ind in 1:n
                        if rand() < β*τ_dict[t_key] # loan probability weighted by distance
                            push!(Reticulation_dict[l_ind_1], l_ind_2)
                            Lexica[l_ind_2][c_ind] = Lexica[l_ind_1][c_ind]
                        end
                    end
                end
            end
        end
        # go one step further in time
        t += 1
    end

    # Turn Lexica dictionary into Matrix
    Lexica_array = Array{Int64, 2}(undef, length(Lexica), n)
    for (lang, words) in Lexica
        Lexica_array[lang, :] .= words
    end


    return Landscape, Lexica_array, Phylogenies, Reticulation_dict
end


function euclidean_distance(x1::S, y1::S, x2::S, y2::S)::Float64 where S<:Real
    sqrt((x1-x2)^2+(y1-y2)^2)
end

function find_new_home(landscape, pos_l)
    border = size(landscape, 1)
    # new coordinates cannot be smaller than 1 or larger than border
    x, y = Tuple(pos_l)
    eligible_cells = CartesianIndex[]
    empty_cells = CartesianIndex[]
    for i in -1:1:1
        for j in -1:1:1
            if i != 0 || j != 0
                new_pos_x = x+i
                new_pos_y = y+j
                if 1 <= new_pos_x <= border && 1 <= new_pos_y <= border
                    my_index = CartesianIndex(new_pos_x, new_pos_y)
                    push!(eligible_cells, my_index)
                    if landscape[my_index] == -1
                        push!(empty_cells, my_index)
                    end
                end
            end
        end
    end
    if length(empty_cells) > 0
        rand(empty_cells)
    else
        rand(eligible_cells)
    end
end

landscape, lexika, phylo, Retdict = SimulateNetwork(1, 500, 100, 0.001, 0.5, 0.005, 0.5, 0.5, 0.5,0.5, 7)

"""
Marisa ToDo:

Write Function that counts origins of words of language
For each langauge return vector of length = #languages
at each language index store share of this langauages vocabulary
"""


function language_origin_statistics(lexika::Array{Int64, 2})::Array{Float64, 2}
    # get number of languages
    num_langs = size(lexika, 1)
    # get number of words
    num_words = size(lexika, 2)
    Sum_stats = zeros(num_langs, num_langs)
    @inbounds for recipient in 1:num_langs
        # count the occurance of the origin languages
        c = countmap(lexika[recipient, :])
        for (k,v) in c
            Sum_stats[recipient, k] =  v/num_words
        end
    end
    return Sum_stats
end
