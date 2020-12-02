# get necessary Packages
#include("./src/MCPhylo.jl")
#using .MCPhylo # use this for tree structures
using Random
Random.seed!(42)
using LinearAlgebra

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


    ToDo: Include Trees
    """
    # check that the grid is big enough to hold proto languages
    @assert gs^2 > k

    # initialize Grid, where languages are placed
    # uninhabited cells are identified by -1
    Landscape = Array{Int64, 2}(undef, gs, gs)
    Landscape .= -1
    # initialize Lexicon
    Lexica = Dict{Int64, Vector{String}}()
    # initialize Phylogenies
    Phylogenies = [Node(string(i)) for i in 1:k]
    indexer = Dict{Int64, Int64}()
    for i in 1:k
        indexer[i] = i
    end
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
        Lexica[l_ind] = [string(l_ind)*"_"*randstring(5) for i in 1:n]
    end

    # start simulating history
    t = 1
    while t < tmax
        # 1. create new children
        curr_k = k
        for l_ind in 1:curr_k
            # language branches of
            if Living[l_ind] # equivalent to Living[l_ind] == true
                if rand() < σ
                    L1 = deepcopy(Lexica[l_ind])
                    L2 = deepcopy(Lexica[l_ind])
                    Living[l_ind] = false
                    push!(Living, true)
                    push!(Living, true)
                    k += 1
                    Lexica[k] = L1
                    # get correct Phylogeny
                    phylo_ind = indexer[l_ind]
                    tree = Phylogenies[phylo_ind]
                    mother_node = find_by_name(tree, string(l_ind))
                    # add new language to tree
                    l1_node = Node(string(k))
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
                    add_child!(mother_node, l2_node)
                    pos_L2 = find_new_home(Landscape, pos_L)
                    if Landscape[pos_L2] != -1
                        # wipe out language living in new home
                        Living[Landscape[pos_L2]] = false
                    end
                    Landscape[pos_L2] = k
                end # end if rand() < σ
            end # end if Living[l_ind]
        end # end for l_ind in 1:k

        # 2. Simulate evolution of words
        for (l_ind, alive) in enumerate(Living)
            if alive
                for c_ind in 1:n
                    if rand() < ρ
                        Lexica[l_ind][c_ind] = string(l_ind)*"_"*randstring(5)
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
                d = euclidean_distance(pos1[1], pos1[2], pos2[1], pos2[2])
                if !(t_key in keys(τ_dict))
                    τ_dict[t_key] = 0
                end
                if τ_dict[t_key] > 0 && rand() < ω*d
                    τ_dict[t_key] = 0
                elseif τ_dict[t_key] == 0 && rand() < α*d
                    τ_dict[t_key] = rand() * (1-d)
                end
                if τ_dict[t_key] > 0
                    for c_ind in 1:n
                        if rand() < β*τ_dict[t_key]
                            Lexica[l_ind_2][c_ind] = Lexica[l_ind_1][c_ind]
                        end
                    end
                end
            end
        end
        # go one step further in time
        t += 1
    end
    return Landscape, Lexica, Phylogenies
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

landscape, lexika, phylos = SimulateNetwork(2, 500, 10, 0.001, 0.5, 0.005, 0.5, 0.5, 0.5,0.5, 7)
