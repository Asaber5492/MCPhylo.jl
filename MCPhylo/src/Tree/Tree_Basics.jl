#=
my_tree:
- Julia version: 1.0.1
- Author: erathorn
- Date: 2019-05-07
=#

#TODO: Automate export of automatically genereated funtions

function Base.summary(io::IO, d::Node)
    summary(io, d.name)
end

function Base.show(io::IO, d::Node)
    print(io, "Tree with root:\n")
    show(io, d.name)
    #print(io, "\n\nLength:\n")
    #show(io, "text/plain", tree_length(d))
    #print(io, "\n\nHeight:\n")
    #show(io, "text/plain", tree_height(d))
    #print(io, "\n\nNumber of leave nodes:\n")
    #show(io, "text/plain",length(get_leaves(d)))
end

function showall(io::IO, d::Node)
  show(io, d)
  print(io, "\nNode:\n")
  show(io, "text/plain", d.name)
  print(io, "\n\n#children:\n")
  show(io, d.nchild)
  print(io, "\n\nbinary:\n")
  show(io, d.binary)
end

Base.:(==)(x::T, y::T) where T<:Node = x.num == y.num


Base.size(x::T) where T<:Node = size(post_order(x))
Base.length(x::T) where T<:Node = x.nchild

Base.getindex(n::T, ind::Int) where T<:Node = ind === 1 ? n.lchild : (ind === 2 ? n.rchild : n.mchild)
Base.unsafe_getindex(n::T, ind::Int) where T<:Node = ind === 1 ? n.lchild : (ind === 2 ? n.rchild : n.mchild)

Base.firstindex(n::T) where T<:Node = 1
Base.lastindex(n::T) where T<:Node = n.nchild
Base.iterate(n::T) where T<:Node = n.lchild, [:rchild, :mchild]
Base.eltype(::Type{T}) where T<:Node = T
function Base.iterate(n::T, chlds) where T<:Node
   if isempty(chlds)
       nothing
   else
       rv = getfield(n, chlds[1])
       !ismissing(rv) ? (rv , chlds[2:end]) : nothing
   end
end
"""
    add_child(mother_node::Node, child::Node)

This function adds a child to the mother node.
The arity of the mother node is increased by `1` and the root
status of the child is set to `False`.
"""
function add_child!(mother_node::Node, child::Node, left::Bool, middle::Bool=false)
    add_child!(mother_node, child)
end # function add_child

"""
    add_child(mother_node::Node, child::Node)

This function adds a child to the mother node.
The arity of the mother node is increased by `1` and the root
status of the child is set to `False`.
"""
function add_child!(mother_node::Node, child::Node)
    push!(mother_node.children, child)
    child.mother = mother_node
    mother_node.nchild += 1
    child.root = false
end # function add_child


"""
    remove_child!(mother_node::Node, index::int)Node

This function removes a child from the list of nodes which are daughters of this
node. The removed node is returned.
"""
function remove_child!(mother_node::Node, left::Bool)::Node
    @assert (mother_node.nchild === 2)
    if left
        rv = popfirst!(mother_node.children)
        rv.mother = missing
    else
        rv = pop!(mother_node.children)
        rv.mother = missing
    end # end if
    mother_node.nchild -= 1
    return rv
end # function

"""
    remove_child!(mother_node::Node, index::int)Node

This function removes a child from the list of nodes which are daughters of this
node. The removed node is returned.
"""
function remove_child!(mother_node::Node, child::Node)::Node
    ind = findfirst(x->x==child, mother_node.children)
    deleteat!(mother_node.children, ind)
    child.mother = missing

    mother_node.nchild -= 1
    return child
end # function

function to_covariance(tree::TreeStochastic, blv::Array)
    to_covariance(tree.value, blv)
end



function to_covariance_ultra(tree::Node) where T<: Real
    mv = tree_height(tree)
    blv = get_branchlength_vector(tree)
    blv ./= mv
    root = deepcopy(tree)
    set_branchlength_vector!(root, blv)
    force_ultrametric(root)

    leaves = get_leaves(root)
    ll = length(leaves)
    covmat = zeros(ll, ll)
    covmatsym = Array{Function,2}(undef, ll, ll)
    covmatsym .= x -> 0.0
    for i in 1:ll
        node1 = leaves[i]
        for j in 1:ll
            if i == j
                symbolicpath = get_path(root, node1)
                path = path_length(root, node1)#get_path(root, node1)
                covmat[i,j] = path#reduce(+, @view blv[path])
                covmatsym[i,j] = x -> reduce(+,x[symbolicpath])
            elseif i>j
                lca = find_lca(root, node1, leaves[j])
                if lca.root != true
                    path = path_length(root, lca)
                    symbolicpath = get_path(root, lca)
                    covmat[i,j] = path
                    covmat[j,i] = path
                    covmatsym[i,j] = x -> reduce(+,x[symbolicpath])
                    covmatsym[j,i] = x -> reduce(+,x[symbolicpath])

                end
            end
        end

    end
    covmat, covmatsym
end



function to_covariance(tree::N, blv::Array{T}) where {N<:Node,T<: Real}
    #rescale_length(tree)
    leaves = get_leaves(tree)
    ll = length(leaves)
    covmat = zeros(T, ll, ll)
    @inbounds @simd for i in 1:ll
        @inbounds for j in 1:ll
            if i >= j
                node1 = leaves[i]
                if i == j
                    #path = get_path(tree, node1)
                    covmat[i,j] = reduce(+, @view blv[get_path(tree, node1)])
                else
                    lca = find_lca(tree, node1, leaves[j])
                    if lca.root != true
                        tmp = reduce(+, @view blv[get_path(tree, lca)])
                        covmat[i,j] = tmp
                        covmat[j,i] = tmp

                    end
                end
            end
        end

    end
    covmat
end


function to_covariance(tree::TreeStochastic)
    blv = get_branchlength_vector(tree)
    to_covariance(tree.value, blv)
end

function to_covariance(tree::Node)
    blv = get_branchlength_vector(tree)
    to_covariance(tree, blv)
end

#function to_covariance(tree::Node)
#    leaves = get_leaves(tree)
#    ll = length(leaves)
#    covmat = zeros(Float64, ll, ll)
#    for i in 1:ll
#        node1 = leaves[i]
#        for j in 1:ll
#            if i == j
#                covmat[i,j] = node_distance(tree, tree, node1, tree)
#            elseif i>j
#                lca = find_lca(tree, node1, leaves[j])
#                if lca.root != true
#                    d = node_distance(tree, tree, lca, tree)
#                    covmat[i,j] = d
#                    covmat[j,i] = d
#                end
#            end

        #end
    #end
    #covmat
#end

function to_distance_matrix(tree::Node)
    leaves = get_leaves(tree)
    ll = length(leaves)
    covmat = zeros(Float64, ll, ll)
    for i in 1:ll
        for j in 1:ll
            if i>j
                d = node_distance(leaves[i], leaves[j])
                distance_mat[i,j] = d
                distance_mat[j,i] = d
            end
        end
    end
    distance_mat
end



"""
    create_tree_from_leaves(leaf_nodes::Vector{T})::Node

This function creates a  random binary tree from a list of leaf nodes.
The root node as access point for the tree is returned.
"""
function create_tree_from_leaves_bin(leaf_nodes::Vector{String}, node_size::Int64 = 1; cu::Bool=false)::Node
    my_node_list::Array{Node,1} = []

    # first create a list of leaf nodes
    for node_name in leaf_nodes
        nn =  Node_ncu(node_name, zeros(Float64, (2, node_size)),missing, missing,missing, missing, 0, true, 0.0, "0", 0, 0.0)
        push!(my_node_list,nn)
    end # for

    # Internal nodes are created using integers as names.
    temp_name::Int = length(my_node_list)+1

    # shuffle the node list to get a random tree
    Random.shuffle!(my_node_list)

    while length(my_node_list) > 2
        # get two nodes
        # create a new mother node to which the two first nodes are added as children
        # add the new mother node to the list and reshuffle
        first_child::Node = pop!(my_node_list)
        first_child.inc_length = rand(Uniform(0.0015,1))#*0.1
        second_child::Node = pop!(my_node_list)
        second_child.inc_length = rand(Uniform(0.0015,1))
        curr_node::Node = Node_ncu(string(temp_name), zeros(Float64, (2, node_size)), missing, missing, missing, missing, 0, true, 0.0, "0", 0,0.0)
        add_child!(curr_node, first_child, true)
        add_child!(curr_node, second_child, false)
        push!(my_node_list, curr_node)
        temp_name += 1
        Random.shuffle!(my_node_list)
    end # while
    root::Node = Node_ncu(string(temp_name), zeros(Float64, (2, node_size)), missing, missing, missing, missing, 0, true, 0.0, "0", 0,0.0)
    lchild = pop!(my_node_list)
    lchild.inc_length = rand(Uniform(0.0015,1))
    #mchild = pop!(my_node_list)
    #mchild.inc_length = rand(Uniform(0.0015,1))
    rchild = pop!(my_node_list)
    rchild.inc_length = rand(Uniform(0.0015,1))
    add_child!(root, lchild, true)
    add_child!(root, rchild, false)
    #add_child!(root, mchild, false, true)

    set_binary!(root)
    number_nodes!(root)

    return root
end # function create_tree_from_leaves



"""
    create_tree_from_leaves(leaf_nodes::Vector{T})::Node

This function creates a  random binary tree from a list of leaf nodes.
The root node as access point for the tree is returned.
"""
function create_tree_from_leaves(leaf_nodes::Vector{String}, node_size::Int64 = 1; cu::Bool=false)::Node
    my_node_list::Array{Node,1} = []

    # first create a list of leaf nodes
    for node_name in leaf_nodes
        nn =  Node_ncu(node_name, zeros(Float64, (2, node_size)),missing, missing,missing, missing, 0, true, 0.0, "0", 0, 0.0)
        push!(my_node_list,nn)
    end # for

    # Internal nodes are created using integers as names.
    temp_name::Int = length(my_node_list)+1

    # shuffle the node list to get a random tree
    Random.shuffle!(my_node_list)

    while length(my_node_list) > 3
        # get two nodes
        # create a new mother node to which the two first nodes are added as children
        # add the new mother node to the list and reshuffle
        first_child::Node = pop!(my_node_list)
        first_child.inc_length = rand(Uniform(0.0015,1))#*0.1
        second_child::Node = pop!(my_node_list)
        second_child.inc_length = rand(Uniform(0.0015,1))
        curr_node::Node = Node_ncu(string(temp_name), zeros(Float64, (2, node_size)), missing, missing, missing, missing, 0, true, 0.0, "0", 0,0.0)
        add_child!(curr_node, first_child, true)
        add_child!(curr_node, second_child, false)
        push!(my_node_list, curr_node)
        temp_name += 1
        Random.shuffle!(my_node_list)
    end # while
    root::Node = Node_ncu(string(temp_name), zeros(Float64, (2, node_size)), missing, missing, missing, missing, 0, true, 0.0, "0", 0,0.0)
    lchild = pop!(my_node_list)
    lchild.inc_length = rand(Uniform(0.0015,1))
    mchild = pop!(my_node_list)
    mchild.inc_length = rand(Uniform(0.0015,1))
    rchild = pop!(my_node_list)
    rchild.inc_length = rand(Uniform(0.0015,1))
    add_child!(root, lchild, true)
    add_child!(root, rchild, false)
    add_child!(root, mchild, false, true)

    set_binary!(root)
    number_nodes!(root)

    return root
end # function create_tree_from_leaves


function create_tree_from_leaves_cu(leaf_nodes::Vector{String}, node_size::Int64 = 1)::Node_cu
    my_node_list::Array{Node_cu,1} = []

    # first create a list of leaf nodes
    for node_name in leaf_nodes
        nn =  Node_cu(node_name, zeros(Float64, (2, node_size)),missing, missing, missing, 0, true, 0.0, "0", 0, 0.0)
        push!(my_node_list,nn)
    end # for

    # Internal nodes are created using integers as names.
    temp_name::Int = length(my_node_list)+1

    # shuffle the node list to get a random tree
    Random.shuffle!(my_node_list)

    while length(my_node_list) != 1
        # get two nodes
        # create a new mother node to which the two first nodes are added as children
        # add the new mother node to the list and reshuffle
        first_child::Node = pop!(my_node_list)
        first_child.inc_length = rand(Uniform(0,1))
        second_child::Node = pop!(my_node_list)
        second_child.inc_length = rand(Uniform(0,1))
        curr_node::Node = Node_cu(string(temp_name), zeros(Float64, (2, node_size)), missing, missing, missing, 0, true, 0.0, "0", 0,0.0)
        add_child!(curr_node, first_child, true)
        add_child!(curr_node, second_child, false)
        push!(my_node_list, curr_node)
        temp_name += 1
        Random.shuffle!(my_node_list)
    end # while
    root = pop!(my_node_list)
    set_binary!(root)
    number_nodes!(root)

    return root
end # function create_tree_from_leaves



function level_order(node::T)::Array{T} where T<: Node
    level = 1
    stack::Array{T} = []
    while level_traverse(node, level, stack)
        level += 1
    end
    stack
end

function level_traverse(node::T, level::Int64, stack::Array{T})::Bool where T <: Node
    if level == 1
        push!(stack, node)
        return true
    else
        boolqueue = [false]
        for child in node.children
            push!(boolqueue, level_traverse(child, level-1, stack))
        end
        return reduce(|, boolqueue)
    end
end

function force_ultrametric(root::T) where T<: Node
    #node2max_depth = Dict{Int64, Int64}()
    po = post_order(root)
    node2max_depth = zeros(length(po))
    #blv = get_branchlength_vector(root)

    for node in po
        if node.nchild != 0
            mv = -1
            for child in node.children
                if node2max_depth[child.num] > mv
                    mv = node2max_depth[child.num]
                end
            end
            node2max_depth[node.num] = mv+1
        else
            node2max_depth[node.num] = 1
        end
    end
    #node2dist = Dict(root.num => 0.0)
    node2dist = zeros(size(node2max_depth))
    tl = tree_height(root)
    nblv = zeros(length(po))
    for node in level_order(root)
        if node.root != true
            nv = (tl - node2dist[node.mother.num])/node2max_depth[node.num]

            nblv[node.num] = nv
            node2dist[node.num] = nv + node2dist[node.mother.num]
        end
    end


    set_branchlength_vector!(root, nblv)
end


"""
    post_order(root::Node, traversal::Vector{Node})::Vector{Node}

This function performs a post order traversal through the tree. It is assumed that `root` is the
root of the tree. Thus, if `root` is not the root, the subtree defined by the root `root` is
used for the post order traversal.
"""
function post_order(root::T, traversal::Vector{T})::Vector{T} where T<:Node
   if root.nchild != 0
        for child in root.children
            post_order(child, traversal)
        end
   end # if
   push!(traversal, root)
   return traversal
end # function post_order_trav


"""
    post_order(root:Node)::Vector{Node}

This function does post order traversal. It is meant as a wrapper. Only the root
node needs to be supplied.
"""
function post_order(root::T)::Vector{T} where T<:Node
    t::Vector{T} = []
    post_order(root, t)
    return t
end # function post_order


"""
    pre_order(root::Node, traversal::Vector{Node})::Vector{Node}

This function performs a pre order traversal through the tree. It is assumed that `root` is the
root of the tree. Thus, if `root` is not the root, the subtree defined by the root `root` is
used for the pre order traversal.
"""
function pre_order(root::T, traversal::Vector{T})::Vector{T} where T<:Node
    push!(traversal, root)
    if root.nchild != 0
        for child in root.children
            pre_order(child, traversal)
        end
    end # if
    return traversal
end # function pre_order!


"""
    pre_order(root:Node)::Vector{Node}

This function does pre order traversal. It is meant as a wrapper. Only the root
node needs to be supplied.
"""
function pre_order(root::T)::Vector{T} where T<:Node
    t::Vector{T} = []
    pre_order(root, t)
    return t
end # function pre_order!


function newick(root::T)  where T<:Node
    newickstring = newick(root, "")
    newickstring = chop(newickstring)
    newickstring = string(newickstring, ";")
    return newickstring
end

function newick(root::T, newickstring::AbstractString) where T<:Node
    if root.nchild != 0
        newickstring = string(newickstring, "(")
        for child in root.children
            newickstring = string(newick(child,newickstring))
        end
        newickstring = chop(newickstring)
        return string(newickstring,")", root.name, ":", root.inc_length,",")

    else
        return string(newickstring, root.name, ":", root.inc_length, ",")
    end
end


"""
    tree_length(root::Node)::Float64

This function calculates the tree_length.
"""
function tree_length(root::T)::Float64  where T<:Node
    return tree_length(root, 0.0)
end # function tree_length

function tree_length(root::T, tl::Float64)::Float64 where T<:Node

    if length(root.children) != 0
        for child in root.children
            tl = tree_length(child, tl)
        end
    end # if
    if root.root !== true
        tl += root.inc_length
    end
    tl
end



function node_height(root::T, mv::Float64)  where T<:Node
    if !root.root
        if isdefined(root, :mother)
            rmh = root.mother.height
        else
            rmh = -Inf
        end
        root.height = rmh+root.inc_length
    end
    if root.nchild != 0
        for child in root.children
            mv = node_height(child, mv)
        end
    else
        if root.height>mv
            mv = root.height
        end
    end # if
    return mv
end

function rescale_length(root::T) where T<:Node
    #mv = tree_height(root)
    #blv = get_branchlength_vector(root)
    #blv ./= mv
    #set_branchlength_vector!(root, blv)
    force_ultrametric(root)
end



"""
    tree_height(root::Node)::Float64

This function calculates the tree height.
"""
function tree_height(root::T)  where T<:Node
    return node_height(root, -Inf)
end


function node_distance(tree::T, node1::T, node2::T, lca::T)::Float64 where T<:Node
    path_length(lca, node1)+path_length(lca,node2)
end

function node_distance(tree::T, node1::T, node2::T)::Float64 where T<:Node
    lca = find_lca(tree, node1, node2)
    node_distance(node1, node2, lca)
end

function get_path(ancestor::T, descendant::T)::Array{Int64} where T<:Node
    path::Array{Int64} = []
    while descendant.binary != ancestor.binary
        push!(path, descendant.num)
        descendant = descendant.mother
    end
    path
end


"""
    path_length(ancestor::Node, descendant::Node)::Float64

Note: The function assumes there is an ancestral relationship between the two nodes.

This function calculates the length of the path separating the ancestor from the
offspring node. The function follows the path specified through the binary
description of the node.
"""
function path_length(ancestor::T, descendant::T)::Float64  where T<:Node
    l::Float64 = 0.0

    while descendant != ancestor
        l += descendant.inc_length
        descendant = descendant.mother
    end # while
    return l
end #function path_length


"""
    get_sister(root::Node, node::Node)::Node

This function gets the sister of `node`. It does so by looking for the respective
binary representation of the sister.
"""
@inline function get_sister(node::T)::T  where T<:Node

    mother = node.mother
    mother.children[findfirst(y-> y!=node, mother.children)]

end # function


"""
    get_mother(root::Node, node::Node)::Node

This function gets the mother of `node`. It does so by looking for the respective
binary representation of the mother node.
"""
@inline function get_mother(node::T)::T  where T<:Node
    return node.mother
end # function

"""
    set_binary!(root::Node)

Assign a binary representation to each node, which specifies the path from the
root to this node via the binary representation of the node.
A left turn is a 1 in binary and a right turn a 0.
"""
function set_binary!(root::T)  where T<:Node
    if root.root
        root.binary = "1"
    end # if
    if root.nchild != 0
        for (ind, node) in enumerate(root.children)
            ind -= 1
            node.binary = string(root.binary, ind)
            set_binary!(node)
        end

    end # if
end # function set_binary

"""
    number_nodes!(root::Node)::Nothing

This function assigns a unique, sequential number to each node.
"""
function number_nodes!(root::T)::Nothing  where T<:Node
    for (index, value) in enumerate(post_order(root))
        value.num = index
    end # for
end # fuction number_nodes


"""
    get_leaves(root::Node)::Vector{Node}

Get all the leaves of this Node. It is meant as a wrapper, only the root node
needs to be supplied
"""
function get_leaves(root::T)::Vector{T}  where T<:Node
    leave_list::Vector{Node} = [i for i in post_order(root) if i.nchild == 0]
    return leave_list
end # function get_leaves


"""
    random_node(root::Node)::Node

This function returns a random node from the tree.
"""
function random_node(root::T)::T  where T<:Node
    post_order_trav = post_order(root)
    return rand(post_order_trav)
end # function random_node


"""
    move!(node1::Node, node2::Node, proportion::Float64)

Change the incomming length of node1 and node2 while keeping there combined length
constant.
"""
function move!(node1::Node, node2::Node, proportion::Float64)
    total::Float64 = node1.inc_length + node2.inc_length
    fp::Float64 = total*proportion
    sp::Float64 = total-fp
    node1.inc_length = fp
    node2.inc_length = sp

end # function move!


"""
    get_branchlength_vector(post_order::Vector{Node})::Vector{Float64}

Return a vector of branch lenghts.
"""
function get_branchlength_vector(post_order::Vector{T})::Vector{Float64}  where T<:Node
    out = zeros(length(post_order)-1)
    @views @simd for i in eachindex(post_order)
        if !post_order[i].root
            out[post_order[i].num]= post_order[i].inc_length
        end
    end
    return out
end # function get_branchlength_vector

function get_branchlength_vector(root::T, out_vec::Vector) where T<:Node

    for child in root.children
        get_branchlength_vector(child, out_vec)
    end

    if !root.root
        out_vec[root.num] = root.inc_length
    end
end

function get_branchlength_vector(root::T)::Vector{Float64}  where T<:Node
    get_branchlength_vector(root, root.blv)
    return root.blv
end # function get_branchlength_vector

function get_branchlength_vector(root::T, vec::Nothing)::Vector{Float64}  where T<:Node
    root.blv = Vector{Float64}(undef, length(post_order(root))-1)
    vec = root.blv
    get_branchlength_vector(root, vec)
    return vec
end


function get_branchlength_vector(t::TreeStochastic)
    get_branchlength_vector(t.value)
end # function



function set_branchlength_vector!(t::TreeStochastic, blenvec::Array{Float64})
    set_branchlength_vector!(t.value, blenvec::Array{Float64})
end # function

function set_branchlength_vector!(t::TreeStochastic, blenvec::ArrayStochastic)
    set_branchlength_vector!(t.value, blenvec.value)
end # function

"""
    set_branchlength_vector!(root::Node, blenvec::Vector{Float64})::Node

This function sets the branch lengths of a tree to the values specified in blenvec.
"""
function set_branchlength_vector!(root::T, blenvec::Array{Float64})  where T<:Node
    any(0 .> blenvec) && throw("this should never happen")
    for child in root.children
        set_branchlength_vector!(child, blenvec)
    end

    @views if root.root !== true
        root.inc_length = blenvec[root.num]
    end
    nothing
end # function set_branchlength_vector!


"""
    get_sum_seperate_length!(root::Node)::Vector{Float64}

This function gets the sum of the branch lengths of the internal branches and the
branches leading to the leave nodes.
"""
function get_sum_seperate_length!(root::T)::Vector{Float64}  where T<:Node
    return get_sum_seperate_length!(post_order(root))
end # function get_sum_seperate_length!


"""
    get_sum_seperate_length!(root::Node)::Vector{Float64}

This function gets the sum of the branch lengths of the internal branches and the
branches leading to the leave nodes.
"""
function get_sum_seperate_length!(post_order::Vector{T})::Vector{Float64}  where T<:Node
    res_int::Float64 = 0.0
    res_leave::Float64 = 0.0
    res_int_log::Float64 = 0.0
    res_leave_log::Float64 = 0.0
    @simd for node in post_order
        if node.nchild !== 0
            # internal branches
            if !node.root
                res_int += node.inc_length
                res_int_log += log(node.inc_length)
            end
        else
            # branches leading to leaves
            res_leave += node.inc_length
            res_leave_log += log(node.inc_length)
        end # if
    end # for
    return [res_int, res_leave, res_int_log, res_leave_log]
end # function get_sum_seperate_length!

function internal_external_map(t::TreeStochastic)::Vector{Int64}
    internal_external_map(t.value)
end

function internal_external_map(root::T)::Vector{Int64}  where T<:Node
    internal_external_map(post_order(root))
end

function internal_external_map(post_order::Vector{T})::Vector{Int64}  where T<:Node
    my_map = Vector{Int64}(undef, length(post_order)-1)
    my_map .= 0
    for node in post_order
        if !node.root
            if node.nchild != 0
                my_map[node.num] = 1
            end
        end
    end
    return my_map
end

function internal_external(root::T)  where T<:Node
    v = root.IntExtMap
    if v === nothing
        v = internal_external_map(root)
        root.IntExtMap = v
    end
    v
end


function find_lca(tree::T, node_l::Array{String, 1})::T  where T<:Node
    find_lca(tree, [find_by_name(tree, i) for i in node_l])
end

function find_lca(tree::T, node_l::Array{T})::T  where T<:Node
    if length(node_l) === 0
        return ""
    elseif length(node_l) === 1
        return node_l[1]
    else
        n1 = popfirst!(node_l)
        n2 = popfirst!(node_l)
        lca = find_lca(tree, n1, n2)
        while length(node_l) !== 0
            n1 = popfirst!(node_l)
            lca = find_lca(lca, n1)
        end
        return lca
    end
end

function find_lca(tree::T, node1::T, node2::T)::T  where T<:Node
    nb = lcp(node1.binary, node2.binary)
    find_binary(tree, nb)
end

function find_num(root::T, num::Int64)  where T<:Node
    rn = Vector{Node}(undef, 1)#MCPhylo.Node()
    find_num(root, num, rn)
    return rn[1]
end

function find_num(root::T, num::Int64, rn::Vector{T})  where T<:Node

    if root.num === num
        rn[1] = root
        rv = true
    else
        rv = false
    end

    if !rv
        for child in root.children
            rv = find_num(child, num,  rn)
        end

    end # if
    return rv
end

function find_binary(root::T, bin::String) where T<:Node
    rn = root
    for ch in bin[2:end]
        rn = rn.children[parse(Int, ch)+1]
    end
    rn
end




#function find_binary(root::T, bin::String)  where T<:Node
#    rn = Vector{Node}(undef, 1)#MCPhylo.Node()
#    find_num(root, bin, rn)
#    return rn[1]
#end

function find_binary(root::T, bin::String, rn::Vector{T})  where T<:Node

    if root.binary == bin
        rn[1] = root
        rv = true
    else
        rv = false
    end

    if !rv
        for child in root.children
            rv = find_binary(child, bin,  rn)
        end

    end # if
    return rv
end



#"""
# proper Markdown comments are not possible
#
#This part creates functions which enable the search for different nodes in the
#tree. It is possible to look for a node via its name, its binary representation
#or to find the root.
#This functionality can be extended by adding more fields to the nodes and the
#meta programmming part here.
#"""
for (sym, my_type) in [(:binary, :String), (:name, :String), (:root ,:Bool), (:num, :Int64)]
    # extend the list to look for more fields in the node
    @eval function $(Symbol(string("find_by_$sym")))(tree::T, identifier::$my_type)::T  where T<:Node
        # create each function and make it so it only accepts the correct type
        local all_nodes = post_order(tree) # make sure all_nodes only belongs to this function
        for node in all_nodes
            if node.$sym == identifier
                # return the node if it is found
                return node
            end # if
        end # for
        # the node is not found. Therefore throw an error!
        throw("The node identified by $identifier is not in the tree.")
    end # function
end







#end # module my_tree
