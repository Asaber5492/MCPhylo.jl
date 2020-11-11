using Distributed
include("./src/MCPhylo.jl")
using .MCPhylo
using Random
using Distances
import Distributions: sampler, _rand!, logpdf, Normal
import Random: GLOBAL_RNG, rand!
using LinearAlgebra

include("BrownianDistr.jl")
#include("TreeSugg.jl")

mt, df = make_tree_with_data("notebook/data-st-64-110.paps.nex"); # load your own nexus file

n_leaves = length(MCPhylo.get_leaves(mt))
n_concs = size(df, 2)
arr = [df[:,:,n.num] for n in MCPhylo.get_leaves(mt)]
arrn = zeros(n_leaves, n_concs)
for (ind, i) in enumerate(arr)
    arrn[ind, :] .= i[1,:]
end

mt2 = deepcopy(mt);

randomize!(mt2);

my_data = Dict{Symbol, Any}(
  :mtree => mt,
  :arrn => arrn,
  :leaves => size(arrn, 1),
  :residuals => size(arrn, 2),
  :nnodes => size(df, 3),
  :μ_raw => ones(n_leaves, n_concs),
  :σ => ones(n_concs)
);



function symtreeprob(d::N)::N where N <: GeneralNode
  move = rand([:NNI, :Slide, :Swing, :EdgeLength])
  tree = deepcopy(d)
  if move == :NNI
    NNI!(tree)
  elseif move == :SPR
    tree = SPR(tree)
  elseif move == :Slide
    slide!(tree)
  elseif move == :Swing
    swing!(tree)
  elseif move == :EdgeLength
    MCPhylo.change_edge_length!(tree)
  else
    throw("Tree move not elegible ")
  end
  return tree
end


function my_ELARR(Σ::Array{Float64,2}, σ::Array{Float64})
    n_concs = length(σ)
    n_leaves = size(Σ, 1)
    Σ_L_Arr = Array{Float64,3}(undef, n_concs, n_leaves, n_leaves)
    @inbounds @simd for i in 1:n_concs
        Σ_L_Arr[i, :, :] .= cholesky(σ[i] .* Σ).L
    end
    Σ_L_Arr
end

function my_muArr(μ::Array{Float64}, n_leaves::Int64)
    n_residuals = size(μ, 1)
    reshape(repeat(μ, outer=n_leaves), n_concs, n_leaves)
end



model =  Model(
    arrn = Stochastic(2,
            (μ_Arr, Σ_Arr, latent) -> BrownianDistr(μ_Arr, Σ_Arr, latent), false),
    μ_Arr = Logical(2, (μ) -> my_muArr(μ.value, my_data[:leaves]), false),
    Σ_Arr = Logical(3, (σ, Σ) -> my_ELARR(Σ.value, σ.value), false),
    μ = Stochastic(1, (μH, σH) -> Normal(μH, σH), true),
    μH = Stochastic(()->Normal(), true),
    σH = Stochastic(()->Exponential(), true),
    σ = Stochastic(1,(λ) -> Exponential(λ), true),
    latent = Stochastic(2, ()->Normal(), false),
    λ = Stochastic(() -> Exponential(), true),
    Σ = Logical(2, (mtree) -> MCPhylo.to_covariance(mtree), false),
    mtree = Stochastic(Node(), () -> CompoundDirichlet(1.0,1.0,0.100,1.0), my_data[:nnodes]+1, true),
    );

inits = [Dict{Symbol, Union{Any, Real}}(
    :mtree => mt,
    :arrn => my_data[:arrn],
    :nnodes => my_data[:nnodes],
    :residuals => my_data[:residuals],
    :leaves => my_data[:leaves],
    :μ_raw => my_data[:μ_raw],
    :latent => rand(my_data[:residuals],my_data[:leaves]),
    :Σ => MCPhylo.to_covariance(mt),
    :μH => rand(),
    :σH => rand(),
    :μ => zeros(my_data[:residuals]),
    :λ => rand(),
    :σ => rand(my_data[:residuals]),
    ),
    Dict{Symbol, Union{Any, Real}}(
        :mtree => mt2,
        :arrn => my_data[:arrn],
        :nnodes => my_data[:nnodes],
        :residuals => my_data[:residuals],
        :leaves => my_data[:leaves],
        :μ_raw => my_data[:μ_raw],
        :Σ => MCPhylo.to_covariance(mt),
        :μH => rand(),
        :σH => rand(),
        :μ => randn(my_data[:residuals]),
        :λ => rand(),
        :σ => rand(my_data[:residuals]),
        )
    ]

scheme = [ABC([:μ,:σ], 1.0, identity, 100, proposal=Normal, kernel=Normal, maxdraw=25, nsim=3, dist=hamming),
          ABC(:latent, 1.0, identity, 100, proposal=Normal, kernel=Normal, maxdraw=25, nsim=3, dist=hamming),
          MCPhylo.ABCT(:mtree, 0.5, identity, 100, proposal=symtreeprob,kernel=Normal, maxdraw=100, nsim=3, dist=hamming),
          #Slice([:μH, :σH, :λ], 0.05, Multivariate)
          #NUTS([:μH, :σH]),# dtype=:Zygote),
          NUTS(:λ)
          ];

setsamplers!(model, scheme)

sim = mcmc(model, my_data, inits, 500, burnin=10,thin=5, chains=1, trees=true)

to_file(sim, "ABC_test")
