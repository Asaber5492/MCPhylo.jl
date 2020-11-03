using Distributed
addprocs(3);
@everywhere include("./src/MCPhylo.jl")
@everywhere using .MCPhylo
@everywhere using Random
@everywhere using Distances
@everywhere import Distributions: sampler, _rand!, logpdf, Normal
@everywhere import Random: GLOBAL_RNG, rand!

@everywhere include("BrownianDistr.jl")
@everywhere include("TreeSugg.jl")

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



@everywhere function symtreeprob(d::N)::N where N <: GeneralNode
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




model =  Model(
    arrn = Stochastic(2,
            (μ, σ, Σ) -> BrownianDistr(μ, σ, Σ), false),
    μ = Stochastic(1, (μH, σH) -> Normal(μH, σH), true),
    μH = Stochastic(()->Normal(), true),
    σH = Stochastic(()->Exponential(), true),
    σ = Stochastic(1,(λ) -> Exponential(λ), true),
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
    :Σ => MCPhylo.to_covariance(mt),
    :μH => rand(),
    :σH => rand(),
    :μ => randn(my_data[:residuals]),
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

scheme = [ABC(:μ, 0.5, identity, 0.5, kernel=Normal, maxdraw=100, nsim=3, dist=hamming),
          ABC(:σ, 0.5, identity, 0.5, kernel=Normal, maxdraw=100, nsim=3, dist=hamming),
          MCPhylo.ABCT(:mtree, 0.5, identity, 0.5, kernel=Normal, proposal=symtreeprob,maxdraw=100, nsim=3, dist=hamming),
          Slice([:μH, :σH, :λ], 0.05, Univariate)
          ];

setsamplers!(model, scheme)

sim = mcmc(model, my_data, inits, 50000, burnin=10000,thin=50, chains=2, trees=true)

to_file(sim, "ABC_test")
