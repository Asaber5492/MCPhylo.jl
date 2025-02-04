########################### Chisq Diagnostic ###########################

function update_hangartner_temp_vars!(u::Matrix{Int64}, X::Matrix{Int64},
                                      t::Int64)
  d = size(X,2)

  for j in 1:d
    u[X[t, j], j] += 1
  end
end

function hangartner_inner(Y::AbstractMatrix, m::Int)
  ## setup temp vars
  n, d = size(Y)

  # Count for each category in each chain
  u = zeros(Int64, m, d)
  v = zeros(Int64, m, d)

  for t in 1:n
    # fill out temp vars
    update_hangartner_temp_vars!(u, Y, t)
  end
  phia, chi_stat, m_tot = weiss_sub(u, v, n)

  return (n * sum(chi_stat), m_tot)
end

function weiss(X::AbstractMatrix{U}) where {U<:Any}
  ## number of iterations, number of chains
  n, d = size(X)

  ## mapping of values to integers
  v_dict = Dict{U, Int}()

  ## max unique categories
  mc = map(c -> length(unique(X[:,c])), 1:d)
  m = length(unique(X))

  ## counter of number of unique values in each chain
  r0 = 0

  ## Count for each category in each chain
  u = zeros(Int, m, d)

  ## Number of times a category did not transition in each chain
  v = zeros(Int, m, d)

  for t in 1:n
    for c in 1:d
      if !(X[t,c] in keys(v_dict))
        r0 += 1
        v_dict[X[t,c]] = r0
      end
      idx1 = v_dict[X[t,c]]
      u[idx1,c] += 1

      if t > 1
        if X[t-1,c] == X[t,c]
          v[idx1, c] += 1
        end
      end
    end
  end
  phia, chi_stat, m_tot = weiss_sub(u, v, n)
  ca = (1 + phia) / (1 - phia)
  stat = (n / ca) * sum(chi_stat)
  pval = NaN
  if ((m_tot - 1) * (d - 1)) >= 1
    pval = 1 - cdf(Chisq((m_tot - 1) * (d - 1)), stat)
  end

  return ( stat, m_tot, pval, ca)
end

function weiss_sub(u::Matrix{Int64}, v::Matrix{Int64}, t::Int)
  m, d = size(u)
  nt = 0.0
  dt = 0.0
  m_tot = 0

  mp = zeros(Float64, m, d)
  ma = zeros(Float64, m)
  phia = 0.0
  ca = 0.0
  df = 0.0

  chi_stat = zeros(Float64, d)

  for j in 1:m
    p1 = 0.0
    p2 = 0.0
    for l in 1:d
      #aggregate
      p1 += v[j, l] / (d * (t - 1))
      p2 += u[j, l] / (d * t)

      #per chain
      mp[j, l] = u[j, l] / t
      ma[j] += u[j, l] / (d * t)
    end
    nt += p1
    dt += p2 ^ 2

    if ma[j] > 0
      m_tot += 1
      for l in 1:d
        chi_stat[l] += (mp[j,l] - ma[j]) ^ 2 / ma[j]
      end
    end
  end
  phia = 1.0 + (1.0 / t) - ((1 - nt) / (1 - dt))
  phia = min(max(phia,0.0),1.0 - eps())
  return (phia, chi_stat, m_tot)
end

function update_billingsley_temp_vars!(f::Array{Int64,3},
                                       X::Matrix{Int64},
                                       t::Int64)
  d = size(X,2)
  for j in 1:d
    if t > 1
      f[X[t-1, j], X[t,j], j] += 1
    end
  end
end

function billingsley_sub(f::Array{Int64,3})
  df = 0.0
  stat = 0.0

  m, d = size(f, 2), size(f, 3)

  # marginal transitions, i.e.
  # number of transitions from each category
  mf = mapslices(sum, f, dims=2)

  # For each category, number of chains for which
  # that category was present
  A = vec(mapslices( (x)-> sum(x.>0), mf, dims=3))

  # For each category, number of categories it
  # transitioned to
  B =  vec(mapslices((x)-> sum(x.>0), mapslices(sum,f,dims=3),dims=2))

  # transition probabilities in each chain
  P = f ./ mf

  # transition probabilities
  mP = reshape((mapslices(sum,f,dims=3) ./ mapslices(sum,mf,dims=3)), Val(2))

  idx = findall(x -> x != 0, A .* B)
  for j in idx
    #df for billingsley
    df += (A[j] - 1) * (B[j] - 1)

    #billingsley
    for k in idx
      if (mP[j,k] > 0.0)
        for l in 1:d
          if mf[j,1,l] > 0 && isfinite(P[j,k,l])
            stat += mf[j,1,l] * (P[j,k,l] - mP[j,k]) ^ 2 / mP[j,k]
          end
        end
      end
    end
  end
  return (stat, df, mP)
end

function bd_inner(Y::AbstractMatrix, m::Int)
  num_iters, num_chains = size(Y)
  # Transition matrix for each chain
  f = zeros(Int64, m, m, num_chains)

  for t in 1:num_iters
    # fill out temp vars
    update_billingsley_temp_vars!(f, Y, t)
  end
  billingsley_sub(f)
end

function simulate_NDARMA(N::Int, p::Int, q::Int, prob::Vector{Float64},
                         phi::Vector{Float64})
  X = zeros(Int64, N)
  X[1:p] = rand(Categorical(prob), p)
  d1 = Multinomial(1, phi)
  d2 = Categorical(prob)
  for t in (p+1):N
    alphabeta = rand(d1)
    eps = rand(d2, q + 1)
    X[t] = sum([X[(t-p):(t-1)]; eps] .* alphabeta)
  end
  return X
end

function simulate_MC(N::Int, P::Matrix{Float64})
  X = zeros(Int64, N)
  n, m = size(P)
  X[1] = sample(1:n)
  for i in 2:N
    X[i] = wsample(1:n, vec(P[X[i-1], :]) )
  end
  return X
end

function diag_all(X::AbstractMatrix{U}, method::Symbol,
                  nsim::Int, start_iter::Int, step_size::Int) where {U<:Any}

  ## number of iterations, number of chains
  n, d = size(X)

  ## mapping of values to integers
  v_dict = Dict{U, Int}()

  ## max unique categories
  mc = map(c -> length(unique(X[:,c])), 1:d)
  m = length(unique(X))

  ## counter of number of unique values in each chain
  r0 = 0

  ## Count for each category in each chain
  u = zeros(Int, m, d)

  ## Number of times a category did not transition in each chain
  v = zeros(Int, m, d)

  ## transition matrix for each chain
  f = zeros(Int, m, m, d)

  result = zeros(Float64, 3, length(start_iter:step_size:n))
  result_iter = 1
  for t in 1:n
    for c in 1:d
      if !(X[t,c] in keys(v_dict))
        r0 += 1
        v_dict[X[t,c]] = r0
      end
      idx1 = v_dict[X[t,c]]
      u[idx1,c] += 1

      if t > 1
        idx2 = v_dict[X[t-1,c]]
        f[idx1, idx2, c] += 1

        if X[t-1,c] == X[t,c]
          v[idx1, c] += 1
        end
      end
    end

    if ((t >= start_iter) && (((t - start_iter) % step_size) == 0))
      phia, chi_stat, m_tot = weiss_sub(u, v, t)
      hot_stat, df, mP = billingsley_sub(f)

      phat = mapslices(sum,u,dims=2)[:,1] / sum(mapslices(sum,u,dims=2))
      ca = (1 + phia) / (1 - phia)
      stat = NaN
      pval = NaN
      df0  = NaN

      if method == :hangartner
        stat = t * sum(chi_stat)
        df0 = (m - 1) * (d - 1)
        if m > 1 && !isnan(stat)
          pval = 1 - cdf(Chisq( (m - 1) * (d - 1)), stat)
        end
      elseif method == :weiss
        stat = (t / ca) * sum(chi_stat)
        df0 = (m - 1) * (d - 1)
        pval = NaN
        if m > 1 && !isnan(stat)
          pval = 1 - cdf(Chisq( (m - 1) * (d - 1)), stat)
        end
      elseif method == :DARBOOT
        stat = t * sum(chi_stat)
        bstats = zeros(Float64, nsim)
        for b in 1:nsim
          Y = hcat([simulate_NDARMA(t, 1, 0, phat, [phia, 1-phia])
                for j in 1:d]...)
          s = hangartner_inner(Y, m)[1]
          bstats[b] = s
        end
        idx = findall(!isnan(bstats))
        df0 = mean(bstats[idx])
        pval = sum(stat .<= bstats[idx])/length(idx)
      elseif method == :MCBOOT
        bstats = zeros(Float64, nsim)
        for b in 1:nsim
          Y = hcat([simulate_MC(t, mP) for j in 1:d]...)
          s = hangartner_inner(Y, m)[1]
          bstats[b] = s
        end
        idx = findall(!isnan(bstats))
        df0 = mean(bstats[idx])
        pval = sum(stat .<= bstats[idx])/length(idx)
      elseif method == :billingsley
        stat = hot_stat
        df0 = df
        if df > 0 && !isnan(hot_stat)
          pval = 1 - cdf(Chisq(df), hot_stat)
        end
      elseif method == :billingsleyBOOT
        stat = hot_stat
        bstats = zeros(Float64, nsim)
        for b in 1:nsim
          Y = hcat([simulate_MC(t, mP) for j in 1:d]...)
          (s,sd) = bd_inner(Y, m)[1:2]
          bstats[b] = s/sd
        end
        idx = findall(!isnan(bstats))
        df0 = mean(bstats[idx])
        pval = sum(stat/df .<= bstats[idx])/length(idx)
      else
        error("Unexpected")
      end
      result[:, result_iter] = [stat, df0, pval]
      result_iter += 1
    end
  end
  return result
end

function discretediag_sub(c::AbstractChains, frac::Real, method::Symbol,
                          nsim::Int, start_iter::Int, step_size::Int)

  num_iters, num_vars, num_chains = size(c.value)

  vals = zeros(Float64, 3 * (num_chains + 1), num_vars)
  plot_vals_stat = zeros(length(start_iter:step_size:num_iters), num_vars)
  plot_vals_pval = zeros(length(start_iter:step_size:num_iters), num_vars)

  ## Between-chain diagnostic
  X = zeros(Int64, num_iters, num_chains)
  for j in 1:length(num_vars)
    X = convert(Array{Int64, 2}, c.value[:,j,:])
    result = diag_all(X, method, nsim, start_iter, step_size)
    plot_vals_stat[:,j] = result[1, :] ./ result[2, :]
    plot_vals_pval[:,j] = result[3, :]
    vals[1:3, j] = result[:, end]
  end

  ## Within-chain diagnostic
  x = zeros(Int64, num_iters)
  Y = zeros(Int64, num_iters, 2)
  for j in 1:num_vars
    for k in 1:num_chains
      x = convert(Array{Int64, 1}, c.value[:,j,k])

      idx1 = 1:round(Int64, frac * num_iters)
      idx2 = round(Int64, num_iters - frac * num_iters + 1):num_iters
      x1 = x[idx1]
      x2 = x[idx2]
      n_min = min(length(x1), length(x2))
      Y = [x1[1:n_min] x2[(end - n_min + 1):end]]

      vals[(3 + 3 * (k - 1) + 1):(3 + 3 * (k - 1) + 3), j] =
        diag_all(Y, method, nsim, n_min, step_size)[:, end]
    end
  end
  return (collect(1:num_vars), vals, plot_vals_stat, plot_vals_pval)

end

function discretediagplot(c::AbstractChains; frac::Real=0.3,
                          method::Symbol=:weiss, nsim::Int=1000,
                          start_iter::Int=100, step_size::Int=10000)

  num_iters, num_vars, num_chains = size(c.value)

  valid_methods = [:hangartner, :weiss, :DARBOOT,
                   :MCBOOT, :billingsley, :billingsleyBOOT]
  if !(method in valid_methods)
    methods_str = join([":$f" for f in valid_methods], ", ")
    throw(ArgumentError("method must be one of ", methods_str))
  end

  if !(0.0 < frac < 1.0)
    throw(ArgumentError("frac must be in (0,1)"))
  end

  if (start_iter > num_iters ) || (step_size > num_iters)
    throw(ArgumentError("start_iter, step_size must be less than $num_iters"))
  end

  V, vals, plot_vals_stat, plot_vals_pval =
    discretediag_sub(c, frac, method, nsim, start_iter, step_size)

  p1 = Plots.plot(repeat(collect(c.range[start_iter:step_size:num_iters])/
                          1000, outer=[length(V)]),
                       vcat([plot_vals_stat[:,j] for j in 1:length(V)]...),
                       seriestype=:line,
                       group=repeat(c.names[V],
                          inner=[length(start_iter:step_size:num_iters)]),
                       xlabel="Iteration (thousands)", ylabel="stat/df",
                       legendtitle="Variable")

  p2 = Plots.plot(repeat(collect(c.range[start_iter:step_size:num_iters])/
                        1000, outer=[length(V)]),
                     vcat([plot_vals_pval[:,j] for j in 1:length(V)]...),
                     seriestype=:line,
                     group=repeat(c.names[V],
                        inner=[length(start_iter:step_size:num_iters)]),
                    xlabel="Iteration (thousands)", ylabel="pval",
                    legendtitle="Variable")


  return [p1, p2]
end
"""
    discretediag(c::AbstractChains; frac::Real=0.3,
                  method::Symbol=:weiss, nsim::Int=1000)

Compute the convergence diagnostic for a discrete variable. Several options are available by choosing method to be one of :hangartner, :weiss, :DARBOOT, MCBOOT, :billinsgley, :billingsleyBOOT. The first four are based off of Pearson’s chi-square test of homogeneity. The diagnostic tests whether the proportion of the categories of the discrete variable are similar in each chain. The last two methods test whether the transition probabilities between each category are similar between each chain. Along with a between chain assessment of convergence, a within-chain assessment is carried out by comparing a specified fraction (frac), or window, of the beginning of a chain to the specified fraction of the end of the chain. For within-chain assessment, users should ensure that there is sufficient separation between the windows to assume that their samples are independent. A non-significant test p-value indicates convergence. Significant p-values indicate non-convergence and the possible need to discard initial samples as a burn-in sequence or to simulate additional samples.

A `ChainSummary` type object with parameters contained in the rows of the `value` field. The first three columns correspond to the test statistic, degrees of freedom, and p-value of the between-chain assessment. The next columns are the test statistic, degrees of freedom, and p-value for each chain of the within-chain assessment.

* `c` : sampler output on which to perform calculations.

* `frac` : proportion of iterations to include in the first window.

* `method` : Specify which method to use. One of :hangartner, :weiss, :DARBOOT, MCBOOT, :billinsgley, :billingsleyBOOT`.

* `nsim` : For the bootstrap methods (:DARBOOT, :MCBOOT, and :billingsleyBOOT) the number of bootstrap simulations.
"""
function discretediag(c::AbstractChains; frac::Real=0.3,
                      method::Symbol=:weiss, nsim::Int=1000)

  num_iters, num_vars, num_chains = size(c.value)

  valid_methods = [:hangartner, :weiss, :DARBOOT,
                   :MCBOOT, :billingsley, :billingsleyBOOT]
  if !(method in valid_methods)
    methods_str = join([":$f" for f in valid_methods], ", ")
    throw(ArgumentError("method must be one of ", methods_str))
  end

  if !(0.0 < frac < 1.0)
    throw(ArgumentError("frac must be in (0,1)"))
  end

  V, vals = discretediag_sub(c, frac, method, nsim,
                             size(c.value,1), size(c.value,1))[1:2]

  hdr = header(c) * "\nChisq Diagnostic:\nEnd Fractions = $frac\n" *
  "method = $method\n"
  ChainSummary(copy(round.(vals,digits=3)'), c.names[V],
               convert(Array{AbstractString, 1},
                       vcat([["stat", "df", "p-value"]
                             for k in 1:(num_chains + 1)]...)), hdr)
end
