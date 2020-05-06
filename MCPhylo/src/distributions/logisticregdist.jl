
mutable struct BrownianPhylo <: DiscreteMatrixDistribution
        mu::Array{Float64,1}
        tree::AbstractNode#NodeS#Node{T,A,B,I} where {T<: Real, A<: AbstractArray, B<:AbstractArray, I<:Integer}
        sigmai::Vector{Float64}
        P::Array{Float64,1}
        scaler::Float64
        chars::Int64
        leaves::Int64
end

minimum(d::BrownianPhylo) = -Inf
maximum(d::BrownianPhylo) = +Inf
Base.size(d::BrownianPhylo) = (d.leaves, d.chars)

function logpdf(d::BrownianPhylo, x::AbstractArray{T, 2})::Float64 where T<:Real
    #blv::Vector{Float64} = get_branchlength_vector(d.tree)
    #cov_fun = to_covariance_func(d.tree)
    covmat = cu(to_covariance(d.tree))
    ms = d.leaves*d.chars
    cu_all = cat([covmat for i=1:d.chars]..., dims=(1,2))
    sigmas_mat = cu(cat([zeros(d.leaves,d.leaves).=i for i in d.sigmai]..., dims=(1,2)))
    #res = mlpdf(cu_all.*sigmas_mat,
    #            Matrix{Float64}(I, ms, ms),
    #            d.mu,
    #            d.P, reshape(x, ms), d.scaler, ms)
    res = mlpdf(cu_all.*sigmas_mat,
                cu(Matrix{Float64}(I, ms, ms)),
                cu(d.mu),
                cu(d.P), cu(reshape(x, ms)), convert(Float32,d.scaler), convert(Float32,ms))
    println(res)
    res
end

@inline function my_bernoulli_logpdf(θ::T, x::S)::T where {S <: Real, T<: Real}
    x == 1 ? θ : log(1-exp(θ))
end

const l2pi = log(2*π)/2

function p_apply_array(f::Function, x::Vector{T})::T where T<:Real
 f(x)
end

@inline function mymvlogpdf_2(v1::T, si::Array{T, 2}, x::Array{T,1})::T where T <: Real
    v1-1/2 * x' * si * x
end



@inline function my_bernoulli_logpdf_cu(θ::T, x::T) where T
    x == 1 ? θ : CUDAnative.log(one(eltype(θ))-CUDAnative.exp(θ))
end

@inline loginvlogit_cu(x::T, λ::T=1.0) where T = -CUDAnative.log((CUDAnative.exp(-λ*x)+one(eltype(x))))


function mlpdf(covmat::CuArray{Float32, 2},
                ey::CuArray{Float32, 2},
                P::CuArray{Float32, 1},
                mu::CuArray{Float32, 1},
                data::CuArray{Float32, 1},
                scaler::Float32,
                nlangs::Float32)::Float64
    LinearAlgebra.LAPACK.potrf!('L', covmat)
    LinearAlgebra.LAPACK.potrs!('L', covmat, ey)
    si = CuArrays.CUBLAS.gemm('T','N', ey, ey)
    r = CuArrays.CUBLAS.symv('L', covmat, P)
    ds = sum(CUDAnative.log.(diag(covmat)))*2
    t1 = r' * si * r
    thetas = loginvlogit_cu.(r + mu, scaler)
    result = sum(my_bernoulli_logpdf_cu.(thetas,data))
    iv = -0.5*ds - 0.5 * t1
    result+(-0.5*log(2*π)+iv)
end




function mlpdf(covmat::Array{Float64, 2},
                ey::Array{Float64, 2},
                P::Array{Float64, 1},
                mu::Array{Float64, 1},
                data::Array{Float64, 1},
                scaler::Float64,
                nlangs::Int64)::Float64
    LinearAlgebra.LAPACK.potrf!('L', covmat)
    LinearAlgebra.LAPACK.potrs!('L', covmat, ey)
    si = LinearAlgebra.BLAS.gemm('T','N', ey, ey)
    r = LinearAlgebra.BLAS.symv('L', covmat, P)
    ds = sum(log.(diag(covmat)))*2
    t1 = r' * si * r
    thetas = loginvlogit.(r + mu, scaler)
    result = sum(my_bernoulli_logpdf.(thetas,data))
    iv = -0.5*ds - 0.5 * t1
    result+(-0.5*log(2*π)+iv)
end




function mlpdf(mu::Array{Float64,2}, cov_fun_mat::Array{Function,2}, blv::Vector{T}, sigmai::Vector{Float64},
               P::Array{Float64,2}, scaler::Float64, n_langs::Int64, chars::Int64, data::Array{Float64,2})::T where T<:Real

    mycov = p_apply_array.(cov_fun_mat, Ref(blv))
    #ch = LinearAlgebra.cholesky(mycov)

    #chl = ch.L

    #ichl = inv(chl)
    #si = transpose(ichl) * ichl
    #ds = logdet(chl)
    #v1 = -n_langs*l2pi-ds
    result::Float64 = zero(Float64)

    @views @inbounds for i=1:chars
        ch = LinearAlgebra.cholesky(sigmai[i]*mycov)
        chl = ch.L
        ichl = inv(chl)
        si = transpose(ichl) * ichl
        ds = logdet(chl)
        v1 = -n_langs*l2pi-ds
        r::Array{Float64,1} = chl * P[:,i]
        result = result + mymvlogpdf_2(v1, si, r)
        for j in 1:n_langs
            result = result + my_bernoulli_logpdf(loginvlogit(r[j] + mu[j], scaler), data[j,i])::Float64
        end
    end
    result
end



function mlpdf_ou(mu::Array{Float64,2}, tree::Node, blv::Vector{T}, sigmai::Vector{Float64},
               P::Array{Float64,2}, scaler, chars::Int64, data::Array{Float64,2})::T where T<:Real

    mycov::Array{T,2} = to_covarianceou(tree, blv)

    ch = cholesky(mycov).L
    rv_a = zeros(T,chars)
    @views @inbounds Base.Threads.@threads for i in 1:chars
        mat = ch*P[:,i]+mu[i,:]
        rv_a[i] = sum(my_bernoulli_logpdf.(loginvlogit.(mat, Ref(scaler)), data[:,i]))
    end
    sum(rv_a)
end



function gradlogpdf(d::BrownianPhylo, x::AbstractArray{T, 2})::Tuple{Float64, Vector{Float64}} where T<:Real

    blv = get_branchlength_vector(d.tree)
    cov_fun = to_covariance_func(d.tree)
    blv = round.(blv, digits=5)
    f(y) =mlpdf(d.mu, cov_fun, y, d.sigmai, d.P,d.scaler, d.leaves, d.chars, x)


    r = Zygote.pullback(f, blv)

    r[1], r[2](1.0)[1]
end
