using Distributions, UnPack, OffsetArrays, CavityTools

struct Model{D,M,M2,O}
    T::Int
    γp::Float64
    λp::Float64
    γi::Float64
    λi::Float64
    μ::M
    τμ::Vector{Int}
    belief::M2
    τbelief::Vector{Int}
    fr::Float64
    dilution::Float64
    distribution::D
    a::O
end

popsize(M::Model) = length(M.τbelief)

function Model(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution) 
    μ = fill(1.0 / 2(T+2), 0:T+1, 0:1, 1:N)
    τμ = fill(T+1, N)
    belief = fill(1/(T+2), 0:T+1, N)
    τbelief = fill(T+1, N)
    a = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    Model(T, γp, λp, γi, λi, μ, τμ, belief, τbelief, fr, dilution, distribution, a)
end

obs(M::Model, ti, taui, oi) = oi ? (((ti <= M.T) == (taui <= M.T)) ? 1.0 - M.fr : M.fr) : 1.0

function update!(M::Model, i)
    @unpack fr, γi, T, a, μ, belief, τμ, τbelief = M
    xi0,∂i,S,oi = rand_disorder(M)
    τi = (1-xi0) * minimum(τ+s for (τ,s) in zip(view(M.τμ,∂i), S); init=T+1)
    ∂out = rand(1:popsize(M), length(∂i))
    τbelief[i] = τi
    μ1, μ0 = μ[:,1,∂i], μ[:,0,∂i]
    μ[:, :, ∂out] .= 0
    for ti = 0:T+1
        ξ = obs(M, ti, τi, oi)
        ξ == 0.0 && continue
        seed = ti == 0 ? γi : 1-γi
        phi = ti != 0 && ti != T+1
        M1,m1full = cavity((@view μ1[ti,:]), *, 1.0)
        M0,m0full = cavity((@view μ0[ti,:]), *, 1.0)
        for (j,m1,m0) in zip(∂out,M1,M0)
            for tj in 0:T+1                
                ν = ξ * seed * (a[ti-tj-1] * m1 - phi * a[ti-tj] * m0)
                #ν = ξ * ((ti == 0 ? 1.0 : 1-γ) * a[ti-tj-1] * m1 - (1-γ) * a[ti-tj] * m0)
                μ[tj, 1, j] += ν * a[tj - ti - 1]
                μ[tj, 0, j] += ν * a[tj - ti]
            end
            τμ[j] = τi
        end
        belief[ti, i] = ξ * seed * (m1full - phi * m0full)
    end

    s = sum(@view belief[:,i])
    @assert s > 0
    belief[:, i] ./= s

    for k in ∂out
        sk = sum(@view μ[:,:,k])
        @assert sk > 0
        μ[:, :, k] ./= sk
    end
end

residual(d::Poisson) = d #residual degree of poiss distribution is poisson with same param
residual(d::Dirac) = Dirac(d.value - 1) #residual degree of rr distribution (delta) is a delta at previous vale
residual(d::DiscreteNonParametric) = DiscreteNonParametric(support(d) .- 1, (probs(d) .* support(d)) / sum(probs(d) .* support(d)))

function rand_disorder(M::Model)
    r = 1.0 / log(1-M.λp)
    delay(r) = floor(Int,log(rand())*r) + 1
    xi0 = (rand() < M.γp);
    d = rand(M.distribution)
    S = [delay(r) for _ = 1:d]
    oi = rand() > M.dilution
    ∂i = rand(1:popsize(M), d)
    return xi0, ∂i, S, oi
end



function pop_dynamics!(M::Model; iterations = 100)
    N = popsize(M)
    for _ = 1:iterations
        for i = 1:N
            update!(M,i)
        end
    end
end


function makeDistrib(degreetype,d)
    if degreetype == "poisson"
        return Poisson(d)
    elseif degreetype == "regular"
        return Dirac(d)
    else degreetype == "ft4"
        min_d, max_d = 3, 150
        d_supp = collect(min_d:max_d)
        d_prob = 1 ./ ((d_supp) .^ 4)
        return degree_dist = DiscreteNonParametric(d_supp, d_prob ./ sum(d_prob))
    end
end