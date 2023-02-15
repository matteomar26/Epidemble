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
    Λ::O
end

popsize(M::Model) = length(M.τbelief)

function Model(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution) 
    μ = fill(1.0 / 2(T+2), 0:T+1, 0:1, 1:N)
    τμ = fill(T+1, N)
    belief = fill(1/(T+2), 0:T+1, N)
    τbelief = fill(T+1, N)
    Λ = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    Model(T, γp, λp, γi, λi, μ, τμ, belief, τbelief, fr, dilution, distribution, Λ)
end

#function obs(M::Model, ti, τi, oi) 
 #   return oi ? (((ti <= M.T) == (τi <= M.T)) ? 1.0 - M.fr : M.fr) : 1.0
#end

function obs(M::Model, ti, τi, oi) 
    if oi == 1  
        return ((ti <= M.T) == (τi <= M.T)) ? 1.0 - M.fr : M.fr 
    elseif oi == 0
        return 1.0
    end
end
 
function normalize!(m)
    s = sum(m)
    @assert s > 0
    m ./= s
end

function update!(M::Model, i)
    @unpack fr, γi, T, Λ, μ, belief, τμ, τbelief = M
    xi0,∂in,S,oi = rand_disorder(M) 
    #we take the minimum of the incoming infections except one neighbour 
    τout, τi = @views cavity(τμ[∂in], min, xi0 ? 0 : T+1) 
    #the minimum over all the neighbours (including the zero patient variable) is the infection time
    τbelief[i] = τi
    #the outgoing elements of population are extracted
    ∂out = rand(1:popsize(M), length(∂in))
    #a copy of the incoming messages is made...
    μ1, μ0 = μ[:,1,∂in], μ[:,0,∂in]
    #... in order not to put them to zero
    μ[:, :, ∂out] .= 0
    belief[:, i] .= 0
    #the outgoing messages are updated
    τμ[∂out] .= τout .+ S
    M1,M0 = zeros(length(∂in)), zeros(length(∂in))
    for ti = 0:T+1
        ξ = obs(M, ti, τi, oi)
        iszero(ξ) && continue
        pseed = iszero(ti) ? γi : 1 - γi
        phi = 0 < ti < T+1
        m1full = cavity!(M1, (@view μ1[ti, :]), *, ξ * pseed)
        m0full = cavity!(M0, (@view μ0[ti, :]), *, ξ * pseed)
        for (j, m1, m0) in zip(∂out, M1, M0)
            for tj in 0:T+1
                ν =  Λ[ti - tj - 1] * m1 - phi * Λ[ti - tj] * m0
                #ν = ξ * ((ti == 0 ? 1.0 : (1 - γi)) * Λ[ti - tj - 1] * m1 - (1 - γi) * Λ[ti - tj] * m0)
                μ[tj, 1, j] += ν * Λ[tj - ti - 1]
                μ[tj, 0, j] += ν * Λ[tj - ti]
            end
        end
        belief[ti, i] = m1full - phi * m0full
    end
    normalize!(@view belief[:, i])
    for j in ∂out
        normalize!(@view μ[:, :, j])
    end
end

residual(d::Poisson) = d #residual degree of poiss distribution is poisson with same param
residual(d::Dirac) = Dirac(d.value - 1) #residual degree of rr distribution (delta) is a delta at previous vale
residual(d::DiscreteNonParametric) = DiscreteNonParametric(support(d) .- 1, (probs(d) .* support(d)) / sum(probs(d) .* support(d)))

geometric(r) = floor(Int, log(rand())*r) + 1

function rand_disorder(M::Model)
    r = 1 / log(1 - M.λp)
    xi0 = rand() < M.γp;
    d = rand(M.distribution)
    S = [geometric(r) for _ in 1:d]
    oi = rand() > M.dilution
    ∂in = rand(1:popsize(M), d)
    (;xi0, ∂in, S, oi)
end


function pop_dynamics!(M::Model; iterations = 100, tol = 1/sqrt(N), callback = (x...)->nothing)
    N = popsize(M)    
    for it = 1:iterations
        s_old = sum(M.belief,dims=2) / N
        for i = 1:N
            update!(M,i)
        end
        callback(it, M)
        s_new = sum(M.belief,dims=2) / N
        if sum(abs.(s_new .- s_old)) <= tol
           return it # we return the iteration at which the pop-dyn converged
        end
    end
    return iterations
end

FatTail(support,k) = DiscreteNonParametric(support, normalize!(1 ./ support .^ k))