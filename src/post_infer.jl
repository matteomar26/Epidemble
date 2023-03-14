mutable struct ParametricModel{D,D2,Taux,M,M1,M2,O,Tλ}
    T::Int
    γp::Float64
    λp::Float64
    γi::Float64
    λi::Tλ
    Paux::Taux
    μ::M
    belief::M2
    ν::M1
    fr::Float64
    dilution::Float64
    distribution::D
    residual::D2
    Λ::O
end

function ParametricModel(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution)
    μ = fill(one(λi) / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    belief = fill(zero(λi), 0:T+1, 0:T+1, N)
    ν = fill(zero(λi), 0:T+1, 0:T+1, 0:T+1, 0:2)
    Paux = fill(zero(λi), 0:1, 0:2)
    Λ = OffsetArray([t <= 0 ? one(λi) : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    ParametricModel(T, γp, λp,γi, λi,Paux, μ, belief, ν,fr, dilution, distribution, residual(distribution), Λ)
end


function update_params!(M,F::Float64,eta)
end

function update_params!(M,F::ComplexF64,eta)
    (eta == 0.0) && return
    @unpack T,Λ = M
    ∂F = F.im / M.λi.im
    M.λi = clamp(M.λi.re - eta * ∂F,0.0,0.99) + im * M.λi.im
    Λ .= OffsetArray([t <= 0 ? 1.0 : (1-M.λi)^t for t = -T-2:T+1], -T-2:T+1)
    M.γi = (sum(M.belief[0,:,:]) / popsize(M)).re
    @show M.λi , M.γi 
end

function avg_err(b)
    N = size(b,3)
    T = size(b,1) - 2
    avg_bel = reshape(sum(sum(b,dims=2),dims=3) ./ (N*(T+2)),T+2) 
    err_bel = sqrt.(reshape(sum(sum(b .^ 2,dims=2),dims=3) ./ (N * (T+2)),T+2) .- (avg_bel .^ 2)) ./ sqrt(N)
    return avg_bel, err_bel
end

FatTail(support,k) = DiscreteNonParametric(support, normalize!(1 ./ support .^ k, 1.0))

function pop_dynamics(M; tot_iterations = 5, tol = 1e-10,eta=1e-1)
    T = M.T
    N = popsize(M)
    F = zero(M.λi)
    Fψi = zero(M.λi)
    F_itoj = zero(M.λi)
    for iterations = 1:tot_iterations
        #avg_old, err_old = avg_err(M.belief |> real)
        F_itoj = zero(M.λi)
        Fψi = zero(M.λi)
        e = 1 #edge counter
        for l = 1:N
            # Extraction of disorder: state of individual i: xi0, delays: sij and sji
            xi0,sij,sji,d,oi = rand_disorder(M.γp,M.λp,M.distribution,M.dilution)
            neighbours = rand(1:N,d)
            for m = 1:d
                res_neigh = [neighbours[1:m-1];neighbours[m+1:end]]
                calculate_ν!(M,res_neigh,xi0,oi)
                #from the un-normalized ν message it is possible to extract the orginal-message 
                #normalization z_i→j 
                # needed for the computation of the Bethe Free energy
                r = 1.0 / log(1-M.λp)
                sij = floor(Int,log(rand())*r) + 1
                sji = floor(Int,log(rand())*r) + 1
                zψij = edge_normalization(M,M.ν,sji)
                F_itoj += log(zψij)
                #Now we can normalize ν
                M.ν ./= zψij    
                # Now we use the ν vector just calculated to extract the new μ.
                # We overwrite the μ in postition μ[:,:,:,:,l]
                update_μ!(M,e,sij,sji)  
                e = mod(e,N) + 1
            end
            zψi = calculate_belief!(M,l,neighbours,xi0,oi)
            Fψi += (0.5 * d - 1) * log(zψi)  
        end
        F = (Fψi - 0.5 * F_itoj) / popsize(M)
        #avg_new, err_new = avg_err(M.belief |> real)
        #if sum(abs.(avg_new .- avg_old) .<= (tol .+ 0.3 .* (err_old .+ err_new))) == length(avg_new) 
         #   return F, iterations
        #end
        update_params!(M,F,eta)
    end
    return F
end


