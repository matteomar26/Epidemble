mutable struct ParametricModel{D,D2,Taux,M,M1,M2,O,Tλ,MO}
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
    obs_list::MO
    obs_range::UnitRange{Int64}
    
end

function ParametricModel(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution, obs_range=T:T)
    μ = fill(one(λi) / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    belief = fill(zero(λi), 0:T+1, 0:T+1, N)
    ν = fill(zero(λi), 0:T+1, 0:T+1, 0:T+1, 0:2)
    Paux = fill(zero(λi), 0:1, 0:2)
    Λ = OffsetArray([t <= 0 ? one(λi) : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    ParametricModel(T, γp, λp,γi, λi,Paux, μ, belief, ν,fr, dilution, distribution, residual(distribution), Λ,fill(false,N),obs_range)
end



function update_lam!(M,F::ComplexF64,eta)
    @unpack T,Λ = M
    ∂F = F.im / M.λi.im
    dλ = - sign(∂F) * eta * M.λi.re #SignDescender
    M.λi = clamp(M.λi.re + dλ,0.0,0.99) + im * M.λi.im
    Λ .= OffsetArray([t <= 0 ? 1.0 : (1-M.λi)^t for t = -T-2:T+1], -T-2:T+1)
end

function update_gam!(M)
    M.γi = (sum(M.belief[0,:,:]) / popsize(M)).re
end

function avg_err(b)
    N = size(b,3)
    T = size(b,1) - 2
    avg_bel = reshape(sum(sum(b,dims=2),dims=3) ./ (N*(T+2)),T+2) 
    err_bel = sqrt.(reshape(sum(sum(b .^ 2,dims=2),dims=3) ./ (N * (T+2)),T+2) .- avg_bel .^ 2) ./ sqrt(N)
    return avg_bel, err_bel
end

function FatTail(support::UnitRange{Int64}, exponent,a)
    p = 1 ./ (collect(support) .^ exponent .+ a)
    return DiscreteNonParametric(collect(support), p / sum(p))
end


function sweep!(M)
    e = 1 #edge counter
    N = popsize(M)
    F_itoj = zero(M.λi)
    Fψi = zero(M.λi)
    for l = 1:N
        # Extraction of disorder: state of individual i: xi0, delays: sij and sji
        xi0,sij,sji,d,oi,ci,ti_obs = rand_disorder(M.γp,M.λp,M.distribution,M.dilution,M.fr,M.obs_range)
        M.obs_list[l] = oi #this is stored for later estimation of AUC
        neighbours = rand(1:N,d)
        for m = 1:d
            res_neigh = [neighbours[1:m-1];neighbours[m+1:end]]
            calculate_ν!(M,res_neigh,xi0,oi,ci,ti_obs)
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
        zψi = calculate_belief!(M,l,neighbours,xi0,oi,ci,ti_obs)
        Fψi += (0.5 * d - 1) * log(zψi)  
    end
    return (Fψi - 0.5 * F_itoj) / N
end

function pop_dynamics(M; tot_iterations = 5, tol = 1e-10, eta = 0.0, infer_lam=false, infer_gam = false,nonlearn_iters=10)
    N, T, F = popsize(M), M.T, zero(M.λi)
    lam_window = zeros(10)
    gam_window = zeros(10)
    for iterations = 0:tot_iterations-1
        wflag = mod(iterations,10)+1
        lam_window[wflag] = M.λi |> real
        gam_window[wflag] = M.γi |> real
        avg_old, err_old = avg_err(M.belief |> real)
        F = sweep!(M)
        avg_new, err_new = avg_err(M.belief |> real)
        infer_lam = check_prior(iterations, infer_lam, lam_window, eta, nonlearn_iters)
        infer_gam = check_prior(iterations, infer_gam, gam_window, eta, nonlearn_iters)
        if (sum(abs.(avg_new .- avg_old) .<= (tol .+ 0.3 .* (err_old .+ err_new))) == length(avg_new)) 
            return F, iterations
        end
        (infer_lam) && (update_lam!(M,F,eta))
        (infer_gam) && (update_gam!(M))
    end
    return F,tot_iterations
end

function check_prior(iterations, infer, window, eta,nonlearn)
    if (iterations >= nonlearn && infer)
        avg = sum(window) / 10
        err = sqrt(sum(window .^ 2)/10 - avg ^ 2)
        if (err/avg <= eta) 
           return false
        end
    end
    return infer
end