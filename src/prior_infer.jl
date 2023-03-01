function pop_dynamics_and_prior(M; tot_iterations = 5, tol = 1e-10, eta = 1e-2)
    T = M.T
    N = popsize(M)
    Paux = fill(0.0, 0:1, 0:2)
    #Precalculation of the function a := (1-λ)^{tθ(t)}, 
    #useful for later (the function a appears
    #  in the inferred time factor node)
    ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    F = 0.0
    for iterations = 1:tot_iterations
        avg_old, err_old = avg_err(M)
        F_itoj = 0.0
        Fψi = 0.0
        alpha = 0.0
        e = 1 #edge counter
        for l = 1:N
            # Extraction of disorder: state of individual i: xi0, delays: sij and sji
            xi0,sij,sji,d,oi = rand_disorder(M.γp,M.λp,M.distribution,M.dilution)
            neighbours = rand(1:N,d)
            for m = 1:d
                res_neigh = [neighbours[1:m-1];neighbours[m+1:end]]
                ν .= 0.0
                calculate_ν!(ν,M,res_neigh,xi0,oi)
                #from the un-normalized ν message it is possible to extract the orginal-message 
                #normalization z_i→j 
                # needed for the computation of the Bethe Free energy
                r = 1.0 / log(1-M.λp)
                sji = floor(Int,log(rand())*r) + 1
                sij = floor(Int,log(rand())*r) + 1
                F_itoj += log(edge_normalization(M,ν,sji))
                #Now we can normalize ν
                ν ./= edge_normalization(M,ν,sji)    
                # Now we use the ν vector just calculated to extract the new μ.
                # We overwrite the μ in postition μ[:,:,:,:,l]
                update_μ!(M,ν,e,sij,sji,Paux)  
                e = mod(e,N) + 1
            end
            zψi = calculate_belief!(M,l,neighbours,xi0,oi) 
            Fψi += (0.5 * d - 1) * log(zψi)   
        end
        F = (Fψi - 0.5 * F_itoj) / popsize(M)
        avg_new, err_new = avg_err(M)
        if sum(abs.(avg_new .- avg_old) .<= (tol .+ 0.3 .* (err_old .+ err_new))) == length(avg_new) 
            return F, iterations
        end
    end
    return F, tot_iterations   
end


function update_∂μ!(M,ν,l,sij,sji,P)
    @unpack T,Λ,μ = M
    μ[:,:,:,:,l] .= 0
    # First we calculate and store the cumulated of ν with respect to 
    # planted time, i.e. the third argument. We call Σ this cumulated 
    Σ = cumsum(ν,dims=3)
    @inbounds for tj = 0:T+1
        for τj = 0:T+1
            #First of all we set to 0 the function we want to update
            #because later we want to sum over it
            P .= 0.0
            for ti = 0:T+1
                #we pre calculate the value of the summed part
                # so not to calculate it twice
                Γ = Σ[ti,tj,min(τj+sji-1,T+1),2] - (τj-sij>=0)*Σ[ti,tj,max(τj-sij,0),2]+(τj+sji<=T+1)*ν[ti,tj,min(τj+sji,T+1),1]+
                    Σ[ti,tj,T+1,0] - Σ[ti,tj,min(τj+sji,T+1),0]
                for c = 0:1
                    P[c,0] += Λ[tj-ti-c] * (τj-sij-1>=0) * Σ[ti,tj,max(τj-sij-1,0),2]
                    P[c,1] += Λ[tj-ti-c] * (τj-sij>=0) * ν[ti,tj,max(τj-sij,0),2]
                    P[c,2] += Λ[tj-ti-c] * Γ
                end
            end
            μ[tj,:,τj,:,l] = P
        end
    end
    S = sum(@view μ[:,:,:,:,l])
    if S == 0.0
        println("sum-zero μ  at $(M.λi), $(M.dilution)")
        return
    end   
    if isnan(S)
        println("NaN in μ")
        return
    end
    #μ[:,:,:,:,l] ./= S; #in the original form the messages are not normalized, but 
    #@show S
end
