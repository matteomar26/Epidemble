using Distributions,UnPack,OffsetArrays


popsize(M) = size(M.belief,3)
obs(M, ti, τi, oi) = oi ? (((ti <= M.T) == (τi <= M.T)) ? 1.0 - M.fr : M.fr) : 1.0


function calculate_ν!(M,neighbours,xi0,oi)
    @unpack T,γi,Λ,μ,ν = M
    if xi0 == 0
        for τi = 1:T+1
            for ti = 0:T+1
                #first we check consistency between
                # the planted time τi and the inferred 
                #time ti by checking the observation constraint
                ξ = obs(M,ti,τi,oi)
                if ξ == 0.0 #if the observation is NOT satisfied
                    continue  # ν = 0
                end
                #Since they both depend on ti only,
                # we precaclulate the prior seed probability
                # of the individual and the value of phi function 
                # which is 1 if 0<ti<T+1 and 0 if ti=0,T+1
                seed = (ti==0 ? γi : (1-γi) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                #now we calculate the four products over
                # μ functions that we need to put in the
                # expression of ν. We call them m1,..,m4
                m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0
                # we initialize the m's to one and then we 
                # loop a product over neighbours
                for k in neighbours 
                    m1 *= μ[ti,1,τi,1,k] + μ[ti,1,τi,2,k]
                    m2 *= μ[ti,0,τi,1,k] + μ[ti,0,τi,2,k]
                    m3 *= μ[ti,1,τi,2,k]
                    m4 *= μ[ti,0,τi,2,k]
                end
                #Now we have everything to calculate ν
                for tj=0:T+1                
                    ν[ti,tj,τi,1] = ξ  * seed * (Λ[ti-tj-1] * m1 - phi * Λ[ti-tj] * m2)
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                    ν[ti,tj,τi,2] = ν[ti,tj,τi,1] + ξ * (τi<T+1) * seed * (phi * Λ[ti-tj] * m4 - Λ[ti-tj-1] * m3)
                end
            end
        end
    else
        # We are now in the case in which the individual is 
        # the zero patient. In this case the computation of 
        # the ν function is a little bit different than before
        # so we separated the cases

        for tj = 0:T+1
            for ti = 0:T+1
                ξ = obs(M,ti,0,oi)
                if ξ == 0.0  #if the observation is NOT satisfied
                    continue
                end
                #we can calculate ν now because it is constant
                # in σ and is nonzero only if τi=0

                #As before we pre-calculate ti-dependent quantities 
                seed = (ti==0 ? γi : (1-γi) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                # We perform the product over neighbours
                m1, m2 = 1.0, 1.0
                for k in neighbours                
                    m1 *= μ[ti,1,0,0,k] + μ[ti,1,0,1,k] + μ[ti,1,0,2,k]
                    m2 *= μ[ti,0,0,0,k] + μ[ti,0,0,1,k] + μ[ti,0,0,2,k]
                end
                #We calculate ν in the zero patient case
                ν[ti,tj,0,:] .= ξ * seed * (Λ[ti-1-tj] * m1 - phi * Λ[ti-tj] * m2)
            end
        end
    end
    if any(isnan,ν)
        println("NaN in ν  at $(M.λi), $(M.dilution)")
        return
    end
    if sum(ν) == 0
        println("sum-zero ν at $(M.λi), $(M.dilution), $(popsize(M)), $(M.fr)")
        return
    end        
end


function calculate_belief!(M,l,neighbours,xi0,oi) 
    @unpack T, belief, γi, μ = M
    belief[:,:,l] .= 0
    if xi0 == 0
        for τi = 1:T+1
            for ti = 0:T+1
                #first we check consistency between
                # the planted time τi and the inferred 
                #time ti by checking the observation constraint
                ξ = obs(M,ti,τi,oi)
                if ξ == 0.0 #if the observation is NOT satisfied
                    continue  # ν = 0
                end
                #Since they both depend on ti only,
                # we precaclulate the prior seed probability
                # of the individual and the value of phi function 
                # which is 1 if 0<ti<T+1 and 0 if ti=0,T+1
                seed = (ti==0 ? γi : (1-γi) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                #now we calculate the four products over
                # μ functions that we need to put in the
                # expression of ν. We call them m1,..,m4
                m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0
                # we initialize the m's to one and then we 
                # loop a product over neighbours
                for k in neighbours 
                    m1 *= μ[ti,1,τi,1,k] + μ[ti,1,τi,2,k]
                    m2 *= μ[ti,0,τi,1,k] + μ[ti,0,τi,2,k]
                    m3 *= μ[ti,1,τi,2,k]
                    m4 *= μ[ti,0,τi,2,k]
                end
                #Now we have everything to calculate ν
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                belief[ti,τi,l] = ξ  * seed * ( m1 - phi * m2) + ξ * (τi<T+1) * seed * (phi *  m4 -  m3)
            end
        end
    else
        # We are now in the case in which the individual is 
        # the zero patient. In this case the computation of 
        # the ν function is a little bit different than before
        # so we separated the cases

        for ti = 0:T+1
            ξ = obs(M,ti,0,oi)
            if ξ == 0.0  #if the observation is NOT satisfied
                continue
            end
            #we can calculate ν now because it is constant
            # in σ and is nonzero only if τi=0

            #As before we pre-calculate ti-dependent quantities 
            seed = (ti==0 ? γi : (1-γi) )
            phi = (ti==0 || ti==T+1) ? 0 : 1
            # We perform the product over neighbours
            m1, m2 = 1.0, 1.0
            for k in neighbours                
                m1 *= μ[ti,1,0,0,k] + μ[ti,1,0,1,k] + μ[ti,1,0,2,k]
                m2 *= μ[ti,0,0,0,k] + μ[ti,0,0,1,k] + μ[ti,0,0,2,k]
            end
            #We calculate ν in the zero patient case
            belief[ti,0,l] = ξ * seed * ( m1 - phi *  m2)
        end
    end
    if any(isnan, @view belief[:,:,l])
        println("NaN in belief")
        return
    end
    S = sum(@view belief[:,:,l])
    if S == 0
        println("sum-zero belief  at $(M.λi), $(M.dilution)")
        return
    end    
    belief[:,:,l] ./= S
    return S
end




residual(d::Poisson) = d #residual degree of poiss distribution is poisson with same param
residual(d::Dirac) = Dirac(d.value - 1) #residual degree of rr distribution (delta) is a delta at previous vale
residual(d::DiscreteNonParametric) = DiscreteNonParametric(support(d) .- 1, (probs(d) .* support(d)) / sum(probs(d) .* support(d)))
residual(d::DiscreteUniform) = Dirac(0)

function rand_disorder(γp, λp, dist, dilution)
    r = 1.0 / log(1-λp)
    sij = floor(Int,log(rand())*r) + 1
    sji = floor(Int,log(rand())*r) + 1
    xi0 = (rand() < γp);
    d = rand(dist)
    oi = rand() > dilution 
    # oi = 1 if the particle is observed, oi = 0 if the particle is not observed 
    return xi0, sij, sji, d, oi
end


function edge_normalization(M,ν,sji)
    tmp = sum(sum(ν,dims=1),dims=2)
    norm = 0.0
    T = M.T
    for taui = 0:T+1
        #norm += max(0,taui-sji) * tmp[0,0,taui,0] +  tmp[0,0,taui,1] + min(T+1,T-taui+sji+1) * tmp[0,0,taui,2]
        norm += max(0,taui-sji) * tmp[0,0,taui,0] + (taui-sji >= 0) * tmp[0,0,taui,1] + (T+2 - max(taui-sji+1,0)) * tmp[0,0,taui,2]
    end
    return norm
end


function avg_err(M)
    N = popsize(M)
    avg_bel = reshape(sum(sum(M.belief,dims=2),dims=3) ./ (N*(M.T+2)),M.T+2) 
    err_bel = sqrt.(reshape(sum(sum(M.belief .^ 2,dims=2),dims=3) ./ (N * (M.T+2)),M.T+2) .- (avg_bel .^ 2)) ./ sqrt(popsize(M))
    return avg_bel, err_bel
end

FatTail(support,k) = DiscreteNonParametric(support, normalize!(1 ./ support .^ k, 1.0))

function pop_dynamics(M; tot_iterations = 5, tol = 1e-10)
    T = M.T
    N = popsize(M)
    F = 0.0
    ∂F = 0.0
    for iterations = 1:tot_iterations
        avg_old, err_old = avg_err(M)
        F_itoj = 0.0
        Fψi = 0.0
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
                update_μ!(M,ν,e,sij,sji)  
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

