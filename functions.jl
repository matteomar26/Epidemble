function calculate_ν!(ν,μ,neighbours,xi0,T)
    if xi0 == 0
        for τi = 0:T+1
            for ti = 0:T+1
                #first we check consistency between
                # the planted time τi and the inferred 
                #time ti by checking the observation constraint            
                if ((ti==T+1)&&(τi<=T)) || ((ti<=T)&&(τi==T+1))  #if the observation is NOT satisfied
                    continue  # ν = 0
                end
                #Since they both depend on ti only,
                # we precaclulate the prior seed probability
                # of the individual and the value of phi function 
                # which is 1 if 0<ti<T+1 and 0 if ti=0,T+1
                seed = (ti==0 ? γ : (1-γ) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                #now we calculate the four products over
                # μ functions that we need to put in the
                # expression of ν. We call them m1,..,m4
                m1, m2, m3, m4 = ones(4)
                # we initialize the m's to one and then we 
                # loop a product over neighbours
                for k in neighbours 
                    m1 *= sum(μ[k,ti,1,τi,1:2])
                    m2 *= sum(μ[k,ti,0,τi,1:2])
                    m3 *= μ[k,ti,1,τi,2]
                    m4 *= μ[k,ti,0,τi,2]
                end
                #Now we have everything to calculate ν
                for tj=0:T+1                
                    ν[ti,tj,τi,1] = seed * (a[ti-tj-1] * m1 - phi * a[ti-tj] * m2)
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                    ν[ti,tj,τi,2] = ν[ti,tj,τi,1] + seed * (phi * a[ti-tj] * m4 - a[ti-tj-1] * m3)
                end
            end
        end
    else
        # We are now in the case in which the individual is 
        # the zero patient. In this case the computation of 
        # the ν function is a little bit different than before
        # so we separated the cases

        for τi = 0:T+1
            for ti = 0:T+1
                if ((ti==T+1)&&(τi<=T)) || ((ti<=T)&&(τi==T+1))  #if the observation is NOT satisfied
                    continue
                end
                #we can calculate ν now because it is constant
                # in σ and is nonzero only if tj=0

                #As before we pre-calculate ti-dependent quantities 
                seed = (ti==0 ? γ : (1-γ) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                # We perform the product over neighbours
                m1, m2 = ones(2)
                for k in neighbours                
                    m1 *= sum(μ[k,ti,1,τi,:])
                    m2 *= sum(μ[k,ti,0,τi,:])
                end
                #We calculate ν in the zero patient case
                ν[ti,0,τi,:] .= seed * (a[ti-1] * m1 - phi * a[ti] * m2)
            end
        end
    end
    if sum(ν) == 0
        println("sum-zero ν")
        return
    end
    ν ./= sum(ν);    
end


function update_μ!(μ,ν,Σ,l,sij,sji,T)
    for tj = 0:T+1
        for τj = 0:T+1
            #First of all we set to 0 the function we want to update
            #because later we want to sum over it
            μ[l,tj,:,τj,:] .= 0
            for ti = 0:T+1
                #we pre calculate the value of the summed part
                # so not to calculate it twice
                Γ = Σ[ti,tj,clamp(τj+sji-1,0,T+1),2] - Σ[ti,tj,clamp(τj-sij,0,T+1),2]+ν[ti,tj,clamp(τj+sji,0,T+1),1]+
                    Σ[ti,tj,T+1,0] - Σ[ti,tj,clamp(τj+sji,0,T+1),0]
                for c = 0:1
                    μ[l,tj,c,τj,0] += a[tj-ti-c] * Σ[ti,tj,clamp(τj-sij-1,0,T+1),2]
                    μ[l,tj,c,τj,1] += a[tj-ti-c] * Σ[ti,tj,clamp(τj-sij,0,T+1),2]
                    μ[l,tj,c,τj,2] += a[tj-ti-c] * Γ
                end
            end
        end
    end
    μ[l,:,:,:,:] ./= sum(μ[l,:,:,:,:]);
end