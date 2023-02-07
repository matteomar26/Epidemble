using Distributions

#=function obs(ti, taui; fr = 0.0, dilution = 0.0)
    if rand() >= dilution
        return (((ti <= T) == (taui<=T)) ? 1.0 - fr : fr)
    else
        return 1.0
    end
end=#

function obs(ti, taui, oi; fr = 0.0)
    if oi == 1
        return (((ti <= T) == (taui<=T)) ? 1.0 - fr : fr)
    else
        return 1.0
    end
end

function calculate_ν!(ν,μ,neighbours,xi0,T,γi,a,oi; fr = 0.0)
    if xi0 == 0
        for τi = 1:T+1
            for ti = 0:T+1
                #first we check consistency between
                # the planted time τi and the inferred 
                #time ti by checking the observation constraint
                ξ = obs(ti,τi,oi,fr=fr)
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
                    m1 *= μ[k,ti,1,τi,1] + μ[k,ti,1,τi,2]
                    m2 *= μ[k,ti,0,τi,1] + μ[k,ti,0,τi,2]
                    m3 *= μ[k,ti,1,τi,2]
                    m4 *= μ[k,ti,0,τi,2]
                end
                #Now we have everything to calculate ν
                for tj=0:T+1                
                    ν[ti,tj,τi,1] = ξ  * seed * (a[ti-tj-1] * m1 - phi * a[ti-tj] * m2)
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                    ν[ti,tj,τi,2] = ν[ti,tj,τi,1] + ξ * (τi<T+1) * seed * (phi * a[ti-tj] * m4 - a[ti-tj-1] * m3)
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
                ξ = obs(ti,0,oi,fr=fr)
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
                    m1 *= μ[k,ti,1,0,0] + μ[k,ti,1,0,1] + μ[k,ti,1,0,2]
                    m2 *= μ[k,ti,0,0,0] + μ[k,ti,0,0,1] + μ[k,ti,0,0,2]
                end
                #We calculate ν in the zero patient case
                ν[ti,tj,0,:] .= ξ * seed * (a[ti-1-tj] * m1 - phi * a[ti-tj] * m2)
            end
        end
    end
    if any(isnan,ν)
        println("NaN in ν")
        return
    end
    if sum(ν) == 0
        println("sum-zero ν")
        return
    end    
    ν ./= sum(ν);    
end


function calculate_belief!(b,μ,neighbours,xi0,T,γi,oi; fr = 0.0)
    b .= 0
    if xi0 == 0
        for τi = 1:T+1
            for ti = 0:T+1
                #first we check consistency between
                # the planted time τi and the inferred 
                #time ti by checking the observation constraint
                ξ = obs(ti,τi,oi,fr=fr)
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
                    m1 *= μ[k,ti,1,τi,1] + μ[k,ti,1,τi,2]
                    m2 *= μ[k,ti,0,τi,1] + μ[k,ti,0,τi,2]
                    m3 *= μ[k,ti,1,τi,2]
                    m4 *= μ[k,ti,0,τi,2]
                end
                #Now we have everything to calculate ν
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                b[ti,τi] = ξ  * seed * ( m1 - phi * m2) + ξ * (τi<T+1) * seed * (phi *  m4 -  m3)
            end
        end
    else
        # We are now in the case in which the individual is 
        # the zero patient. In this case the computation of 
        # the ν function is a little bit different than before
        # so we separated the cases

        for ti = 0:T+1
            ξ = obs(ti,0,oi,fr=fr)
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
                m1 *= μ[k,ti,1,0,0] + μ[k,ti,1,0,1] + μ[k,ti,1,0,2]
                m2 *= μ[k,ti,0,0,0] + μ[k,ti,0,0,1] + μ[k,ti,0,0,2]
            end
            #We calculate ν in the zero patient case
            b[ti,0] = ξ * seed * ( m1 - phi *  m2)
        end
    end
    if any(isnan,b)
        println("NaN in b")
        return
    end
    if sum(b) == 0
        println("sum-zero b")
        return
    end    
    b ./= sum(b);
end



function update_μ!(μ,ν,Σ,l,sij,sji,T,a,P)
    μ[l,:,:,:,:] .= 0
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
                    P[c,0] += a[tj-ti-c] * (τj-sij-1>=0) * Σ[ti,tj,max(τj-sij-1,0),2]
                    P[c,1] += a[tj-ti-c] * (τj-sij>=0) * ν[ti,tj,max(τj-sij,0),2]
                    P[c,2] += a[tj-ti-c] * Γ
                end
            end
            μ[l,tj,:,τj,:] = P
        end
    end
    S = sum(@view μ[l,:,:,:,:])
    if S == 0.0
        println("sum-zero μ")
        return
    end    
    if isnan(S)
        println("NaN in μ")
        return
    end
    μ[l,:,:,:,:] ./= S;
end

function update_marginal!(marg,l,ν1,ν2,Σ,sij,sji,T)
    marg[l,:,:] .= 0.0
    for ti = 0:T+1
        for τi = 0:T+1
            for tj = 0:T+1
                Γ = Σ[tj,ti,min(τi+sij-1,T+1),2]-(τi-sji>=0)*Σ[tj,ti,max(τi-sji,0),2]+(τi+sij<=T+1)*ν2[tj,ti,min(τi+sij,T+1),1]
                   +Σ[tj,ti,T+1,0]-Σ[tj,ti,min(τi+sij,T+1),0]
                marg[l,ti,τi] += ν1[ti,tj,τi,0]*(τi-sji-1>=0)*Σ[tj,ti,max(τi-sji-1,0),2] +
                              ν1[ti,tj,τi,1]*(τi-sji>=0)*ν2[tj,ti,max(τi-sji,0),2] + ν1[ti,tj,τi,2]*Γ 
            end
        end
    end
    marg[l,:,:] ./= sum(@view marg[l,:,:])
end

residual(d::Poisson) = d #residual degree of poiss distribution is poisson with same param
residual(d::Dirac) = Dirac(d.value - 1) #residual degree of rr distribution (delta) is a delta at previous vale
residual(d::DiscreteNonParametric) = DiscreteNonParametric(support(d) .- 1, (probs(d) .* support(d)) / sum(probs(d) .* support(d)))

function rand_disorder(γp, λp, dist)
    r = 1.0 / log(1-λp)
    sij = floor(Int,log(rand())*r) + 1
    sji = floor(Int,log(rand())*r) + 1
    xi0 = (rand() < γp);
    d = rand(dist)
    return xi0, sij, sji, d
end

function pop_dynamics(N, T, λp, λi, γp, γi, degree_dist; tot_iterations = 5, fr = 0.0, dilution = 0.0)
    μ = fill(1.0 / (6*(T+2)^2), 1:N, 0:T+1, 0:1, 0:T+1, 0:2)
    Paux = fill(0.0, 0:1, 0:2)
    res_dist = residual(degree_dist) #we calculate the distribution of the residual degree
    
    #Precalculation of the function a := (1-λ)^{tθ(t)}, 
    #useful for later (the function a appears
    #  in the inferred time factor node)
    a = Dict{Int,Float64}(zip(-T-2:T+1,[ t<=0 ? 1.0 : (1-λi)^t for t = -T-2:T+1]));
    
    ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    for iterations = 1:tot_iterations
        for l = 1:N
            # Extraction of disorder: state of individual i: xi0, delays: sij and sji

            xi0,sij,sji,d = rand_disorder(γp,λp,res_dist)

            # Initialization of ν=0
            ν .= 0.0
            #Extraction of d-1 μ's from population
            neighbours = rand(1:N,d)
            oi = rand() > dilution # oi = 1 if the particle is observed, oi = 0 if the particle is not observed 
            #Beginning of calculations: we start by calculating the ν: 
            calculate_ν!(ν,μ,neighbours,xi0,T,γi,a,oi,fr=fr)

            # Now we use the ν vector just calculated to extract the new μ.
            # We overwrite the μ in postition μ[l,:,:,:,:]

            # First we calculate and store the cumulated of ν with respect to 
            # planted time, i.e. the third argument. We call Σ this cumulated 
            Σ = cumsum(ν,dims=3)
            
            #then we call the update μ function
            update_μ!(μ,ν,Σ,l,sij,sji,T,a,Paux)     
        end
    end
    
    p = fill(0.0, 0:T+1, 0:T+1, 0:T+1)
    marg = fill(0.0, 1:N, 0:T+1, 0:T+1)

    # Now we take out converged population of μ and use it to extract marginals.
    for l = 1:N
        xi0,sij,sji,d = rand_disorder(γp,λp,degree_dist)
        neighbours = rand(1:N,d)
        oi = rand() > dilution # oi = 1 if the particle is observed, oi = 0 if the particle is not observed 
        calculate_belief!(view(marg,l,:,:),μ,neighbours,xi0,T,γi,oi; fr) 
    end
    return marg
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