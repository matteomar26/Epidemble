#obs(ti, taui) = (ti == taui)
obs(ti, taui) = ((ti <= T) == (taui<=T))

function calculate_ν!(ν,μ,neighbours,xi0,T,γi,a)
    if xi0 == 0
        for τi = 0:T+1
            for ti = 0:T+1
                #first we check consistency between
                # the planted time τi and the inferred 
                #time ti by checking the observation constraint            
                #if ((ti==T+1)&&(τi<=T)) || ((ti<=T)&&(τi==T+1))  
                #if (ti <= T) != (τi <= T) #if the observation is NOT satisfied
                if !obs(ti,τi) #if the observation is NOT satisfied
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
                    ν[ti,tj,τi,1] = seed * (a(ti-tj-1) * m1 - phi * a(ti-tj) * m2)
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                    ν[ti,tj,τi,2] = ν[ti,tj,τi,1] + (τi<T+1) * seed * (phi * a(ti-tj) * m4 - a(ti-tj-1) * m3)
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
                if !obs(ti,0)  #if the observation is NOT satisfied
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
                ν[ti,tj,0,:] .= seed * (a(ti-1-tj) * m1 - phi * a(ti-tj) * m2)
            end
        end
    end
    if any(isnan.(ν))
        println("NaN in ν")
        return
    end
    if sum(ν) == 0
        println("sum-zero ν")
        return
    end    
    ν ./= sum(ν);    
end


function update_μ!(μ,ν,Σ,l,sij,sji,T,a)
    μ[l,:,:,:,:] .= 0
    for tj = 0:T+1
        for τj = 0:T+1
            #First of all we set to 0 the function we want to update
            #because later we want to sum over it
            for ti = 0:T+1
                #we pre calculate the value of the summed part
                # so not to calculate it twice
                Γ = Σ[ti,tj,min(τj+sji-1,T+1),2] - (τj-sij>=0)*Σ[ti,tj,max(τj-sij,0),2]+(τj+sji<=T+1)*ν[ti,tj,min(τj+sji,T+1),1]+
                    Σ[ti,tj,T+1,0] - Σ[ti,tj,min(τj+sji,T+1),0]
                for c = 0:1
                    μ[l,tj,c,τj,0] += a(tj-ti-c) * (τj-sij-1>=0) * Σ[ti,tj,max(τj-sij-1,0),2]
                    μ[l,tj,c,τj,1] += a(tj-ti-c) * (τj-sij>=0) * ν[ti,tj,max(τj-sij,0),2]
                    μ[l,tj,c,τj,2] += a(tj-ti-c) * Γ
                end
            end
        end
    end
    if any(isnan.(μ))
        println("NaN in μ")
        return
    end
    if sum(μ[l,:,:,:,:]) == 0
        println("sum-zero μ")
        return
    end    
    μ[l,:,:,:,:] ./= sum(@view μ[l,:,:,:,:]);
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

function rand_disorder(γp,λp)
    r = 1.0 / log(1-λp)
    sij = floor(Int,log(rand())*r) + 1
    sji = floor(Int,log(rand())*r) + 1
    xi0 = (rand() < γp);
    return xi0,sij,sji
end



function pop_dynamics(N, T, λp, λi, γp, γi, d; tot_iterations = 5000)
    μ = fill(1.0 / (6*(T+2)^2), 1:N, 0:T+1, 0:1, 0:T+1, 0:2)

    #Precalculation of the function a := (1-λ)^{tθ(t)}, 
    #useful for later (the function a appears
    #  in the inferred time factor node)

    pcache = [(1-λi)^t for t = 1:T+1]
    a(t) = t <= 0 ? 1.0 : pcache[t]
    
    
    ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    for iterations = 1:tot_iterations
        # Extraction of disorder: state of individual i: xi0, delays: sij and sji

        xi0,sij,sji = rand_disorder(γp,λp)

        # Initialization of ν=0
        ν .= 0.0

        #Extraction of d-1 μ's from population
        neighbours = rand(1:N,d-1)

        #Beginning of calculations: we start by calculating the ν: 
        calculate_ν!(ν,μ,neighbours,xi0,T,γi,a)

        # Now we use the ν vector just calculated to extract the new μ.
        # We extract a population index that we call "l".
        # We overwrite the μ in postition μ[l,:,:,:,:]
        l = rand(1:N);

        # First we calculate and store the cumulated of ν with respect to 
        # planted time, i.e. the third argument. We call Σ this cumulated 
        Σ = cumsum(ν,dims=3)
        update_μ!(μ,ν,Σ,l,sij,sji,T,a)     
    end
    
    p = fill(0.0, 0:T+1, 0:T+1, 0:T+1)
    marg = fill(0.0, 1:N, 0:T+1, 0:T+1)


    # Now we take out converged population of μ and use it to extract marginals.
    # First we extract two ν's and then we combine it in order to obtain a marginal.
    # In order to extract a ν we have to extract d-1 μ's. Therefore we extract two groups of 
    # d-1 μ's and from them we calculate the two ν's. We also have to extract disorder.
    for l = 1:N
        group1 = rand(1:N,d-1) #groups of neighbours 
        group2 = rand(1:N,d-1)

        xi0,sij,sji = rand_disorder(γp,λp) #planted disorder
        xj0 = (rand() < γp);

        ν1 = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
        ν2 = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)

        calculate_ν!(ν1,μ,group1,xi0,T,γi,a)
        calculate_ν!(ν2,μ,group2,xj0,T,γi,a)

        #Once the ν are calculated we have to cumulate with respect the third argument
        Σ = cumsum(ν2,dims=3)
        update_marginal!(marg,l,ν1,ν2,Σ,sij,sji,T)
    end
    return marg
end


function TracePhaseDiagram(γvalues, λvalues, N, T, d; tot_iterations=10000)
    p_infer = zeros(length(γvalues),length(λvalues))
    pr = Progress(length(γvalues) * length(λvalues))
    Threads.@threads for (γcount,λcount) in collect(product(1:length(γvalues),1:length(λvalues)))
        λi = λp = λvalues[λcount]
        γi = γp = γvalues[γcount]
        marg = pop_dynamics(N, T, λp, λi, γp, γi, d, tot_iterations = tot_iterations)
        marg2D = reshape((sum(marg,dims=1)./ N),T+2,T+2);
        # we sum over the trace of the 2D marginal to find the probability to infere correctly
        p_infer[γcount,λcount] = sum([marg2D[t,t] for t=1:T+2])
        ProgressMeter.next!(pr)
    end
    return p_infer
end

function AUCPhaseDiagram(γvalues, λvalues, N, T, d; tot_iterations=10000)
    p_infer = zeros(length(γvalues),length(λvalues))
    pr = Progress(length(γvalues) * length(λvalues))
    for (γcount,λcount) in collect(product(1:length(γvalues),1:length(λvalues)))
        λi = λp = λvalues[λcount]
        γi = γp = γvalues[γcount]
        marg = pop_dynamics(N, T, λp, λi, γp, γi, d, tot_iterations = tot_iterations)
        # we sum over the trace of the 2D marginal to find the probability to infere correctly
        p_infer[γcount,λcount] = avgAUC(0,marg)
        #ProgressMeter.next!(pr)#, showvalues=[(:F,sum(avF))])
    end
    return p_infer
end


function avgAUC(t, marg)
    N = size(marg,1)
    T = size(marg,2) - 2
    AUC = 0
    popcount = 0
    for l = 1:N-1
        result = 0
        count = 0
        for τi = 0:t
            for τj = t+1:T+1
                if (sum(marg[l, τi, :]) == 0 || sum(marg[l+1, τj, :]) == 0)
                    continue
                end
                count += 1
                pi = sum(marg[l, τi, 0:t])
                pj = sum(marg[l+1, τj, 0:t])
                result += (pi == pj) 
            end
        end
        if count == 0
            @assert result == 0
        else
            popcount += 1
            AUC += (result/count)
        end
    end
    @show AUC
    AUC/popcount
end