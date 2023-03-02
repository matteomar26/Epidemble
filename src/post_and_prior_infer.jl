struct ParametricModel{D,D2,Taux,M,M1,M2,O}
    T::Int
    γp::Float64
    λp::Float64
    infer_params::Vector{Float64}
    Paux::Taux
    Paux∂::Taux
    μ::M
    ∂μ::M
    belief::M2
    ν::M1
    ∂ν::M1
    fr::Float64
    dilution::Float64
    distribution::D
    residual::D2
    Λ::O
    ∂Λ::O
end

function ParametricModel(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution) 
    ∂μ = fill(1.0 / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    μ = fill(1.0 / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    belief = fill(0.0, 0:T+1, 0:T+1, N)
    ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    ∂ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    Paux = fill(0.0, 0:1, 0:2)
    Paux∂ = fill(0.0, 0:1, 0:2)
    ∂Λ = OffsetArray([t <= 0 ? 0.0 : t * ((1-λi)^(t-1)) for t = -T-2:T+1], -T-2:T+1)
    Λ = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    Model(T, γp, λp,[γi, λi],Paux, Paux∂, μ, ∂μ, belief, ν,∂ν, fr, dilution, distribution, residual(distribution), Λ, ∂Λ)
end

function update_μ!(M::ParametricModel,ν,l,sij,sji)
    @unpack T,Λ,∂Λ,μ,∂μ,Paux,Paux∂ = M
    ∂μ[:,:,:,:,l] .= 0
    μ[:,:,:,:,l] .= 0
    # First we calculate and store the cumulated of ν with respect to 
    # planted time, i.e. the third argument. We call Σ this cumulated 
    for tj = 0:T+1
        for τj = 0:T+1
            #First of all we set to 0 the function we want to update
            #because later we want to sum over it
            Paux .= 0.0
            Paux∂ .= 0.0
            for ti = 0:T+1
                #we pre calculate the value of the summed part
                # so not to calculate it twice
                Γ = Σ[ti,tj,min(τj+sji-1,T+1),2] - (τj-sij>=0)*Σ[ti,tj,max(τj-sij,0),2]+(τj+sji<=T+1)*ν[ti,tj,min(τj+sji,T+1),1]+
                    Σ[ti,tj,T+1,0] - Σ[ti,tj,min(τj+sji,T+1),0]
                for c = 0:1
                    Paux[c,0] += Λ[tj-ti-c] * (τj-sij-1>=0) * Σ[ti,tj,max(τj-sij-1,0),2]
                    Paux[c,1] += Λ[tj-ti-c] * (τj-sij>=0) * ν[ti,tj,max(τj-sij,0),2]
                    Paux[c,2] += Λ[tj-ti-c] * Γ
                    Paux∂[c,0] += ∂Λ[tj-ti-c] * (τj-sij-1>=0) * Σ[ti,tj,max(τj-sij-1,0),2]
                    Paux∂[c,1] += ∂Λ[tj-ti-c] * (τj-sij>=0) * ν[ti,tj,max(τj-sij,0),2]
                    Paux∂[c,2] += ∂Λ[tj-ti-c] * Γ                    
                end
            end
            μ[tj,:,τj,:,l] = Paux
            ∂μ[tj,:,τj,:,l] = Paux∂
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


function compute_∂zψi(M::ParametricModel,l,neighbours,xi0,oi) 
    @unpack T, γi, μ, ∂μ = M
    ∂zψi = 0.0
    if xi0 == 0
        for τi = 1:T+1
            for ti = 0:T+1
                ξ = obs(M,ti,τi,oi)
                if ξ == 0.0 #if the observation is NOT satisfied
                    continue  # ν = 0
                end
                seed = (ti==0 ? γi : (1-γi) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                m1, m2, m3, m4 = 0.0, 0.0, 0.0, 0.0
                for j in neighbours
                    m1∂ = ∂μ[ti,1,τi,1,j] + ∂μ[ti,1,τi,2,j]
                    m2∂ = ∂μ[ti,0,τi,1,j] + ∂μ[ti,0,τi,2,j]
                    m3∂ = ∂μ[ti,1,τi,2,j]
                    m4∂ = ∂μ[ti,0,τi,2,j]
                    for k in neighbours 
                        (k == j) && (continue)
                        m1∂ *= μ[ti,1,τi,1,k] + μ[ti,1,τi,2,k]
                        m2∂ *= μ[ti,0,τi,1,k] + μ[ti,0,τi,2,k]
                        m3∂ *= μ[ti,1,τi,2,k]
                        m4∂ *= μ[ti,0,τi,2,k]
                    end
                    m1 += m1∂ 
                    m2 += m2∂
                    m3 += m3∂
                    m4 += m4∂
                end
                ∂zψi += ξ  * seed * ( m1 - phi * m2) + ξ * (τi<T+1) * seed * (phi *  m4 -  m3)
            end
        end
    else
        # the zero patient case. 
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
            m1, m2 = 0.0, 0.0
            for j in neighbours
                m1∂ = ∂μ[ti,1,0,0,j] + ∂μ[ti,1,0,1,j] + ∂μ[ti,1,0,2,j]
                m2∂ = ∂μ[ti,0,0,0,j] + ∂μ[ti,0,0,1,j] + ∂μ[ti,0,0,2,j]
                for k in neighbours                
                    (k == j) && (continue)
                    m1∂ *= μ[ti,1,0,0,k] + μ[ti,1,0,1,k] + μ[ti,1,0,2,k]
                    m2∂ *= μ[ti,0,0,0,k] + μ[ti,0,0,1,k] + μ[ti,0,0,2,k]
                end
            end
            #We calculate ν in the zero patient case
            ∂zψi += ξ * seed * ( m1 - phi *  m2)
        end
    end    
    if ∂zψi == 0
        println("sum-zero partial deriv. belief  at $(M.λi), $(M.dilution)")
        return
    end    
    return ∂zψi 
end

function compute_∂zψij(M::ParametricModel,neighbours,xi0,oi)
    @unpack T,γi,Λ,μ,∂μ = M
    ∂zψij = 0.0
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
