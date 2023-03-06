mutable struct ParametricModel{D,D2,Taux,M,M1,M2,O}
    T::Int
    γp::Float64
    λp::Float64
    γi::Float64
    λi::Float64
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
    eta::Float64
end

function ParametricModel(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution, eta=1e-2)
    ∂μ = fill(1.0 / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    μ = fill(1.0 / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    belief = fill(0.0, 0:T+1, 0:T+1, N)
    ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    ∂ν = fill(0.0, 0:T+1, 0:T+1, 0:T+1, 0:2)
    Paux = fill(0.0, 0:1, 0:2)
    Paux∂ = fill(0.0, 0:1, 0:2)
    ∂Λ = OffsetArray([t <= 0 ? 0.0 : t * ((1-λi)^(t-1)) for t = -T-2:T+1], -T-2:T+1)
    Λ = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    ParametricModel(T, γp, λp,γi, λi,Paux, Paux∂, μ, ∂μ, belief, ν,∂ν, fr, dilution, distribution, residual(distribution), Λ, ∂Λ, eta)
end

function update_μ!(M::ParametricModel,l,sij,sji)
    @unpack T,Λ,∂Λ,μ,∂μ,Paux,Paux∂,ν = M
    ∂μ[:,:,:,:,l] .= 0
    μ[:,:,:,:,l] .= 0
    Σ = cumsum(ν,dims=3)
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


function ∂zψi(M::ParametricModel,l,neighbours,xi0,oi) 
    @unpack T, γi, μ, ∂μ = M
    ∂z = 0.0
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
                ∂z += ξ  * seed * ( m1 - phi * m2) + ξ * (τi<T+1) * seed * (phi *  m4 -  m3)
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
                m1 += m1∂ 
                m2 += m2∂
            end
            ∂z += ξ * seed * ( m1 - phi *  m2)
        end
    end    
    return ∂z 
end

function ∂zψij(M::ParametricModel,neighbours,xi0,oi,sji)
    @unpack T,γi,Λ,∂Λ,μ,∂μ,∂ν = M
    ∂ν .= 0.0
    if xi0 == 0
        for τi = 1:T+1
            for ti = 0:T+1
                ξ = obs(M,ti,τi,oi)
                if ξ == 0.0 #if the observation is NOT satisfied
                    continue  # ν = 0
                end
                seed = (ti==0 ? γi : (1-γi) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                #first part of Leibniz rule: derivative of lambda
                m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0
                for k in neighbours 
                    m1 *= μ[ti,1,τi,1,k] + μ[ti,1,τi,2,k]
                    m2 *= μ[ti,0,τi,1,k] + μ[ti,0,τi,2,k]
                    m3 *= μ[ti,1,τi,2,k]
                    m4 *= μ[ti,0,τi,2,k]
                end
                for tj=0:T+1                
                    ∂ν[ti,tj,τi,1] = ξ  * seed * (∂Λ[ti-tj-1] * m1 - phi * ∂Λ[ti-tj] * m2)
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                    ∂ν[ti,tj,τi,2] = ∂ν[ti,tj,τi,1] + ξ * (τi<T+1) * seed * (phi * ∂Λ[ti-tj] * m4 - ∂Λ[ti-tj-1] * m3)
                end
                #second part: derivative of the message
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
                for tj=0:T+1
                    tmp = ξ  * seed * (Λ[ti-tj-1] * m1 - phi * Λ[ti-tj] * m2)
                    ∂ν[ti,tj,τi,1] += tmp
                    # We use the fact that ν for σ=2 is just ν at σ=1 plus a term
                    ∂ν[ti,tj,τi,2] += tmp + ξ * (τi<T+1) * seed * (phi * Λ[ti-tj] * m4 - Λ[ti-tj-1] * m3)
                end
            end
        end
    else
        #zero patient case

        for tj = 0:T+1
            for ti = 0:T+1
                ξ = obs(M,ti,0,oi)
                if ξ == 0.0  #if the observation is NOT satisfied
                    continue
                end
                
                seed = (ti==0 ? γi : (1-γi) )
                phi = (ti==0 || ti==T+1) ? 0 : 1
                
                #First part of Leibniz rule
                m1, m2 = 1.0, 1.0
                for k in neighbours                
                    m1 *= μ[ti,1,0,0,k] + μ[ti,1,0,1,k] + μ[ti,1,0,2,k]
                    m2 *= μ[ti,0,0,0,k] + μ[ti,0,0,1,k] + μ[ti,0,0,2,k]
                end
                ∂ν[ti,tj,0,:] .= ξ * seed * (∂Λ[ti-1-tj] * m1 - phi * ∂Λ[ti-tj] * m2)
                m1, m2 = 0.0, 0.0
                for j in neighbours
                    m1∂ = ∂μ[ti,1,0,0,j] + ∂μ[ti,1,0,1,j] + ∂μ[ti,1,0,2,j]
                    m2∂ = ∂μ[ti,0,0,0,j] + ∂μ[ti,0,0,1,j] + ∂μ[ti,0,0,2,j]
                    for k in neighbours
                        (k==j) && (continue)
                        m1∂ *= μ[ti,1,0,0,k] + μ[ti,1,0,1,k] + μ[ti,1,0,2,k]
                        m2∂ *= μ[ti,0,0,0,k] + μ[ti,0,0,1,k] + μ[ti,0,0,2,k]
                    end
                    m1 += m1∂
                    m2 += m2∂
                end
                ∂ν[ti,tj,0,:] .+= ξ * seed * (Λ[ti-1-tj] * m1 - phi * Λ[ti-tj] * m2)
            end
        end
    end
    if any(isnan,∂ν)
        println("NaN in ∂ν  at $(M.λi), $(M.dilution)")
        return
    end
    if sum(∂ν) == 0
        println("sum-zero ∂ν at $(M.λi), $(M.dilution), $(popsize(M)), $(M.fr)")
        return
    end  
    return edge_normalization(M,∂ν,sji)
end

function update_params!(M::ParametricModel,∂F)
    @unpack T,Λ,∂Λ,λi,eta = M
    @show ∂F
    M.λi -= eta * ∂F
    ∂Λ = OffsetArray([t <= 0 ? 0.0 : t * ((1-λi)^(t-1)) for t = -T-2:T+1], -T-2:T+1)
    Λ = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
end