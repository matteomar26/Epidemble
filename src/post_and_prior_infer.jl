struct ParametricModel{D,D2,Taux,M,M1,M2,O}
    T::Int
    γp::Float64
    λp::Float64
    γi::Float64
    λi::Float64
    Paux::Taux
    Paux∂::Taux
    μ::M
    ∂μ::M1
    belief::M2
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
    Paux = fill(0.0, 0:1, 0:2)
    Paux∂ = fill(0.0, 0:1, 0:2)
    ∂Λ = OffsetArray([t <= 0 ? 0.0 : t * ((1-λi)^(t-1)) for t = -T-2:T+1], -T-2:T+1)
    Λ = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    Model(T, γp, λp, γi, λi,Paux, Paux∂, μ, ∂μ, belief, fr, dilution, distribution, residual(distribution), Λ, ∂Λ)
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
