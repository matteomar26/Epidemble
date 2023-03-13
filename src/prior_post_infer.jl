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
    Λ = OffsetArray([t <= 0 ? 1.0 : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    ParametricModel(T, γp, λp,γi, λi,Paux, μ, belief, ν,fr, dilution, distribution, residual(distribution), Λ)
end


function update_params!(M,F::Float64,eta)
end

function update_params!(M,F::ComplexF64,eta)
    @unpack T,Λ = M
    ∂F = F.im / M.λi.im
    M.λi = clamp(M.λi.re - eta * ∂F,0.0,0.99) + im * M.λi.im
    Λ .= OffsetArray([t <= 0 ? 1.0 : (1-M.λi)^t for t = -T-2:T+1], -T-2:T+1)
end