mutable struct ParametricModel{D,D2,Taux,M,M1,M2,O,Tλ,MO,Tsympt}
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
    distribution::D
    residual::D2
    Λ::O
    obs_list::MO
    obs_range::UnitRange{Int64}
    field::Float64
    p_sympt_pla::Float64
    p_test_pla::Float64
    p_sympt_inf::Tsympt
    p_test_inf::Float64
end

function ParametricModel(; N, T, γp, λp, γi=γp, λi=λp, fr=0.0, dilution=0.0, distribution, obs_range=T:T,field=0.0,p_sympt_pla=0.0,p_test_pla=1-dilution,p_sympt_inf=p_sympt_pla,p_test_inf=p_test_pla)
    μ = fill(one(λi * p_sympt_inf) / (6*(T+2)^2), 0:T+1, 0:1, 0:T+1, 0:2, 1:N)
    belief = fill(zero(λi * p_sympt_inf), 0:T+1, 0:T+1, N)
    ν = fill(zero(λi * p_sympt_inf), 0:T+1, 0:T+1, 0:T+1, 0:2)
    Paux = fill(zero(λi * p_sympt_inf), 0:1, 0:2)
    Λ = OffsetArray([t <= 0 ? one(λi) : (1-λi)^t for t = -T-2:T+1], -T-2:T+1)
    ParametricModel(T, γp, λp,γi, λi,Paux, μ, belief, ν,fr, distribution, residual(distribution), Λ,fill(false,N),obs_range,field,p_sympt_pla,p_test_pla,p_sympt_inf,p_test_inf)
end



function update_lam!(M,F::ComplexF64,eta)
    @unpack T,Λ = M
    ∂F = F.im / M.λi.im
    dλ = - sign(∂F) * eta * M.λi.re #SignDescender
    M.λi = clamp(M.λi.re + dλ,0.0,0.99) + im * M.λi.im
    Λ .= OffsetArray([t <= 0 ? 1.0 : (1-M.λi)^t for t = -T-2:T+1], -T-2:T+1)
end

function update_sympt!(M,F::ComplexF64,eta)
    @unpack T,Λ = M
    ∂F = F.im / M.p_sympt_inf.im
    dsympt = - sign(∂F) * eta * M.p_sympt_inf.re #SignDescender
    M.p_sympt_inf = clamp(M.p_sympt_inf.re + dsympt,0.0,0.99) + im * M.p_sympt_inf.im
end

function update_gam!(M)
    M.γi = (sum(M.belief[0,:,:]) / popsize(M)).re
end

#=function avg_err(b)
    N = size(b,3)
    T = size(b,1) - 2
    avg_bel = reshape(sum(sum(b,dims=2),dims=3) ./ (N*(T+2)),T+2) 
    err_bel = sqrt.(reshape(sum(sum(b .^ 2,dims=2),dims=3) ./ (N * (T+2)),T+2) .- avg_bel .^ 2) ./ sqrt(N)
    return avg_bel, err_bel
end=#

function avg_err(b)
    N = size(b,3)
    T = size(b,1) - 2
    avg_bel = reshape(sum(b, dims=3) ./ N, T+2, T+2) 
    err_bel = sqrt.(reshape(sum(b .^ 2, dims=3) ./ N, T+2, T+2) .- avg_bel .^ 2) ./ sqrt(N)
    return avg_bel, err_bel
end

function FatTail(support::UnitRange{Int64}, exponent,a)
    p = 1 ./ (collect(support) .^ exponent .+ a)
    return DiscreteNonParametric(collect(support), p / sum(p))
end


function sweep!(M)
    e = 1 #edge counter
    N = popsize(M)
    F_itoj = zero(M.λi * M.p_sympt_inf)
    Fψi = zero(M.λi * M.p_sympt_inf)
    for l = 1:N
        # Extraction of disorder: state of individual i: xi0, delays: sij and sji
        xi0,sij,sji,d,oi,sympt,ci,ti_obs = rand_disorder(M,M.distribution)
        M.obs_list[l] = oi #this is stored for later estimation of AUC
        neighbours = rand(1:N,d)
        for m = 1:d
            res_neigh = [neighbours[1:m-1];neighbours[m+1:end]]
            calculate_logν!(M,res_neigh,xi0,oi,sympt,ci,ti_obs)
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
        zψi = calculate_logbelief!(M,l,neighbours,xi0,oi,sympt,ci,ti_obs)
        Fψi += (0.5 * d - 1) * log(zψi) 
    end
    
    return (Fψi - 0.5 * F_itoj) / N 
end

#=function pop_dynamicsObsolete(M; tot_iterations = 5, tol = 1e-10, eta = 0.0, infer_lam=false, infer_gam = false,infer_sympt=false,nonlearn_iters=10)
    N, T, F = popsize(M), M.T, zero(M.λi * M.p_sympt_inf)
    lam_window = zeros(10)
    gam_window = zeros(10)
    sympt_window = zeros(10)
    for iterations = 0:tot_iterations-1
        wflag = mod(iterations,10)+1
        lam_window[wflag] = M.λi |> real
        gam_window[wflag] = M.γi |> real
        sympt_window[wflag] = M.p_sympt_inf |> real
        avg_old, err_old = avg_err(M.belief |> real)
        F = sweep!(M)
        avg_new, err_new = avg_err(M.belief |> real)
        infer_lam = check_prior(iterations, infer_lam, lam_window, eta, nonlearn_iters)
        infer_gam = check_prior(iterations, infer_gam, gam_window, eta, nonlearn_iters)
        infer_sympt = check_prior(iterations, infer_sympt, sympt_window, eta, nonlearn_iters)
        converged = check_convergenceObsolete(avg_new, err_new, avg_old, err_old, tol) 
        #@show converged
        #@show abs.(avg_new .- avg_old)
        #@show (err_old .+ err_new)
        if converged  & !infer_lam & !infer_gam & !infer_sympt #if we don't have to infer 
            return F, iterations+1
        end
        (infer_lam) && (update_lam!(M,F,eta))
        (infer_gam) && (update_gam!(M))
        (infer_sympt) && (update_sympt!(M,F,eta))
    end
    return F |> real , tot_iterations
end=#


function pop_dynamics(M; tot_iterations = 5, tol = 1e-10, eta = 0.0, infer_lam=false, infer_gam = false,infer_sympt=false,nonlearn_iters=10)
    N, T, F = popsize(M), M.T, zero(M.λi * M.p_sympt_inf)
    F_window = zeros(10)
    converged = false
    lam_window = zeros(10)
    gam_window = zeros(10)
    sympt_window = zeros(10)
    for iterations = 0:tot_iterations-1
        wflag = mod(iterations,10)+1
        lam_window[wflag] = M.λi |> real
        gam_window[wflag] = M.γi |> real
        sympt_window[wflag] = M.p_sympt_inf |> real
        F = sweep!(M) 
        F_window[mod(iterations,length(F_window))+1] = (F |> real)
        infer_lam = check_prior(iterations, infer_lam, lam_window, eta, nonlearn_iters)
        infer_gam = check_prior(iterations, infer_gam, gam_window, eta, nonlearn_iters)
        infer_sympt = check_prior(iterations, infer_sympt, sympt_window, eta, nonlearn_iters)
        (iterations > length(F_window)) && (converged = check_convergence(F_window,2/sqrt(N)))
        if converged  & !infer_lam & !infer_gam & !infer_sympt #if we don't have to infer 
            return F_window |> real, iterations+1
        end
        (infer_lam) && (update_lam!(M,F,eta))
        (infer_gam) && (update_gam!(M))
        (infer_sympt) && (update_sympt!(M,F,eta))
    end
    return F_window |> real , tot_iterations
end



function check_prior(iterations, infer, window, eta,nonlearn)
    if (iterations >= nonlearn && infer) # if we need to infer + we we have iterated more than the nonlearn treshold
        avg = sum(window) / 10
        err = sqrt(sum(window .^ 2)/10 - avg ^ 2)
        if (err/avg <= eta) 
           return false
        end
    end
    return infer
end

function check_convergence(window,tol)
    l = length(window)
    avg = sum(window) / l
    variance = sum(window .^ 2)/l - avg ^ 2
    err = abs(variance) > 1e-15 ? sqrt(variance) : 0.0
    if (err <= tol) 
        return true
    else 
        return false
    end
end

#=function check_convergenceObsolete(avg_new, err_new, avg_old, err_old, tol)
    prod(abs.(avg_new .- avg_old) .<= (tol .+ (err_old .+ err_new)))
end=#

function prior_entropy(M)
    N = popsize(M)
    s = 0
    r = 1.0 / log(1-M.λp)
    T = M.T
    planted_times = rand(0:T+1,N)
    for iters = 1:10
        s = 0
        for i = 1:N
            xi0 = (rand() < M.γp); #zero patient
            d = rand(M.distribution) # number of neighbours
            neighb = rand(1:N,d)
            #fill the entering delays
            delays_out = zeros(d)
            for j in 1:d
                delays_out[j] = floor(Int,log(rand())*r) + 1 
            end
            #compute the new planted time
            ti = Int((!xi0) * min(minimum(planted_times[neighb] .+ delays_out),T+1)) 
            S1 = S2 = 0
            for j in 1:d
                tj = planted_times[neighb[j]]
                #@show tj
                theta_ij = ((ti - tj - 1) >= 0 )
                S1 += theta_ij ? (ti - tj - 1) : 0
                S2 += theta_ij
            end
            #@show ti, S1, S2, psi(M,ti,S1,S2)
            s -= log(psi(M,ti,S1,S2))
            planted_times[i] = ti
        end
    end
    return s/N,planted_times
end