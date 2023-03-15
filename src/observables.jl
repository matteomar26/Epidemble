function diag(marg2D)
    T = size(marg2D,1) - 2
    sum([marg2D[t,t] for t=1:T+2])
end

function L1(marg2D) 
    T = size(marg2D,1) - 2
    [sum(marg2D[1:t,1:t]) + sum(marg2D[t+1:end,t+1:end]) for t=1:T+1]
end


#=function L1bis(marg)  #old function used to debug the measures L1, MSE, Overlap
    N = size(marg,1)
    T = size(marg,2) - 2
    p_agree = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for l = 1 : N
            for τi = 0 : T + 1
                (sum(marg[l, :, τi]) == 0) && continue
                pi_inf = sum(marg[l,0:t,τi]) 
                p_agree[t] += (τi <= t ? pi_inf : 1 - pi_inf) #is the planted infected? if so take the prob of being infected 
            end
        end
    end
    p_agree ./= N
    return [p_agree[t] for t = 0:T]
end=#



function MSE(marg)
    N = size(marg,3)
    T = size(marg,1) - 2
    sq_err = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for l = 1 : N
            for τi = 0 : T + 1 
                (sum(marg[:, τi,l]) == 0) && continue
                xi_pt = (τi <= t)
                pi_inf = sum(marg[0:t,τi,l]) 
                sq_err[t] += (pi_inf - xi_pt) ^ 2 
            end
        end
    end
    sq_err ./= N
    return [sq_err[t] for t = 0:T]
end



function avgOverlap(marg)
    N = size(marg,3)
    T = size(marg,1) - 2
    overlap = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for l = 1 : N
            for τi = 0 : T + 1
                (sum(marg[:, τi, l]) == 0) && continue
                xi_pt = (τi <= t)
                xi_inf = sum(marg[0:t,τi,l]) > 0.5
                overlap[t] += (xi_pt == xi_inf)
            end
        end
    end
    overlap ./= N
    return [overlap[t] for t = 0:T]
end

function avgAUC(marg)
    N = size(marg,3)
    T = size(marg,1) - 2
    AUC = OffsetArrays.OffsetArray(zeros(T+1),-1)
    count = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for l = 1 :  N 
        for m = l + 1 : min(N,l+400)
            result = 0
            for τi = 0 : T 
                (sum(marg[:, τi, l]) == 0 ) && continue
                for τj = τi + 1 : T + 1
                    (sum(marg[:, τj, m]) == 0) && continue
                    # at the perfect inference for t=τj you would sum the diagonal
                    for t = τi : τj - 1
                        count[t] += 1
                        pi = sum(marg[0:t, τi, l])
                        pj = sum(marg[0:t, τj, m])
                        if pi ≈ pj
                            AUC[t] += 1/2
                        elseif pi > pj
                            AUC[t] += 1
                        end
                    end
                end
            end
        end
    end
    AUC ./= count
    return [AUC[t] for t = 0:T]
end

function avg_ninf(marg2D)
    return sum(sum(marg2D,dims=2)'[1:end-1])  #number of infected expected by the inference scheme.
end

function save_values!(inf_out,marg,conv)
    marg2D = reshape(sum(marg,dims=3) ./ N, T+2,T+2)
    inf_out[1] = conv[2] #number of iterations
    inf_out[2:T+2] .= avgAUC(marg)
    inf_out[T+3 : 2*T + 3] .= avgOverlap(marg)
    inf_out[2*T + 4 : 3*T + 4] .= L1(marg2D)
    inf_out[3*T + 5 : 4*T + 5] .= MSE(marg)
    inf_out[4*T + 6] = conv[1] |> real #free energy 
end


function inf_vs_dil_mism(γ, λRange, λp, N, T, degree_dist, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 6) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λRange[λcount]
        dilution = dilRange[dilcount]
        γi = γp = γ
        M = Model(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = tot_iterations)
        marg = M.belief;
        save_values!(@view(inf_out[λcount,dilcount,:]), marg, conv)
    end
    return inf_out
end

function inf_vs_gam_learn(γRange, λRange, γi, λi, N, T, degree_dist, fr , dil ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(γRange), 4*T + 6) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,γcount) in collect(product(1:length(λRange),1:length(γRange)))
        λp = λRange[λcount]
        dilution = dil
        γp = γRange[γcount]
        M = ParametricModel(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi + 0.001im, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3)
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1)
        marg = M.belief |> real;
        save_values!(@view(inf_out[λcount,γcount,:]), marg, conv)
    end
    return inf_out
end


function inf_vs_dil_mismγ(λ, γRange, γp, N, T, degree_dist, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(γRange),length(dilRange), 4*T + 6) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (γcount,dilcount) in collect(product(1:length(γRange),1:length(dilRange)))
        γi = γRange[γcount]
        dilution = dilRange[dilcount]
        λi = λp = λ
        M = Model(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = tot_iterations)
        marg = M.belief;
        save_values!(@view(inf_out[γcount,dilcount,:]), marg, conv)
    end
    return inf_out
end


function inf_vs_dil_optimal(γ, λRange, N, T, degree_dist, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 6) # 2 value for conv and fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λp = λRange[λcount]
        dilution = dilRange[dilcount]
        γi = γp = γ
        M = Model(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = tot_iterations)
        marg = M.belief;
        save_values!(@view(inf_out[λcount,dilcount,:]),marg,conv)
    end
    return inf_out
end
