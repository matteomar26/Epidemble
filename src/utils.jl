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
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3,infer_lam=true, infer_gam=true)        
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1,infer_lam=true, infer_gam=true)
        marg = M.belief |> real;
        save_values!(@view(inf_out[λcount,γcount,:]), marg, conv)
    end
    return inf_out
end

function inf_vs_gam_learnλ(dilRange, λRange, γ, λi, N, T, degree_dist, fr , dilution; tot_iterations=1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 6) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λp = λRange[λcount]
        dilution = dilRange[dilcount]
        M = ParametricModel(N = N, T = T, γp = γ, λp = λp, γi=γ, λi=λi + 0.001im, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3,infer_lam=true, infer_gam=false)
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1,infer_lam=true, infer_gam=false)
        marg = M.belief |> real;
        save_values!(@view(inf_out[λcount,dilcount,:]), marg, conv)
    end
    return inf_out
end


function inf_vs_gam_learnγ(dilRange, γRange, λ, γi, N, T, degree_dist, fr , dilution; tot_iterations=1)
    inf_out = zeros(length(γRange),length(dilRange), 4*T + 6) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (γcount,dilcount) in collect(product(1:length(γRange),1:length(dilRange)))
        γp = γRange[γcount]
        dilution = dilRange[dilcount]
        M = ParametricModel(N = N, T = T, γp = γp, λp = λ, γi=γi, λi=λ, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3,infer_lam=false, infer_gam=true)
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1,infer_lam=false, infer_gam=true)
        marg = M.belief |> real;
        save_values!(@view(inf_out[γcount,dilcount,:]), marg, conv)
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