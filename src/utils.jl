function save_values!(inf_out,M,conv,count_obs)
    marg = M.belief |> real;
    marg2D = reshape(sum(marg,dims=3) ./ N, T+2,T+2)
    inf_out[1] = conv[2] #number of iterations
    inf_out[2:T+2] .= avgAUC(marg,M.obs_list,count_obs=count_obs)
    inf_out[T+3 : 2*T + 3] .= avgOverlap(marg)
    inf_out[2*T + 4 : 3*T + 4] .= L1(marg2D)
    inf_out[3*T + 5 : 4*T + 5] .= MSE(marg)
    #inf_out[4*T + 6] = conv[1] |> real #free energy 
    inf_out[4*T + 7] = M.λi |> real 
    inf_out[4*T + 8] = M.γi |> real
    #inf_out[4*T + 9] = (U - conv[1]) |> real #entropy
end



function inf_vs_dil_optimal(γ, λRange, N, T, degree_dist, fr , dilRange ; tot_iterations = 1, count_obs = false)
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 8) # 2 value for conv and fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λp = λRange[λcount]
        dilution = dilRange[dilcount]
        γi = γp = γ
        M = ParametricModel(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi, fr=fr, dilution=dilution, distribution=degree_dist) ;
        unif_initializ!(M)
        conv = logpop_dynamics(M, tot_iterations = tot_iterations)
        save_values!(@view(inf_out[λcount,dilcount,:]),M,conv,count_obs)
    end
    return inf_out
end

function fr_vs_dil_optimal(γ, λ, N, T, degree_dist, frRange , dilRange ; tot_iterations = 1, count_obs = false)
    inf_out = zeros(length(frRange),length(dilRange), 4*T + 8) # 2 value for conv and fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (frcount,dilcount) in collect(product(1:length(frRange),1:length(dilRange)))
        λi = λp = λ
        dilution = dilRange[dilcount]
        fr = frRange[frcount]
        γi = γp = γ
        M = ParametricModel(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi, fr=fr, dilution=dilution, distribution=degree_dist) ;
        unif_initializ!(M)
        conv = pop_dynamics(M, tot_iterations = tot_iterations)
        save_values!(@view(inf_out[frcount,dilcount,:]),M,conv,count_obs)
    end
    return inf_out
end


function inf_vs_dil_mismλ(γ, λRange, λp, N, T, degree_dist, fr , dilRange ; tot_iterations = 1, count_obs = true )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 8) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λRange[λcount]
        dilution = dilRange[dilcount]
        γi = γp = γ
        M = ParametricModel(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = tot_iterations, infer_lam=false, infer_gam=false)
        save_values!(@view(inf_out[λcount,dilcount,:]), M, conv, count_obs)
    end
    return inf_out
end

function inf_vs_dil_mismγ(λ, γRange, γp, N, T, degree_dist, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(γRange),length(dilRange), 4*T + 8) # 2 value for conv and Fe and 4(T+1) values for the AUC,overlap,L1,MSE
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


function inf_vs_dil_learnλ(dilRange, λRange, γ, λi, N, T, degree_dist, fr , dilution; tot_iterations=1, count_obs=true)
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 8) # 4 values for conv and Fe and lambdai and gammai and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λp = λRange[λcount]
        dilution = dilRange[dilcount]
        M = ParametricModel(N = N, T = T, γp = γ, λp = λp, γi=γ, λi=λi + 0.001im, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3,infer_lam=true, infer_gam=false)
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1,infer_lam=true, infer_gam=false)
        
        save_values!(@view(inf_out[λcount,dilcount,:]), M, conv,count_obs)
    end
    return inf_out
end


function inf_vs_gam_learnγ(dilRange, γRange, λ, γi, N, T, degree_dist, fr , dilution; tot_iterations=1)
    inf_out = zeros(length(γRange),length(dilRange), 4*T + 8) # 4 values for conv and Fe and lambdai and gammai and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (γcount,dilcount) in collect(product(1:length(γRange),1:length(dilRange)))
        γp = γRange[γcount]
        dilution = dilRange[dilcount]
        M = ParametricModel(N = N, T = T, γp = γp, λp = λ, γi=γi, λi=λ, fr=fr, dilution=dilution, distribution=degree_dist) ;
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3,infer_lam=false, infer_gam=true)
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1,infer_lam=false, infer_gam=true)
        save_values!(@view(inf_out[γcount,dilcount,:]), M, conv)
    end
    return inf_out
end

function gam_vs_lam_learn(γRange, λRange, dilution, N, T, degree_dist,  fr; λi = 0.5,γi = 0.5, tot_iterations = 1, count_obs = false, obs_range=T:T)
    inf_out = zeros(length(λRange),length(γRange), 4*T + 8) # 4 values for conv and Fe and lambdai and gammai and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,γcount) in collect(product(1:length(λRange),1:length(γRange)))
        λp = λRange[λcount]
        γp = γRange[γcount]
        M = ParametricModel(N = N, T = T, γp = γp, λp = λp, γi=γi, λi=λi + 0.001im, fr=fr, dilution=dilution, distribution=degree_dist,obs_range=obs_range) ;
        conv = pop_dynamics(M, tot_iterations = 5, eta = 0.3,infer_lam=true, infer_gam=true)
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.03,infer_lam=true, infer_gam=true)
        save_values!(@view(inf_out[λcount,γcount,:]), M, conv, count_obs)
    end
    return inf_out
end

function gam_vs_lam_optimal(γRange, λRange, dilution, N, T, degree_dist,  fr; tot_iterations = 1, count_obs = false, obs_range=T:T)
    inf_out = zeros(length(λRange),length(γRange), 4*T + 8) # 4 values for conv and Fe and lambdai and gammai and 4(T+1) values for the AUC,overlap,L1,MSE
    Threads.@threads for (λcount,γcount) in collect(product(1:length(λRange),1:length(γRange)))
        λp = λRange[λcount]
        γp = γRange[γcount]
        M = ParametricModel(N = N, T = T, γp = γp, λp = λp, γi=γp, λi=λp, fr=fr, dilution=dilution, distribution=degree_dist,obs_range=obs_range) ;
        conv = pop_dynamics(M, tot_iterations = tot_iterations, eta = 0.1,infer_lam=false, infer_gam=false)
        save_values!(@view(inf_out[λcount,γcount,:]), M, conv, count_obs)
    end
    return inf_out
end