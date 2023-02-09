function compute2Dmarg(M::Model)
    N, T = popsize(M), M.T
    omarg2D = fill(0.0, 0:T+1, 0:T+1)
    for i in 1:N
        for t in 0:T+1
            omarg2D[t, M.τbelief[i]] += M.belief[t,i]/N
        end
    end
    return omarg2D.parent
end


function diag(M::Model)
    marg2D = compute2Dmarg(M)
    T = size(marg2D,1) - 2
    sum([marg2D[t,t] for t=1:T+2])
end

function L1(M::Model)
    marg2D = compute2Dmarg(M)
    T = size(marg2D,1) - 2
    [sum(marg2D[1:t,1:t]) + sum(marg2D[t+1:end,t+1:end]) for t=1:T+1]
end


function MSE(M::Model)
    N, T = popsize(M), M.T
    sq_err = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for i = 1 : N
            τi = M.τbelief[i]
            xi_pt = (τi <= t)
            pi_inf = sum(M.belief[0:t,i]) 
            sq_err[t] += (pi_inf - xi_pt) ^ 2 
        end
    end
    sq_err ./= N
    return [sq_err[t] for t = 0:T]
end



function avgOverlap(M::Model)
    N, T = popsize(M), M.T
    overlap = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for i = 1 : N
            τi = M.τbelief[i]
            xi_pt = (τi <= t)
            xi_inf = sum(M.belief[0:t,i]) > 0.5
            overlap[t] += (xi_pt == xi_inf)
        end
    end
    overlap ./= N
    return [overlap[t] for t = 0:T]
end


function avgAUC(M::Model)
    N, T = popsize(M), M.T
    AUC = OffsetArrays.OffsetArray(zeros(T+1),-1)
    count = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for i = 1 :  N 
        for j = i + 1 : min(N,i+400)
            result = 0
            τi = M.τbelief[i]
            τj = M.τbelief[j]
            (τj <= τi) && continue 
            for t = τi : τj - 1
                count[t] += 1
                pi = sum(M.belief[0:t,i])
                pj = sum(M.belief[0:t,j])
                if pi ≈ pj
                    AUC[t] += 1/2
                elseif pi > pj
                    AUC[t] += 1
                end
            end
        end
    end
    AUC ./= count
    return [AUC[t] for t = 0:T]
end

function avg_ninf(M::Model)
    marg2D = compute2Dmarg(M)
    return sum(sum(marg2D,dims=2)'[1:end-1])  #number of infected expected by the inference scheme.
end

function save_values!(inf_out,marg,marg2D)
    inf_out[1] = avg_ninf(marg2D) #number of infected
    inf_out[2:T+2] .= avgAUC(marg)
    inf_out[T+3 : 2*T + 3] .= avgOverlap(marg)
    inf_out[2*T + 4 : 3*T + 4] .= L1(marg2D)
    inf_out[3*T + 5 : 4*T + 5] .= MSE(marg)
end


function inf_vs_dil_mism(γ, λRange, λp, N, T, degreetype, d, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 5) # 1 value for n_inf and 4(T+1) values for the AUC,overlap,L1,MSE
    degree_dist = makeDistrib(degreetype, d)
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λRange[λcount]
        dilution = dilRange[dilcount]
        γi = γp = γ
        marg = pop_dynamics(N, T, λp, λi, γp, γi, degree_dist, tot_iterations = tot_iterations, fr=fr, dilution=dilution)
        marg2D = reshape((sum(marg,dims=1)./ N),T+2,T+2);
        save_values!(@view(inf_out[λcount,dilcount,:]), marg, marg2D)
    end
    return inf_out
end



function inf_vs_dil_optimal(γ, λRange, N, T, degreetype, d, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 5) # 1 value for n_inf and 4(T+1) values for the AUC,overlap,L1,MSE
    degree_dist = makeDistrib(degreetype, d)
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λp = λRange[λcount]
        dilution = dilRange[dilcount]
        γi = γp = γ
        marg = pop_dynamics(N, T, λp, λi, γp, γi, degree_dist, tot_iterations = tot_iterations, fr=fr, dilution=dilution)
        marg2D = reshape((sum(marg,dims=1)./ N),T+2,T+2);
        save_values!(@view(inf_out[λcount,dilcount,:]),marg,marg2D)
    end
    return inf_out
end
