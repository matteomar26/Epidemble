function diag(marg2D)
    T = size(marg2D,1) - 2
    sum([marg2D[t,t] for t=1:T+2])
end

function L1(marg2D) 
    T = size(marg2D,1) - 2
    [sum(marg2D[1:t,1:t]) + sum(marg2D[t+1:end,t+1:end]) for t=1:T+1]
end


#=function L1bis(marg)
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
    N = size(marg,1)
    T = size(marg,2) - 2
    sq_err = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for l = 1 : N
            for τi = 0 : T + 1 
                (sum(marg[l, :, τi]) == 0) && continue
                xi_pt = (τi <= t)
                pi_inf = sum(marg[l,0:t,τi]) 
                sq_err[t] += (pi_inf - xi_pt) ^ 2 
            end
        end
    end
    sq_err ./= N
    return [sq_err[t] for t = 0:T]
end



function avgOverlap(marg)
    N = size(marg,1)
    T = size(marg,2) - 2
    overlap = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for t = 0 : T 
        for l = 1 : N
            for τi = 0 : T + 1
                (sum(marg[l, :, τi]) == 0) && continue
                xi_pt = (τi <= t)
                xi_inf = sum(marg[l,0:t,τi]) > 0.5
                overlap[t] += (xi_pt == xi_inf)
            end
        end
    end
    overlap ./= N
    return [overlap[t] for t = 0:T]
end

function avgAUC(marg)
    N = size(marg,1)
    T = size(marg,2) - 2
    AUC = OffsetArrays.OffsetArray(zeros(T+1),-1)
    count = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for l = 1 :  N 
        for m = l + 1 : min(N,l+400)
            result = 0
            for τi = 0 : T 
                (sum(marg[l, :, τi]) == 0 ) && continue
                for τj = τi + 1 : T + 1
                    (sum(marg[m, :, τj]) == 0) && continue
                    # at the perfect inference for t=τj you would sum the diagonal
                    for t = τi : τj - 1
                        count[t] += 1
                        pi = sum(marg[l, 0:t, τi])
                        pj = sum(marg[m, 0:t, τj])
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


function inf_vs_dil_mism(γ, λRange, λp, N, T, degreetype, d, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 5) # 1 value for pdiag and 4(T+1) values for the AUC,overlap,L1,MSE
    degree_dist = makeDistrib(degreetype, d)
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λRange[λcount]
        γi = γp = γ
       # @show λi λp
        dilution = dilRange[dilcount]
        marg = pop_dynamics(N, T, λp, λi, γp, γi, degree_dist, tot_iterations = tot_iterations, fr=fr, dilution=dilution)
        marg2D = reshape((sum(marg,dims=1)./ N),T+2,T+2);
        # we sum over the trace of the 2D marginal to find the probability to infere correctly
        inf_out[λcount,dilcount,1] = sum([marg2D[t,t] for t=1:T+2])
        inf_out[λcount,dilcount,2:T+2] .= avgAUC(marg)
        inf_out[λcount,dilcount, T+3 : 2*T + 3] .= avgOverlap(marg)
        inf_out[λcount,dilcount, 2*T + 4 : 3*T + 4] .= L1(marg2D)
        inf_out[λcount,dilcount, 3*T + 5 : 4*T + 5] .= MSE(marg)
        #ProgressMeter.next!(pr)
    end
    return inf_out
end



function inf_vs_dil(γ, λRange, N, T, degreetype, d, fr , dilRange ; tot_iterations = 1 )
    inf_out = zeros(length(λRange),length(dilRange), 4*T + 5) # 1 value for pdiag and 4(T+1) values for the AUC,overlap,L1,MSE
    degree_dist = makeDistrib(degreetype, d)
    Threads.@threads for (λcount,dilcount) in collect(product(1:length(λRange),1:length(dilRange)))
        λi = λp = λRange[λcount]
        γi = γp = γ
       # @show λi λp
        dilution = dilRange[dilcount]
        marg = pop_dynamics(N, T, λp, λi, γp, γi, degree_dist, tot_iterations = tot_iterations, fr=fr, dilution=dilution)
        marg2D = reshape((sum(marg,dims=1)./ N),T+2,T+2);
        # we sum over the trace of the 2D marginal to find the probability to infere correctly
        inf_out[λcount,dilcount,1] = sum([marg2D[t,t] for t=1:T+2])
        inf_out[λcount,dilcount,2:T+2] .= avgAUC(marg)
        inf_out[λcount,dilcount, T+3 : 2*T + 3] .= avgOverlap(marg)
        inf_out[λcount,dilcount, 2*T + 4 : 3*T + 4] .= L1(marg2D)
        inf_out[λcount,dilcount, 3*T + 5 : 4*T + 5] .= MSE(marg)
        #ProgressMeter.next!(pr)
    end
    return inf_out
end
