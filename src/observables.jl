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

function avgAUC(marg,obs_list; count_obs=true)
    N = size(marg,3)
    T = size(marg,1) - 2
    AUC = OffsetArrays.OffsetArray(zeros(T+1),-1)
    count = OffsetArrays.OffsetArray(zeros(T+1),-1)
    for l = 1 :  N 
        for m = l + 1 : min(N,l+800)
            if ((count_obs == false) && (obs_list[l] || obs_list[m]))
                continue
            else
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
    end
    AUC ./= count
    return [AUC[t] for t = 0:T]
end

function avg_ninf(marg2D)
    return sum(sum(marg2D,dims=2)'[1:end-1])  #number of infected expected by the inference scheme.
end

