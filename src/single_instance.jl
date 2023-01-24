using PyCall

@pyimport sib
function sibyl(N, T, Λ, O, γ, λ ; maxit = 400, tol = 1e-14)
    
    contacts = [(i-1,j-1,t, λ) for t in 1:T for (i,j,v) in zip(findnz(Λ.A)...)];
    obs = [[(i,-1,t) for t=1:T for i=0:N-1];
           [(i-1,s,t) for (i,s,t,p) in O]]
    sort!(obs, lt=((i1,s1,t1),(i2,s2,t2))->(t1<t2))
    prob_sus = 0.5
    prob_seed = γ
    pseed = prob_seed / (2 - prob_seed)
    psus = prob_sus * (1 - pseed)
    params = sib.Params(prob_r=sib.Exponential(mu=0), pseed=pseed, psus=psus,pautoinf=1e-10)
    f = sib.FactorGraph(contacts=contacts, observations=obs, params=params)
    sib.iterate(f, maxit=maxit,tol=tol)
    sib.iterate(f, maxit=maxit, damping=0.5, tol=tol)
    sib.iterate(f, maxit=maxit, damping=0.9, tol=tol)
    p_sib=[collect(n.bt) for n in f.nodes]
    m_sib = zeros(N, T)
    for i=1:N
        m_sib[i,1] = p_sib[i][1] 
        for t=2:T
            m_sib[i,t] = m_sib[i,t-1] + p_sib[i][t]
        end
    end 
    return m_sib
end

function tpr(xtrue, rank) 
    cumsum(xtrue[rank]) ./( cumsum(xtrue[rank])[end])
end


function fpr(xtrue, rank) 
    N = size(rank,1)
    return (range(1,N,length=N) .- cumsum(xtrue[rank])) ./ (range(1,N,length=N) .- cumsum(xtrue[rank]) )[end]
end


function ROC(xtrue, p)
    N = size(xtrue,1)
    rank = sortperm(p, rev=true)
    return fpr(xtrue, rank) , tpr(xtrue, rank)
end

    
function AUROC(ROC)
    N = size(ROC[1],1) 
    AU = 0
    for t = 1:N-1
        AU += ROC[2][t] * (ROC[1][t+1] - ROC[1][t])
    end
    return AU
end

function sample!(x, G, λi, γi) 
    N, T = size(x)
    for i=1:N
        x[i,1] = rand() < γi
    end
    for t = 2:T
        for i = 1:N
            if x[i,t-1] == 1
                x[i,t] = 1
                continue
            end
            r = 1
            for j in inedges(G,i)
                r *= 1 - λi * x[j.src,t-1] 
            end
            x[i,t] = rand() > r
        end
    end
    x
end