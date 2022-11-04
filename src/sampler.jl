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