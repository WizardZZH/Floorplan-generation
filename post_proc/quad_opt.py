from cvxopt  import solvers, matrix 
import numpy as np

def is_connected(idx_1,idx_2,com_emb):
    for room in com_emb:
        for rec in room:
            if idx_1 in rec and idx_2 in rec:
                id_1 = rec.index(idx_1)
                id_2 = rec.index(idx_2)
                if id_1>id_2:
                    id_1,id_2 = id_2,id_1
                if  id_2-id_1==1:
                    return 1
                if id_1==0 and id_2==len(rec)-1:
                    return 1
    return 0


def quad_prog(coor,order,com_emb):
    delta = -10
    length = coor.shape[0]
    P = matrix(np.eye(length))
    q = matrix(coor*(-1))

    G = []
    A = []
    for i in range(length-1):
        for j in range(i+1,length):
            idx_t =  i
            idx_n =  j
            equ = np.zeros(length)
            if order[idx_t]>=0 and order[idx_n]>=0:
                if order[idx_t]== order[idx_n]:
                    c = is_connected(idx_t,idx_n,com_emb)
                    if c==1:
                        equ[idx_t] = 1
                        equ[idx_n] = - 1
                        A.append(equ)
                else:
                    if  order[idx_t]> order[idx_n]:
                        equ[idx_t] = -1
                        equ[idx_n] = 1
                    if  order[idx_t] < order[idx_n]:
                        equ[idx_t] = 1
                        equ[idx_n] = -1
                    G.append(equ)

    if len(G) <=3:
        return 0,1
    h = matrix(np.ones(len(G))*delta)
    b = matrix(np.zeros(len(A)))
    G = matrix(np.array(G))
    A = matrix(np.array(A))
    #print(P,q,G,h,A,b)
    sol = solvers.qp(P,q,G,h,A,b)
    sol_x = np.array(sol['x']).reshape(-1).tolist()
    return sol_x,0