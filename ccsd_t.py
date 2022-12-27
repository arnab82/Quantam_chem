import numpy as np
import time
start=time.time()
from ccsd_rhf import *

def cind_r1r2(p,q,r1,r2,nofe):
    #   arg: running indices p,q and r1, r2 are whether oo or ov or vo or vv.
    #   returns compound index according to the convenience.
    return (r2*(p-nofe*(p>=nofe)))+q-nofe*(q>=nofe)
for i in range(o):
    for a in range(v):
        D[cind_r1r2(i,a,o,v,nofe)]=F_so_oo[i][i]-F_so_vv[a][a]
print("the denominator MATRIX IS ",D,"\n\n")


def connected_triples_a():
    con_a=np.zeros([o,o,o,v,v,v])
    for i in range(o):
        for j in range(o):
            for k in range(o):
                for a in range(v):
                    for b in range(v):
                        for c in range(v):
                            for e in range(v):
                                con_a[i][j][k][a][b][c]+=td[j][k][a][e]*ASObasis_vovv[e][i][b][c]
    return con_a
con_a=connected_triples_a()
def connected_triples_b():
    con_b=np.zeros([o,o,o,v,v,v])
    for i in range(o):
        for j in range(o):
            for k in range(o):
                for a in range(v):
                    for b in range(v):
                        for c in range(v):
                            for m in range(o):
                                con_b[i][j][k][a][b][c]+=td[i][m][b][c]*ASObasis_ovoo[m][a][j][k]
    return con_b
con_b=connected_triples_b()

def disconnected():
    dis=np.zeros([o,o,o,v,v,v])
    for i in range(o):
        for j in range(o):
            for k in range(o):
                for a in range(v):
                    for b in range(v):
                        for c in range(v):
                            dis[i][j][k][a][b][c]+=ts[i][a]*ASObasis_oovv[j][k][b][c]
    return dis
dis=disconnected()
def connected_triples():
    tt_connected=np.zeros([o,o,o,v,v,v])
    for i in range(o):
        for j in range(o):
            for k in range(o):
                for a in range(v):
                    for b in range(v):
                        for c in range(v):
                            tt_connected[i][j][k][a][b][c]=(1/(D[cind_r1r2(i,a,o,v,nofe)]+D[cind_r1r2(j,b,o,v,nofe)]+D[cind_r1r2(k,c,o,v,nofe)]))*((con_a[i][j][k][a][b][c]-con_a[j][i][k][a][b][c]-con_a[k][j][i][a][b][c])-(con_a[i][j][k][b][a][c]-con_a[j][i][k][b][a][c]-con_a[k][j][i][b][a][c])-(con_a[i][j][k][c][b][a]+con_a[j][i][k][c][b][a]-con_a[k][j][i][c][b][a])- (con_b[i][j][k][a][b][c]-con_b[j][i][k][a][b][c]-con_b[k][j][i][a][b][c])+(con_b[i][j][k][b][a][c]-con_b[j][i][k][b][a][c]-con_b[k][j][i][b][a][c])+(con_b[i][j][k][c][b][a]+con_b[j][i][k][c][b][a]-con_b[k][j][i][c][b][a]))
    return tt_connected
tt_connected=connected_triples()
def disconnected_triples():
    tt_disconnected=np.zeros([o,o,o,v,v,v])
    for i in range(o):
        for j in range(o):
            for k in range(o):
                for a in range(v):
                    for b in range(v):
                        for c in range(v):
                            tt_disconnected[i][j][k][a][b][c]=(1/(D[cind_r1r2(i,a,o,v,nofe)]+D[cind_r1r2(j,b,o,v,nofe)]+D[cind_r1r2(k,c,o,v,nofe)]))*((dis[i][j][k][a][b][c]-dis[j][i][k][a][b][c]-dis[k][j][i][a][b][c])-(dis[i][j][k][b][a][c]-dis[j][i][k][b][a][c]-dis[k][j][i][b][a][c])-(dis[i][j][k][c][b][a]-dis[j][i][k][c][b][a]-dis[k][j][i][c][b][a]))
    return tt_disconnected

tt_disconnected=disconnected_triples()
    ##################### Energy Calculation #######################
E_t=0.0
for i in range(o):
    for j in range(o):
        for k in range(o):
            for a in range(v):
                for b in range(v):
                    for c in range(v):
                        E_t+=(1/36)*(tt_connected[i][j][k][a][b][c])*(D[cind_r1r2(i,a,o,v,nofe)]+D[cind_r1r2(j,b,o,v,nofe)]+D[cind_r1r2(k,c,o,v,nofe)])*(tt_connected[i][j][k][a][b][c]+tt_disconnected[i][j][k][a][b][c])
print("The value of triples correction to the scf eneregy",E_t)
end=time.time()
print(f"the runtime of the program is {end-start}")