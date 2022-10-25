import numpy as np
from mp2_python import *
o=n_elec
v=2*nbasis-n_elec
ov=o*v
Hcore_so=np.zeros([2*nbasis,2*nbasis])
H_cis_so=np.zeros([o*v,o*v])
for i in range(o):
    for j in range(o):
        for a in range(o,2*nbasis):
            for b in range(o,2*nbasis):
                H_cis_so[cind_r1r2(i,a,o,v,n_elec)][cind_r1r2(j,b,o,v,n_elec)]=((F_so[a][b])*(i==j))-((F_so[i][j])*(a==b))+(ASObasis[a][j][i][b])

if __name__=="__main__":
    wH_cis_so,vH_cis_so=np.linalg.eigh(H_cis_so)
    print("\n\n\n\nCIS Excitation Energies:\n\n")
    print("\t iteration no.\t\t\t\tHartree\t\t\t\teV\n-----------------------------------------------------------------------------------------------\n\n\n")
    for i in range(o*v):
        print("\t"+str(i)+"\t\t\t"+str(wH_cis_so[i])+"\t\t\t"+str((wH_cis_so[i])*27.2110000013)+"\n")
