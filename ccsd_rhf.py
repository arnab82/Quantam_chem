import numpy as np 
import time
start=time.time()
from mp2_python import *
nofe=n_elec
nocc=int((nofe//2)+(nofe%2))
print(nofe)
E_ccsd=0
def mo_to_aso(newtwoe):
    nbasis=int(len(newtwoe))
    ASObasis=np.zeros([2*nbasis,2*nbasis,2*nbasis,2*nbasis])
    for i in range(2*nbasis):
        for j in range(2*nbasis):
            for k in range(2*nbasis):
                for l in range(2*nbasis):
                    ASObasis[i][j][k][l]=(newtwoe[i//2][k//2][j//2][l//2])*(i%2==k%2)*(j%2==l%2)-(newtwoe[i//2][l//2][j//2][k//2])*(i%2==l%2)*(j%2==k%2)
    return ASObasis
ASObasis=mo_to_aso(newtwoe)
print(np.shape(ASObasis))
print("THE VALUE OF ASOBASIS IS",ASObasis)

def mo_to_cso(newtwoe):
    nbasis=int(len(newtwoe))
    CSObasis=np.zeros([2*nbasis,2*nbasis,2*nbasis,2*nbasis])
    for i in range(2*nbasis):
        for j in range(2*nbasis):
            for k in range(2*nbasis):
                for l in range(2*nbasis):
                    CSObasis[i][j][k][l]=(newtwoe[i//2][k//2][j//2][l//2])*(i%2==k%2)*(j%2==l%2)
    return CSObasis
def Mat2_aotoMat2_mo(C,twoe):
    nbasis=int(len(twoe))
    temp1=np.zeros([nbasis,nbasis])
    Mat2_mo=np.zeros([nbasis,nbasis])
    temp1=np.einsum("ip,ij->pj", C, twoe)
    Mat2_mo=np.einsum("jq,pj->pq", C, temp1)
    return Mat2_mo
def Mat2_motoMat2_so(Mat2_mo):
    nbasis=int(len(Mat2_mo))
    Mat2_so=np.zeros([2*nbasis,2*nbasis])
    for i in range(2*nbasis):
        #for j in range(2*nbasis):
            Mat2_so[i][i]=Mat2_mo[i//2][i//2]
    return Mat2_so
#print(Mat2_aotoMat2_mo(c,fock))
#print("size of the matrix is",np.shape(Mat2_aotoMat2_mo(c,fock)))
F_so=np.zeros([2*nbasis,2*nbasis])
F_so=Mat2_motoMat2_so(Mat2_aotoMat2_mo(c,fock))
#print("the value of fock matrix in spin basisis ",F_so)# matched
def cind_r1r2(p,q,r1,r2,nofe):
    #   arg: running indices p,q and r1, r2 are whether oo or ov or vo or vv.
    #   returns compound index according to the comnvenience.
    return (r2*(p-nofe*(p>=nofe)))+q-nofe*(q>=nofe)
o=nofe
print(o)
v=2*nbasis-nofe
print(v)
maxiter=500
E_ccsd=0
F_so_oo=np.zeros([o,o])
F_so_vv=np.zeros([v,v])
F_so_ov=np.zeros([o,v])
F_so_vo=np.zeros([v,o])
ASObasis_oooo=np.zeros([o,o,o,o])
ASObasis_oovv=np.zeros([o,o,v,v])
ASObasis_ooov=np.zeros([o,o,o,v])
ASObasis_oovo=np.zeros([o,o,v,o])
ASObasis_vvoo=np.zeros([v,v,o,o])
ASObasis_vvvv=np.zeros([v,v,v,v])
ASObasis_vvov=np.zeros([v,v,o,v])
ASObasis_vvvo=np.zeros([v,v,v,o])
ASObasis_ovoo=np.zeros([o,v,o,o])
ASObasis_ovvv=np.zeros([o,v,v,v])
ASObasis_ovov=np.zeros([o,v,o,v])
ASObasis_ovvo=np.zeros([o,v,v,o])
ASObasis_vooo=np.zeros([v,o,o,o])
ASObasis_vovv=np.zeros([v,o,v,v])
ASObasis_voov=np.zeros([v,o,o,v])
ASObasis_vovo=np.zeros([v,o,v,o])
#For the partitioning of F_so matrix
for i in range(o):
    for j in range(o):
        F_so_oo[i][j]=F_so[i][j]
#print(F_so_oo,"\n\n\n")
#print(np.shape(F_so_oo))
for a in range(v):
    for b in range(v):
        F_so_vv[a][b]=F_so[a+nofe][b+nofe]
#print(F_so_vv,"\n\n\n")
#print(np.shape(F_so_vv))
for i in range(o):
    for a in range(v):
        F_so_ov[i][a]=F_so[i][a+nofe]
#print(F_so_ov,"\n\n\n")
#print(np.shape(F_so_ov))
for b in range(v):
    for j in range(o):
        F_so_vo[b][j]=F_so[b+nofe][j]
#print(F_so_vo,"\n\n\n")
#print(np.shape(F_so_vo))
#For the partitioning of ASObasis matrix
for i in range(o):
    for j in range(o):
        for m in range(o):
            for n in range(o):
                ASObasis_oooo[i][j][m][n]=ASObasis[i][j][m][n]
for i in range(o):
    for j in range(o):
        for m in range(o):
            for b in range(v):
                ASObasis_ooov[i][j][m][b]=ASObasis[i][j][m][b+nofe]
for i in range(o):
    for j in range(o):
        for a in range(v):
            for m in range(o):
                ASObasis_oovo[i][j][a][m]=ASObasis[i][j][a+nofe][m]
for i in range(o):
    for a in range(v):
        for j in range(o):
            for m in range(o):
                ASObasis_ovoo[i][a][j][m]=ASObasis[i][a+nofe][j][m]
for a in range(v):
    for j in range(o):
        for m in range(o):
            for n in range(o):
                ASObasis_vooo[a][j][m][n]=ASObasis[a+nofe][j][m][n]
for a in range(v):
    for b in range(v):
        for m in range(o):
            for n in range(o):
                ASObasis_vvoo[a][b][m][n]=ASObasis[a+nofe][b+nofe][m][n]
for i in range(o):
    for a in range(v):
        for b in range(v):
            for n in range(o):
                ASObasis_ovvo[i][a][b][n]=ASObasis[i][a+nofe][b+nofe][n]
for i in range(o):
    for a in range(v):
        for m in range(o):
            for b in range(v):
                ASObasis_ovov[i][a][m][b]=ASObasis[i][a+nofe][m][b+nofe]
for i in range(o):
    for j in range(o):
        for a in range(v):
            for b in range(v):
                ASObasis_oovv[i][j][a][b]=ASObasis[i][j][a+nofe][b+nofe]
for a in range(v):
    for b in range(v):
        for e in range(v):
            for f in range(v):
                ASObasis_vvvv[a][b][e][f]=ASObasis[a+nofe][b+nofe][e+nofe][f+nofe]
for a in range(v):
    for b in range(v):
        for m in range(o):
            for f in range(v):
                ASObasis_vvov[a][b][m][f]=ASObasis[a+nofe][b+nofe][m][f+nofe]
for a in range(v):
    for b in range(v):
        for e in range(v):
            for n in range(o):
                ASObasis_vvvo[a][b][e][n]=ASObasis[a+nofe][b+nofe][e+nofe][n]
for i in range(o):
    for b in range(v):
        for e in range(v):
            for f in range(v):
                ASObasis_ovvv[i][b][e][f]=ASObasis[i][b+nofe][e+nofe][f+nofe]
for a in range(v):
    for j in range(o):
        for e in range(v):
            for f in range(v):
                ASObasis_vovv[a][j][e][f]=ASObasis[a+nofe][j][e+nofe][f+nofe]
for a in range(v):
    for j in range(o):
        for m in range(o):
            for f in range(v):
                ASObasis_voov[a][j][m][f]=ASObasis[a+nofe][j][m][f+nofe]
for a in range(v):
    for j in range(o):
        for e in range(v):
            for n in range(o):
                ASObasis_vovo[a][j][e][n]=ASObasis[a+nofe][j][e+nofe][n]

#print("the value of double excitation operator is",ASObasis_oovv,"\n\n")

D=np.zeros([o*v])
tao=np.zeros([o,o,v,v])
taobar=np.zeros([o,o,v,v])
ts=np.zeros([o,v])
tsnew=np.zeros([o,v])
td=np.zeros([o,o,v,v])
tdnew=np.zeros([o,o,v,v])
Fae=np.zeros([v,v])
Fmi=np.zeros([o,o])
Fme=np.zeros([o,v])
Wmnij=np.zeros([o,o,o,o])
Wabef=np.zeros([v,v,v,v])
Wmbej=np.zeros([o,v,v,o])
for i in range(o):
    for j in range(o):
        for a in range(v):
            for b in range(v):
                td[i][j][a][b]=(ASObasis_oovv[i][j][a][b])/((E[i//2]+E[j//2]-E[(a+nofe)//2]-E[(b+nofe)//2]))
#print("the value of double excitation operator is ",td,"\n\n")
E_mp2_so=0
for i in range(o):
    for j in range(o):
        for a in range(v):
            for b in range(v):
                E_mp2_so+=0.25*(ASObasis_oovv[i][j][a][b])*td[i][j][a][b]
print(E_mp2_so)
#Step #3: Calculate the CC Intermediates
def cind_r1r2(p,q,r1,r2,nofe):
    #   arg: running indices p,q and r1, r2 are whether oo or ov or vo or vv.
    #   returns compound index according to the convenience.
    return (r2*(p-nofe*(p>=nofe)))+q-nofe*(q>=nofe)
for i in range(o):
    for a in range(v):
        D[cind_r1r2(i,a,o,v,nofe)]=F_so_oo[i][i]-F_so_vv[a][a]
print("the denominator MATRIX IS ",D,"\n\n")
print(len(D))
#iteration
for iteration in range(maxiter):
	#formation of tao
	for i in range(o):
		for j in range(o):
			for a in range(v):
				for b in range(v):
					tao[i][j][a][b]=td[i][j][a][b] + (ts[i][a]*ts[j][b])-(ts[i][b]*ts[j][a])
	print("the value of tao is",tao)
	print("the value of size of tao is",np.shape(tao))
	print(np.where(tao==0.000631533771))
	exit(0)
	#formation of taobar
	for i in range(o):
		for j in range(o):
			for a in range(v):
				for b in range(v):
					taobar[i][j][a][b]=td[i][j][a][b] +0.5*(ts[i][a]*ts[j][b]-ts[i][b]*ts[j][a])
	print("the value of taobar is",taobar)
	#intermediates
	# for Fae
	Fae_a=np.zeros([v,v])
	Fae_b=np.zeros([v,v])
	Fae_c=np.zeros([v,v])
	Fae_a=np.einsum('me,ma->ae',ts,F_so_ov)
	Fae_b=np.einsum('mf,mafe->ae',ts,ASObasis_ovvv)
	Fae_c=np.einsum('mnaf,mnef->ae',taobar,ASObasis_oovv)
    #for Fmi
	Fmi_a=np.zeros([o,o])
	Fmi_b=np.zeros([o,o])
	Fmi_c=np.zeros([o,o])
	Fmi_a=np.einsum('ie,me->mi',ts,F_so_ov)
	Fmi_b=np.einsum('ne,mnie->mi',ts,ASObasis_ooov)
	Fmi_c=np.einsum('inef,mnef->mi',taobar,ASObasis_oovv)
    #for Fme
	Fme_a=np.zeros([o,v])
	Fme_a=np.einsum('nf,mnef->me',ts,ASObasis_oovv)
    #evaluationg F intermediates
	for a in range(v):
		for e in range(v):
			Fae[a][e]=(1-(a==e))*F_so_vv[a][e] - 0.5*Fae_a[a][e] + Fae_b[a][e] - 0.5*Fae_c[a][e]
	for m in range(o):
		for i in range(o):
			Fmi[m][i] = (1-(m==i))*F_so_oo[m][i] + 0.5*Fmi_a[m][i] + Fmi_b[m][i] + 0.5*Fmi_c[m][i]
	for m in range(o):
		for e in range(v):
			Fme[m][e] = F_so_ov[m][e] + Fme_a[m][e]
	print("the fmi matrix is ",Fmi)
	print("the fme matrix is ",Fme)
	print("the fae matrix is ",Fae)
    #W intermediates
	Wmnij_a=np.zeros([o,o,o,o])
	Wmnij_b=np.zeros([o,o,o,o])
	Wmnij_a=np.einsum('je,mnie->mnij',ts,ASObasis_ooov)
	Wmnij_b=np.einsum('ijef,mnef->mnij',tao,ASObasis_oovv)
	Wabef_a=np.zeros([v,v,v,v])
	Wabef_b=np.zeros([v,v,v,v])
	Wabef_a=np.einsum('mb,amef->abef',ts,ASObasis_vovv)
	Wabef_b=np.einsum('mnab,mnef->abef',tao,ASObasis_oovv)	
	Wmbej_a=np.zeros([o,v,v,o])
	Wmbej_b=np.zeros([o,v,v,o])
	Wmbej_c=np.zeros([o,v,v,o])
	Wmbej_a=np.einsum('jf,mbef->mbej',ts,ASObasis_ovvv)
	Wmbej_b=np.einsum('nb,mnej->mbej',ts,ASObasis_oovo)
	Wmbej_c=np.einsum('jnfb,mnef->mbej',(0.5*td+np.einsum('jf,nb->jnfb',ts,ts)),ASObasis_oovv)
    #evaluating W intermediates
	for m in range(o):
		for n in range(o):
			for i in range(o):
				for j in range(o):
					Wmnij[m][n][i][j]=ASObasis_oooo[m][n][i][j]+ Wmnij_a[m][n][i][j] - Wmnij_a[m][n][j][i] + 0.25*Wmnij_b[m][n][i][j]
	for a in range(v):
		for b in range(v):
			for e in range(v):
				for f in range(v):
					Wabef[a][b][e][f]=ASObasis_vvvv[a][b][e][f]-Wabef_a[a][b][e][f]+Wabef_a[b][a][e][f] +0.25*Wabef_b[a][b][e][f]
	for m in range(o):
		for b in range(v):
			for e in range(v):
				for j in range(o):
					Wmbej[m][b][e][j]=ASObasis_ovvo[m][b][e][j]+Wmbej_a[m][b][e][j] - Wmbej_b[m][b][e][j] - Wmbej_c[m][b][e][j]
#For T1
	T1=np.zeros([o,v])
	T1=T1+np.einsum('ie,ae->ia',ts,Fae)
	T1=T1-np.einsum('ma,mi->ia',ts,Fmi)
	T1=T1+np.einsum('imae,me->ia',td,Fme)
	T1=T1-np.einsum('nf,naif->ia',ts,ASObasis_ovov)
	T1=T1-0.5*(np.einsum('imef,maef->ia',td,ASObasis_ovvv))
	T1=T1-0.5*(np.einsum('mnae,nmei->ia',td,ASObasis_oovo))
	print("the value of T1_a is ",T1)
#for T2
	T2_a=np.zeros([o,o,v,v])
	T2_b=np.zeros([o,o,v,v])
	T2_c=np.zeros([o,o,v,v])
	T2_d=np.zeros([o,o,v,v])
	T2_e=np.zeros([o,o,v,v])
	T2_f=np.zeros([o,o,v,v])
	T2_g=np.zeros([o,o,v,v])
	T2_a=np.einsum('ijae,be->ijab',td,(Fae-0.5*(np.einsum('mb,me->be',ts,Fme))))
	T2_b=np.einsum('imab,mj->ijab',td,(Fmi-0.5*(np.einsum('je,me->mj',ts,Fme))))
	T2_c=np.einsum('mnab,mnij->ijab',tao,Wmnij)
	T2_d=np.einsum('ijef,abef->ijab',tao,Wabef)
	T2_e=np.einsum('imae,mbej->ijab',td,Wmbej)-np.einsum('ie,ma,mbej->ijab',ts,ts,ASObasis_ovvo)
	T2_f=np.einsum('ie,abej->ijab',ts,ASObasis_vvvo)
	T2_g=np.einsum('ma,mbij->ijab',ts,ASObasis_ovoo)
	#print("the value of T2_g is",T2_g)
#Writing T1 Equation
	for i in range(o):
		for a in range(v):
			tsnew[i][a] = (1/D[cind_r1r2(i,a,o,v,nofe)]) * (F_so_ov[i][a] + T1[i][a])
	print("the tsnew matrix value is",tsnew,"\n\n")
#writing T2 Equation
	for i in range(o):
		for j in range(o):
			for a in range(v):
				for b in range(v):
					tdnew[i][j][a][b]=(1/(D[cind_r1r2(i,a,o,v,nofe)]+D[cind_r1r2(j,b,o,v,nofe)]))*(ASObasis_oovv[i][j][a][b] + T2_a[i][j][a][b] - T2_a[i][j][b][a] - T2_b[i][j][a][b] + T2_b[j][i][a][b] + 0.5*T2_c[i][j][a][b] + 0.5*T2_d[i][j][a][b] + T2_e[i][j][a][b] - T2_e[j][i][a][b] - T2_e[i][j][b][a] + T2_e[j][i][b][a] + T2_f[i][j][a][b] - T2_f[j][i][a][b] - T2_g[i][j][a][b] + T2_g[i][j][b][a])
	print("the tdnew matrix value is",tsnew,"\n\n")
#E_ccsd calculation
	E_ccsd_a=0
	E_ccsd_b=0
	E_ccsd_c=0
	for i in range(o):
		for a in range(v):
			E_ccsd_a+=F_so_ov[i][a]*tsnew[i][a]
			for j in range(o):				
				for b in range(v):
					E_ccsd_b+=ASObasis_oovv[i][j][a][b]*tdnew[i][j][a][b]
					E_ccsd_c+=ASObasis_oovv[i][j][a][b]*tsnew[i][a]*tsnew[j][b]
	print("the singles energy is", E_ccsd_c)
	print("the doubles energy is", E_ccsd_b)
	E_ccsdnew=E_ccsd_a+(0.25*E_ccsd_b)+(0.5*E_ccsd_c)
	print("the ccsd energy is", E_ccsdnew)
	exit(0)
	#Check for convergence
	if abs(E_ccsdnew-E_ccsd)<=10**(-14) and abs(np.std(tsnew-ts))<=10**(-14) and abs(np.std(tdnew-td))<=10**(-12):
			print("SUCCESS! Coupled Cluster SCF converged.")
			ts=tsnew
			td=tdnew
			E_ccsd=E_ccsdnew
			print("E_ccsd = " + str(E_ccsd) + ".")
			break
	if iteration==maxiter-1:
		print("Maximum iterations reached. CCSD SCF did not converge.")
		exit(0)
	Delta_E_ccsd=E_ccsdnew-E_ccsd
	Delta_ts=np.std(tsnew-ts)
	Delta_td=np.std(tdnew-td)

	if iteration==0:
		print('\n\n\n')
		print("\n\n\n\nCCSD SCF Iterations:\n--------------------")
		print('\n\n\tIteration(s)\t\tE_ccsd(new)\t\tDelta_E_ccsd\t\tDelta_ts\t\tDelta_td\n')
	print('\n\t'+str(iteration+1)+'\t\t'+str(E_ccsdnew)+'\t'+str(Delta_E_ccsd)+'\t'+str(Delta_ts)+'\t\t'+str(Delta_td)+'\n')
	print("CCSD SCF energy deviations=" + str(E_ccsdnew-E_ccsd) + ", singles=" + str(abs(np.std(tsnew-ts))) + " and doubles=" + str(abs(np.std(tdnew-td))) + " in " + str(iteration+1) + " iteration(s).")
	ts=tsnew
	td=tdnew
	E_ccsd=E_ccsdnew
end=time.time()
print(f"the runtime of the program is {end-start}")
