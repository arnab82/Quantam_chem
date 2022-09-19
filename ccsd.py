import numpy as np 
from mp2 import *



nofe=n_elec
nocc=int((nofe//2)+(nofe%2))
print(nofe)
E_ccsd=0
def mo_to_aso(newtwoe):
    nbasis=int(len(newtwoe))
    #print(nbasis)
    ASObasis=np.zeros([2*nbasis,2*nbasis,2*nbasis,2*nbasis])
    for i in range(2*nbasis):
        for j in range(2*nbasis):
            for k in range(2*nbasis):
                for l in range(2*nbasis):
                    ASObasis[i][j][k][l]=(newtwoe[i//2][k//2][j//2][l//2])*(i%2==k%2)*(j%2==l%2)-(newtwoe[i//2][l//2][j//2][k//2])*(i%2==l%2)*(j%2==k%2)
    return ASObasis
ASObasis=mo_to_aso(newtwoe)
#print(ASObasis)

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
    temp1=np.zeros([2*nbasis,2*nbasis])
    Mat2_mo=np.zeros([2*nbasis,2*nbasis])
    temp1=np.einsum("ip,ij->pj", C, twoe)
    Mat2_mo=np.einsum("jq,pj->pq", C, temp1)
    return Mat2_mo

def Mat2_motoMat2_so(Mat2_mo):
    nbasis=int(len(Mat2_mo))
    Mat2_so=np.zeros([2*nbasis,2*nbasis])
    for i in range(2*nbasis):
        #for j in range(2*mumax):
            Mat2_so[i][i]=Mat2_mo[i//2][i//2]
    return Mat2_so


F_so=np.zeros([2*nbasis,2*nbasis])
F_so=Mat2_motoMat2_so(Mat2_aotoMat2_mo(c,fock))
#print(F_so)
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

for a in range(v):
    for b in range(v):
        F_so_vv[a][b]=F_so[a+nofe][b+nofe]

for i in range(o):
    for a in range(v):
        F_so_ov[i][a]=F_so[i][a+nofe]

for b in range(v):
    for j in range(o):
        F_so_vo[b][j]=F_so[b+nofe][j]

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

D=np.zeros([o*v])
tao=np.zeros([o,v,o,v])
taobar=np.zeros([o,v,o,v])
ts=np.zeros([o,v])
tsnew=np.zeros([o,v])
td=np.zeros([o,v,o,v])
tdnew=np.zeros([o,v,o,v])
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
                td[i][a][j][b]=(ASObasis_oovv[i][j][a][b])/((E[i//2]+E[j//2]-E[(a+nofe)//2]-E[(b+nofe)//2]))

E_mp2_so=0

for i in range(o):
    for j in range(o):
        for a in range(v):
            for b in range(v):
                E_mp2_so+=0.25*(ASObasis[i][j][a][b])*td[i][a][j][b]
print(E_mp2_so)

#Step #3: Calculate the CC Intermediates
'''for i in range(o):
    for a in range(v):
        D=F_so_oo[i][i]-F_so_vv[a][a]'''
        #print(D)
def cind_r1r2(p,q,r1,r2,nofe):
    #   arg: running indices p,q and r1, r2 are whether oo or ov or vo or vv.
    #   returns compound index according to the convenience.

    return (r2*(p-nofe*(p>=nofe)))+q-nofe*(q>=nofe)

for i in range(o):
    for a in range(v):
        D[cind_r1r2(i,a,o,v,nofe)]=F_so_oo[i][i]-F_so_vv[a][a]
        #print(D)

#iteration
for iteration in range(maxiter):
	#formation of tao
	for i in range(o):
		for a in range(v):
			for j in range(o):
				for b in range(v):
					tao[i][a][j][b]=td[i][a][j][b] + (ts[i][a]+ts[i][a]*ts[j][b]-ts[i][b]*ts[j][a])
	#formation of taobar
	for i in range(o):
		for a in range(v):
			for j in range(o):
				for b in range(v):
					taobar[i][a][j][b]=td[i][a][j][b] +0.5*(ts[i][a]+ts[i][a]*ts[j][b]-ts[i][b]*ts[j][a])

	#intermediates
	# for Fae
	Fae_a=np.zeros([v,v])
	Fae_b=np.zeros([v,v])
	Fae_c=np.zeros([v,v])

	Fae_a=np.einsum('ma,me->ae',ts,F_so_ov)
	Fae_b=np.einsum('mf,mafe->ae',ts,ASObasis_ovvv)
	Fae_c=np.einsum('manf,mnef->ae',taobar,ASObasis_oovv)

    #for Fmi
	Fmi_a=np.zeros([o,o])
	Fmi_b=np.zeros([o,o])
	Fmi_c=np.zeros([o,o])

	Fmi_a=np.einsum('ie,me->mi',ts,F_so_ov)
	Fmi_b=np.einsum('ne,mnie->mi',ts,ASObasis_ooov)
	Fmi_c=np.einsum('ienf,mnef->mi',taobar,ASObasis_oovv)

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
    #W intermediates
	Wmnij_a=np.zeros([o,o,o,o])
	Wmnij_b=np.zeros([o,o,o,o])
	Wmnij_a=np.einsum('je,mnie->mnij',ts,ASObasis_ooov)
	Wmnij_b=np.einsum('iejf,mnef->mnij',tao,ASObasis_oovv)

	Wabef_a=np.zeros([v,v,v,v])
	Wabef_b=np.zeros([v,v,v,v])
	Wabef_a=np.einsum('mb,amef->abef',ts,ASObasis_vovv)
	Wabef_b=np.einsum('manb,mnef->abef',tao,ASObasis_oovv)
	
	Wmbej_a=np.zeros([o,v,v,o])
	Wmbej_b=np.zeros([o,v,v,o])
	Wmbej_c=np.zeros([o,v,v,o])
	Wmbej_a=np.einsum('jf,mbef->mbej',ts,ASObasis_ovvv)
	Wmbej_b=np.einsum('nb,mnej->mbej',ts,ASObasis_oovo)
	Wmbej_c=np.einsum('jfnb,mnef->mbej',(0.5*td+np.einsum('jf,nb->jfnb',ts,ts)),ASObasis_oovv)

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
					Wmbej[m][b][e][j]=ASObasis_ovvo[m][b][e][f]+Wmbej_a[m][b][e][j] - Wmbej_b[m][b][e][j] - Wmbej_c[m][b][e][j]
#For T1
	T1_a=np.zeros([o,v])
	T1_b=np.zeros([o,v])
	T1_c=np.zeros([o,v])
	T1_d=np.zeros([o,v])
	T1_e=np.zeros([o,v])
	T1_f=np.zeros([o,v])

	T1_a=np.einsum('ie,ae->ia',ts,Fae)
	T1_b=np.einsum('ma,mi->ia',ts,Fmi)
	T1_c=np.einsum('iame,me->ia',td,Fme)
	T1_d=np.einsum('nf,naif->ia',ts,ASObasis_ovov)
	T1_e=np.einsum('iemf,maef->ia',td,ASObasis_ovvv)
	T1_f=np.einsum('mane,nmei->ia',td,ASObasis_oovo)
#for T2
	T2_a=np.zeros([o,v,o,v])
	T2_b=np.zeros([o,v,o,v])
	T2_c=np.zeros([o,v,o,v])
	T2_d=np.zeros([o,v,o,v])
	T2_e=np.zeros([o,v,o,v])
	T2_f=np.zeros([o,v,o,v])
	T2_g=np.zeros([o,v,o,v])

	T2_a=np.einsum('iaje,be->iajb',td,(Fae-0.5*(np.einsum('mb,me->be',ts,Fme))))
	T2_b=np.einsum('iamb,mj->iajb',td,(Fmi-0.5*(np.einsum('je,me->jm',ts,Fme))))
	T2_c=np.einsum('manb,mnij->iajb',tao,Wmnij)
	T2_d=np.einsum('iejf,abef->iajb',tao,Wabef)
	T2_e=np.einsum('iame,mbej->iajb',td,Wmbej)-np.einsum('ie,ma,mbej->iajb',ts,ts,ASObasis_ovvo)
	T2_f=np.einsum('ie,abej->iajb',ts,ASObasis_vvvo)
	T2_g=np.einsum('ma,mbij->iajb',ts,ASObasis_ovoo)
#Writing T1 Equation

	for i in range(o):
		for a in range(v):
			tsnew[i][a] = (1/D[cind_r1r2(i,a,o,v,nofe)]) * (F_so_ov[i][a] + T1_a[i][a] - T1_b[i][a] + T1_c[i][a] - T1_d[i][a] - 0.5*T1_e[i][a] - 0.5*T1_f[i][a])
#writing T2 Equation
	for i in range(o):
		for a in range(v):
			for j in range(o):
				for b in range(v):
					tdnew[i][a][j][b]=(1/(D[cind_r1r2(i,a,o,v,nofe)]+D[cind_r1r2(j,b,o,v,nofe)]))*(ASObasis_oovv[i][j][a][b] + T2_a[i][a][j][b] - T2_a[i][b][j][a] - T2_b[i][a][j][b] + T2_b[j][a][i][b] + 0.5*T2_c[i][a][j][b] + 0.5*T2_d[i][a][j][b] + T2_e[i][a][j][b] - T2_e[j][a][i][b] - T2_e[i][b][j][a] + T2_e[j][b][i][a] + T2_f[i][a][j][b] - T2_f[j][a][i][b] - T2_g[i][a][j][b] + T2_g[i][b][j][a])

#E_ccsd calculation

	E_ccsd_a=0
	E_ccsd_b=0
	E_ccsd_c=0

	for i in range(o):
		for a in range(v):
			E_ccsd_a+=F_so_ov[i][a]*tsnew[i][a]
			for j in range(o):
				for b in range(v):
					E_ccsd_b+=ASObasis_ovov[i][a][j][b]*tdnew[i][a][j][b]
					E_ccsd_c+=ASObasis_ovov[i][a][j][b]*tsnew[i][a]*tsnew[j][b]
	E_ccsdnew=E_ccsd_a+(0.25*E_ccsd_b)+(0.5*E_ccsd_c)
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
