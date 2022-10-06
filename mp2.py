import numpy as np 


def read_enuc():
	with open('enuc.dat','r') as f:
		file_content=f.readlines()
		enuc=file_content[0].split()
		enuc=enuc[0]
	return float(enuc) 

#overlap matrix
def read_overlap():
	with open('s.dat','r') as f:
		file_content=f.readlines()
		nbasis=file_content[-1].split()
		nbasis=int(nbasis[0])
		s=np.zeros((nbasis,nbasis))
		for line in file_content:
			line=line.split()
			i=int(line[0])-1
			j=int(line[1])-1
			s[i,j]=float(line[2])
			s[j,i]=float(line[2])
	return s 
#kinetic energy
def read_kinetic():
	with open('t.dat','r') as f:
		file_content=f.readlines()
		nbasis=file_content[-1].split()
		nbasis=int(nbasis[0])
		t=np.zeros((nbasis,nbasis))
		for line in file_content:
			line=line.split()
			i=int(line[0])-1
			j=int(line[1])-1
			t[i,j]=float(line[2])
			t[j,i]=float(line[2])
	return t 
#potential energy
def read_potential():
	with open('v.dat','r') as f:
		file_content=f.readlines()
		nbasis=file_content[-1].split()
		nbasis=int(nbasis[0])
		v=np.zeros((nbasis,nbasis))
		for line in file_content:
			line=line.split()
			i=int(line[0])-1
			j=int(line[1])-1
			v[i,j]=float(line[2])
			v[j,i]=float(line[2])
	return v
'''def compound_index(i,j,k,l):         # a function which will give unique code for each set of atomic orbitals
    if i >=j: ij = i*(i+1)/2 + j
    else: ij = j*(j+1)/2 +i
    if k >=l: kl = k*(k+1)/2 + l
    else: kl = l*(l+1)/2 + k
    if ij >= kl: ijkl = ij*(ij+1)/2 + kl
    else: ijkl = kl*(kl+1)/2 + ij
    return ijkl'''
#two e integral
def read_2e():
	with open('eri.dat','r') as f:
		file_content=f.readlines()
		nbasis=file_content[-1].split()
		nbasis=int(nbasis[0])
		twoe=np.zeros((nbasis,nbasis,nbasis,nbasis))
		for line in file_content:
			line=line.split()
			i=int(line[0])-1
			j=int(line[1])-1
			k=int(line[2])-1
			l=int(line[3])-1
			twoe[i,j,k,l]=float(line[4])
			twoe[i,j,l,k]=float(line[4])
			twoe[j,i,k,l]=float(line[4])
			twoe[j,i,l,k]=float(line[4])
			twoe[k,l,i,j]=float(line[4])
			twoe[l,k,i,j]=float(line[4])
			twoe[k,l,j,i]=float(line[4])
			twoe[l,k,j,i]=float(line[4])
			print(i, j, k, l,'\n')
	return twoe

'''def read_2e():
	with open('eri.dat','r') as f:
		file_content=f.readlines()
		nbasis=file_content[-1].split()
		nbasis=int(nbasis[0])
		twoe=np.zeros((nbasis,nbasis,nbasis,nbasis))
		unique_code = []
		value=[]
		for line in file_content:
			line=line.split()
			i=int(line[0])-1
			j=int(line[1])-1
			k=int(line[2])-1
			l=int(line[3])-1
			ijkl=compound_index(i,j,k,l)
        	unique_code.append(ijkl)
        	value.append(float(line[4]))

	for i in range(nbasis):       # to generate all the two electron integrals ( with both zero and non zero values)
  		for j in range(nbasis):
    		for k in range(nbasis):
      			for l in range(nbasis):
           			ijkl=compound_index(i+1,j+1,k+1,l+1) # function is recalled corresponding to this set of atomic orbitals
           			if ijkl in unique_code:    # condition to check whether this unique number matches with the number in list
               			twoe=unique_code.index(ijkl) 
    return twoe '''         		
enuc=read_enuc()
twoe=read_2e()
t=read_kinetic()
s=read_overlap()
v=read_potential()
#print(twoe)
#print(s)
core_h=t+v 
#fock matrix
q,L=np.linalg.eigh(s)
q_half=np.power(q,-0.5)
q_half=np.diag(q_half)
s_half=np.matmul(L,q_half)
s_half=np.matmul(s_half,L.T)
nbasis=s.shape[0]
print(nbasis)
'''P=np.zeros(nbasis**2) #Density matrix
P.shape=(nbasis,nbasis) #P is nbasis x nbasis
fock=np.zeros((nbasis,nbasis))
for i in range(nbasis):
	for j in range(nbasis):
		for k  in range(nbasis):
			for l in range(nbasis):
				fock[i,j]+=P[k,l]*((2*twoe[i,j,k,l])-twoe[i,l,k,j])
fock=fock+core_h'''


fock=np.matmul(s_half.T,core_h)
fock=np.matmul(fock,s_half)
print(fock)
#density matrix and energy

energy=0.0
for n in range(100):
	f_dash = np.einsum("ij,jk,kl->il", s_half, fock , s_half.T)
	eps,c_dash=np.linalg.eigh(f_dash)
	c=np.matmul(np.conj(s_half),c_dash)
	#print(c)
	D=np.zeros((nbasis,nbasis))
	n_elec=10
	no=int(n_elec/2)
	for i in range(nbasis):
		for j in range(nbasis):
			for m in range(no):
				D[i,j]+=c[i,m]*c[j,m]
	#print(D)
	new_energy=0.0 
	for i in range(nbasis):
		for j in range(nbasis):
			new_energy+=D[i,j]*(fock[i,j]+core_h[i,j])

	del_e=new_energy-energy
	energy=new_energy
	if np.abs(del_e)<=1e-12:
		break
	print("iteration=",n,"energy= ", new_energy+enuc," delta_e= ",np.around(del_e,decimals=12))
#Compute the New Fock Matrix
	new_fock=np.zeros((nbasis,nbasis))
	for i in range(nbasis):
			for j in range(nbasis):
				new_fock[i,j]=core_h[i,j]
				for k  in range(nbasis):
					for l in range(nbasis):
						new_fock[i,j]+=D[k,l]*((2*twoe[i,j,k,l])-twoe[i,l,k,j])
	fock=new_fock
	#print(fock)

#transforming two electron integrals-A0 to MO :The Noddy Algorithm
'''newtwoe=np.zeros_like(twoe)
newtwoe.shape=(nbasis,nbasis,nbasis,nbasis)
for p in range(nbasis):
    for q in range(nbasis):
        for r in range(nbasis):
            for s in range(nbasis):
                val=0.0
                for i in range(len(c)):
                    cip=c[i][p]
                    for j in range(len(c)):
                        cjq=c[j][q]
                        for k in range(len(c)):
                            ckr=c[k][r]
                            for l in range(len(c)):
                                cls=c[l][s]
                                #newjk[p][q][r][s]+=C[i][p]*C[j][q]* \
                                # jk[i][j][k][l]*C[k][r]*C[l][s]
                                val+=cip*cjq*
                                 twoe[i][j][k][l]*ckr*cls
                newtwoe[p][q][r][s]+=val'''
#the smarter algorithm
newtwoe=np.zeros_like(twoe)
newtwoe.shape=(nbasis,nbasis,nbasis,nbasis)
temp1=np.einsum("ip,ijkl->pjkl", c, twoe)
temp2=np.einsum("jq,pjkl->pqkl", c, temp1)
temp3=np.einsum("kr,pqkl->pqrl", c, temp2)
newtwoe=np.einsum("ls,pqrl->pqrs", c, temp3)



# Compute mp2 energy
E_i,X_i=np.linalg.eigh(fock)
#print(E_i)
E=np.array(E_i)
#print(E)
emp2=0.0
ndocc=5
for i in range(ndocc):
    for a in range(ndocc,nbasis):
        for j in range(ndocc):
            for b in range(ndocc,nbasis):
                emp2+=newtwoe[i][a][j][b]*(2*newtwoe[i][a][j][b]-newtwoe[i][b][j][a])\
                   / (E[i]+E[j]-E[a]-E[b])

print("MP2 correlation E = ",emp2) 
print("----------------------------------------------") 
print("\tMP2 Total Energy:\t",emp2+enuc+new_energy) 
print ("----------------------------------------------") 
fock=new_fock

