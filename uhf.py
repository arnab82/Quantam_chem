
import numpy as np 
import time
start=time.time()
from pyscf import gto 
mol = gto.Mole(spin =1,charge=0)
#mol.atom = "O 0.000000000000 -0.143225816552 0.000000000000;H 1.638036840407 1.136548822547 -0.000000000000;H -1.638036840407 1.136548822547 -0.000000000000"
mol.atom="N -0.000000 0.000000 0.141610;H  0.000000 0.806598 -0.495633;H -0.000000 -0.806598 -0.495633"
#mol.unit = "Bohr"
mol.basis = "sto3g"
#mol.basis = "aug-cc-pVDZ"
#mol.basis = "cc-pVDZ"
mol.build()
enuc  = mol.energy_nuc()
s = np.array(mol.intor("int1e_ovlp"))
t = mol.intor("int1e_kin")
v = mol.intor("int1e_nuc")
twoe = mol.intor("int2e")
nbasis = s.shape[0]
n_elec = 9
mult=mol.spin +1
no = int(n_elec/2)
core_h=t+v 
n_alpha=int((n_elec+mult-1)/2)
print(n_alpha)
n_beta=int(n_elec-n_alpha)
print(n_beta)
def make_density(C,n):
	nbasis = C.shape[0]
	D = np.zeros_like(C)
	for i in range(nbasis):
		for j in range(nbasis):
			for m in range(n):
				D[i,j]+=C[i,m]*C[j,m]
	return D

def make_fock(D,D_V,H,twoe):
	nbasis = D.shape[0]
	fock=np.zeros((nbasis,nbasis))
	for i in range(nbasis):
			for j in range(nbasis):
				for k  in range(nbasis):
					for l in range(nbasis):
						fock[i,j]+=((D[l,k]*twoe[i,j,k,l])-(D_V[l,k]*twoe[i,l,k,j]))
	return fock+H
def scf_energy(D,Fock_alpha,Fock_beta,D_alpha,D_beta,core_h):
	nbasis = D.shape[0]
	new_energy=0.0 
	for i in range(nbasis):
		for j in range(nbasis):
			new_energy+=0.5*((D[j,i]*core_h[i,j])+(D_alpha[j,i]*Fock_alpha[i,j])+(D_beta[j,i]*Fock_beta[i,j]))
	return new_energy
#s=(s+s.T)/2
q,L=np.linalg.eigh(s)
q_half = np.zeros_like(s)
for i in range(nbasis):
	q_half[i,i] = q[i]**-0.5
temp = np.matmul(L , q_half)
s_half = np.matmul(temp , L.T)
init_f = np.einsum("ij,jk,kl->il", s_half, core_h , s_half.T)
e,C0 = np.linalg.eigh(init_f)
C = np.matmul(s_half,C0)
D_alpha = np.zeros([nbasis,nbasis])
D_beta = np.zeros([nbasis,nbasis])
'''for i in range(n_alpha):
	D_alpha[i,i]=1.0
for i in range(n_beta):  
	D_beta[i,i]=1.0'''
energy=0.0
D_alpha=make_density(C,n_alpha)
D_beta=make_density(C,n_beta)
D=D_alpha+D_beta
for i in range(100):
    Fock_alpha=make_fock(D,D_alpha,core_h,twoe)
    #Fock_alpha=(Fock_alpha+Fock_alpha.T)/2
    Fock_beta=make_fock(D,D_beta,core_h,twoe)
    #Fock_beta=(Fock_beta+Fock_beta.T)/2
    Fock_alpha_d = np.einsum("ij,jk,kl->il", s_half.T, Fock_alpha , s_half)
    Fock_beta_d = np.einsum("ij,jk,kl->il", s_half.T, Fock_beta , s_half)
    #Fock_alpha_d=(Fock_alpha_d+Fock_alpha_d.T)/2
    #Fock_beta_d=(Fock_beta_d+Fock_beta_d.T)/2
    eps_alpha,c_alpha_d=np.linalg.eigh(Fock_alpha_d)
    eps_beta,c_beta_d=np.linalg.eigh(Fock_beta_d)
    C_alpha=np.matmul(s_half,c_alpha_d)
    C_beta=np.matmul(s_half,c_beta_d)
    D_alpha=make_density(C_alpha,n_alpha)
    D_beta=make_density(C_beta,n_beta)
    D=D_alpha+D_beta
    hf_energy = scf_energy(D,Fock_alpha,Fock_beta,D_alpha,D_beta,core_h)
    del_e = hf_energy - energy
    energy=hf_energy
    if np.abs(del_e)<=1e-12:
	    break
    print("iteration=",i,"energy= ",energy+enuc," delta_e= ",np.around(del_e,decimals=12))
