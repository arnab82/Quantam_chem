import numpy as np 
import time
start=time.time()
from pyscf import gto 
"""import sys
inFile = sys.argv[1]
with open(inFile,'r') as i:
	content = i.readlines()
input_file =[]
for line in content:
	v_line=line.strip()
	if len(v_line)>0:
		input_file.append(v_line.split())

Level_of_theory = input_file[0][0]
basis_set = input_file[0][1]
unit = input_file[0][2]
charge, multiplicity = input_file[1]
for i in range(2):
	input_file.pop(0)
geom_file = input_file
Atoms = []
for i in range(len(geom_file)):
	Atoms.append(geom_file[i][0])
#print(Atoms)
geom_raw = geom_file
for i in range(len(geom_file)):
	geom_raw[i].pop(0)

geom = ''
atomline = ''
for i in range(len(geom_raw)):
	atomline += Atoms[i]+" "
	for j in range(len(geom_raw[i])):
		if j!=(len(geom_raw[i])-1):
			atomline += geom_raw[i][j]+" "
		else:
			atomline += geom_raw[i][j]
		
	if (i == len(geom_raw)-1):
		geom += atomline +""
	else:
		geom += atomline +";"
	atomline = ''
print(geom)
print(basis_set)"""
def make_density(C,no):
	nbasis = C.shape[0]
	D = np.zeros_like(C)
	for i in range(nbasis):
		for j in range(nbasis):
			for m in range(no):
				D[i,j]+=C[i,m]*C[j,m]
	return D
def make_fock(D,H,twoe):
	nbasis = D.shape[0]
	fock=np.zeros((nbasis,nbasis))
	for i in range(nbasis):
			for j in range(nbasis):
				for k  in range(nbasis):
					for l in range(nbasis):
						fock[i,j]+=D[k,l]* (2.0*twoe[i,j,k,l]-twoe[i,k,j,l])
	return fock+H
def scf_energy(D,fock,core_h):
	nbasis = D.shape[0]
	new_energy=0.0 
	for i in range(nbasis):
		for j in range(nbasis):
			new_energy+=D[i,j]*(fock[i,j]+core_h[i,j])
	return new_energy

#print(len(basis_set))
mol = gto.Mole(spin = 0,charge=0)
#mol.atom = geom
mol.atom = "O 0.000000000000 -0.143225816552 0.000000000000;H 1.638036840407 1.136548822547 -0.000000000000;H -1.638036840407 1.136548822547 -0.000000000000"
mol.unit = "Bohr"
mol.basis = "sto3g"
#mol.basis = "cc-pVDZ"
mol.build()

enuc  = mol.energy_nuc()
s = np.array(mol.intor("int1e_ovlp"))
t = mol.intor("int1e_kin")
v = mol.intor("int1e_nuc")
twoe = mol.intor("int2e")
nbasis = s.shape[0]
n_elec = 10
no = int(n_elec/2)
core_h=t+v 
#fock matrix
q,L=np.linalg.eigh(s)
q_half = np.zeros_like(s)
for i in range(nbasis):
	q_half[i,i] = q[i]**-0.5
temp = np.matmul(L , q_half)
s_half = np.matmul(temp , L.transpose())
init_f = np.einsum("ij,jk,kl->il", s_half, core_h , s_half.T)
e,C0 = np.linalg.eigh(init_f)
C = np.matmul(s_half,C0)
D = np.zeros_like(C)
for i in range(no):
	D[i,i]=1.0
energy=0.0
for n in range(100):
	fock = make_fock(D,core_h,twoe)
	f_dash = np.einsum("ij,jk,kl->il", s_half, fock , s_half.T)
	eps,c_dash=np.linalg.eigh(f_dash)
	C=np.matmul(s_half,c_dash)
	D = make_density(C,no)
	hf_energy = scf_energy(D,fock,core_h)
	del_e = hf_energy - energy
	energy=hf_energy
	if np.abs(del_e)<=1e-12:
		break
	print("iteration=",n,"energy= ",energy+enuc," delta_e= ",np.around(del_e,decimals=12))
c = C
#print(c)
newtwoe=np.zeros_like(twoe)
newtwoe.shape=(nbasis,nbasis,nbasis,nbasis)
temp1=np.einsum("ip,ijkl->pjkl", c, twoe)
temp2=np.einsum("jq,pjkl->pqkl", c, temp1)
temp3=np.einsum("kr,pqkl->pqrl", c, temp2)
newtwoe=np.einsum("ls,pqrl->pqrs", c, temp3)
#print("the value of newtwoe is",newtwoe)
#print("the value of twoe is",twoe)
E=np.array(eps)
emp2=0.0
ndocc=5
print(len(c))
for i in range(ndocc):
    for a in range(ndocc,nbasis):
        for j in range(ndocc):
            for b in range(ndocc,nbasis):
                emp2+=newtwoe[i][a][j][b]*(2*newtwoe[i][a][j][b]-newtwoe[i][b][j][a])/(E[i]+E[j]-E[a]-E[b])
print("MP2 correlation E = ",emp2) 
end=time.time()
print(f"the runtime of the program is {end-start}")
#print("the value of coefficient matrix is ",c)
