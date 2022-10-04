using PyCall
using MKL
using LinearAlgebra,Statistics
pyscf=PyCall.pyimport("pyscf")
mol=pyscf.gto.M()
#=struct Atom
    id::Integer
    symbol::String
    xyz::Array{Float64,1}
end=#
atoms="O 0.000000000000 -0.143225816552 0.000000000000;H 1.638036840407 1.136548822547 -0.000000000000;H -1.638036840407 1.136548822547 -0.000000000000"
#atoms=[]
#push!(atoms,Atom(1,"O",[0.000000000000,-0.143225816552,0.000000000000]))
#push!(atoms,Atom(2,"H",[1.638036840407,1.136548822547,-0.000000000000]))
#push!(atoms,Atom(3,"H",[-1.638036840407,1.136548822547,-0.000000000000]))
mol.charge=0
mol.unit = "Bohr"
mol.spin=0
mol.build(
	atom= atoms,
	basis = "sto3g")
#println(mol.atom)
#h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
#S = mol.intor("int1e_ovlp")
#eri = mol.intor("int2e",aosym="s8")
#nuclear_repulsion = mol.energy_nuc()
#constant = nuclear_repulsion
#println(S)
#println(eri)
#println(h1e)
#println(constant)
print(mol.atom)
print(mol.basis)
function pyscf_1e()
    h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    return h1e
end

function pyscf_overlap()
    S = mol.intor("int1e_ovlp")
    return S
end

function pyscf_2e()
    eri = mol.intor("int2e")
    return eri
end

function pyscf_nucr()
    nuclear_repulsion = mol.energy_nuc()
    constant = nuclear_repulsion
    return constant
end

h1e=pyscf_1e()
eri=pyscf_2e()
constant=pyscf_nucr()
S=pyscf_overlap()

println("the one electron integral is ",h1e ,"\n")
#println("the two electron integral is ",eri,"\n")
println(" shape of two electron integral is",size(eri),"\n")
println("the nuclear nuclear repulsion term is ",constant,"\n")
println("the overlap integral is",S)
   
s=(S+transpose(S))/2
nbasis=7
q,L=LinearAlgebra.eigen(s)
println("the value of q is",q,"\n")
println("the value of L is",L,"\n")

q_half=[]
for i in 1:lastindex(q)
    push!(q_half,q[i]^(-0.5))
end

println("the q_half value is ",q_half,"\n")
q_half=Diagonal(q_half)
s_half=*(L,q_half)
s_half=*(s_half,transpose(L))
global fock= zeros(Float64,nbasis,nbasis)
println("the shape of fock matrix is",size(fock))
s_half=transpose(s_half)
println("the s_half matrix is",s_half,"\n")
println("the shape of s_half is ",size(s_half),"\n")
fock=*(s_half,h1e)
fock=*(fock,s_half)
#println("the value of fock matrix is ",fock)
list_e=[0.0,]

#scf initialisation 
for n in 1:100
    fock=(fock+transpose(fock))/2
    eps,c_dash=LinearAlgebra.eigen(fock)
    c=*(conj(s_half),c_dash)
    #println("the value of c matrix is ",c,"\n")
    println("the shape of c matrix is ",size(c),"\n")
    D=zeros(Float64, nbasis,nbasis)
    println("the shape of D is ",size(D),"\n")
    n_elec=10
    no=Int64((n_elec)/2)
    #println("testing purpose value of c is   ",c[2,4],"\n")
    println("the no of occupied orbital is ",no)
    #println("the last index of c is  ",lastindex(c),"\n")
    println("the value of nbasis is ",nbasis,"\n")
    for i in 1:nbasis
        for j in 1:nbasis
            for m in 1:no
                D[i,j]+=(*(c[i,m],c[j,m]))
                #push!(D[i,j],(*(c[i,m],c[j,m])))
                #push!(D,(*(c[i,m],c[j,m])))
            end
        end
    end    
    #println("the density matrix is ", D,"\n")  
#energy compute 
    n_hf_energy=[]
    for i in 1:nbasis
        for j in 1:nbasis
            x=(*(D[i,j],(fock[i,j]+h1e[i,j])))
            push!(n_hf_energy,x)
        end
    end
    push!(list_e,sum(n_hf_energy)) 
    #println(list_e)
    del_e=0.0
    for i in list_e
        del_e=list_e[lastindex(list_e)]-list_e[lastindex(list_e)-1]
    end
    #println(del_e)
    if (abs(del_e))<=10^(-12)
        break
    end
    println("iteration= ",n,"    energy= ", sum(n_hf_energy)+constant,"         delta_e= ",round(del_e,digits=12))
    #println("the value of final hf energy is  ",sum(n_hf_energy)+constant,"\n")
#Compute the New Fock Matrix
    new_fock=zeros(Float64,nbasis,nbasis)
    for i in 1:nbasis
        for j in 1:nbasis
            new_fock[i,j]=h1e[i,j]
            for k in 1:nbasis
                for l in 1:nbasis
                new_fock[i,j]+=(*(D[k,l],((2*eri[i,j,k,l])-eri[i,l,k,j])))
                #push!(new_fock[i,j],(*(D[k,l],(*((2*twoe[i,j,k,l])-twoe[i,l,k,j])))))
                end
            end
        end
    end 
    global fock=new_fock  
end        

#println("the value of updated fock matrix is ",fock,"\n")

    


