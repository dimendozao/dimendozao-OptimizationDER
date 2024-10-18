# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:11:01 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from amplpy import AMPL

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Gen.csv")

case='SA'
city='Bog'
city1='BOG'
problem='MPF'

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
means=mat['clustermeans']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=mat['clusternode']


H=np.size(means,axis=1)

num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for k in range(num_lines):
        sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*means[cnode[branch['j'][k]-1][0]-1][h]
    
pdm=np.real(sd)
qdm=np.imag(sd)

ngen=np.sum(bus['type']==2)
pgen=np.zeros(num_nodes)
qgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

vmax=vmax+0.1
vmin=vmin-0.1

if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]
        
vmax[iref]=1
vmin[iref]=1

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

zk=np.zeros(num_lines,dtype='complex')
yk=np.zeros(num_lines,dtype='complex')

fr=np.zeros(num_lines,dtype='int')
to=np.zeros(num_lines,dtype='int')

for k in range(num_lines):
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    zk[k]=branch['r'][k] + 1j*branch['x'][k]
    yk[k]=1/zk[k]
    
qvnmax=vmax**2
qvnmin=vmin**2

qikmax=np.ones(num_lines)
qikmin=np.zeros(num_lines)

pkmax=np.ones(num_lines)
pkmin=np.zeros(num_lines)
qkmax=np.ones(num_lines)
qkmin=np.zeros(num_lines)

zr=np.real(zk)
zi=np.imag(zk)

fr+=1
to+=1

"----- Optimization model -----"

ampl = AMPL()


ampl.eval(
    r"""
    param nn;
    param nl;
    param nh;    
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nl").set(num_lines)
ampl.get_parameter("nh").set(H)

ampl.eval(
    r"""
    set N=1..nn;
    set L=1..nl;
    set H=1..nh;            
"""
)

ampl.eval(
    r"""   
    param fr {L};
    param to {L};
    param zr {L};
    param zi {L};
    param pd {H,N};
    param qd {H,N};
    param qvnmax{N};
    param qvnmin{N};    
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};       
"""
)
ampl.get_parameter("fr").set_values(fr)
ampl.get_parameter("to").set_values(to)
ampl.get_parameter("zr").set_values(zr)
ampl.get_parameter("zi").set_values(zi)
ampl.get_parameter("pd").set_values(pdm)
ampl.get_parameter("qd").set_values(qdm)
ampl.get_parameter("qvnmax").set_values(qvnmax)
ampl.get_parameter("qvnmin").set_values(qvnmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
               

ampl.eval(
    r"""
    var  qvn{h in H,i in N} >= qvnmin[i], <= qvnmax[i];
    var  qik{h in H,k in L};
    var  pk{h in H,k in L};
    var  qk{h in H,k in L};
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    
    
    minimize Losses:
          sum{h in H,i in N,k in L: fr[k]=i} pk[h,k]
        + sum{h in H,i in N,k in L: to[k]=i} zr[k]*qik[h,k]-sum{h in H,i in N,k in L: to[k]=i} pk[h,k]
        + sum{h in H,i in N,k in L: fr[k]=i} qk[h,k]
        + sum{h in H,i in N,k in L: to[k]=i} zi[k]*qik[h,k]-sum{h in H,i in N,k in L: to[k]=i} qk[h,k];        
    
    subject to PB {h in H,i in N}: 
        pg[h,i]-pd[h,i] = sum {k in L: fr[k]=i} pk[h,k]
        + sum{k in L: to[k]=i} zr[k]*qik[h,k]
        - sum{k in L: to[k]=i} pk[h,k];
      
    subject to QB {h in H,i in N}: 
        qg[h,i]-qd[h,i] = sum{k in L: fr[k]=i} qk[h,k]
        + sum{k in L: to[k]=i} zi[k]*qik[h,k]
        - sum{k in L: to[k]=i} qk[h,k];
        
    subject to C1 {h in H, k in L}:
        qvn[h,fr[k]]-qvn[h,to[k]]==2*(pk[h,k]*zr[k]+qk[h,k]*zi[k])-qik[h,k]*(zr[k]*zr[k] + zi[k]*zi[k]);
        
    subject to SOC {h in H, k in L}:
        qik[h,k]*qvn[h,fr[k]]==pk[h,k]*pk[h,k]+qk[h,k]*qk[h,k];        
        
"""
)

    
"-------Problem/solver Setup--------"

ampl.option["solver"] = "ipopt"
ampl.solve()

"-------------Print solution------------"
pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
qvno= np.zeros([H,num_nodes])
qiko= np.zeros([H,num_lines])
pko= np.zeros([H,num_lines])
qko= np.zeros([H,num_lines])

for h in range(H):
    for i in range (num_nodes):
        qvno[h][i]=ampl.get_variable('qvn')[h+1,i+1].value()        
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()
    
    for k in range (num_lines):
        qiko[h][k]=ampl.get_variable('qik')[h+1,k+1].value()        
        pko[h][k]=ampl.get_variable('pk')[h+1,k+1].value()
        qko[h][k]=ampl.get_variable('qk')[h+1,k+1].value()


vo=np.sqrt(qvno)

plt.plot(vo)

Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for k in range(num_lines):        
        Equp[h][fr[k]-1]+=pko[h][k]
        Equp[h][to[k]-1]+=zr[k]*qiko[h][k]-pko[h][k]
        Equq[h][fr[k]-1]+=qko[h][k]
        Equq[h][to[k]-1]+=zi[k]*qiko[h][k]-qko[h][k]


pho=np.zeros([H,num_nodes])
beta=np.zeros([H,num_lines])
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        beta[h][k]=np.arctan((zr[k]*qko[h][k]-zi[k]*pko[h][k])/(qvno[h][fr[k]-1]-zr[k]*pko[h][k]-zi[k]*qko[h][k]))
        pho[h][to[k]-1]=pho[h][fr[k]-1]-beta[h][k]


t=np.zeros(H)
t[0]=ampl.getValue('_solve_elapsed_time')


out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t)).T
output=np.hstack((vo,pho,Equp,Equq,out1))
df=pd.DataFrame(output)


columns=[]
for i in range(num_nodes):
    columns.append('v'+str(i+1))
for i in range(num_nodes):    
    columns.append('ph'+str(i+1))
for i in range(num_nodes):    
    columns.append('eqp'+str(i+1))
for i in range(num_nodes):    
    columns.append('eqq'+str(i+1))


columns.append('pl')
columns.append('ql')
columns.append('pg')
columns.append('qg')
columns.append('t')
    
df.columns=columns

solvlist=[0]*H
solvlist[0]='AI'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")