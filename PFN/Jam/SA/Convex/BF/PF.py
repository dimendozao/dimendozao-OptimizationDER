# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:05:39 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cvx
from scipy.io import loadmat

case='SA'
city='Jam'
city1='JAM'
problem='PFN'
case1='SA_J23'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Gen.csv')

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
dmeans=np.squeeze(mat['clustermeans']).T

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=np.squeeze(mat['clusternode'])

cnode[0]=1
cnode=cnode-1

dem=np.mean(dmeans,axis=0)

num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
for i in range(num_nodes):
    sd[i]=sd[i]*dem[cnode[i]]    
pdm=np.real(sd)
qdm=np.imag(sd)

ngen=np.sum(bus['type']==2)
pgen=np.zeros(num_nodes)
qgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]
vmax=vmax+0.1
vmin=vmin-0.1         
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


"----- Optimization model -----"
pgref = cvx.Variable(num_nodes)
qgref = cvx.Variable(num_nodes)
qvn= cvx.Variable(num_nodes)
qik= cvx.Variable(num_lines)
pk= cvx.Variable(num_lines)
qk= cvx.Variable(num_lines)

"-------Constraint Construction-------- "

EqNp = num_nodes*[0]
EqNq = num_nodes*[0]

for k in range(num_lines):
    EqNp[fr[k]]+=pk[k]
    EqNp[to[k]]+=(np.real(zk[k])*qik[k])-pk[k]
    EqNq[fr[k]]+=qk[k]
    EqNq[to[k]]+=(np.imag(zk[k])*qik[k])-qk[k]

res=[]

res +=  [qvn>=qvnmin]
res +=  [qvn<=qvnmax]
res +=  [pgref>=prefmin]
res +=  [pgref<=prefmax]
res +=  [qgref>=qrefmin]
res +=  [qgref<=qrefmax] 

for k in range(num_lines):   
    res+= [qvn[fr[k]]-qvn[to[k]]==2*(pk[k]*np.real(zk[k])+qk[k]*np.imag(zk[k]))-qik[k]*(np.square(np.abs(zk[k])))]
    vip= qvn[branch['i'][k]-1]+qik[k]
    vim= qvn[branch['i'][k]-1]-qik[k]
    st= cvx.vstack([2*pk[k],2*qk[k],vim])
    res += [cvx.SOC(vip,st)] 

res +=  [cvx.hstack(EqNp)==pgref+pgen-pdm]
res +=  [cvx.hstack(EqNq)==qgref+qgen-qdm]         
          
"-------Objective definition--------"
#obj = cvx.Minimize(1)
#obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
obj = cvx.Minimize(cvx.sum(EqNp)+cvx.sum(EqNq))

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)

OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)
"-------------Print solution------------"

"----- Print results -----"
t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

qvno=qvn.value
qiko= qik.value
pko= pk.value
qko= qk.value

pgo=pgref.value
qgo=qgref.value
sgo=np.sqrt(np.square(pgo)+np.square(qgo))


vo=np.sqrt(qvno)

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = num_nodes*[0]
Equq = num_nodes*[0]

for k in range(num_lines):
    Equp[fr[k]]+=pko[k]
    Equp[to[k]]+=(np.real(zk[k])*qiko[k])-pko[k]
    Equq[fr[k]]+=qko[k]
    Equq[to[k]]+=(np.imag(zk[k])*qiko[k])-qko[k]


pho=np.zeros(num_nodes)
beta=np.zeros(num_lines)
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

for k in range(num_lines):
    beta[k]=np.arctan((np.real(zk[k])*qko[k]-np.imag(zk[k])*pko[k])/(qvno[fr[k]]-np.real(zk[k])*pko[k]-np.imag(zk[k])*qko[k]))
    pho[to[k]]=pho[fr[k]]-beta[k]


output=np.vstack((vo,pho,Equp,Equq,ploss,qloss,pgo,qgo,t)).T
df=pd.DataFrame(output)
columns=[]
columns.append('v')
columns.append('ph')
columns.append('eqp')
columns.append('eqq')
columns.append('pl')
columns.append('ql')
columns.append('pg')
columns.append('qg')
columns.append('t')
    
df.columns=columns
solvlist=[0]*num_nodes
solvlist[0]='CM'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")