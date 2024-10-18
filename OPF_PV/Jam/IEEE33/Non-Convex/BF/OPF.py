# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 08:23:54 2023

@author: diego
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat


case='IEEE33'
city='Jam'
city1='JAM'
problem='OPF_PV'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\MeansRAD_'+city1+'.mat')
imeans=np.squeeze(mat['means'])

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
dmeans=np.squeeze(mat['clustermeans']).T

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=np.squeeze(mat['clusternode'])

cnode[0]=1
cnode=cnode-1

rad=np.mean(imeans)
dem=np.mean(dmeans,axis=0)

num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    
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

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))
pveff=0.8   

"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar(num_nodes,lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar(num_nodes,lb=qrefmin,ub=qrefmax,name='qgref')

qvn= m.addMVar(num_nodes,lb=qvnmin,ub=qvnmax,name='qvn')
qik= m.addMVar(num_lines,name='qik')
pk= m.addMVar(num_lines,name='pk')
qk= m.addMVar(num_lines,name='qk')

pv= m.addMVar(num_nodes,lb=0,ub=pvcmax,name='pv')
ppv= m.addMVar(num_nodes,lb=0,ub=pvcmax,name='ppv')
zpv= m.addMVar(num_nodes,name='zpv',vtype=GRB.BINARY)

"-------Constraint Construction-------- "

EqNp = num_nodes*[0]
EqNq = num_nodes*[0]

for k in range(num_lines):
    EqNp[fr[k]]+=pk[k]
    EqNp[to[k]]+=(np.real(zk[k])*qik[k])-pk[k]
    EqNq[fr[k]]+=qk[k]
    EqNq[to[k]]+=(np.imag(zk[k])*qik[k])-qk[k]
    
    
m.addConstrs((qvn[fr[k]]-qvn[to[k]]==2*(pk[k]*np.real(zk[k])+qk[k]*np.imag(zk[k]))-qik[k]*(np.square(np.abs(zk[k]))) for k in range(num_lines)),name='c-ph')
m.addConstrs((qik[k]*qvn[fr[k]]==pk[k]*pk[k]+qk[k]*qk[k] for k in range(num_lines)),name='c-ivs1')
m.addConstrs((EqNp[i]==pgref[i]+(ppv[i]*rad*pveff)+pgen[i]-(pdm[i]*dem[cnode[i]]) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[i]==qgref[i]+qgen[i]-(qdm[i]*dem[cnode[i]]) for i in range(num_nodes)), name='c-pf-q')

m.addConstrs((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)), name='c-pv1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)), name='c-pv2')
m.addConstrs((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)), name='c-pv3')
m.addConstr(zpv.sum()==npv)
m.addConstr(ppv.sum()==pvcmax)

"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj=gp.quicksum(EqNp[i] for i in range(num_nodes))+gp.quicksum(EqNq[i] for i in range(num_nodes))
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('Aggregate',0)
m.setParam('NonConvex',2)
#m.setParam('Presolve',0)
m.setParam('MIPFocus',2)
#m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 120);

m.optimize()
"-------------Print solution------------"

pgo=pgref.X
qgo=qgref.X
qvno=qvn.X
qiko=qik.X
pko=pk.X
qko=qk.X
pvo=pv.X
ppvo=ppv.X
zpvo=zpv.X

vo=np.sqrt(qvno)

plt.plot(vo)

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


t=np.zeros(num_nodes)
t[0]=m.Runtime


output=np.vstack((vo,pho,Equp,Equq,ploss,qloss,pgo,qgo,pvo,ppvo,zpvo,t)).T
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
columns.append('pv')
columns.append('ppv')
columns.append('zpv')
columns.append('t')
    
df.columns=columns

solvlist=[0]*num_nodes
solvlist[0]='G'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")    
