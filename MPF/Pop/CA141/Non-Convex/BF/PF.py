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

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

case='CA141'
city='Pop'
city1='POP'
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

prefmax=np.tile(prefmax,(H,1))
prefmin=np.tile(prefmin,(H,1))

qrefmax=np.tile(qrefmax,(H,1))
qrefmin=np.tile(qrefmin,(H,1))

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

qvnmax=np.tile(qvnmax,(H,1))
qvnmin=np.tile(qvnmin,(H,1))

qikmax=np.ones(num_lines)
qikmin=np.zeros(num_lines)

pkmax=np.ones(num_lines)
pkmin=np.zeros(num_lines)
qkmax=np.ones(num_lines)
qkmin=np.zeros(num_lines)


"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

qvn= m.addMVar((H,num_nodes),lb=qvnmin,ub=qvnmax,name='qvn')
qik= m.addMVar((H,num_lines),name='qik')
pk= m.addMVar((H,num_lines),name='pk')
qk= m.addMVar((H,num_lines),name='qk')

"-------Constraint Construction-------- "

EqNp = [[0] * num_nodes for h in range(H)] 
EqNq = [[0] * num_nodes for h in range(H)] 

for h in range(H):
    for k in range(num_lines):
        EqNp[h][fr[k]]+=pk[h][k]
        EqNp[h][to[k]]+=(np.real(zk[k])*qik[h][k])-pk[h][k]
        EqNq[h][fr[k]]+=qk[h][k]
        EqNq[h][to[k]]+=(np.imag(zk[k])*qik[h][k])-qk[h][k]
    
    
m.addConstrs((qvn[h][fr[k]]-qvn[h][to[k]]==2*(pk[h][k]*np.real(zk[k])+qk[h][k]*np.imag(zk[k]))-qik[h][k]*(np.square(np.abs(zk[k]))) for h in range(H) for k in range(num_lines)),name='c-ph')
m.addConstrs((qik[h][k]*qvn[h][fr[k]]==pk[h][k]*pk[h][k]+qk[h][k]*qk[h][k] for h in range(H) for k in range(num_lines)),name='c-ncsocp')

            
m.addConstrs((EqNp[h][i]==pgref[h][i]+pgen[i]-pdm[h][i] for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-qdm[h][i] for h in range(H) for i in range(num_nodes)), name='c-pf-q')

"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj=gp.quicksum(EqNp[h][i] for h in range(H) for i in range(num_nodes))+gp.quicksum(EqNq[h][i] for h in range(H) for i in range(num_nodes))
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
m.setParam('NonConvex',2)
#m.setParam('Presolve',-1)
m.setParam('MIPFocus',2)
m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 120);

m.optimize()
"-------------Print solution------------"
all_vars = m.getVars()
values = m.getAttr("X", all_vars)
names = m.getAttr("VarName", all_vars)

pgov=np.array(values[:(num_nodes*H)])
qgov=np.array(values[num_nodes*H:(2*num_nodes*H)])
qvnov=np.array(values[2*num_nodes*H:3*num_nodes*H])
qikov=np.array(values[3*num_nodes*H:(3*num_nodes*H)+(num_lines*H)])
pkov=np.array(values[(3*num_nodes*H)+(H*num_lines):(3*num_nodes*H)+(2*num_lines*H)])
qkov=np.array(values[(3*num_nodes*H)+(2*num_lines*H):(3*num_nodes*H)+(3*num_lines*H)])


pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
qvno=np.zeros([H,num_nodes])
qiko=np.zeros([H,num_lines])
pko=np.zeros([H,num_lines])
qko=np.zeros([H,num_lines])


k=0
for h in range(H):
    for i in range(num_nodes):
        pgo[h][i]=pgov[k]
        qgo[h][i]=qgov[k]
        qvno[h][i]=qvnov[k]
        k+=1
        

vo=np.sqrt(qvno)

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

kk=0
for h in range(H):
    for k in range(num_lines):
        qiko[h][k]=qikov[kk]
        pko[h][k]=pkov[kk]
        qko[h][k]=qkov[kk]
        kk+=1

for h in range(H):
    for k in range(num_lines):        
        Equp[h][fr[k]]+=pko[h][k]
        Equp[h][to[k]]+=(np.real(zk[k])*qiko[h][k])-pko[h][k]
        Equq[h][fr[k]]+=qko[h][k]
        Equq[h][to[k]]+=(np.imag(zk[k])*qiko[h][k])-qko[h][k]


pho=np.zeros([H,num_nodes])
beta=np.zeros([H,num_lines])
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        beta[h][k]=np.arctan((np.real(zk[k])*qko[h][k]-np.imag(zk[k])*pko[h][k])/(qvno[h][fr[k]]-np.real(zk[k])*pko[h][k]-np.imag(zk[k])*qko[h][k]))
        pho[h][to[k]]=pho[h][fr[k]]-beta[h][k]


t=np.zeros(H)
t[0]=m.Runtime


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
solvlist[0]='G'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")
