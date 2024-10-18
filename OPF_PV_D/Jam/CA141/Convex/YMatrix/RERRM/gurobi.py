# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:37:33 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.io import loadmat


case='CA141'
city='Bog'
city1='BOG'
problem='OPF_PV'


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\MeansRAD_'+city1+'.mat')
imeans=np.squeeze(mat['means'])

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
dmeans=np.squeeze(mat['clustermeans']).T

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=np.squeeze(mat['clusternode'])

cnode[0]=1
cnode=cnode-1

H=len(imeans)
num_lines = len(branch)
num_nodes=len(bus)
ncluster=np.size(dmeans,axis=1)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    
sdh=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for i in range(num_nodes):
       sdh[h][i]=sd[i]*dmeans[h][cnode[i]]

pdh=np.real(sdh)
qdh=np.imag(sdh)

ngen=np.sum(bus['type']==2)
pgen=np.zeros(num_nodes)
qgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

vrmax=np.zeros(num_nodes)
vrmin=np.zeros(num_nodes)

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

vrmax=np.array([vmax[i] for i in range(num_nodes)])
vrmin=np.array([vmin[i] for i in range(num_nodes)])

vimax=np.array([vmax[i] for i in range(num_nodes)])
vimin=np.array([-vmax[i] for i in range(num_nodes)])

vimax[iref]=0
vimin[iref]=0

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

ym=np.zeros([num_nodes,num_nodes],dtype='complex')

idx2=np.zeros([num_nodes,num_nodes])

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])
    idx2[fr][to]=1
    idx2[to][fr]=1    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    
ymr=np.real(ym)
ymi=np.imag(ym)

idx1=idx2+np.eye(num_nodes)

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx1[i][j]==1:
            if i==j:
                cnmax[i][i]=vmax[i]*vmax[i]
                cnmin[i][i]=vmin[i]*vmin[i]
            else:
                cnmax[i][j]=vmax[i]*vmax[j]
                cnmin[i][j]=vmin[i]*vmin[j]
                snmax[i][j]=vmax[i]*vmax[j]
                snmin[i][j]=-vmax[i]*vmax[j]

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))
pveff=0.8

"----- Optimization model -----"

m = gp.Model("PF-rect")
pgref = m.addMVar((H,num_nodes),lb=0,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=0,ub=qrefmax,name='qgref')

cn= m.addMVar((H,num_nodes,num_nodes),lb=cnmin,ub=cnmax,name='cn')
sn= m.addMVar((H,num_nodes,num_nodes),lb=snmin,ub=snmax,name='sn')

pv= m.addMVar(num_nodes,lb=0,ub=pvcmax,name='pv')
ppv= m.addMVar(num_nodes,lb=0,ub=pvcmax,name='ppv')
zpv= m.addMVar(num_nodes,vtype=GRB.BINARY,name='zpv')

pl= m.addMVar(H,name='pl')
ql= m.addMVar(H,name='ql')

"-------Constraint Construction-------- "

EqNp = [num_nodes*[0] for h in range(H)]
EqNq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for i in range(num_nodes):        
        for j in range(num_lines):
            EqNp[h][i]+=cn[h][i][j]*ymr[i][j]+sn[h][i][j]*ymi[i][j]
            EqNq[h][i]+=sn[h][i][j]*ymr[i][j]-cn[h][i][j]*ymi[i][j]
            if idx2[i][j]:
                m.addConstr(cn[h][i][j]==cn[h][j][i],name='c-her1')
                m.addConstr(sn[h][i][j]==-sn[h][j][i],name='c-her2')

m.addConstrs((pgref[h][i]+(ppv[i]*imeans[h]*pveff)+pgen[i]-pdh[h]==EqNp[h][i] for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((qgref[h][i]+qgen[i]-qdh[h]==EqNq[h][i] for h in range(H) for i in range(num_nodes)), name='c-pf-q')
m.addConstrs((cn[h][i][i]*cn[h][j][j]>=cn[h][i][j]*cn[h][i][j]+sn[h][i][j]*sn[h][i][j] for h in range(H) for i in range(num_nodes) for j in range(num_nodes) if idx2[i][j]==1),name='c-socp')

m.addConstrs((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)),name='c-pvlin1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)),name='c-pvlin2')
m.addConstrs((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)),name='c-pvlin3')
m.addConstr(zpv.sum()==npv)
m.addConstr(ppv.sum()==pvcmax)

m.addConstrs((gp.quicksum(EqNp[h][i] for i in range(num_nodes))==pl[h] for h in range(H)), name='c-ppl')
m.addConstrs((gp.quicksum(EqNq[h][i] for i in range(num_nodes))==ql[h] for h in range(H)), name='c-pql')

"-------Objective definition--------"

obj=pl.sum()+ql.sum()
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('NonConvex',1)
m.setParam('Presolve',0)
m.setParam('Aggregate',0)
m.setParam('BarHomogeneous',1)
m.setParam('MIPFocus',3)
#m.setParam('MIPGap',1e-6)
#m.setParam("TimeLimit", 120);

m.optimize()
"----- Print results -----"

m.computeIIS()
if m.IISMinimal:
    print("IIS is minimal\n")
else:
    print("IIS is not minimal\n")

print("\nThe following constraint(s) cannot be satisfied:")

for c in m.getConstrs():
    if c.IISConstr:
        print(c.ConstrName)

for v in m.getVars():
    if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
    if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')