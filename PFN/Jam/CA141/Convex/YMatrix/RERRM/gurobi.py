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


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

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

vrmax=np.zeros(num_nodes)
vrmin=np.zeros(num_nodes)

if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]
        

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

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    
ymr=np.real(ym)
ymi=np.imag(ym)

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if not np.abs(ym[i][j])==0:
            if i==j:
                cnmax[i][i]=vmax[i]*vmax[i]
                cnmin[i][i]=vmin[i]*vmin[i]
            else:
                cnmax[i][j]=vmax[i]*vmax[j]
                cnmin[i][j]=vmin[i]*vmin[j]
                snmax[i][j]=vmax[i]*vmax[j]
                snmin[i][j]=-vmax[i]*vmax[j]


"----- Optimization model -----"

m = gp.Model("PF-rect")
pgref = m.addMVar(num_nodes,lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar(num_nodes,lb=qrefmin,ub=qrefmax,name='qgref')

cn= m.addMVar((num_nodes,num_nodes),lb=cnmin,ub=cnmax,name='cn')
sn= m.addMVar((num_nodes,num_nodes),lb=snmin,ub=snmax,name='sn')


"-------Constraint Construction-------- "


EqNp = [0] * num_nodes
EqNq = [0] * num_nodes 

for i in range(num_nodes):        
    for j in range(num_lines):
        EqNp[i]+=cn[i][j]*ymr[i][j]+sn[i][j]*ymi[i][j]
        EqNq[i]+=sn[i][j]*ymr[i][j]-cn[i][j]*ymi[i][j]
        if i<j:
            m.addConstr(cn[i][j]==cn[j][i],name='c-her1')
            m.addConstr(sn[i][j]==-sn[j][i],name='c-her2')

m.addConstrs((pgref[i]+pgen[i]-pdm[i]==EqNp[i] for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((qgref[i]+qgen[i]-qdm[i]==EqNq[i] for i in range(num_nodes)), name='c-pf-q')
m.addConstrs((cn[i][i]*cn[j][j]>=cn[i][j]*cn[i][j]+sn[i][j]*sn[i][j] for i in range(num_nodes) for j in range(num_nodes)),name='c-socp')

"-------Objective definition--------"

obj=gp.quicksum(EqNp[i] for i in range(num_nodes))+gp.quicksum(EqNq[i] for i in range(num_nodes))
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('NonConvex',1)
#m.setParam('Presolve',0)
m.setParam('BarHomogeneous',1)
m.setParam('MIPFocus',2)
m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 120);

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