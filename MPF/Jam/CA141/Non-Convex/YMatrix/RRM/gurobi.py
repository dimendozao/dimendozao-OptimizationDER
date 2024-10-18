# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:50:10 2024

@author: diego
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sp

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

case='CA141'
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
        if i==j:
            cnmax[i][i]=vmax[i]*vmax[i]
            cnmin[i][i]=vmin[i]*vmin[i]
        else:
            cnmax[i][j]=vmax[i]*vmax[j]
            cnmin[i][j]=vmin[i]*vmin[j]
            snmax[i][j]=vmax[i]*vmax[j]
            snmin[i][j]=-vmax[i]*vmax[j]

prefmax=np.tile(prefmax,(H,1))
prefmin=np.tile(prefmin,(H,1))
qrefmax=np.tile(qrefmax,(H,1))
qrefmin=np.tile(qrefmin,(H,1))

cnmax=np.tile(cnmax,(H,1,1))
cnmin=np.tile(cnmin,(H,1,1))

snmax=np.tile(snmax,(H,1,1))
snmin=np.tile(snmin,(H,1,1))

idx=np.abs(ym)!=0

nz=np.sum(idx)

ix=np.zeros(nz,dtype=int)
jx=np.zeros(nz,dtype=int)
yrx=np.zeros(nz)
yix=np.zeros(nz)

k=0

for i in range(num_nodes):
    for j in range(num_nodes):
        if not np.abs(ym[i][j])==0:
            ix[k]=i
            jx[k]=j
            yrx[k]=np.real(ym[i][j])
            yix[k]=np.imag(ym[i][j])
            k+=1
            
yrs=sp.coo_matrix((yrx,(ix,jx)),shape=(num_nodes,num_nodes))  
yis=sp.coo_matrix((yix,(ix,jx)),shape=(num_nodes,num_nodes))  
            
"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

cn= m.addMVar((H,num_nodes,num_nodes),lb=cnmin,ub=cnmax,name='cn')
sn= m.addMVar((H,num_nodes,num_nodes),lb=snmin,ub=snmax,name='sn')

"-------Constraint Construction-------- "

m.addConstrs(((cn[h][i][j]*cn[h][i][j])+(sn[h][i][j]*sn[h][i][j])==cn[h][i][i]*cn[h][j][j] for h in range(H) for i in range(num_nodes) for j in range(num_nodes) if idx[i][j] and j>i), name='c-soc')        
m.addConstrs((cn[h]==cn[h].transpose() for h in range(H)), name='c-her-1')
m.addConstrs((sn[h]==-sn[h].transpose() for h in range(H)), name='c-her-2')

m.addConstrs((cn[h]@yrs+sn[h]@yis==pgref[h]+pgen-pdm[h] for h in range(H)), name='c-pf-p')
m.addConstrs((sn[h]@yrs-cn[h]@yis==qgref[h]+qgen-qdm[h] for h in range(H)), name='c-pf-q')

EqNp=[[0]*num_nodes for h in range(H)]
EqNq=[[0]*num_nodes for h in range(H)]

pl=[0]*H
ql=[0]*H

for h in range(H):
    EqNp[h]=cn[h]@yrs+sn[h]@yis
    EqNq[h]=sn[h]@yrs-cn[h]@yis
    pl[h]=gp.quicksum(EqNp[h])
    ql[h]=gp.quicksum(EqNq[h])

"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj= gp.quicksum(pl)+gp.quicksum(ql)
#m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
m.setParam('NonConvex',2)
#m.setParam('Presolve',-1)
m.setParam('MIPFocus',2)
m.setParam('MIPGap',1e-6)
#m.setParam("TimeLimit", 120);

m.optimize()