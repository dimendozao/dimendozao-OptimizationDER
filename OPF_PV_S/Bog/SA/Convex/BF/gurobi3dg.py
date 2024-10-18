# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:27:25 2024

@author: diego
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Gen.csv")

mat=loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\Bog\\ClusterMeans_BOG.mat')


clusters=mat['clustermeans']

mat=loadmat("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\Bog\\MeansRAD_BOG.mat")

radmean=mat['means'].T

H=clusters.shape[1]

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

qvnhmax=np.zeros([H,num_nodes])
qvnhmin=np.zeros([H,num_nodes])
prefhmax=np.zeros([H,num_nodes])
prefhmin=np.zeros([H,num_nodes])
qrefhmax=np.zeros([H,num_nodes])
qrefhmin=np.zeros([H,num_nodes])


for h in range(H):
    qvnhmax[h]=qvnmax
    qvnhmin[h]=qvnmin
    prefhmax[h]=prefmax
    prefhmin[h]=prefmin
    qrefhmax[h]=qrefmax
    qrefhmin[h]=qrefmin


cpv=2.5
npv=3


"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar((H,num_nodes),lb=prefhmin,ub=prefhmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefhmin,ub=qrefhmax,name='qgref')

qvn= m.addMVar((H,num_nodes),lb=qvnhmin,ub=qvnhmax,name='qvn')
qik= m.addMVar((H,num_lines),name='qik')
pk= m.addMVar((H,num_lines),name='pk')
qk= m.addMVar((H,num_lines),name='qk')
z=  m.addMVar(num_nodes,vtype=gp.GRB.BINARY,name='z')
pv= m.addMVar(num_nodes,lb=0,ub=cpv,name='pv')
ppv= m.addMVar(num_nodes,lb=0,ub=cpv,name='ppv')

"-------Constraint Construction-------- "

EqNp = [num_nodes*[0] for h in range(H)]
EqNq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for k in range(num_lines):
        EqNp[h][fr[k]]+=pk[h][k]
        EqNp[h][to[k]]+=(np.real(zk[k])*qik[h][k])-pk[h][k]
        EqNq[h][fr[k]]+=qk[h][k]
        EqNq[h][to[k]]+=(np.imag(zk[k])*qik[h][k])-qk[h][k]
    
    
m.addConstrs((qvn[h][fr[k]]-qvn[h][to[k]]==2*(pk[h][k]*np.real(zk[k])+qk[h][k]*np.imag(zk[k]))-qik[h][k]*(np.square(np.abs(zk[k]))) for h in range(H) for k in range(num_lines)),name='c-ph')
m.addConstrs((qik[h][k]*qvn[h][fr[k]]>=pk[h][k]*pk[h][k]+qk[h][k]*qk[h][k] for h in range(H) for k in range(num_lines)),name='c-ivs1')
m.addConstrs((EqNp[h][i]==pgref[h][i]+(ppv[i]*radmean[h])+pgen[i]-(pdm[i]*clusters[1][h]) for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-(qdm[i]*clusters[1][h]) for h in range(H) for i in range(num_nodes)), name='c-pf-q')
m.addConstrs((ppv[i]<=z[i]*cpv for i in range(num_nodes)), name='c-pv1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)), name='c-pv2')
m.addConstrs((ppv[i]>=pv[i]-cpv*(1-z[i]) for i in range(num_nodes)), name='c-pv3')
m.addConstr(gp.quicksum(z[i] for i in range(num_nodes))==npv)

"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj=gp.quicksum(EqNp[h][i] for h in range(H) for i in range(num_nodes))+gp.quicksum(EqNq[h][i] for h in range(H) for i in range(num_nodes))
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('NonConvex',2)
#m.setParam('Presolve',-1)
m.setParam('MIPFocus',2)
m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 120);

m.optimize()

"-------------Print solution------------"
all_vars = m.getVars()
values = m.getAttr("X", all_vars)
names = m.getAttr("VarName", all_vars)

pgov=np.array(values[:num_nodes*H])
qgov=np.array(values[num_nodes*H:(2*num_nodes*H)])
qvnov=np.array(values[2*num_nodes*H:3*num_nodes*H])
qikov=np.array(values[3*num_nodes*H:((3*num_nodes*H)+(num_lines*H))])
pkov=np.array(values[((3*num_nodes*H)+(num_lines*H)):((3*num_nodes*H)+(2*num_lines*H))])
qkov=np.array(values[((3*num_nodes*H)+(2*num_lines*H)):((3*num_nodes*H)+(3*num_lines*H))])
zo= np.array(values[((3*num_nodes*H)+(3*num_lines*H)):((3*num_nodes*H)+(3*num_lines*H))+(num_nodes)])
pvo= np.array(values[((3*num_nodes*H)+(3*num_lines*H))+(num_nodes):((3*num_nodes*H)+(3*num_lines*H))+(2*num_nodes)])
ppvo= np.array(values[((3*num_nodes*H)+(3*num_lines*H))+(2*num_nodes):((3*num_nodes*H)+(3*num_lines*H))+(3*num_nodes)])

pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
qvno=np.zeros([H,num_nodes])
qiko=np.zeros([H,num_lines])
pko=np.zeros([H,num_lines])
qko=np.zeros([H,num_lines])
pdmo=np.zeros([H,num_nodes])

k=0
for h in range(H):
    for i in range(num_nodes):
        pgo[h][i]=pgov[k]
        qgo[h][i]=qgov[k]
        qvno[h][i]=qvnov[k]
        pdmo[h][i]=pdm[i]*clusters[1][h]
        k+=1

k=0
for h in range(H):
    for i in range(num_lines):
        qiko[h][i]=qikov[k]
        pko[h][i]=pkov[k]
        qko[h][i]=qkov[k]        
        k+=1 
        
plt.plot(np.sum(pgo,axis=1))
plt.plot(np.multiply(radmean[:,0],np.sum(ppvo)))
plt.plot(np.sum(pdmo,axis=1))
plt.legend(['pg','pv','pd'])