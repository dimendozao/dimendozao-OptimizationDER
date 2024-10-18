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
city='Bog'
city1='BOG'
problem='OPF_PV_D'


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

H=len(imeans)
num_lines = len(branch)
num_nodes=len(bus)
ncluster=np.size(dmeans,axis=1)
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

vmax=np.tile(vmax,(H,1))
vmin=np.tile(vmin,(H,1))

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

qikmax=np.ones(num_lines)
qikmin=np.zeros(num_lines)

pkmax=np.ones(num_lines)
pkmin=np.zeros(num_lines)
qkmax=np.ones(num_lines)
qkmin=np.zeros(num_lines)

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(pdm)

pveff=0.8

"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

qvn= m.addMVar((H,num_nodes),lb=qvnmin,ub=qvnmax,name='qvn')
qik= m.addMVar((H,num_lines),name='qik')
pk= m.addMVar((H,num_lines),name='pk')
qk= m.addMVar((H,num_lines),name='qk')

pv= m.addMVar(num_nodes,name='pv')
ppv= m.addMVar(num_nodes,name='ppv')
zpv= m.addMVar(num_nodes,name='zpv',vtype=GRB.BINARY)

pl= m.addMVar(H,name='pl')
ql= m.addMVar(H,name='ql')

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
m.addConstrs((qik[h][k]*qvn[h][fr[k]]>=pk[h][k]*pk[h][k]+qk[h][k]*qk[h][k] for h in range(H) for k in range(num_lines)),name='c-socp')

            
m.addConstrs((EqNp[h][i]==pgref[h][i]+(ppv[i]*imeans[h]*pveff)+pgen[i]-(pdm[i]*dmeans[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-(qdm[i]*dmeans[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-q')

m.addConstrs((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)), name='c-pv1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)), name='c-pv2')
m.addConstrs((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)), name='c-pv3')
m.addConstr(zpv.sum()==npv)
m.addConstr(ppv.sum()==pvcmax)

m.addConstrs((gp.quicksum(EqNp[h][i] for i in range(num_nodes))==pl[h] for h in range(H)), name='c-ppl')
m.addConstrs((gp.quicksum(EqNq[h][i] for i in range(num_nodes))==ql[h] for h in range(H)), name='c-pql')

"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj=pl.sum()+ql.sum()
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
m.setParam('NonConvex',1)
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
qvnov=np.array(values[(2*num_nodes*H):(3*num_nodes*H)])
qikov=np.array(values[(3*num_nodes*H):(3*num_nodes*H)+(num_lines*H)])
pkov=np.array(values[(3*num_nodes*H)+(num_lines*H):(3*num_nodes*H)+(2*num_lines*H)])
qkov=np.array(values[(3*num_nodes*H)+(2*num_lines*H):(3*num_nodes*H)+(3*num_lines*H)])
pvo=np.array(values[(3*num_nodes*H)+(3*num_lines*H):(3*num_nodes*H)+(3*num_lines*H)+(num_nodes)])
ppvo=np.array(values[(3*num_nodes*H)+(3*num_lines*H)+(num_nodes):(3*num_nodes*H)+(3*num_lines*H)+(2*num_nodes)])
zpvo=np.array(values[(3*num_nodes*H)+(3*num_lines*H)+(2*num_nodes):(3*num_nodes*H)+(3*num_lines*H)+(3*num_nodes)])


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

kk=0
for h in range(H):
    for k in range(num_lines):
        qiko[h][k]=qikov[kk]
        pko[h][k]=pkov[kk]
        qko[h][k]=qkov[kk]
        kk+=1    

vo=np.sqrt(qvno)

plt.plot(vo)

Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

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

ppvout=np.zeros([H,num_nodes])
zpvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=ppvo
    zpvout[h]=zpvo


out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,imeans)).T
output=np.hstack((vo,pho,Equp,Equq,out1,dmeans,ppvout,zpvout))
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
columns.append('ic')
for i in range(ncluster):    
    columns.append('dc_c'+str(i+1))
for i in range(num_nodes):    
    columns.append('ppv'+str(i+1))
for i in range(num_nodes):    
    columns.append('zpv'+str(i+1))
    
df.columns=columns

solvlist=[0]*H
solvlist[0]='G'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")
