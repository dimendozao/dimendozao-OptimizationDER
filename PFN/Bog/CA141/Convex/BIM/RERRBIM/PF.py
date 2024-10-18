# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:34:07 2024

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
problem='PFN'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')

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
ylr=np.zeros(num_lines)
yli=np.zeros(num_lines)
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    ylr[k]=np.real(1/(branch['r'][k] + 1j*branch['x'][k]))
    yli[k]=np.imag(1/(branch['r'][k] + 1j*branch['x'][k]))
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

umax=vmax**2
umin=vmin**2        

umax[iref]=1
umin[iref]=1

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

wrmax=np.zeros(num_lines)
wrmin=np.zeros(num_lines)
wimax=np.zeros(num_lines)
wimin=np.zeros(num_lines)
    
for k in range(num_lines):        
    wrmax[k]=vmax[fr[k]]*vmax[to[k]]
    wrmin[k]=vmin[fr[k]]*vmin[to[k]]
    wimax[k]=vmax[fr[k]]*vmax[to[k]]
    wimin[k]=-vmax[fr[k]]*vmax[to[k]]


"----- Optimization model -----"

m = gp.Model("PF-rect")
pgref = m.addMVar(num_nodes,lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar(num_nodes,lb=qrefmin,ub=qrefmax,name='qgref')

u= m.addMVar(num_nodes,lb=umin,ub=umax,name='u')
wr= m.addMVar(num_lines,lb=wrmin,ub=wrmax,name='wr')
wi= m.addMVar(num_lines,lb=wimin,ub=wimax,name='wi')

"-------Constraint Construction-------- "

EqNp = [0] * num_nodes
EqNq = [0] * num_nodes 

for i in range(num_nodes):        
    for k in range(num_lines):
        if i==fr[k]:
            EqNp[i]+=(u[i]*ylr[k])-(wr[k]*ylr[k])-(wi[k]*yli[k])
            EqNq[i]+=(-u[i]*yli[k])+(wr[k]*yli[k])-(wi[k]*ylr[k])
        if i==to[k]:            
            EqNp[i]+=(u[i]*ylr[k])-(wr[k]*ylr[k])+(wi[k]*yli[k])
            EqNq[i]+=(-u[i]*yli[k])+(wr[k]*yli[k])+(wi[k]*ylr[k])    

m.addConstrs((EqNp[i]==pgref[i]+pgen[i]-pdm[i] for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[i]==qgref[i]+qgen[i]-qdm[i] for i in range(num_nodes)), name='c-pf-q')

m.addConstrs((u[fr[k]]*u[to[k]]>=wr[k]*wr[k]+wi[k]*wi[k] for k in range(num_lines)),name='c-socp')    
    
"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj=gp.quicksum(EqNp[i] for i in range(num_nodes))+gp.quicksum(EqNq[i] for i in range(num_nodes))
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('NonConvex',1)
#m.setParam('BarConvTol',1e-15)
m.setParam('Presolve',0)
m.setParam('Aggregate',0)
m.setParam('BarHomogeneous',1)
m.setParam('MIPFocus',2)
m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 120);

m.optimize()
"----- Print results -----"

all_vars = m.getVars()
values = m.getAttr("X", all_vars)
names = m.getAttr("VarName", all_vars)

pgo=np.array(values[:num_nodes])
qgo=np.array(values[num_nodes:(2*num_nodes)])
uo=np.array(values[2*num_nodes:3*num_nodes])
wro=np.array(values[3*num_nodes:(3*num_nodes)+(num_lines)])
wio=np.array(values[(3*num_nodes)+(num_lines):(3*num_nodes)+(2*num_lines)])


vo=np.sqrt(uo)

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = num_nodes*[0]
Equq = num_nodes*[0]


for i in range(num_nodes):        
    for k in range(num_lines):
        if i==fr[k]:
            Equp[i]+=(uo[i]*ylr[k])-(wro[k]*ylr[k])-(wio[k]*yli[k])
            Equq[i]+=(-uo[i]*yli[k])+(wro[k]*yli[k])-(wio[k]*ylr[k])
        if i==to[k]:            
            Equp[i]+=(uo[i]*ylr[k])-(wro[k]*ylr[k])+(wio[k]*yli[k])
            Equq[i]+=(-uo[i]*yli[k])+(wro[k]*yli[k])+(wio[k]*ylr[k])  

ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

ph=np.zeros(num_nodes)

for k in range(num_lines):
    ph[to[k]]=ph[fr[k]]-np.angle(wro[k]+1j*wio[k])

t=np.zeros(num_nodes)
t[0]=m.Runtime
    
# vrec=(vo*np.cos(ph)+1j*vo*np.sin(ph))

# sinfeas1=np.zeros(num_nodes,dtype='complex')
# sinfeas2=np.zeros(num_nodes,dtype='complex')
# pinfeas1=np.zeros(num_nodes)
# pinfeas2=np.zeros(num_nodes)
# qinfeas1=np.zeros(num_nodes)
# qinfeas2=np.zeros(num_nodes)


# pinfeas1=pgo-Equp-np.real(sd)
# qinfeas1=qgo-Equq-np.imag(sd)

# ym=np.zeros([num_nodes,num_nodes],dtype='complex')
# for k in range(num_lines):
#     ym[fr[k]][to[k]]=-(ylr[k]+1j*yli[k])
#     ym[to[k]][fr[k]]=-(ylr[k]+1j*yli[k])
    
# for i in range(num_nodes):
#     ym[i][i]=-np.sum(ym[i])

# equ2= np.multiply(vrec,np.matmul(ym.conjugate(),vrec.conjugate()))   
# pinfeas2=pgo-np.real(equ2)-np.real(sd)
# qinfeas2=qgo-np.imag(equ2)-np.imag(sd)
   
#output=np.vstack((vo,ph,Equp,Equq,ploss,qloss,pgo,qgo,pinfeas1,pinfeas2,qinfeas1,qinfeas2,t)).T

output=np.vstack((vo,ph,Equp,Equq,ploss,qloss,pgo,qgo,t)).T

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
# columns.append('pinf1')
# columns.append('pinf2')
# columns.append('qinf1')
# columns.append('qinf2')
columns.append('t')

    
df.columns=columns

solvlist=[0]*num_nodes
solvlist[0]='G'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")
