# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:26:47 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.io import loadmat


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Gen.csv")

case='IEEE69'
city='Jam'
city1='JAM'
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



if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

vmax=vmax+0.1
vmin=vmin-0.1

umax=vmax**2
umin=vmin**2        

umax[iref]=1
umin[iref]=1

umax=np.tile(umax,(H,1))
umin=np.tile(umin,(H,1))

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=np.maximum(0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]])


prefmax=np.tile(prefmax,(H,1))
prefmin=np.tile(prefmin,(H,1))

qrefmax=np.tile(qrefmax,(H,1))
qrefmin=np.tile(qrefmin,(H,1))

wrmax=np.zeros(num_lines)
wrmin=np.zeros(num_lines)
wimax=np.zeros(num_lines)
wimin=np.zeros(num_lines)

fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1        
    wrmax[k]=vmax[fr[k]]*vmax[to[k]]
    wrmin[k]=vmin[fr[k]]*vmin[to[k]]
    wimax[k]=vmax[fr[k]]*vmax[to[k]]
    wimin[k]=-vmax[fr[k]]*vmax[to[k]]

wrmax=np.tile(wrmax,(H,1))
wrmin=np.tile(wrmin,(H,1))
wimax=np.tile(wimax,(H,1))
wimin=np.tile(wimin,(H,1))

"----- Optimization model -----"

m = gp.Model("PF-rect")
pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

u= m.addMVar((H,num_nodes),lb=umin,ub=umax,name='u')
wr= m.addMVar((H,num_lines),lb=wrmin,ub=wrmax,name='wr')
wi= m.addMVar((H,num_lines),lb=wimin,ub=wimax,name='wi')

"-------Constraint Construction-------- "

EqNp = [[0] * num_nodes for h in range(H)] 
EqNq = [[0] * num_nodes for h in range(H)] 

for h in range(H):
    for i in range(num_nodes):        
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                EqNp[h][i]+=(u[h][i]*yr)-(wr[h][k]*yr)-(wi[h][k]*yi)
                EqNq[h][i]+=(-u[h][i]*yi)+(wr[h][k]*yi)-(wi[h][k]*yr)
            if i==(branch['j'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                EqNp[h][i]+=(u[h][i]*yr)-(wr[h][k]*yr)+(wi[h][k]*yi)
                EqNq[h][i]+=(-u[h][i]*yi)+(wr[h][k]*yi)+(wi[h][k]*yr)    

m.addConstrs((EqNp[h][i]==pgref[h][i]+pgen[i]-pdm[h][i] for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-qdm[h][i] for h in range(H) for i in range(num_nodes)), name='c-pf-q')
m.addConstrs((u[h][fr[k]]*u[h][to[k]]>=wr[h][k]*wr[h][k]+wi[h][k]*wi[h][k] for h in range(H) for k in range(num_lines)),name='c-socp')   

#m.addConstrs((gp.quicksum(EqNp[h][i] for i in range(num_nodes))>=0 for h in range(H)),name='c-pos-pl')
#m.addConstrs((gp.quicksum(EqNq[h][i] for i in range(num_nodes))>=0 for h in range(H)),name='c-pos-ql')   

"-------Objective definition--------"
#obj=1
obj = pgref.sum() + qgref.sum()
#obj=gp.quicksum(EqNp[h][i] for h in range(H) for i in range(num_nodes))+gp.quicksum(EqNq[h][i] for h in range(H) for i in range(num_nodes))
m.setObjective(obj, GRB.MINIMIZE)



"-------Problem/solver Setup--------"
#m.setParam('NonConvex',1)
m.setParam('Aggregate',0)
m.setParam('Presolve',0)
m.setParam('BarConvTol',1e-15)
m.setParam('NumericFocus',3)
m.setParam('BarHomogeneous',1)
m.setParam('MIPFocus',2)
m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 120);

m.optimize()
"----- Print results -----"

t=np.zeros(H)
t[0]=m.Runtime

all_vars = m.getVars()
values = m.getAttr("X", all_vars)
names = m.getAttr("VarName", all_vars)

pgov=np.array(values[:num_nodes*H])
qgov=np.array(values[num_nodes*H:(2*num_nodes*H)])
uov=np.array(values[2*num_nodes*H:3*num_nodes*H])
wrov=np.array(values[3*num_nodes*H:(3*num_nodes*H)+(num_lines*H)])
wiov=np.array(values[(3*num_nodes*H)+(num_lines*H):(3*num_nodes*H)+(2*num_lines*H)])

uo=np.zeros([H,num_nodes])
pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
wro=np.zeros([H,num_lines])
wio=np.zeros([H,num_lines])

kk=0
for h in range(H):
    for i in range(num_nodes):
        uo[h][i]=uov[kk]
        pgo[h][i]=pgov[kk]
        qgo[h][i]=qgov[kk]
        kk+=1
    
kk=0
for h in range(H):
    for k in range(num_lines):
        wro[h][k]=wrov[kk]
        wio[h][k]=wiov[kk]
        kk+=1 

v=np.zeros([H,num_nodes])

for h in range(H):
    v[h]=np.sqrt(uo[h])

plt.plot(v)

Equp =[[0] * num_nodes for h in range(H)]
Equq =[[0] * num_nodes for h in range(H)]
 

for h in range(H):
    for i in range(num_nodes):
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                Equp[h][i]+=(uo[h][i]*yr)-(wro[h][k]*yr)-(wio[h][k]*yi)
                Equq[h][i]+=(-uo[h][i]*yi)+(wro[h][k]*yi)-(wio[h][k]*yr)            
            if i==(branch['j'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                Equp[h][i]+=(uo[h][i]*yr)-(wro[h][k]*yr)+(wio[h][k]*yi)
                Equq[h][i]+=(-uo[h][i]*yi)+(wro[h][k]*yi)+(wio[h][k]*yr)       

ploss=np.zeros(H)
qloss=np.zeros(H)

ph=np.zeros([H,num_nodes])

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        to=branch['j'][k]-1
        fr=branch['i'][k]-1
        ph[h][to]=ph[h][fr]-np.angle(wro[h][k]+1j*wio[h][k])
   
out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t)).T
output=np.hstack((v,ph,Equp,Equq,out1))
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