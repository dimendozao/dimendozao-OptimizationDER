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
problem='OPF_PV'


case='CA141'
city='Bog'
city1='BOG'
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

H=len(imeans)
num_lines = len(branch)
num_nodes=len(bus)
ncluster=np.size(dmeans,axis=1)
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

umax=np.tile(umax,(H,1))
umin=np.tile(umin,(H,1))

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

wrmax=np.zeros(num_lines)
wrmin=np.zeros(num_lines)
wimax=np.zeros(num_lines)
wimin=np.zeros(num_lines)
    
for k in range(num_lines):        
    wrmax[k]=vmax[fr[k]]*vmax[to[k]]
    wrmin[k]=vmin[fr[k]]*vmin[to[k]]
    wimax[k]=vmax[fr[k]]*vmax[to[k]]
    wimin[k]=-vmax[fr[k]]*vmax[to[k]]
    
wrmax=np.tile(wrmax,(H,1))
wrmin=np.tile(wrmin,(H,1))

wimax=np.tile(wimax,(H,1))
wimin=np.tile(wimin,(H,1))

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))
pveff=0.8

"----- Optimization model -----"

m = gp.Model("PF-rect")
pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

u= m.addMVar((H,num_nodes),lb=umin,ub=umax,name='u')
wr= m.addMVar((H,num_lines),lb=wrmin,ub=wrmax,name='wr')
wi= m.addMVar((H,num_lines),lb=wimin,ub=wimax,name='wi')


pv= m.addMVar(num_nodes,lb=0,ub=pvcmax,name='pv')
ppv= m.addMVar(num_nodes,lb=0,ub=pvcmax,name='pv')
zpv= m.addMVar(num_nodes,vtype=GRB.BINARY,name='zpv')

pl= m.addMVar(H,name='pl')
ql= m.addMVar(H,name='ql')

"-------Constraint Construction-------- "

EqNp = [num_nodes*[0] for h in range(H)]
EqNq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for i in range(num_nodes):        
        for k in range(num_lines):
            if i==fr[k]:
                EqNp[h][i]+=(u[h][i]*ylr[k])-(wr[h][k]*ylr[k])-(wi[h][k]*yli[k])
                EqNq[h][i]+=(-u[h][i]*yli[k])+(wr[h][k]*yli[k])-(wi[h][k]*ylr[k])
            if i==to[k]:            
                EqNp[h][i]+=(u[h][i]*ylr[k])-(wr[h][k]*ylr[k])+(wi[h][k]*yli[k])
                EqNq[h][i]+=(-u[h][i]*yli[k])+(wr[h][k]*yli[k])+(wi[h][k]*ylr[k])    

m.addConstrs((EqNp[h][i]==pgref[h][i]+(ppv[i]*imeans[h]*pveff)+pgen[i]-(pdm[i]*dmeans[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-(qdm[i]*dmeans[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-q')

m.addConstrs((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)), name='c-pv1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)), name='c-pv2')
m.addConstrs((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)), name='c-pv3')
m.addConstr(zpv.sum()==npv)
m.addConstr(ppv.sum()==pvcmax)

m.addConstrs((u[h][fr[k]]*u[h][to[k]]>=wr[h][k]*wr[h][k]+wi[h][k]*wi[h][k] for h in range(H) for k in range(num_lines)),name='c-socp') 

m.addConstrs((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)),name='c-pvlin1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)),name='c-pvlin2')
m.addConstrs((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)),name='c-pvlin3')
m.addConstr(zpv.sum()==npv)
m.addConstr(ppv.sum()==pvcmax)

m.addConstrs((gp.quicksum(EqNp[h][i] for i in range(num_nodes))==pl[h] for h in range(H)), name='c-ppl')
m.addConstrs((gp.quicksum(EqNq[h][i] for i in range(num_nodes))==ql[h] for h in range(H)), name='c-pql')   
    
"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
#obj=gp.quicksum(EqNp[i] for i in range(num_nodes))+gp.quicksum(EqNq[i] for i in range(num_nodes))
obj=pl.sum()+ql.sum()
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
m.setParam('NonConvex',1)
#m.setParam('Presolve',-1)
m.setParam('MIPFocus',3)
m.setParam('MIPGap',1e-6)
#m.setParam("TimeLimit", 120);

m.optimize()
"-------------Print solution------------"

all_vars = m.getVars()
values = m.getAttr("X", all_vars)
names = m.getAttr("VarName", all_vars)

pgov=np.array(values[:num_nodes*H])
qgov=np.array(values[num_nodes*H:(2*num_nodes*H)])
uov=np.array(values[(2*num_nodes*H):(3*num_nodes*H)])
wrov=np.array(values[(3*num_nodes*H):(3*num_nodes*H)+(num_lines*H)])
wiov=np.array(values[(3*num_nodes*H)+(num_lines*H):(3*num_nodes*H)+(2*num_lines*H)])
pvo=np.array(values[(3*num_nodes*H)+(2*num_lines*H):(3*num_nodes*H)+(2*num_lines*H)+num_nodes])
ppvo=np.array(values[(3*num_nodes*H)+(2*num_lines*H)+num_nodes:(3*num_nodes*H)+(2*num_lines*H)+(2*num_nodes)])
zpvo=np.array(values[(3*num_nodes*H)+(2*num_lines*H)+(2*num_nodes):(3*num_nodes*H)+(2*num_lines*H)+(3*num_nodes)])

pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
uo=np.zeros([H,num_nodes])
wro=np.zeros([H,num_lines])
wio=np.zeros([H,num_lines])


k=0
for h in range(H):
    for i in range(num_nodes):
        pgo[h][i]=pgov[k]
        qgo[h][i]=qgov[k]
        uo[h][i]=uov[k]        
        k+=1

kk=0
for h in range(H):
    for k in range(num_lines):
        wro[h][k]=wrov[kk]
        wio[h][k]=wiov[kk]        
        kk+=1

vo=np.sqrt(uo)

plt.plot(vo)


Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for i in range(num_nodes):        
        for k in range(num_lines):
            if i==fr[k]:
                Equp[h][i]+=(uo[h][i]*ylr[k])-(wro[h][k]*ylr[k])-(wio[h][k]*yli[k])
                Equq[h][i]+=(-uo[h][i]*yli[k])+(wro[h][k]*yli[k])-(wio[h][k]*ylr[k])
            if i==to[k]:            
                Equp[h][i]+=(uo[h][i]*ylr[k])-(wro[h][k]*ylr[k])+(wio[h][k]*yli[k])
                Equq[h][i]+=(-uo[h][i]*yli[k])+(wro[h][k]*yli[k])+(wio[h][k]*ylr[k])  

pho=np.zeros([H,num_nodes])
beta=np.zeros([H,num_lines])
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        pho[h][to[k]]=pho[h][fr[k]]-np.angle(wro[h][k]+1j*wio[h][k])

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
