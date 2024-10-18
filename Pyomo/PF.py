# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:23:01 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyomo.environ as pyo
from amplpy import modules

os.environ['NEOS_EMAIL'] = 'dimendozao@unal.edu.co' 

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69RBranch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69RBus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69RGen.csv")

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

phmin=np.ones(num_nodes)*(-np.pi)
phmax=np.ones(num_nodes)*(np.pi)

phmin[iref]=0
phmax[iref]=0

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

qvnmax=vmax**2
qvnmin=vmin**2

qvlmax=vmax[1:]**2
qvlmin=vmin[1:]**2

phmmin=np.ones(num_lines)*(-np.pi)
phmmax=np.ones(num_lines)*(np.pi)

#yr = dict(enumerate(np.real(ym), 1))
#yi = dict(enumerate(np.imag(ym), 1))
yr=np.real(ym)
yi=np.imag(ym)

cphmax=1
cphmin=0
sphmax=0.5
sphmin=-0.5

"----- Optimization model -----"
model = pyo.ConcreteModel()

model.n =   pyo.Param(initialize=num_nodes)
model.l =   pyo.Param(initialize=num_lines)

model.i = pyo.RangeSet(model.n)
model.k = pyo.RangeSet(model.l)

def yr_init(model, i, j):
    return yr[i-1][j-1]

def yi_init(model, i, j):
    return yi[i-1][j-1]

def pd_init(model, i):
    return pdm[i-1]

def qd_init(model, i):
    return qdm[i-1]

def pgen_init(model, i):
    return pgen[i-1]

def qgen_init(model, i):
    return qgen[i-1]
    
model.yr =  pyo.Param(model.i,model.i,initialize=yr_init)
model.yi =  pyo.Param(model.i,model.i,initialize=yi_init)
model.pd =  pyo.Param(model.i,initialize=pd_init)
model.qd =  pyo.Param(model.i,initialize=qd_init)
model.pgen =  pyo.Param(model.i,initialize=pgen_init)
model.qgen =  pyo.Param(model.i,initialize=qgen_init)

def pgrefb(model, i):
    return (prefmin[i-1], prefmax[i-1])

def qgrefb(model, i):
    return (qrefmin[i-1], qrefmax[i-1])

def vb(model, i):
    return (vmin[i-1], vmax[i-1])

def phb(model, i):
    return (phmin[i-1], phmax[i-1])

def qvnb(model, i):
    return (qvnmin[i-1], qvnmax[i-1])

def qvlb(model, k):
    return (qvlmin[k-1], qvlmax[k-1])

def phmb(model, k):
    return (phmmin[k-1], phmmax[k-1])

model.pgref= pyo.Var(model.i,domain=pyo.NonNegativeReals,bounds=pgrefb)
model.qgref= pyo.Var(model.i,domain=pyo.Reals,bounds=qgrefb)

model.v= pyo.Var(model.i,domain=pyo.NonNegativeReals,bounds=vb)
model.ph= pyo.Var(model.i,domain=pyo.Reals,bounds=phb)


model.qvn= pyo.Var(model.i,domain=pyo.NonNegativeReals,bounds=qvnb)
model.qvl= pyo.Var(model.k,domain=pyo.NonNegativeReals,bounds=qvlb)
model.phm= pyo.Var(model.k,domain=pyo.Reals,bounds=phmb)

"-------Constraint Construction-------- "

model.res = pyo.ConstraintList()

for i in model.i:
    model.res.add(model.qvn[i]==model.v[i]*model.v[i])
    
    
for k in model.k:
    fr=branch['i'][k-1]
    to=branch['j'][k-1]
    model.res.add(model.qvl[k]==model.v[fr]*model.v[to])
    model.res.add(model.phm[k]==model.ph[fr]-model.ph[to])

EqNp = num_nodes*[0]
EqNq = num_nodes*[0]
    
for i in model.i:
    EqNp[i-1]+=(model.qvn[i]*model.yr[i,i])
    EqNq[i-1]+=(-model.qvn[i]*model.yi[i,i])
    for k in model.k:
        fr=branch['i'][k-1]
        to=branch['j'][k-1]
        if i==fr:
            EqNp[i-1]+=(model.qvl[k]*model.yr[fr,to]*pyo.cos(model.phm[k]))+(model.qvl[k]*model.yi[fr,to]*pyo.sin(model.phm[k]))
            EqNq[i-1]+=-(model.qvl[k]*model.yi[fr,to]*pyo.cos(model.phm[k]))+(model.qvl[k]*model.yr[fr,to]*pyo.sin(model.phm[k]))
        if i==to:
            EqNp[i-1]+=(model.qvl[k]*model.yr[to,fr]*pyo.cos(model.phm[k]))-(model.qvl[k]*model.yi[to,fr]*pyo.sin(model.phm[k]))
            EqNq[i-1]+=-(model.qvl[k]*model.yi[to,fr]*pyo.cos(model.phm[k]))-(model.qvl[k]*model.yr[to,fr]*pyo.sin(model.phm[k]))


for i in model.i:
    model.res.add(EqNp[i-1]==model.pgref[i]+model.pgen[i]-model.pd[i])
    model.res.add(EqNq[i-1]==model.qgref[i]+model.qgen[i]-model.qd[i])

model.res.add(pyo.quicksum(EqNp[i-1] for i in model.i)>=0)
model.res.add(pyo.quicksum(EqNq[i-1] for i in model.i)>=0)

"-------Objective definition--------"

def ObjRule(model):
    #return pyo.quicksum(model.pgref[i]+model.qgref[i] for i in model.i)
    return pyo.quicksum(EqNp[i-1]+EqNq[i-1] for i in model.i)

model.obj1 = pyo.Objective(rule=ObjRule)

"-------Problem/solver Setup--------"
#opt = pyo.SolverFactory("ipopt",executable='C:\\Users\\diego\\Downloads\\Ipopt-3.14.12-win64-msvs2019-md\\Ipopt-3.14.12-win64-msvs2019-md\\bin\\')
#results=opt.solve(model)

#solver = pyo.SolverFactory('ipopt',executable=modules.find("ipopt"), solve_io="nl")
#results=solver(model,tee=True)

solver_manager = pyo.SolverManagerFactory('neos')
opt = pyo.SolverFactory('ipopt', tee=True)       
results=solver_manager.solve(model,opt=opt,tee=True)

results.write()

"-------------Print solution------------"
pgo=np.zeros(num_nodes)
qgo=np.zeros(num_nodes)
vo=np.zeros(num_nodes)
pho=np.zeros(num_nodes)
qvno=np.zeros(num_nodes)

for i in model.i:
    pgo[i-1]=model.pgref[i].value
    qgo[i-1]=model.pgref[i].value
    vo[i-1]=model.v[i].value
    pho[i-1]=model.pho[i].value
    qvno[i-1]=model.qvn[i].value
    
qvlo=np.zeros(num_lines)
phmo=np.zeros(num_lines)

for k in model.k:
    qvlo[k-1]=model.qvl[k].value
    phmo[k-1]=model.phm[k].value

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)
qlo=np.sum(qgo)+np.sum(qgen)-np.sum(qdm)

Equp = num_nodes*[0]
Equq = num_nodes*[0]

for i in range(num_nodes):
    Equp[i]+=(qvno[i]*yr[i][i])
    Equq[i]+=(-qvno[i]*yi[i][i])
    for k in range(num_lines):
        fr=branch['i'][k]-1
        to=branch['j'][k]-1
        if i==fr:
            Equp[i]+=(qvlo[k]*yr[fr][to]*np.cos(phmo[k]))+(qvlo[k]*yi[fr][to]*np.sin(phmo[k]))
            Equq[i]+=-(qvlo[k]*yi[fr][to]*np.cos(phmo[k]))+(qvlo[k]*yr[fr][to]*np.sin(phmo[k]))
        if i==to:
            Equp[i]+=(qvlo[k]*yr[to][fr]*np.cos(phmo[k]))-(qvlo[k]*yi[to][fr]*np.sin(phmo[k]))
            Equq[i]+=-(qvlo[k]*yi[to][fr]*np.cos(phmo[k]))-(qvlo[k]*yr[to][fr]*np.sin(phmo[k]))


ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

t=np.zeros(num_nodes)


output=np.vstack((vo,pho,Equp,Equq,ploss,qloss,pgo,qgo,t)).T
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
columns.append('t')
    
df.columns=columns
df.to_excel("Results.xlsx")



