# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:04:23 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cvx


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Gen.csv")


num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

ngen=np.sum(bus['type']==2)
sgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

sd=np.zeros(num_nodes,dtype='complex')

if ngen>0:
    for i in range(ngen):
        sgen[bus['i'][i]-1]=gen['pi'][i]
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

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

yl=np.zeros(num_lines,dtype='complex')
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    yl[k]=1/(branch['r'][k] + 1j*branch['x'][k])
    

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
sref = cvx.Variable(num_nodes,complex=True)
u= cvx.Variable(num_nodes)
w= cvx.Variable(num_lines,complex=True)

"-------Constraint Construction-------- "

EqN = [0] * num_nodes 
res=[]

res =   [u>=umin]
res +=  [u<=umax]

res +=   [cvx.real(w)>=wrmin]
res +=  [cvx.real(w)<=wrmax]

res +=   [cvx.imag(w)>=wimin]
res +=  [cvx.imag(w)<=wimax]


for i in range(num_nodes):        
    for k in range(num_lines):
        if i==fr[k]:            
            EqN[i]+=(u[i]-w[k])*yl[k].conjugate()
        if i==to[k]:
            EqN[i]+=(u[i]-cvx.conj(w[k]))*yl[k].conjugate()    

res +=  [cvx.hstack(EqN)==sref+sgen-sd]        
res +=  [cvx.real(sref)>=prefmin]
res +=  [cvx.real(sref)<=prefmax]
res +=  [cvx.imag(sref)>=qrefmin]
res +=  [cvx.imag(sref)<=qrefmax]
       
    
for k in range(num_lines):
    up= u[fr[k]]+u[to[k]]
    um= u[fr[k]]-u[to[k]]
    st= cvx.vstack([2*cvx.real(w[k]),2*cvx.imag(w[k]),um])
    res += [cvx.SOC(up,st)]
        

"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.real(cvx.sum(sref))+cvx.imag(cvx.sum(sref)))
obj = cvx.Minimize(cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
#OPFSOC.solve(solver=cvx.CPLEX,verbose=True) 
#print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
v = np.sqrt(uo)
wo= w.value
pg=np.real(sref.value)
qg=np.imag(sref.value)
sg=np.sqrt(np.square(pg)+np.square(qg))
sgref=sref.value

plt.plot(v)

Equ =[0] * num_nodes
 

pf=np.zeros(num_nodes)
pf[iref]=pg[iref]/sg[iref]


for i in range(num_nodes):
    for k in range(num_lines):
        if i==(branch['i'][k]-1):
            Equ[i]+=(uo[i]-wo[k])*yl[k].conjugate()
        if i==(branch['j'][k]-1):
            Equ[i]+=(uo[i]-wo[k].conjugate())*yl[k].conjugate()    

Equp=np.real(np.array(Equ))
Equq=np.imag(np.array(Equ))
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

ph=np.zeros(num_nodes)

for k in range(num_lines):
    ph[to[k]]=ph[fr[k]]-np.angle(wo[k])
   
output=np.vstack((v,ph,Equp,Equq,ploss,qloss,pg,qg,t)).T
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
solvlist=[0]*num_nodes
solvlist[0]='CM'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")