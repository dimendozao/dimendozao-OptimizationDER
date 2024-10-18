# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:26:47 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cvx
from scipy.io import loadmat

case='IEEE33'
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

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]

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




"----- Optimization model -----"
pgref = cvx.Variable(num_nodes)
qgref = cvx.Variable(num_nodes)
u= cvx.Variable(num_nodes)
wr= cvx.Variable(num_lines)
wi= cvx.Variable(num_lines)

"-------Constraint Construction-------- "

EqNp = [0] * num_nodes
EqNq = [0] * num_nodes 
res=[]

res =   [u>=umin]
res +=  [u<=umax]

for i in range(num_nodes):        
    for k in range(num_lines):
        if i==(branch['i'][k]-1):
            y  = 1/(branch['r'][k] + 1j*branch['x'][k])
            yr=np.real(y)
            yi=np.imag(y)
            EqNp[i]+=(u[i]*yr)-(wr[k]*yr)-(wi[k]*yi)
            EqNq[i]+=(-u[i]*yi)+(wr[k]*yi)-(wi[k]*yr)
        if i==(branch['j'][k]-1):
            y  = 1/(branch['r'][k] + 1j*branch['x'][k])
            yr=np.real(y)
            yi=np.imag(y)
            EqNp[i]+=(u[i]*yr)-(wr[k]*yr)+(wi[k]*yi)
            EqNq[i]+=(-u[i]*yi)+(wr[k]*yi)+(wi[k]*yr)    

res +=  [cvx.hstack(EqNp)==pgref+pgen-pdm]
res +=  [cvx.hstack(EqNq)==qgref+qgen-qdm]  
res +=  [pgref>=prefmin]
res +=  [pgref<=prefmax]
res +=  [qgref>=qrefmin]
res +=  [qgref<=qrefmax]
    
for k in range(num_lines):
    up= u[branch['i'][k]-1]+u[branch['j'][k]-1]
    um= u[branch['i'][k]-1]-u[branch['j'][k]-1]
    st= cvx.vstack([2*wr[k],2*wi[k],um])
    res += [cvx.SOC(up,st)]      
        




"-------Objective definition--------"
#obj = cvx.Minimize(cvx.real(cvx.sum(sref+sgen-sd-cvx.hstack(EqN)))+cvx.imag(cvx.sum(sref+sgen-sd-cvx.hstack(EqN))))
#obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
obj = cvx.Minimize(cvx.sum(EqNp)+cvx.sum(EqNq))
#obj = cvx.Minimize(cvx.abs(sref[0])+cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(1)
"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)

OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_FREE'},verbose=True)    
print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)


"----- Print results -----"
t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
v = np.sqrt(uo)
wro= wr.value
wio= wi.value
pg=pgref.value
qg=qgref.value
sg=np.sqrt(np.square(pg)+np.square(qg))


plt.plot(v)

Equp =[0] * num_nodes
Equq =[0] * num_nodes
 

pf=np.zeros(num_nodes)
pf[iref]=pg[iref]/sg[iref]


for i in range(num_nodes):
    for k in range(num_lines):
        if i==(branch['i'][k]-1):
            y  = 1/(branch['r'][k] + 1j*branch['x'][k])
            yr=np.real(y)
            yi=np.imag(y)
            Equp[i]+=(uo[i]*yr)-(wro[k]*yr)-(wio[k]*yi)
            Equq[i]+=(-uo[i]*yi)+(wro[k]*yi)-(wio[k]*yr)            
        if i==(branch['j'][k]-1):
            y  = 1/(branch['r'][k] + 1j*branch['x'][k])
            yr=np.real(y)
            yi=np.imag(y)
            Equp[i]+=(uo[i]*yr)-(wro[k]*yr)+(wio[k]*yi)
            Equq[i]+=(-uo[i]*yi)+(wro[k]*yi)+(wio[k]*yr)       

ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

ph=np.zeros(num_nodes)

for k in range(num_lines):
    to=branch['j'][k]-1
    fr=branch['i'][k]-1
    ph[to]=ph[fr]-np.angle(wro[k]+1j*wio[k])
   
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