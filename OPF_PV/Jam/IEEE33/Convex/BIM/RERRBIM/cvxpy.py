# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 07:59:46 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import cvxpy as cvx

case='IEEE33'
city='Jam'
city1='JAM'
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

rad=np.mean(imeans)
dem=np.mean(dmeans,axis=0)

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

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))

pveff=0.8

"----- Optimization model -----"
pgref = cvx.Variable(num_nodes, nonneg=True)
qgref = cvx.Variable(num_nodes, nonneg=True)

u= cvx.Variable(num_nodes,nonneg=True)
wr= cvx.Variable(num_lines,nonneg=True)
wi= cvx.Variable(num_lines)

pv= cvx.Variable(num_nodes,nonneg=True)
ppv= cvx.Variable(num_nodes,nonneg=True)
zpv= cvx.Variable(num_nodes,boolean=True)

pl=cvx.Variable(nonneg=True)
ql=cvx.Variable(nonneg=True)

"-------Constraint Construction-------- "

EqNp = num_nodes*[0]
EqNq = num_nodes*[0]

res=[]

res +=  [pgref>=prefmin]
res +=  [pgref<=prefmax]

res +=  [qgref>=qrefmin]
res +=  [qgref<=qrefmax]

res +=  [u>=umin]
res +=  [u<=umax]

res +=  [wr>=wrmin]
res +=  [wr<=wrmax]

res +=  [wi>=wimin]
res +=  [wi<=wimax]


for i in range(num_nodes):        
    for k in range(num_lines):
        if i==fr[k]:            
            EqNp[i]+=(u[i]*ylr[k])-(wr[k]*ylr[k])-(wi[k]*yli[k])
            EqNq[i]+=(-u[i]*yli[k])+(wr[k]*yli[k])-(wi[k]*ylr[k])
        if i==to[k]:
            EqNp[i]+=(u[i]*ylr[k])-(wr[k]*ylr[k])+(wi[k]*yli[k])
            EqNq[i]+=(-u[i]*yli[k])+(wr[k]*yli[k])+(wi[k]*ylr[k])    
 
for i in range(num_nodes):  
    res +=  [EqNp[i]==pgref[i]+(ppv[i]*rad*pveff)+pgen[i]-(pdm[i]*dem[cnode[i]])]
    res +=  [EqNq[i]==qgref[i]+qgen[i]-(qdm[i]*dem[cnode[i]])]          


for k in range(num_lines):
    up= u[fr[k]]+u[to[k]]
    um= u[fr[k]]-u[to[k]]
    st= cvx.vstack([2*wr[k],2*wi[k],um])
    res += [cvx.SOC(up,st)]   

res += [pl==cvx.sum(EqNp)]
res += [ql==cvx.sum(EqNq)]
    
res += [ppv<=zpv*pvcmax]
res += [ppv<=pv]
res += [ppv>=pv-pvcmax*(1-zpv)]
res += [cvx.sum(zpv)==npv]
res += [cvx.sum(ppv)==pvcmax]
res += [pv<=pvcmax]
res += [ppv<=pvcmax]
    
    

"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.real(cvx.sum(sref))+cvx.imag(cvx.sum(sref)))
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(cvx.abs(cvx.sum(EqNp))+cvx.abs(cvx.sum(EqNq)))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
#print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
vo = np.sqrt(uo)
wro= wr.value
wio= wi.value
pgo=pgref.value
qgo=qgref.value

pvo=pv.value
ppvo=ppv.value
zpvo=zpv.value

plt.plot(vo)

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

pho=np.zeros(num_nodes)
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

for k in range(num_lines):
    pho[to[k]]=pho[fr[k]]-np.angle(wro[k]+1j*wio[k])
   

   
output=np.vstack((vo,pho,Equp,Equq,ploss,qloss,pgo,qgo,pvo,ppvo,zpvo,t)).T
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
columns.append('pv')
columns.append('ppv')
columns.append('zpv')
columns.append('t')
    
df.columns=columns

solvlist=[0]*num_nodes
solvlist[0]='CM'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")


