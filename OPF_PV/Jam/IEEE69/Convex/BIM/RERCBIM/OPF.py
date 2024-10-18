# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:04:23 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cvx
from scipy.io import loadmat

case='IEEE69'
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
qrefmin[iref]=np.max([0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]])

yl=np.zeros(num_lines,dtype='complex')
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    yl[k]=np.conj(1/(branch['r'][k] + 1j*branch['x'][k]))
    

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
sref = cvx.Variable(num_nodes,complex=True)
u= cvx.Variable(num_nodes,complex=True)
w= cvx.Variable(num_lines,complex=True)

pv= cvx.Variable(num_nodes,complex=True)
ppv= cvx.Variable(num_nodes,complex=True)
zpv= cvx.Variable(num_nodes,boolean=True)

pl=cvx.Variable(nonneg=True)
ql=cvx.Variable(nonneg=True)

"-------Constraint Construction-------- "

EqN = num_nodes*[0]


res=[]

res +=  [cvx.real(sref)>=prefmin]
res +=  [cvx.real(sref)<=prefmax]
res +=  [cvx.imag(sref)>=qrefmin]
res +=  [cvx.imag(sref)<=qrefmax]

res +=  [cvx.real(u)>=umin]
res +=  [cvx.real(u)<=umax]

res +=  [cvx.imag(u)==0]
res +=  [cvx.imag(u)==0]

res +=  [cvx.real(w)>=wrmin]
res +=  [cvx.real(w)<=wrmax]

res +=  [cvx.imag(w)>=wimin]
res +=  [cvx.imag(w)<=wimax]

for i in range(num_nodes):        
    for k in range(num_lines):
        if i==fr[k]:            
            EqN[i]+=(u[i]*yl[k])-(w[k]*yl[k])
        if i==to[k]:
            EqN[i]+=(u[i]*yl[k])-(cvx.conj(w[k])*yl[k])    
 
for i in range(num_nodes):  
    res +=  [sref[i]+(ppv[i]*rad*pveff)+sgen[i]-(sd[i]*dem[cnode[i]])==EqN[i]]       

for k in range(num_lines):
    up= cvx.real(u[fr[k]])+cvx.real(u[to[k]])
    um= cvx.real(u[fr[k]])-cvx.real(u[to[k]])
    st= cvx.vstack([2*cvx.real(w[k]),2*cvx.imag(w[k]),um])
    res += [cvx.SOC(up,st)]
    
res +=[pl==cvx.real(cvx.sum(EqN))]
res +=[ql==cvx.imag(cvx.sum(EqN))]

res += [cvx.real(ppv)<=zpv*pvcmax]
res += [cvx.real(ppv)<=cvx.real(pv)]
res += [cvx.real(ppv)>=cvx.real(pv)-pvcmax*(1-zpv)]
res += [cvx.sum(zpv)==npv]
res += [cvx.sum(cvx.real(ppv))==pvcmax]
res += [cvx.real(pv)<=pvcmax]
res += [cvx.real(pv)>=0]
res += [cvx.imag(pv)==0]
res += [cvx.real(ppv)<=pvcmax]
res += [cvx.real(ppv)>=0]
res += [cvx.imag(ppv)==0]   



"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.real(cvx.sum(sref))+cvx.imag(cvx.sum(sref)))
obj = cvx.Minimize(pl+ql)
#obj = cvx.Minimize(cvx.abs(cvx.sum(EqN)))
#obj = cvx.Minimize(cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
#print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
vo = np.sqrt(np.real(uo))
wo= w.value
pgo=np.real(sref.value)
qgo=np.imag(sref.value)


pvo=np.real(pv.value)
ppvo=np.real(ppv.value)
zpvo=zpv.value

plt.plot(vo)

Equ=np.zeros(num_nodes,dtype=complex)
 


for i in range(num_nodes):
    for k in range(num_lines):
        if i==(branch['i'][k]-1):
            Equ[i]+=(uo[i]-wo[k])*yl[k]
        if i==(branch['j'][k]-1):
            Equ[i]+=(uo[i]-wo[k].conjugate())*yl[k]    

Equp=np.real(Equ)
Equq=np.imag(Equ)


pho=np.zeros(num_nodes)
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

for k in range(num_lines):
    pho[to[k]]=pho[fr[k]]-np.angle(wo[k])

   
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
