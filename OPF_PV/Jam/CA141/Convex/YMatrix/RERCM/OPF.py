# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:11:35 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cvx
from scipy.io import loadmat


case='CA141'
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

for k in range(num_lines):
    sd[branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])

idem=np.zeros(num_nodes)

for i in range(num_nodes):
    idem[i]=dem[cnode[i]]
    
if ngen>0:
    for i in range(ngen):
        sgen[bus['i'][i]-1]=gen['pi'][i]
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

vmax=vmax+0.1
vmin=vmin-0.1
        
vmax[iref]=1
vmin[iref]=1

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=np.maximum(0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]])

ym=np.zeros([num_nodes,num_nodes],dtype='complex')

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])


idx1=np.abs(ym)!=0
idx1=idx1-np.eye(num_nodes)

wrmax=np.zeros([num_nodes,num_nodes])
wrmin=np.zeros([num_nodes,num_nodes])
wimax=np.zeros([num_nodes,num_nodes])
wimin=np.zeros([num_nodes,num_nodes])
    
for i in range(num_nodes):
    for j in range(num_nodes):
        if i==j:
            wrmax[i][j]=vmax[i]*vmax[j]
            wrmin[i][j]=vmin[i]*vmin[j]
        else:            
            wrmax[i][j]=vmax[i]*vmax[j]
            wrmin[i][j]=vmin[i]*vmin[j]
            wimax[i][j]=vmax[i]*vmax[j]
            wimin[i][j]=-vmax[i]*vmax[j]    

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))

pveff=0.8
"----- Optimization model -----"

sref = cvx.Variable(num_nodes,complex=True,name='sref')
w = cvx.Variable([num_nodes,num_nodes],hermitian=True,name='wij')

pv= cvx.Variable(num_nodes,complex=True)
ppv= cvx.Variable(num_nodes,complex=True)
zpv= cvx.Variable(num_nodes,boolean=True)

pl=cvx.Variable(nonneg=True)
ql=cvx.Variable(nonneg=True)
"-------Constraint Construction-------- "
res=[]

res +=  [cvx.real(sref)>=prefmin]
res +=  [cvx.real(sref)<=prefmax]
res +=  [cvx.imag(sref)>=qrefmin]
res +=  [cvx.imag(sref)<=qrefmax]

res+=   [cvx.real(w)<=wrmax]
res+=   [cvx.real(w)>=wrmin]
res+=   [cvx.imag(w)<=wimax]
res+=   [cvx.imag(w)>=wimin]   

res+=[sref+(ppv*rad*pveff)+sgen-(np.multiply(sd,idem))==cvx.diag(w@ym.conjugate())]

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx1[i][j] and j>i:
            up=cvx.real(w[i][i])+cvx.real(w[j][j])
            um=cvx.real(w[i][i])-cvx.real(w[j][j])
            st=cvx.vstack([2*cvx.real(w[i][j]),2*cvx.imag(w[i][j]),um])
            res += [cvx.SOC(up,st)]    
       
        
res += [pl==cvx.real(cvx.trace(w@ym.conjugate()))]
res += [ql==cvx.imag(cvx.trace(w@ym.conjugate()))]

   
#res +=  [cvx.real(cvx.sum(sref))+cvx.real(cvx.sum(ppv*rad))-cvx.real(cvx.sum(sd))>=0]
#res +=  [cvx.imag(cvx.sum(sref))-cvx.imag(cvx.sum(sd))>=0]

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
#obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
obj = cvx.Minimize(cvx.abs(cvx.trace(w@ym.conjugate())))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,accept_unknown=True,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON','MSK_IPAR_INFEAS_REPORT_AUTO':'MSK_ON'},verbose=True)    
#OPFSOC.solve(solver=cvx.ECOS,verbose=True)
#OPFSOC.solve(solver=cvx.GUROBI,verbose=True)
#print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

wo= w.value
    
uo=np.real(np.diag(wo))
vo= np.sqrt(uo)

pgo=np.real(sref.value)
qgo=np.imag(sref.value)


pvo=np.real(pv.value)
ppvo=np.real(ppv.value)
zpvo=zpv.value

plt.plot(vo)


Equp=np.diag(np.real(wo@ym.conjugate()))
Equq=np.diag(np.imag(wo@ym.conjugate()))
    
pho=np.zeros(num_nodes)
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

for i in range(num_lines):
    to=branch['j'][i]-1
    fr=branch['i'][i]-1
    pho[to]=pho[fr]-np.angle(wo[fr][to])
   

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

