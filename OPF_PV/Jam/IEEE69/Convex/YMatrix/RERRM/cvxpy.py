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
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])
    
idem=np.zeros(num_nodes)

for i in range(num_nodes):
    idem[i]=dem[cnode[i]]
    
pdm=np.real(sd)
qdm=np.imag(sd)

ngen=np.sum(bus['type']==2)
pgen=np.zeros(num_nodes)
qgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

vrmax=np.zeros(num_nodes)
vrmin=np.zeros(num_nodes)

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

vrmax=np.array([vmax[i] for i in range(num_nodes)])
vrmin=np.array([vmin[i] for i in range(num_nodes)])

vimax=np.array([vmax[i] for i in range(num_nodes)])
vimin=np.array([-vmax[i] for i in range(num_nodes)])

vimax[iref]=0
vimin[iref]=0

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

ym=np.zeros([num_nodes,num_nodes],dtype='complex')

idx=np.zeros([num_nodes,num_nodes])

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])
    idx[to][fr]=1
    idx[fr][to]=1    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    idx[i][i]=1
    
ymr=np.real(ym)
ymi=np.imag(ym)

idx2=[]

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx[i][j]:
            idx2.append((i,j))
            
cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx[i][j]:
            if i==j:
                cnmax[i][i]=vmax[i]*vmax[i]
                cnmin[i][i]=vmin[i]*vmin[i]
            else:
                cnmax[i][j]=vmax[i]*vmax[j]
                cnmin[i][j]=vmin[i]*vmin[j]
                snmax[i][j]=vmax[i]*vmax[j]
                snmin[i][j]=-vmax[i]*vmax[j]    

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))
pveff=0.8    
"----- Optimization model -----"

pref = cvx.Variable(num_nodes,nonneg=True)
qref = cvx.Variable(num_nodes,nonneg=True)

cn = cvx.Variable((num_nodes,num_nodes),sparsity=idx2)
sn = cvx.Variable((num_nodes,num_nodes),sparsity=idx2)

pv= cvx.Variable(num_nodes,nonneg=True)
ppv= cvx.Variable(num_nodes,nonneg=True)
zpv= cvx.Variable(num_nodes,boolean=True)

"-------Constraint Construction-------- "
res=[]

res +=  [pref>=prefmin]
res +=  [pref<=prefmax]
res +=  [qref>=qrefmin]
res +=  [qref<=qrefmax]

res +=   [cn<=cnmax]
res +=   [sn<=snmax]
res +=   [cn>=cnmin]
res +=   [sn>=snmin]



res += [pref+(ppv*rad*pveff)+pgen-(np.multiply(pdm,idem))==cvx.diag(cn@ymr)+cvx.diag(sn@ymi)]        
res += [qref+qgen-(np.multiply(qdm,idem))==cvx.diag(sn@ymr)-cvx.diag(cn@ymi)] 




for i in range(num_nodes):
    for j in range(num_nodes):
        if idx[i][j] and j>i:
            up=cn[i][i]+cn[j][j]
            um=cn[i][i]-cn[j][j]
            st=cvx.vstack([2*cn[i][j],2*sn[i][j],um])
            res += [cvx.SOC(up,st)]
            res += [cn[i][j]==cn[j][i]]
            res += [sn[i][j]==-sn[j][i]]                   
       
        
pl=cvx.sum(cvx.diag(cn@ymr)+cvx.diag(sn@ymi))
ql=cvx.sum(cvx.diag(sn@ymr)-cvx.diag(cn@ymi))

res +=[pl>=0]
res +=[ql>=0]

res += [cvx.sum(pref)+cvx.sum(ppv*rad*pveff)-cvx.sum(np.multiply(pdm,idem))>=0]
res += [cvx.sum(qref)-cvx.sum(np.multiply(qdm,idem))>=0]


res += [ppv<=zpv*pvcmax]
res += [ppv<=pv]
res += [ppv>=pv-pvcmax*(1-zpv)]
res += [cvx.sum(zpv)==npv]
res += [cvx.sum(ppv)==pvcmax]
res += [pv<=pvcmax]
res += [ppv<=pvcmax] 
           
"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.sum(pref)+cvx.sum(qref))
obj = cvx.Minimize(pl+ql)
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON','MSK_IPAR_INFEAS_REPORT_AUTO':'MSK_ON'},verbose=True)    
#OPFSOC.solve(solver=cvx.CPLEX,verbose=True)    


"----- Print results -----"

t=np.zeros(num_nodes)
t[0]=OPFSOC.solver_stats.solve_time

cno= cn.value
sno= sn.value
pgo=np.zeros(num_nodes)
qgo=np.zeros(num_nodes)

uo=np.diag(cno)
v= np.sqrt(uo)

pg=pref.value
qg=qref.value
sg=np.sqrt(np.square(pg)+np.square(qg))

pvo=pv.value
ppvo=ppv.value
zpvo=zpv.value

plt.plot(v)

pf=np.zeros(num_nodes)
pf[iref]=pg[iref]/sg[iref]


Equp=np.diag(cno@ymr)+np.diag(sno@ymi)
Equq=np.diag(sno@ymr)-np.diag(cno@ymi)
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

ph=np.zeros(num_nodes)

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx[i][j] and j>i:
            ph[j]=ph[i]-np.angle(cno[i][j]+1j*sno[i][j])
   
output=np.vstack((v,ph,Equp,Equq,ploss,qloss,pg,qg,pvo,ppvo,zpvo,t)).T
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

