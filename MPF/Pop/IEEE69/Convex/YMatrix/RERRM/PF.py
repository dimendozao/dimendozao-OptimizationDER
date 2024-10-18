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


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Gen.csv")

case='IEEE69'
city='Pop'
city1='POP'
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

vmax=np.tile(vmax,(H,1))
vmin=np.tile(vmin,(H,1))

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

ym=np.zeros([num_nodes,num_nodes],dtype='complex')

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    
ymr=np.real(ym)
ymi=np.imag(ym)

cnmax=np.zeros([H,num_nodes,num_nodes])
cnmin=np.zeros([H,num_nodes,num_nodes])
snmax=np.zeros([H,num_nodes,num_nodes])
snmin=np.zeros([H,num_nodes,num_nodes])

for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i==j:
                cnmax[h][i][i]=vmax[h][i]*vmax[h][i]
                cnmin[h][i][i]=vmin[h][i]*vmin[h][i]
            else:
                cnmax[h][i][j]=vmax[h][i]*vmax[h][j]
                cnmin[h][i][j]=vmin[h][i]*vmin[h][j]
                snmax[h][i][j]=vmax[h][i]*vmax[h][j]
                snmin[h][i][j]=-vmax[h][i]*vmax[h][j]    

idx=np.abs(ym)!=0 

idx1=[]

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx[i][j]:
            idx1.append((i,j))     
"----- Optimization model -----"

pref = cvx.Variable((H,num_nodes))
qref = cvx.Variable((H,num_nodes))

cn=[0]*H
sn=[0]*H

for h in range(H):
    cn[h] = cvx.Variable((num_nodes,num_nodes),sparsity=idx1)
    sn[h] = cvx.Variable((num_nodes,num_nodes),sparsity=idx1)    

"-------Constraint Construction-------- "
res=[]

for h in range(H):
    res+=   [cn[h]<=cnmax[h]]
    res+=   [cn[h]>=cnmin[h]]
    res+=   [sn[h]<=snmax[h]]
    res+=   [sn[h]>=snmin[h]]
    res+=   [pref[h]+pgen-pdm[h]==cvx.diag(cn[h]@ymr)+cvx.diag(sn[h]@ymi)]        
    res+=   [qref[h]+qgen-qdm[h]==cvx.diag(sn[h]@ymr)-cvx.diag(cn[h]@ymi)]
    
    
res +=  [pref>=prefmin]
res +=  [pref<=prefmax]
res +=  [qref>=qrefmin]
res +=  [qref<=qrefmax]

for h  in range(H):
    res +=  [cn[h]==cn[h].T]
    res +=  [sn[h]==-sn[h].T]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j!=i and idx[i][j]:
                up=cn[h][i][i]+cn[h][j][j]
                um=cn[h][i][i]-cn[h][j][j]
                st=cvx.vstack([2*cn[h][i][j],2*sn[h][i][j],cn[h][i][i]-cn[h][j][j]])
                res += [cvx.SOC(up,st)]
                
pl=[0]*H
ql=[0]*H

for h in range(H):
    pl[h]=cvx.trace(cn[h]@ymr)+cvx.trace(sn[h]@ymi)
    ql[h]=cvx.trace(sn[h]@ymr)-cvx.trace(cn[h]@ymi)
           
"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.sum(pref)+cvx.sum(qref))
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,accept_unknown=True,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time

cno=np.zeros([H,num_nodes,num_nodes])
sno=np.zeros([H,num_nodes,num_nodes])
v=np.zeros([H,num_nodes])

for h in range(H):
    cno[h]= cn[h].value
    sno[h]= sn[h].value
    v[h]=np.sqrt(np.diag(cno[h]))
    
pg=pref.value
qg=qref.value
sg=np.sqrt(np.square(pg)+np.square(qg))

plt.plot(v)

Equp =[[0] * num_nodes for h in range(H)]
Equq =[[0] * num_nodes for h in range(H)]

for h in range(H):
    Equp[h]=np.diag(ymr@cno[h])+np.diag(ymi@sno[h])
    Equq[h]=np.diag(ymr@sno[h])-np.diag(ymi@cno[h])
    
ploss=np.zeros(H)
qloss=np.zeros(H)

ph=np.zeros([H,num_nodes])

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        to=branch['j'][k]-1
        fr=branch['i'][k]-1
        ph[h][to]=ph[h][fr]-np.angle(cno[h][fr][to]+1j*sno[h][fr][to])

  
out1=np.vstack((ploss,qloss,pg[:,iref],qg[:,iref],t)).T
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
solvlist[0]='CM'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")

