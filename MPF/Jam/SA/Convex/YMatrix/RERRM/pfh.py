# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:05:18 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pad
import cvxpy as cvx
from scipy.io import loadmat


"----- Read the database -----"
branch = pad.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Branch.csv")
bus= pad.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Bus.csv")
gen= pad.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Gen.csv")

case='SA'
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
    
ymr=np.real(ym)
ymi=np.imag(ym)

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])


for i in range(num_nodes):
    for j in range(num_nodes):
        if i==j:
            cnmax[i][i]=vmax[i]*vmax[i]
            cnmin[i][i]=vmin[i]*vmin[i]
        else:
            cnmax[i][j]=vmax[i]*vmax[j]
            cnmin[i][j]=vmin[i]*vmin[j]
            snmax[i][j]=vmax[i]*vmax[j]
            snmin[i][j]=-vmax[i]*vmax[j]    

idx=np.abs(ym)!=0 

idx1=[]

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx[i][j]:
            idx1.append((i,j)) 

cnmax=np.multiply(cnmax,idx)
cnmin=np.multiply(cnmin,idx)

snimax=np.multiply(snmax,idx)
snmin=np.multiply(snmin,idx) 


    
cno=np.zeros([H,num_nodes,num_nodes])
sno=np.zeros([H,num_nodes,num_nodes])
pgo=np.zeros([H,num_nodes]) 
qgo=np.zeros([H,num_nodes])
v=np.zeros([H,num_nodes])
t1=np.zeros(H)
"----- Optimization model -----"

pref = cvx.Variable(num_nodes)
qref = cvx.Variable(num_nodes)
cn = cvx.Variable((num_nodes,num_nodes),sparsity=idx1)
sn = cvx.Variable((num_nodes,num_nodes),sparsity=idx1)

pd=cvx.Parameter(num_nodes)
qd=cvx.Parameter(num_nodes)

"-------Constraint Construction-------- "
res=[]


res+=   [cn<=cnmax]
res+=   [cn>=cnmin]
res+=   [sn<=snmax]
res+=   [sn>=snmin]
res+=   [pref+pgen-pd==cvx.diag(cn@ymr)+cvx.diag(sn@ymi)]        
res+=   [qref+qgen-qd==cvx.diag(sn@ymr)-cvx.diag(cn@ymi)] 
    
    
res +=  [pref>=prefmin]
res +=  [pref<=prefmax]
res +=  [qref>=qrefmin]
res +=  [qref<=qrefmax]

res +=  [cn==cn.T]
res +=  [sn==-sn.T]


for i in range(num_nodes):
    for j in range(num_nodes):
        if not j==i and idx[i][j]:
            up=cn[i][i]+cn[j][j]
            um=cn[i][i]-cn[j][j]
            st=cvx.vstack([2*cn[i][j],2*sn[i][j],um])
            res += [cvx.SOC(up,st)]
            
                    


pl=cvx.trace(cn@ymr)+cvx.trace(sn@ymi)
ql=cvx.trace(sn@ymr)-cvx.trace(cn@ymi)

#res +=[pl>=0]
#res +=[ql>=0]  
           
"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.sum(pref)+cvx.sum(qref))
obj = cvx.Minimize(pl+ql)
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
for h in range(H):
    pd.value=pdm[h]
    qd.value=qdm[h]
    OPFSOC.solve(solver=cvx.MOSEK,canon_backend=cvx.SCIPY_CANON_BACKEND,enforce_dpp=True,accept_unknown=True,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
    # OPFSOC.solve(solver=cvx.SCS,verbose=True)
    # OPFSOC.solve(solver=cvx.GUROBI,env=env,verbose=True)
    "----- Print results -----"
    t1[h]=OPFSOC.solver_stats.solve_time
    cno[h]= cn.value
    sno[h]= sn.value
    pgo[h]= pref.value
    qgo[h]= qref.value
    
t=np.zeros(H)
t[0]=np.sum(t1)


for h in range(H):
    v[h]=np.sqrt(np.diag(cno[h]))
    

plt.plot(v)

Equp =[[0] * num_nodes for h in range(H)]
Equq =[[0] * num_nodes for h in range(H)]

for h in range(H):
    Equp[h]=np.diag(cno[h]@ymr)+np.diag(sno[h]@ymi)
    Equq[h]=np.diag(sno[h]@ymr)-np.diag(cno[h]@ymi)
    
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

  
out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t)).T
output=np.hstack((v,ph,Equp,Equq,out1))
df=pad.DataFrame(output)

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
solvlist[0]='CM-H'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")