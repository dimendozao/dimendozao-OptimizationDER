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
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Gen.csv")

case='SA'
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

ngen=np.sum(bus['type']==2)
sgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

vmax=vmax+0.1
vmin=vmin-0.1

sd=np.zeros([H,num_nodes],dtype='complex')

if ngen>0:
    for i in range(ngen):
        sgen[bus['i'][i]-1]=gen['pi'][i]
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

umax=vmax**2
umin=vmin**2        

umax[iref]=1
umin[iref]=1

umax=np.tile(umax,(H,1))
umin=np.tile(umin,(H,1))

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=np.maximum(0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]])

prefmax=np.tile(prefmax,(H,1))
prefmin=np.tile(prefmin,(H,1))

qrefmax=np.tile(qrefmax,(H,1))
qrefmin=np.tile(qrefmin,(H,1))

for h in range(H):
    for k in range(num_lines):
        sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*means[cnode[branch['j'][k]-1][0]-1][h]


ym=np.zeros([num_nodes,num_nodes],dtype='complex')

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    
idx=np.abs(ym)!=0
idx=idx-np.eye(num_nodes)
    
"----- Optimization model -----"

sref = cvx.Variable((H,num_nodes),complex=True)

w=[0]*H

for h in range(H):
    w[h] = cvx.Variable((num_nodes,num_nodes),hermitian=True)

"-------Constraint Construction-------- "
res=[]



for h in range(H):
    #res+=   [w[h]>>0]
    res+=   [cvx.real(cvx.diag(w[h]))<=umax[h]]
    res+=   [cvx.real(cvx.diag(w[h]))>=umin[h]]
    res+=   [sref[h]+sgen-sd[h]==cvx.diag(w[h]@ym.conjugate())]        


res +=  [cvx.real(sref)>=prefmin]
res +=  [cvx.real(sref)<=prefmax]
res +=  [cvx.imag(sref)>=qrefmin]
res +=  [cvx.imag(sref)<=qrefmax]


for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            if idx[i][j]:
                up=cvx.real(w[h][i][i]+w[h][j][j])
                um=cvx.real(w[h][i][i]-w[h][j][j])
                st=cvx.vstack([2*cvx.real(w[h][i][j]),2*cvx.imag(w[h][i][j]),cvx.real(w[h][i][i]-w[h][j][j])])
                res += [cvx.SOC(up,st)]    
       
pl=[0]*H
ql=[0]*H

for h in range(H):
    pl[h]=cvx.real(cvx.trace(w[h]@ym.conjugate()))
    ql[h]=cvx.imag(cvx.trace(w[h]@ym.conjugate()))        

"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.real(cvx.sum(sref))+cvx.imag(cvx.sum(sref)))
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time

wo=np.zeros([H,num_nodes,num_nodes],dtype='complex')
uo=np.zeros([H,num_nodes])

for h in range(H):
    wo[h]= w[h].value
    uo[h]= np.real(np.diag(wo[h]))

v= np.sqrt(uo)
pg=np.real(sref.value)
qg=np.imag(sref.value)
sg=np.sqrt(np.square(pg)+np.square(qg))
sgref=sref.value

plt.plot(v)

Equp =[[0] * num_nodes for h in range(H)]
Equq =[[0] * num_nodes for h in range(H)]


for h in range(H):
    Equp[h]=np.diag(np.real(ym.conjugate()@wo[h]))
    Equq[h]=np.diag(np.imag(ym.conjugate()@wo[h]))

ploss=np.zeros(H)
qloss=np.zeros(H)

ph=np.zeros([H,num_nodes])

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        to=branch['j'][k]-1
        fr=branch['i'][k]-1
        ph[h][to]=ph[h][fr]-np.angle(wo[h][to][fr])

   
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

