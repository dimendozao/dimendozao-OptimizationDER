# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:23:21 2024

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
city='Bog'
city1='BOG'
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

for h in range(H):
    for k in range(num_lines):
        sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*means[cnode[branch['j'][k]-1][0]-1][h]

if ngen>0:
    for i in range(ngen):
        sgen[bus['i'][i]-1]=gen['pi'][i]
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

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
    
wrmax=np.zeros([num_nodes,num_nodes])
wimax=np.zeros([num_nodes,num_nodes])
wrmin=np.zeros([num_nodes,num_nodes])
wimin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        wrmax[i][j]=vmax[i]*vmax[j]
        wrmin[i][j]=vmin[i]*vmin[j]
        if not i==j:
            wimax[i][j]=vmax[i]*vmax[j]
            wimin[i][j]=-vmax[i]*vmax[j]
        else:
            wimax[i][j]=0
            wimin[i][j]=0
            
    
wo=np.zeros([H,num_nodes,num_nodes],dtype='complex')
sgo=np.zeros([H,num_nodes],dtype='complex')
t1=np.zeros(H)
uo=np.zeros([H,num_nodes])
"----- Optimization model -----"

sref = cvx.Variable(num_nodes,complex=True)
w = cvx.Variable((num_nodes,num_nodes),hermitian=True)
sdm=cvx.Parameter(num_nodes,complex=True)
y = cvx.Parameter((num_nodes,num_nodes),complex=True)

"-------Constraint Construction-------- "
res=[]

res+=   [w>>0]

res+=   [cvx.real(w)<=wrmax]
res+=   [cvx.real(w)>=wrmin]
res+=   [cvx.imag(w)<=wimax]
res+=   [cvx.imag(w)>=wimin]

res+=   [sref+sgen-sdm==cvx.diag(w@cvx.conj(y))]        


res +=  [cvx.real(sref)>=prefmin]
res +=  [cvx.real(sref)<=prefmax]
res +=  [cvx.imag(sref)>=qrefmin]
res +=  [cvx.imag(sref)<=qrefmax]

# up=[]
# rw=[]
# iw=[]
# um=[]

# for i in range(num_nodes):
#     for j in range(num_nodes):
#         if not j==i and not np.abs(ym[i][j])==0:
#             up.append(cvx.real(w[i][i])+cvx.real(w[j][j]))
#             rw.append(2*cvx.real(w[i][j]))
#             iw.append(2*cvx.imag(w[i][j]))
#             um.append(cvx.real(w[i][i])-cvx.real(w[j][j]))
            
# up=cvx.hstack(up)      
# rw=cvx.hstack(rw)
# iw=cvx.hstack(iw)
# um=cvx.hstack(um)

# res += [cvx.SOC(up,cvx.vstack([rw,iw,um]))]    
       

pl=cvx.real(cvx.trace(w@cvx.conj(y)))
ql=cvx.imag(cvx.trace(w@cvx.conj(y)))
res +=[pl>=0]
res +=[ql>=0]    

"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.real(cvx.sum(sref))+cvx.imag(cvx.sum(sref)))
obj = cvx.Minimize(pl+ql)
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
y.value=ym
# env = gurobipy.Env()
# env.setParam('BarHomogeneous', 1)
# env.setParam('BarQCPConvTol', 1e-20)
# env.setParam('Aggregate',0)
# env.setParam('Presolve',0)
# env.setParam('ScaleFlag',2)
# env.setParam('NumericFocus',3)
for h in range(H):
    sdm.value=sd[h]
    OPFSOC.solve(solver=cvx.MOSEK,canon_backend=cvx.SCIPY_CANON_BACKEND,enforce_dpp=True,accept_unknown=True,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
    # OPFSOC.solve(solver=cvx.SCS,verbose=True)
    # OPFSOC.solve(solver=cvx.GUROBI,env=env,verbose=True)
    "----- Print results -----"
    t1[h]=OPFSOC.solver_stats.solve_time
    wo[h]= w.value
    sgo[h]= sref.value
    uo[h] = np.real(np.diag(wo[h]))


v= np.sqrt(uo)
pg=np.real(sgo)
qg=np.imag(sgo)
sg=np.sqrt(np.square(pg)+np.square(qg))

plt.plot(v)

Equp =[[0] * num_nodes for h in range(H)]
Equq =[[0] * num_nodes for h in range(H)]


for h in range(H):
    Equp[h]=np.diag(np.real(wo[h]@ym.conjugate()))
    Equq[h]=np.diag(np.imag(wo[h]@ym.conjugate()))

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

t=np.zeros(H)
t[0]=np.sum(t1)
   
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
solvlist[0]='CM-H'


df.insert(len(df.columns),'Solver',solvlist)
#df.to_excel("Results.xlsx")