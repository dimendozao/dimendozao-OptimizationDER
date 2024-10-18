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

sd=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for k in range(num_lines):
        sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*means[cnode[branch['j'][k]-1][0]-1][h]


fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

wrmax=np.zeros(num_lines)
wrmin=np.zeros(num_lines)
wimax=np.zeros(num_lines)
wimin=np.zeros(num_lines)
    
for k in range(num_lines):
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1        
    wrmax[k]=vmax[fr[k]]*vmax[to[k]]
    wrmin[k]=vmin[fr[k]]*vmin[to[k]]
    wimax[k]=vmax[fr[k]]*vmax[to[k]]
    wimin[k]=-vmax[fr[k]]*vmax[to[k]]
    
wrmax=np.tile(wrmax,(H,1))
wrmin=np.tile(wrmin,(H,1))

wimax=np.tile(wimax,(H,1))
wimin=np.tile(wimin,(H,1))

"----- Optimization model -----"
sref = cvx.Variable((H,num_nodes),complex=True)
u= cvx.Variable((H,num_nodes))
w= cvx.Variable((H,num_lines),complex=True)

"-------Constraint Construction-------- "

EqN = [[0] * num_nodes for h in range(H)] 

res=[]

res +=  [u>=umin]
res +=  [u<=umax]

res +=  [cvx.real(w)>=wrmin]
res +=  [cvx.imag(w)>=wimin]

res +=  [cvx.real(w)<=wrmax]
res +=  [cvx.imag(w)<=wimax]

for h in range(H):
    for i in range(num_nodes):        
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k]) 
                EqN[h][i]+=(u[h][i]-w[h][k])*y.conjugate()
            if i==(branch['j'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                EqN[h][i]+=(u[h][i]-cvx.conj(w[h][k]))*y.conjugate()    

for h in range(H):
    res +=  [cvx.hstack(EqN[h])==sref[h]+sgen-sd[h]] 
    #res +=  [sref[h][iref+1:]==prefmin[h][iref+1:]]

    
res +=  [cvx.real(sref)>=prefmin]
res +=  [cvx.real(sref)<=prefmax]
res +=  [cvx.imag(sref)>=qrefmin]
res +=  [cvx.imag(sref)<=qrefmax]
       
for h in range(H):    
    for k in range(num_lines):
        up= u[h][branch['i'][k]-1]+u[h][branch['j'][k]-1]
        um= u[h][branch['i'][k]-1]-u[h][branch['j'][k]-1]
        st= cvx.vstack([2*cvx.real(w[h][k]),2*cvx.imag(w[h][k]),um])
        res += [cvx.SOC(up,st)]
        

pl=[0]*H
ql=[0]*H

for h in range(H):
    pl[h]=cvx.real(cvx.sum(EqN[h]))
    ql[h]=cvx.imag(cvx.sum(EqN[h]))
#     res += [pl[h]>=0] 
#     res += [ql[h]>=0]        


"-------Objective definition--------"
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(cvx.real(cvx.sum(sref))+cvx.imag(cvx.sum(sref)))
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,accept_unknown=True,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
v = np.sqrt(uo)
wo= w.value
pg=np.real(sref.value)
qg=np.imag(sref.value)
sg=np.sqrt(np.square(pg)+np.square(qg))
sgref=sref.value

plt.plot(v)

Equ =[[0] * num_nodes for h in range(H)]
 

for h in range(H):
    for i in range(num_nodes):
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k]) 
                Equ[h][i]+=(uo[h][i]-wo[h][k])*y.conjugate()
            if i==(branch['j'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                Equ[h][i]+=(uo[h][i]-wo[h][k].conjugate())*y.conjugate()    

Equp=np.real(np.array(Equ))
Equq=np.imag(np.array(Equ))

ploss=np.zeros(H)
qloss=np.zeros(H)

ph=np.zeros([H,num_nodes])

for h in range(H):
    ploss[h]=np.real(np.sum(Equ[h]))
    qloss[h]=np.imag(np.sum(Equ[h]))
    for k in range(num_lines):
        to=branch['j'][k]-1
        fr=branch['i'][k]-1
        ph[h][to]=ph[h][fr]-np.angle(wo[h][k])
   

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