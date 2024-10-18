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


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Gen.csv")

case='IEEE33'
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

umax=np.tile(umax,(H,1))
umin=np.tile(umin,(H,1))

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

"----- Optimization model -----"
pgref = cvx.Variable((H,num_nodes))
qgref = cvx.Variable((H,num_nodes))
u= cvx.Variable((H,num_nodes))
wr= cvx.Variable((H,num_lines))
wi= cvx.Variable((H,num_lines))

"-------Constraint Construction-------- "

EqNp = [[0] * num_nodes for h in range(H)] 
EqNq = [[0] * num_nodes for h in range(H)] 

res=[]

res +=  [u>=umin]
res +=  [u<=umax]

for h in range(H):
    for i in range(num_nodes):        
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                EqNp[h][i]+=(u[h][i]*yr)-(wr[h][k]*yr)-(wi[h][k]*yi)
                EqNq[h][i]+=(-u[h][i]*yi)+(wr[h][k]*yi)-(wi[h][k]*yr)
            if i==(branch['j'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                EqNp[h][i]+=(u[h][i]*yr)-(wr[h][k]*yr)+(wi[h][k]*yi)
                EqNq[h][i]+=(-u[h][i]*yi)+(wr[h][k]*yi)+(wi[h][k]*yr)    

for h in range(H):
    res +=  [cvx.hstack(EqNp[h])==pgref[h]+pgen-pdm[h]]
    res +=  [cvx.hstack(EqNq[h])==qgref[h]+qgen-qdm[h]]  

res +=  [pgref>=prefmin]
res +=  [pgref<=prefmax]
res +=  [qgref>=qrefmin]
res +=  [qgref<=qrefmax]

for h in range(H):    
    for k in range(num_lines):
        up= u[h][branch['i'][k]-1]+u[h][branch['j'][k]-1]
        um= u[h][branch['i'][k]-1]-u[h][branch['j'][k]-1]
        st= cvx.vstack([2*wr[h][k],2*wi[h][k],um])
        res += [cvx.SOC(up,st)]      
        


pl=[0]*H
ql=[0]*H

for h in range(H):
    pl[h]=cvx.sum(EqNp[h])
    ql[h]=cvx.sum(EqNq[h])

"-------Objective definition--------"
#obj = cvx.Minimize(cvx.real(cvx.sum(sref+sgen-sd-cvx.hstack(EqN)))+cvx.imag(cvx.sum(sref+sgen-sd-cvx.hstack(EqN))))
#obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(cvx.abs(sref[0])+cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(1)
"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)

OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)


"----- Print results -----"
t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
v = np.sqrt(uo)
wro= wr.value
wio= wi.value
pg=pgref.value
qg=qgref.value
sg=np.sqrt(np.square(pg)+np.square(qg))


plt.plot(v)

Equp =[[0] * num_nodes for h in range(H)]
Equq =[[0] * num_nodes for h in range(H)]
 

for h in range(H):
    for i in range(num_nodes):
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                Equp[h][i]+=(uo[h][i]*yr)-(wro[h][k]*yr)-(wio[h][k]*yi)
                Equq[h][i]+=(-uo[h][i]*yi)+(wro[h][k]*yi)-(wio[h][k]*yr)            
            if i==(branch['j'][k]-1):
                y  = 1/(branch['r'][k] + 1j*branch['x'][k])
                yr=np.real(y)
                yi=np.imag(y)
                Equp[h][i]+=(uo[h][i]*yr)-(wro[h][k]*yr)+(wio[h][k]*yi)
                Equq[h][i]+=(-uo[h][i]*yi)+(wro[h][k]*yi)+(wio[h][k]*yr)       

ploss=np.zeros(H)
qloss=np.zeros(H)

ph=np.zeros([H,num_nodes])

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        to=branch['j'][k]-1
        fr=branch['i'][k]-1
        ph[h][to]=ph[h][fr]-np.angle(wro[h][k]+1j*wio[h][k])
   
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