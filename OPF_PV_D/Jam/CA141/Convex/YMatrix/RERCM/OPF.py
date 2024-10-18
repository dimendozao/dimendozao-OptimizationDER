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
problem='OPF_PV_D'

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

H=len(imeans)
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
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]

sdh=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for i in range(num_nodes):
       sdh[h][i]=sd[i]*dmeans[h][cnode[i]]

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

sref = cvx.Variable([H,num_nodes],complex=True,name='sref')
w=[0]*H

for h in range(H):
    w[h] = cvx.Variable([num_nodes,num_nodes],hermitian=True,name='wij')

pv= cvx.Variable(num_nodes,complex=True)
ppv= cvx.Variable(num_nodes,complex=True)
zpv= cvx.Variable(num_nodes,boolean=True)

pl=cvx.Variable(H,nonneg=True)
ql=cvx.Variable(H,nonneg=True)
"-------Constraint Construction-------- "
res=[]

for h in range(H):
    res+=   [cvx.real(w[h])<=wrmax]
    res+=   [cvx.real(w[h])>=wrmin]
    res+=   [cvx.imag(w[h])<=wimax]
    res+=   [cvx.imag(w[h])>=wimin]   
    
    res+=[sref[h]+(ppv*imeans[h]*pveff)+sgen-sdh[h]==cvx.diag(w[h]@ym.conjugate())]        
    
    
    res +=  [cvx.real(sref[h])>=prefmin]
    res +=  [cvx.real(sref[h])<=prefmax]
    res +=  [cvx.imag(sref[h])>=qrefmin]
    res +=  [cvx.imag(sref[h])<=qrefmax]
    
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if idx1[i][j]:
                up=cvx.real(w[h][i][i])+cvx.real(w[h][j][j])
                um=cvx.real(w[h][i][i])-cvx.real(w[h][j][j])
                st=cvx.vstack([2*cvx.real(w[h][i][j]),2*cvx.imag(w[h][i][j]),um])
                res += [cvx.SOC(up,st)]    
           
            
    res += [pl[h]==cvx.real(cvx.trace(w[h]@ym.conjugate()))]
    res += [ql[h]==cvx.imag(cvx.trace(w[h]@ym.conjugate()))]
    
       
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
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,accept_unknown=True,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON','MSK_IPAR_INFEAS_REPORT_AUTO':'MSK_ON'},verbose=True)    
#OPFSOC.solve(solver=cvx.ECOS,verbose=True)
#OPFSOC.solve(solver=cvx.GUROBI,verbose=True)
#print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time

wo=np.zeros([H,num_nodes,num_nodes],dtype='complex')

uo=np.zeros([H,num_nodes])
for h in range(H):
    wo[h]= w[h].value    
    uo[h]=np.real(np.diag(wo[h]))
vo= np.sqrt(uo)

pgo=np.real(sref.value)
qgo=np.imag(sref.value)


pvo=np.real(pv.value)
ppvo=np.real(ppv.value)
zpvo=zpv.value

plt.plot(vo)

Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    Equp[h]=np.diag(np.real(wo[h]@ym.conjugate()))
    Equq[h]=np.diag(np.imag(wo[h]@ym.conjugate()))
    
pho=np.zeros([H,num_nodes])
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for i in range(num_lines):
        to=branch['j'][i]-1
        fr=branch['i'][i]-1
        pho[h][to]=pho[h][fr]-np.angle(wo[h][to][fr])
   
ppvout=np.zeros([H,num_nodes])
zpvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=ppvo
    zpvout[h]=zpvo
   
out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,imeans)).T
output=np.hstack((vo,pho,Equp,Equq,out1,dmeans,ppvout,zpvout))
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
columns.append('ic')
for i in range(ncluster):    
    columns.append('dc_c'+str(i+1))
for i in range(num_nodes):    
    columns.append('ppv'+str(i+1))
for i in range(num_nodes):    
    columns.append('zpv'+str(i+1))
    
df.columns=columns

solvlist=[0]*H
solvlist[0]='CM'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")

