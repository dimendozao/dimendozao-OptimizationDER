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

case='IEEE33'
city='Bog'
city1='BOG'
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

H=len(imeans)
num_lines = len(branch)
num_nodes=len(bus)
ncluster=np.size(dmeans,axis=1)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    
sdh=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for i in range(num_nodes):
       sdh[h][i]=sd[i]*dmeans[h][cnode[i]]

pdh=np.real(sdh)
qdh=np.imag(sdh)


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

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    
ymr=np.real(ym)
ymi=np.imag(ym)

idx1=np.abs(ym)!=0
idx2=[]

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx1[i][j]:
            idx2.append((i,j))
            
cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if not np.real(ym[i][j])==0:
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

pref = cvx.Variable([H,num_nodes],nonneg=True)
qref = cvx.Variable([H,num_nodes],nonneg=True)

cn=[0]*H
sn=[0]*H

for h in range(H):
    cn[h] = cvx.Variable((num_nodes,num_nodes),sparsity=idx2)
    sn[h] = cvx.Variable((num_nodes,num_nodes),sparsity=idx2)

pv= cvx.Variable(num_nodes,nonneg=True)
ppv= cvx.Variable(num_nodes,nonneg=True)
zpv= cvx.Variable(num_nodes,boolean=True)

pl=cvx.Variable(H,nonneg=True)
ql=cvx.Variable(H,nonneg=True)

"-------Constraint Construction-------- "
res=[]

for h in range(H):
    res+=   [cn[h]<=cnmax]
    res+=   [sn[h]<=snmax]
    res+=   [cn[h]>=cnmin]
    res+=   [sn[h]>=snmin]
    
    res+= [pref[h]+(ppv*imeans[h]*pveff)+pgen-pdh[h]==cvx.diag(cn[h]@ymr)+cvx.diag(sn[h]@ymi)]        
    res+= [qref[h]+qgen-qdh[h]==cvx.diag(sn[h]@ymr)-cvx.diag(cn[h]@ymi)] 
    
    
    res +=  [pref[h]>=prefmin]
    res +=  [pref[h]<=prefmax]
    res +=  [qref[h]>=qrefmin]
    res +=  [qref[h]<=qrefmax]
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if idx1[i][j] and j>i:
                up=cn[h][i][i]+cn[h][j][j]
                um=cn[h][i][i]-cn[h][j][j]
                st=cvx.vstack([2*cn[h][i][j],2*sn[h][i][j],um])
                res += [cvx.SOC(up,st)]    
                res += [cn[h][i][j]==cn[h][j][i]]
                res += [sn[h][i][j]==-sn[h][j][i]]                        
           
            
    res += [pl[h]==cvx.sum(cvx.diag(cn[h]@ymr)+cvx.diag(sn[h]@ymi))]
    res += [ql[h]==cvx.sum(cvx.diag(sn[h]@ymr)-cvx.diag(cn[h]@ymi))]    
   
    
    #res += [cvx.sum(pref[h])+cvx.sum(ppv*imeans[h]*pveff)-cvx.sum(pdh[h])>=0]
    #res += [cvx.sum(qref[h])-cvx.sum(qdh[h])>=0]


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
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON','MSK_IPAR_INFEAS_REPORT_AUTO':'MSK_ON'},verbose=True)    
#OPFSOC.solve(solver=cvx.CPLEX,verbose=True)    


"----- Print results -----"

t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time
cno=np.zeros([H,num_nodes,num_nodes])
sno=np.zeros([H,num_nodes,num_nodes])

pgo=pref.value
qgo=qref.value


uo=np.zeros([H,num_nodes])

for h in range(H):
    cno[h]=cn[h].value
    sno[h]=sn[h].value
    uo[h]=np.diag(cno[h])
   

vo= np.sqrt(uo)
plt.plot(vo)



pvo=pv.value
ppvo=ppv.value
zpvo=zpv.value

Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    Equp[h]=np.diag(cno[h]@ymr)+np.diag(sno[h]@ymi)
    Equq[h]=np.diag(sno[h]@ymr)-np.diag(cno[h]@ymi)


pho=np.zeros([H,num_nodes])

ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for i in range(num_nodes):
        for j in range(num_nodes):
            if idx1[i][j] and j>i:
                pho[h][j]=pho[h][i]-np.angle(cno[h][i][j]+1j*sno[h][i][j])
   
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

