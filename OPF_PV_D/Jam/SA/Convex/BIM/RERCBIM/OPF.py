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

case='SA'
case1='SA_J23'
city='Jam'
city1='JAM'
problem='OPF_PV_D'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Gen.csv')

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

if ngen>0:
    for i in range(ngen):
        sgen[bus['i'][i]-1]=gen['pi'][i]
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

vmax=vmax+0.1
vmin=vmin-0.1

vmax[iref]=1
vmin[iref]=1

umax=vmax**2
umin=vmin**2        

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=np.max([0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]])

yl=np.zeros(num_lines,dtype='complex')
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    yl[k]=np.conj(1/(branch['r'][k] + 1j*branch['x'][k]))
    

wrmax=np.zeros(num_lines)
wrmin=np.zeros(num_lines)
wimax=np.zeros(num_lines)
wimin=np.zeros(num_lines)
    
for k in range(num_lines):        
    wrmax[k]=vmax[fr[k]]*vmax[to[k]]
    wrmin[k]=vmin[fr[k]]*vmin[to[k]]
    wimax[k]=vmax[fr[k]]*vmax[to[k]]
    wimin[k]=-vmax[fr[k]]*vmax[to[k]]

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))

pveff=0.8

"----- Optimization model -----"
sref = cvx.Variable([H,num_nodes],complex=True)
u= cvx.Variable([H,num_nodes],complex=True)
w= cvx.Variable([H,num_lines],complex=True)

pv= cvx.Variable(num_nodes,complex=True)
ppv= cvx.Variable(num_nodes,complex=True)
zpv= cvx.Variable(num_nodes,boolean=True)

sl=cvx.Variable(H,complex=True)


"-------Constraint Construction-------- "

EqN = [num_nodes*[0] for h in range(H)]

res=[]

for h in range(H):
    res +=  [cvx.real(u[h])>=umin]
    res +=  [cvx.real(u[h])<=umax]
    
    res +=  [cvx.imag(u[h])==0]
    res +=  [cvx.imag(u[h])==0]
    
    res +=  [cvx.real(w[h])>=wrmin]
    res +=  [cvx.real(w[h])<=wrmax]
    
    res +=  [cvx.imag(w[h])>=wimin]
    res +=  [cvx.imag(w[h])<=wimax]

    for i in range(num_nodes):        
        for k in range(num_lines):
            if i==fr[k]:            
                EqN[h][i]+=(u[h][i]*yl[k])-(w[h][k]*yl[k])
            if i==to[k]:
                EqN[h][i]+=(u[h][i]*yl[k])-(cvx.conj(w[h][k])*yl[k])    
 
    for i in range(num_nodes):  
        res +=  [sref[h][i]+(ppv[i]*imeans[h]*pveff)+sgen[i]-(sd[i]*dmeans[h][cnode[i]])==EqN[h][i]]       

    res +=  [cvx.real(sref[h])>=prefmin]
    res +=  [cvx.real(sref[h])<=prefmax]
    res +=  [cvx.imag(sref[h])>=qrefmin]
    res +=  [cvx.imag(sref[h])<=qrefmax]
       
    
    for k in range(num_lines):
        up= cvx.real(u[h][fr[k]])+cvx.real(u[h][to[k]])
        um= cvx.real(u[h][fr[k]])-cvx.real(u[h][to[k]])
        st= cvx.vstack([2*cvx.real(w[h][k]),2*cvx.imag(w[h][k]),um])
        res += [cvx.SOC(up,st)]
        #res += [cvx.real(u[h][fr[k]])>=cvx.real(u[h][to[k]])]
        
    res +=[sl[h]==cvx.sum(EqN[h])]
    res +=[cvx.real(sl[h])>=0]
    res +=[cvx.imag(sl[h])>=0]

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
obj = cvx.Minimize(cvx.abs(cvx.sum(sl)))
#obj = cvx.Minimize(cvx.abs(cvx.sum(EqN)))
#obj = cvx.Minimize(cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
#obj = cvx.Minimize(1)

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_PRIMAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_ON'},verbose=True)    
#print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

"----- Print results -----"

t=np.zeros(H)
t[0]=OPFSOC.solver_stats.solve_time

uo=u.value
vo = np.sqrt(np.real(uo))
wo= w.value
pgo=np.real(sref.value)
qgo=np.imag(sref.value)


pvo=np.real(pv.value)
ppvo=np.real(ppv.value)
zpvo=zpv.value

plt.plot(vo)

Equ=np.zeros([H,num_nodes],dtype=complex)
 

for h in range(H):
    for i in range(num_nodes):
        for k in range(num_lines):
            if i==(branch['i'][k]-1):
                Equ[h][i]+=(uo[h][i]-wo[h][k])*yl[k]
            if i==(branch['j'][k]-1):
                Equ[h][i]+=(uo[h][i]-wo[h][k].conjugate())*yl[k]    

Equp=np.real(Equ)
Equq=np.imag(Equ)


pho=np.zeros([H,num_nodes])
ploss=np.zeros(H)
qloss=np.zeros(H)


for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        pho[h][to[k]]=pho[h][fr[k]]-np.angle(wo[h][k])

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
