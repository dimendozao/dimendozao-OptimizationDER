# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 08:23:54 2023

@author: diego
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

case='CA141'
city='Pop'
city1='POP'
problem='OPF_BESS_I'

prob_sol='OPF_PV_D'


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

pvall=pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+prob_sol+'\\'+city+'\\'+case+'\\bestsol.csv')

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

ppv=pvall['ppv'].to_numpy()
    
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

zk=np.zeros(num_lines,dtype='complex')
yk=np.zeros(num_lines,dtype='complex')

fr=np.zeros(num_lines,dtype='int')
to=np.zeros(num_lines,dtype='int')

for k in range(num_lines):
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    zk[k]=branch['r'][k] + 1j*branch['x'][k]
    yk[k]=1/zk[k]
    
qvnmax=vmax**2
qvnmin=vmin**2

qikmax=np.ones(num_lines)
qikmin=np.zeros(num_lines)

pkmax=np.ones(num_lines)
pkmin=np.zeros(num_lines)
qkmax=np.ones(num_lines)
qkmin=np.zeros(num_lines)

npv=np.floor(0.1*num_nodes)

pveff=0.8

pbcmax=np.sum(pdm)*1

"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

qvn= m.addMVar((H,num_nodes),lb=qvnmin,ub=qvnmax,name='qvn')
qik= m.addMVar((H,num_lines),lb=-GRB.INFINITY,name='qik')
pk= m.addMVar((H,num_lines),lb=-GRB.INFINITY,name='pk')
qk= m.addMVar((H,num_lines),lb=-GRB.INFINITY,name='qk')

pbc= m.addMVar(1,lb=0,ub=pbcmax,name='pbc')
pb= m.addMVar(H,lb=-pbcmax,ub=pbcmax,name='pb')
ppb= m.addMVar((H,num_nodes),lb=-pbcmax,ub=pbcmax,name='ppb')
zb= m.addMVar((H,num_nodes),name='zb',vtype=GRB.BINARY)
cap= m.addMVar(H,name='cap')
cap0= m.addVar(name='cap0')

pl= m.addMVar(H,name='pl')
ql= m.addMVar(H,name='ql')

"-------Constraint Construction-------- "

EqNp = [num_nodes*[0] for h in range(H)]
EqNq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for k in range(num_lines):
        EqNp[h][fr[k]]+=pk[h][k]
        EqNp[h][to[k]]+=(np.real(zk[k])*qik[h][k])-pk[h][k]
        EqNq[h][fr[k]]+=qk[h][k]
        EqNq[h][to[k]]+=(np.imag(zk[k])*qik[h][k])-qk[h][k]
    
    
m.addConstrs((qvn[h][fr[k]]-qvn[h][to[k]]==2*(pk[h][k]*np.real(zk[k])+qk[h][k]*np.imag(zk[k]))-qik[h][k]*(np.square(np.abs(zk[k]))) for h in range(H) for k in range(num_lines)),name='c-ph')
m.addConstrs((qik[h][k]*qvn[h][fr[k]]>=pk[h][k]*pk[h][k]+qk[h][k]*qk[h][k] for h in range(H) for k in range(num_lines)),name='c-socp')

            
m.addConstrs((EqNp[h][i]==pgref[h][i]+(ppv[i]*imeans[h]*pveff)+ppb[h][i]+pgen[i]-(pdm[i]*dmeans[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-(qdm[i]*dmeans[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-q')

m.addConstrs((ppb[h][i]<=zb[h][i]*pbcmax for h in range(H) for i in range(num_nodes)), name='c-pb1')
m.addConstrs((ppb[h][i]>=zb[h][i]*-pbcmax for h in range(H) for i in range(num_nodes)), name='c-pb2')
m.addConstrs((ppb[h][i]<=pb[h]+pbcmax*(1-zb[h][i]) for h in range(H) for i in range(num_nodes)), name='c-pb3')
m.addConstrs((ppb[h][i]>=pb[h]-pbcmax*(1-zb[h][i]) for h in range(H) for i in range(num_nodes)), name='c-pb4')
m.addConstrs((pb[h]>=-pbc for h in range(H)), name='c-pb5')
m.addConstrs((pb[h]<=pbc for h in range(H)), name='c-pb6')
m.addConstrs((gp.quicksum(zb[h][i] for i in range(num_nodes))==1 for h in range(H)),name='c-pb7')
m.addConstr(cap[0]==cap0-pb[0],name='c-cap0')
m.addConstr(cap[H-1]==cap0,name='c-cap024')
m.addConstrs((cap[h+1]==cap[h]-pb[h+1] for h in range(H-1)),name='c-caph')
m.addConstrs((cap[h]<=pbc for h in range(H)),name='c-capup')
m.addConstr(cap0<=pbc,name='c-cap0up')



m.addConstrs((gp.quicksum(EqNp[h][i] for i in range(num_nodes))==pl[h] for h in range(H)), name='c-ppl')
m.addConstrs((gp.quicksum(EqNq[h][i] for i in range(num_nodes))==ql[h] for h in range(H)), name='c-pql')



"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj=pl.sum()
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('NonConvex',1)
#m.setParam('Presolve',-1)
#m.setParam('Seed',100)
m.setParam('Threads',12)
#m.setParam('ConcurrentMIP',6)
m.setParam('MIPFocus',3)
m.setParam('MIPGap',1e-6)
m.setParam("TimeLimit", 21600)

m.optimize()
"-------------Print solution------------"
pgo=pgref.X
qgo=qgref.X
qvno=qvn.X
qiko=qik.X
pko=pk.X
qko=qk.X
pbco=pbc.X[0]
pbo=pb.X
ppbo=ppb.X
zbo=zb.X
cap0o=cap0.X
capo=cap.X

vo=np.sqrt(qvno)

plt.plot(vo)

Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for k in range(num_lines):
        Equp[h][fr[k]]+=pko[h][k]
        Equp[h][to[k]]+=(np.real(zk[k])*qiko[h][k])-pko[h][k]
        Equq[h][fr[k]]+=qko[h][k]
        Equq[h][to[k]]+=(np.imag(zk[k])*qiko[h][k])-qko[h][k]


pho=np.zeros([H,num_nodes])
beta=np.zeros([H,num_lines])
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])
    for k in range(num_lines):
        beta[h][k]=np.arctan((np.real(zk[k])*qko[h][k]-np.imag(zk[k])*pko[h][k])/(qvno[h][fr[k]]-np.real(zk[k])*pko[h][k]-np.imag(zk[k])*qko[h][k]))
        pho[h][to[k]]=pho[h][fr[k]]-beta[h][k]


t=np.zeros(H)
t[0]=m.Runtime

gap=np.zeros(H)
gap[0]=m.MIPGap

pbcout=np.zeros(H)
pbcout[0]=pbco

ppvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=ppv
   
cap0out=np.zeros(H)
cap0out[0]=cap0o

out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,gap,imeans,pbcout,pbo,cap0out,capo)).T
output=np.hstack((vo,pho,Equp,Equq,out1,zbo,dmeans,ppvout))
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
columns.append('Gap')
columns.append('ic')
columns.append('pbc')
columns.append('pb')
columns.append('cap0')
columns.append('caph')
for i in range(num_nodes):    
    columns.append('zb'+str(i+1))
for i in range(ncluster):    
    columns.append('dc_c'+str(i+1))
for i in range(num_nodes):    
    columns.append('ppv'+str(i+1))


    
df.columns=columns

solvlist=[0]*H
solvlist[0]='G'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")
