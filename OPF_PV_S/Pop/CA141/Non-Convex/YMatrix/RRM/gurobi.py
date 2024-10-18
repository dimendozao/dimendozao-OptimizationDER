# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:50:10 2024

@author: diego
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sp

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

case='CA141'
city='Bog'
city1='BOG'
problem='MPF'

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=mat['clusternode']
cnode[0]=1

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\PWLxyDem_'+city1+'.mat')
xdempwl=mat['xpwl']
ydempwl=mat['ypwl']

c1 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC1.csv')
c2 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC2.csv')
c3 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC3.csv')
c4 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC4.csv')
c5 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC5.csv')


irr = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\'+'ParamTable.csv')

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\NSTparamDem_'+city1+'.mat')
dparams=mat['params']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\NSTparamRAD_'+city1+'.mat')
iparams=mat['params']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\PWLxyRAD_'+city1+'.mat')
xirrpwl=mat['xpwl']
yirrpwl=mat['ypwl']

adists=['Exponential','Fisk','Logistic','Log-N','Normal','Rayleigh','Weibull'];
aclust=['c1','c2','c3','c4','c5']

ndists=len(adists)
ncluster=len(dparams)
H=np.size(dparams,axis=1)

cparameters=pd.concat([c1,c2,c3,c4,c5],keys=aclust)

bestfitsd=[[0]*H for i in range(ncluster)]

bestfitsi=[10]*H 
nihours=int(len(irr)/2)
ihours=np.zeros(H)


for h in range(H):
    for hh in range(nihours):
        if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
            ihours[h]=1
    
    
    
for h in range(H):
    for hh in range(nihours):
        if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
            ihours[h]=1
            for j in range(ndists):
                if adists[j] in irr['bestparams1'][2*hh] and irr['bestparams1'][2*hh].find(adists[j])==0:
                   bestfitsi[h]=j 
                   
    for i in range(ncluster):
        for j in range(ndists):
            if adists[j] in cparameters['bestparams1'][aclust[i],2*h] and cparameters['bestparams1'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd[i][h]=j



num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])
    
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

prefmax=np.tile(prefmax,(H,1))
prefmin=np.tile(prefmin,(H,1))
qrefmax=np.tile(qrefmax,(H,1))
qrefmin=np.tile(qrefmin,(H,1))

idx=np.abs(ym)!=0

nz=np.sum(idx)

ix=np.zeros(nz,dtype=int)
jx=np.zeros(nz,dtype=int)
yrx=np.zeros(nz)
yix=np.zeros(nz)

k=0

for i in range(num_nodes):
    for j in range(num_nodes):
        if not np.abs(ym[i][j])==0:
            ix[k]=i
            jx[k]=j
            yrx[k]=np.real(ym[i][j])
            yix[k]=np.imag(ym[i][j])
            k+=1
            
yrs=sp.coo_matrix((yrx,(ix,jx)),shape=(num_nodes,num_nodes))  
yis=sp.coo_matrix((yix,(ix,jx)),shape=(num_nodes,num_nodes))

npv=np.floor(0.1*num_nodes)
#npv=1
pvcmax=0.5*np.sum(pdm)
pveff=0.8

#nlv=-10

"pwl downsampling"

ndpwl=np.size(xdempwl,axis=3)
nipwl=np.size(xirrpwl,axis=2)

dsfactor=2

#dxdpwl=np.zeros([H,ncluster,ndists,nddpwl])
#dxipwl=np.zeros([H,ndists,ndipwl])

dxdpwl=xdempwl[:,:,:,0:ndpwl:dsfactor]
dxipwl=xirrpwl[:,:,0:nipwl:dsfactor]
dydpwl=ydempwl[:,:,:,0:ndpwl:dsfactor]
dyipwl=yirrpwl[:,:,0:nipwl:dsfactor]  
            
"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar(H,lb=0,ub=prefmax[0,0],name='pgref')
qgref = m.addMVar(H,lb=0,ub=qrefmax[0,0],name='qgref')

cn= m.addMVar((H,num_nodes,num_nodes),lb=cnmin,ub=cnmax,name='cn')
sn= m.addMVar((H,num_nodes,num_nodes),lb=snmin,ub=snmax,name='sn')

pv= m.addMVar(num_nodes,ub=pvcmax,name='pv')
ppv= m.addMVar(num_nodes,ub=pvcmax,name='ppv')
zpv= m.addMVar(num_nodes,name='zpv',vtype=GRB.BINARY)
pic= m.addMVar((H,num_nodes),ub=pvcmax,name='pic')

dc= m.addMVar((H,ncluster),ub=2,name='dc')
ic= m.addMVar(H,ub=2,name='ic')

pl= m.addMVar(H,name='pl')
ql= m.addMVar(H,name='ql')

probdem=m.addMVar((H,ncluster),ub=1,name='prob_dem')
probirr=m.addMVar(H,ub=1,name='prob_irr')

"-------Constraint Construction-------- "

m.addConstrs(((cn[h][i][j]*cn[h][i][j])+(sn[h][i][j]*sn[h][i][j])==cn[h][i][i]*cn[h][j][j] for h in range(H) for i in range(num_nodes) for j in range(num_nodes) if (idx[i][j] and j!=i)), name='c-soc')        

m.addConstrs((cn[h][i][j]==cn[h][j][i] for h in range(H) for i in range(num_nodes) for j in range(num_nodes) if (idx[i][j] and j!=i)), name='c-her-1')
m.addConstrs((sn[h][i][j]==-sn[h][j][i] for h in range(H) for i in range(num_nodes) for j in range(num_nodes) if (idx[i][j] and j!=i)), name='c-her-2')

EqNp=[[0]*num_nodes for h in range(H)]
EqNq=[[0]*num_nodes for h in range(H)]

pl=[0]*H
ql=[0]*H

for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            if idx[i][j]:
                EqNp[h][i]+=cn[h][i][j]*ymr[i][j]+sn[h][i][j]*ymi[i][j]
                EqNq[h][i]+=sn[h][i][j]*ymr[i][j]-cn[h][i][j]*ymi[i][j]
    pl[h]=gp.quicksum(EqNp[h])
    ql[h]=gp.quicksum(EqNq[h])



m.addConstrs((EqNp[h][0]==pgref[h]+(pic[h][0]*pveff)+pgen[0]-pdm[0] for h in range(H)), name='c-pf0-p')
m.addConstrs((EqNq[h][0]==qgref[h]+qgen[0]-qdm[0] for h in range(H)), name='c-pf0-q')
            
m.addConstrs((EqNp[h][i]==(pic[h][i]*pveff)+pgen[i]-(pdm[i]*dc[h][cnode[i]-1]) for h in range(H) for i in range(num_nodes) if i>0), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgen[i]-(qdm[i]*dc[h][cnode[i]-1]) for h in range(H) for i in range(num_nodes) if i>0), name='c-pf-q')

m.addConstrs((pic[h,i]==ppv[i]*ic[h] for h in range(H) for i in range(num_nodes) if bestfitsi[h]!=10), name='c-pic')
m.addConstrs((pic[h,i]==0 for h in range(H) for i in range(num_nodes) if bestfitsi[h]==10), name='c-pic0')
m.addConstrs((ic[h]==0 for h in range(H) if bestfitsi[h]==10), name='c-ic0')

m.addConstrs((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)), name='c-pv1')
m.addConstrs((ppv[i]<=pv[i] for i in range(num_nodes)), name='c-pv2')
m.addConstrs((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)), name='c-pv3')

m.addConstr(zpv.sum()==npv)
m.addConstr(ppv.sum()==pvcmax)

for h in range(H):
    for i in range(ncluster):        
        m.addGenConstrPWL(dc[h][i],probdem[h][i],dxdpwl[h][i][bestfitsd[i][h]],dydpwl[h][i][bestfitsd[i][h]]) # p    
    
    if bestfitsi[h]!=10:
        m.addGenConstrPWL(ic[h],probirr[h],dxipwl[h][bestfitsi[h]],dyipwl[h][bestfitsi[h]]) # p        
    else:        
        m.addConstr(probirr[h]==0)

"-------Objective definition--------"
#obj=1
#obj = pgref.sum() + qgref.sum()
obj= gp.quicksum(pl)+gp.quicksum(ql)-probdem.sum()+probirr.sum()
#m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.setParam('DualReductions',0)
m.setParam('FuncNonlinear',1)
m.setParam('NonConvex',2)
#m.setParam('Aggregate',0)
#m.setParam('Presolve',0)
m.setParam('MIPFocus',3)
m.setParam('MIPGap',1e-6)
#m.setParam("TimeLimit", 120);

m.optimize()