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
import scipy.stats as ss


case='CA141'
city='Jam'
city1='JAM'
problem='OPF_PV_S'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')


mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=np.squeeze(mat['clusternode'])

cnode[0]=1
cnode=cnode-1

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


#npv=np.floor(0.1*num_nodes)
npv=5
pvcmax=0.5*np.sum(pdm)
pveff=0.8

#nlv=-10

"pwl downsampling"

ndpwl=np.size(xdempwl,axis=3)
nipwl=np.size(xirrpwl,axis=2)

dsfactor=1

#dxdpwl=np.zeros([H,ncluster,ndists,nddpwl])
#dxipwl=np.zeros([H,ndists,ndipwl])

dxdpwl=xdempwl[:,:,:,0:ndpwl:dsfactor]
dxipwl=xirrpwl[:,:,0:nipwl:dsfactor]
dydpwl=ydempwl[:,:,:,0:ndpwl:dsfactor]
dyipwl=yirrpwl[:,:,0:nipwl:dsfactor]

mindc=np.zeros([H,ncluster])
maxdc=np.zeros([H,ncluster])

minic=np.zeros(H)
maxic=np.zeros(H)

for h in range(H):
    for c in range(ncluster):
        dist=bestfitsd[c][h]
        mindc[h][c]=np.min(xdempwl[h][c][dist])
        maxdc[h][c]=np.max(xdempwl[h][c][dist])
    dist=bestfitsi[h]
    if dist!=10:
        minic[h]=np.min(xirrpwl[h][dist])
        maxic[h]=np.max(xirrpwl[h][dist])

"----- Optimization model -----"
m = gp.Model("PF-rect")

pgref = m.addMVar((H,num_nodes),lb=prefmin,ub=prefmax,name='pgref')
qgref = m.addMVar((H,num_nodes),lb=qrefmin,ub=qrefmax,name='qgref')

qvn= m.addMVar((H,num_nodes),lb=qvnmin,ub=qvnmax,name='qvn')
qik= m.addMVar((H,num_lines),lb=-GRB.INFINITY,name='qik')
pk= m.addMVar((H,num_lines),lb=-GRB.INFINITY,name='pk')
qk= m.addMVar((H,num_lines),lb=-GRB.INFINITY,name='qk')

pv= m.addMVar(num_nodes,name='pv')
ppv= m.addMVar(num_nodes,name='ppv')
zpv= m.addMVar(num_nodes,name='zpv',vtype=GRB.BINARY)
pic= m.addMVar((H,num_nodes),name='pic')

dc= m.addMVar((H,ncluster),lb=mindc,ub=maxdc,name='dc')
ic= m.addMVar(H,lb=minic,ub=maxic,name='ic')

pl= m.addMVar(H,name='pl')
ql= m.addMVar(H,name='ql')

probdem=m.addMVar((H,ncluster),ub=1,name='prob_dem')
probirr=m.addMVar(H,ub=1,name='prob_irr')

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

m.addConstrs((EqNp[h][i]==pgref[h][i]+(pic[h][i]*pveff)+pgen[i]-(pdm[i]*dc[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-p')
m.addConstrs((EqNq[h][i]==qgref[h][i]+qgen[i]-(qdm[i]*dc[h][cnode[i]]) for h in range(H) for i in range(num_nodes)), name='c-pf-q')


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
        
m.addConstrs((gp.quicksum(EqNp[h][i] for i in range(num_nodes))==pl[h] for h in range(H)), name='c-ppl')
m.addConstrs((gp.quicksum(EqNq[h][i] for i in range(num_nodes))==ql[h] for h in range(H)), name='c-pql')              
    
"-------Objective definition--------"
#obj=gp.quicksum(EqNp[h][i] for h in range(H) for i in range(num_nodes))+gp.quicksum(EqNq[h][i] for h in range(H) for i in range(num_nodes))-probdem.sum()+probirr.sum()
obj=pl.sum()+ql.sum()-probdem.sum()+probirr.sum()
m.setObjective(obj, GRB.MINIMIZE)

"-------Problem/solver Setup--------"
#m.tune()
# #m.setParam('DualReductions',0)
# #m.setParam('FuncNonlinear',1)
m.setParam('NonConvex',2)
# m.setParam('Aggregate',0)
# m.setParam('Presolve',0)
m.setParam('MIPFocus',3)
m.setParam('MIPGap',1e-5)
# m.setParam("TimeLimit", 120);
# m.read('tune0.prm')
m.optimize()
"-------------Print solution------------"

pgo=pgref.X
qgo=qgref.X
qvno=qvn.X
qiko=qik.X
pko=pk.X
qko=qk.X
pico=pic.X
ico=ic.X
dco=dc.X
probdo=probdem.X
probio=probirr.X
pvo=pv.X
ppvo=ppv.X
zpvo=zpv.X


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

npwlout=np.zeros(H)
npwlout[0]=np.size(dxdpwl,axis=3)

ppvout=np.zeros([H,num_nodes])
zpvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=ppvo
    zpvout[h]=zpvo

probdco= np.zeros([H,ncluster])
probico=  np.zeros(H)

for h in range(H):        
    for i in range(ncluster):            
        distribution=adists[bestfitsd[i][h]]
        "Exponential"
        if distribution==adists[0]:                    
            probdco[h][i]=1-np.exp(-dparams[i][h][0]*dco[h][i])            
        "Fisk  (Beta is always positive)"
        if distribution==adists[1]:
            probdco[h][i]=1/(1+np.power(dco[h][i]/dparams[i][h][2],-dparams[i][h][3]))                          
        "Logistic"                        
        if distribution==adists[2]:                
            probdco[h][i]=1/(1+np.exp((-dco[h][i]+dparams[i][h][4])/dparams[i][h][5]))
        "LogNorm"                        
        if distribution==adists[3]:                
            probdco[h][i]=ss.norm.cdf(np.log(dco[h][i]),dparams[i][h][6],dparams[i][h][7])
        "Normal"
        if distribution==adists[4]:                
            probdco[h][i]=ss.norm.cdf(dco[h][i],dparams[i][h][8],dparams[i][h][9])
        "Rayleigh"
        if distribution==adists[5]:                
            probdco[h][i]=1-np.exp(-np.square(dco[h][i])/(2*np.square(dparams[i][h][10])))
        "Weibull"
        if distribution==adists[6]:                
            probdco[h][i]=1-np.exp(-np.power(dco[h][i]*dparams[i][h][12],dparams[i][h][13]))
            
    if bestfitsi[h]!=10:
        distribution=adists[bestfitsi[h]]
        "Exponential"
        if distribution==adists[0]:               
            probico[h]=1-np.exp(-iparams[h][0]*ico[h])            
        "Fisk  (Beta is always positive)"
        if distribution==adists[1]:
            probico[h]=1/(1+np.power(ico[h]/iparams[h][2],-iparams[h][3]))                          
        "Logistic"                        
        if distribution==adists[2]:                
            probico[h]=1/(1+np.exp((-ico[h]+iparams[h][4])/iparams[h][5]))
        "LogNorm"                        
        if distribution==adists[3]:                
            probico[h]=ss.norm.cdf(np.log(ico[h]),iparams[h][6],iparams[h][7])
        "Normal"
        if distribution==adists[4]:                
            probico[h]=ss.norm.cdf(ico[h],iparams[h][8],iparams[h][9])
        "Rayleigh"
        if distribution==adists[5]:                
            probico[h]=1-np.exp(-np.square(ico[h])/(2*np.square(iparams[h][10])))
        "Weibull"
        if distribution==adists[6]:                
            probico[h]=1-np.exp(-np.power(ico[h]*iparams[h][12],iparams[h][13]))


out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,npwlout,ico,probio,probico)).T
output=np.hstack((vo,pho,Equp,Equq,out1,dco,probdo,probdco,ppvout,zpvout))
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
columns.append('npwl_prob')
columns.append('ic')
columns.append('prob_ic_opt')
columns.append('prob_ic_cal')
for i in range(ncluster):    
    columns.append('dc_c'+str(i+1))
for i in range(ncluster):    
    columns.append('prob_dc_opt'+str(i+1))
for i in range(ncluster):    
    columns.append('prob_dc_cal'+str(i+1))
for i in range(num_nodes):    
    columns.append('ppv'+str(i+1))
for i in range(num_nodes):    
    columns.append('zpv'+str(i+1))
    
df.columns=columns

solvlist=[0]*H
solvlist[0]='G'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("ResultsPWL.xlsx")




    
    
        
  


    
