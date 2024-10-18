# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:51:10 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import cvxpy as cvx


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

bestfitsd=np.zeros([H,ncluster],dtype=int)


bestfitsi=10*np.ones(H,dtype=int)


bestfitsd1=np.zeros([H,ncluster],dtype=int)
bestfitsd2=np.zeros([H,ncluster],dtype=int)
bestfitsd3=np.zeros([H,ncluster],dtype=int)


bestfitsi1=10*np.ones(H,dtype=int)
bestfitsi2=10*np.ones(H,dtype=int)
bestfitsi3=10*np.ones(H,dtype=int)
 
nihours=int(len(irr)/2)
ihours=np.zeros(H)


for h in range(H):
    for hh in range(nihours):
        if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
            ihours[h]=1
    
okfitsdem=[1,1,1,0,0,1,1]
okfitsirr=[1,0,1,0,0,1,1]     
    
for h in range(H):
    for hh in range(nihours):
        if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
            ihours[h]=1
            for j in range(ndists):
                if adists[j] in irr['bestparams1'][2*hh] and irr['bestparams1'][2*hh].find(adists[j])==0:
                   bestfitsi1[h]=j
                if adists[j] in irr['bestparams2'][2*hh] and irr['bestparams2'][2*hh].find(adists[j])==0:
                   bestfitsi2[h]=j
                if adists[j] in irr['bestparams3'][2*hh] and irr['bestparams3'][2*hh].find(adists[j])==0:
                   bestfitsi3[h]=j
                   
    for i in range(ncluster):
        for j in range(ndists):
            if adists[j] in cparameters['bestparams1'][aclust[i],2*h] and cparameters['bestparams1'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd1[h][i]=j
            if adists[j] in cparameters['bestparams2'][aclust[i],2*h] and cparameters['bestparams2'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd2[h][i]=j
            if adists[j] in cparameters['bestparams3'][aclust[i],2*h] and cparameters['bestparams3'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd3[h][i]=j

nten=np.sum(bestfitsi1==10)                

for h in range(H):
    for i in range(ncluster):
        if okfitsdem[bestfitsd1[h][i]]==1:          
            bestfitsd[h][i]=bestfitsd1[h][i]
        elif okfitsdem[bestfitsd2[h][i]]==1:
            bestfitsd[h][i]=bestfitsd2[h][i]
        elif okfitsdem[bestfitsd3[h][i]]==1:
            bestfitsd[h][i]=bestfitsd3[h][i]
        else:
            bestfitsd[h][i]=10
    if bestfitsi1[h]!=10:
        if okfitsirr[bestfitsi1[h]]==1:
            bestfitsi[h]=bestfitsi1[h]            
        elif okfitsirr[bestfitsi2[h]]==1:
            bestfitsi[h]=bestfitsi2[h]
        elif okfitsirr[bestfitsi3[h]]==1:
            bestfitsi[h]=bestfitsi3[h]
        else:
            if bestfitsi1[h]==1 and (iparams[h][3]<0 or iparams[h][3]>1):
                bestfitsi[h]=bestfitsi1[h]
            elif bestfitsi2[h]==1 and (iparams[h][3]<0 or iparams[h][3]>1):
                bestfitsi[h]=bestfitsi2[h]
            elif bestfitsi3[h]==1 and (iparams[h][3]<0 or iparams[h][3]>1):
                bestfitsi[h]=bestfitsi3[h]
            else:
                bestfitsi[h]=10
            
if np.sum(bestfitsd==10)>0 or np.sum(bestfitsi==10)>nten :
    ok=0
    print('No todas las distribuciones se pueden modelar sin PWLA')
else:
    ok=1


if ok==1:        
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
    
    
    npv=np.floor(0.1*num_nodes)
    #npv=1
    pvcmax=0.5*np.sum(pdm)
    pveff=0.8
    nlv=-10
    
    npwl=10
    icxg=np.linspace(0,2,num=npwl)
    icxm=np.zeros([npwl,npwl])
    pvxg=np.linspace(0,pvcmax,num=npwl)
    pvxm=np.zeros([npwl,npwl])
    pvhx=np.zeros([npwl,npwl])
    for i in range(npwl):
        pvxm[i]=pvxg[i]*np.ones(npwl)
        icxm[i]=icxg[i]*np.ones(npwl)
    
    icxm=icxm.T
    pvhx=np.multiply(pvxm,icxm)
            
            
    "----- Optimization model -----"
    
    pgref = cvx.Variable((H,num_nodes),nonneg=True,name='pgref')
    qgref = cvx.Variable((H,num_nodes),nonneg=True,name='qgref')

    qvn= cvx.Variable((H,num_nodes),nonneg=True,name='qvn')
    qik= cvx.Variable((H,num_lines),name='qik')
    pk= cvx.Variable((H,num_lines),name='pk')
    qk= cvx.Variable((H,num_lines),name='qk')
    
    pv= cvx.Variable(num_nodes,nonneg=True,name='pv')
    ppv= cvx.Variable([H,num_nodes],nonneg=True,name='ppv')
    pvh= cvx.Variable([H,num_nodes],nonneg=True,name='pvh')
    zpv= cvx.Variable(num_nodes,boolean=True,name='zpv')
    

    dc= cvx.Variable((H,num_nodes),nonneg=True,name='dc')
    ic= cvx.Variable(H,nonneg=True,name='ic')

       
    lambdas=[[0]*num_nodes for i in range(H)]
    deltax=[0]*H
    deltae=[0]*H
    xis=[0]*H
    etas=[0]*H
    
    for h in range(H):
        for i in range(num_nodes):
            lambdas[h][i]=  cvx.Variable([npwl,npwl],nonneg=True)
        
        deltax[h]= cvx.Variable((num_nodes,npwl-1),boolean=True)
        deltae[h]= cvx.Variable((num_nodes,npwl-1),boolean=True)
        xis[h]=    cvx.Variable((num_nodes,npwl),nonneg=True)
        etas[h]=   cvx.Variable((num_nodes,npwl),nonneg=True)
    
    "-------Constraint Construction-------- "

    EqNp = [num_nodes*[0] for h in range(H)]
    EqNq = [num_nodes*[0] for h in range(H)]
    
    probd=  [[0]*num_nodes for i in range(H)]
    hprobd= [0]*H 
    probi=  [0]*H 
    
    pl=  [0]*H
    ql=  [0]*H
    
    res=[]

    for h in range(H):
        for k in range(num_lines):
            EqNp[h][fr[k]]+=pk[h][k]
            EqNp[h][to[k]]+=(np.real(zk[k])*qik[h][k])-pk[h][k]
            EqNq[h][fr[k]]+=qk[h][k]
            EqNq[h][to[k]]+=(np.imag(zk[k])*qik[h][k])-qk[h][k]
            res += [qvn[h][fr[k]]-qvn[h][to[k]]==2*(pk[h][k]*np.real(zk[k])+qk[h][k]*np.imag(zk[k]))-qik[h][k]*(np.square(np.abs(zk[k])))]
            up = qik[h][k]+qvn[h][fr[k]]
            um = qik[h][k]+qvn[h][fr[k]]
            st = cvx.vstack([2*pk[h][k],2*qk[h][k],um])
            res += [cvx.SOC(up,st)]
            
    
    
    
    "Variable Bounds"
    res += [qvn<=qvnmax]
    res += [qvn>=qvnmin]
    res += [pgref<=prefmax]
    res += [pgref>=prefmin]
    res += [qgref<=qrefmax]
    res += [qgref>=qrefmin]
    res += [cvx.sum(zpv)==npv]
    res += [cvx.sum(pv)==pvcmax]
    
    for i in range(num_nodes):
        res +=  [pv[i]<=pvcmax]
    
    for h in range(H):        
        for i in range(num_nodes): 
            "Power Flow constraints"
            res +=  [pgref[h][i]+(ppv[h][i]*pveff)+pgen[i]-(pdm[i]*dc[h][i])==EqNp[h][i]]
            res +=  [qgref[h][i]+qgen[i]-(qdm[i]*dc[h][i])==EqNq[h][i]]
            "Pv integer linearization"
            res +=  [ppv[h][i]<=pvcmax*zpv[i]]
            res +=  [ppv[h][i]<=pvh[h][i]]
            res +=  [ppv[h][i]>=pvh[h][i]-(pvcmax*(1-zpv[i]))]
            
            
            "Piecewise linear constraints"
            
            res +=  [pv[i]==cvx.sum(cvx.multiply(pvxm,lambdas[h][i]))]                               
        
            if not bestfitsi[h]==10:
                res += [ic[h]==cvx.sum(cvx.multiply(icxm,lambdas[h][i]))]
                res += [pvh[h][i]==cvx.sum(cvx.multiply(pvhx,lambdas[h][i]))]
            else:
                res += [ic[h]==0]
                res += [pvh[h][i]==0]
        
            res += [cvx.sum(lambdas[h][i])==1]
            res += [xis[h][i]==cvx.sum(lambdas[h][i],axis=0)]
            res += [etas[h][i]==cvx.sum(lambdas[h][i],axis=1)]
            res += [cvx.sum(deltax[h][i])==1]
            res += [cvx.sum(deltae[h][i])==1]
            
            "Probabilities calculation"
            if i>0:
                clusternode=cnode[i][0]-1
                distribution=adists[bestfitsd[h][clusternode]]
                if distribution==adists[0]:                    
                    a=-dparams[clusternode][h][0]*dc[h][i]
                    probd[h][i]=1-cvx.exp(a)  #expon concave (max p)
                if distribution==adists[1]:                    
                    probd[h][i]=dparams[clusternode][h][3]*cvx.log(dc[h][i]) # Fisk concave (max p)
                if distribution==adists[2]:
                    a=(-dc[h][i]+dparams[clusternode][h][4])/dparams[clusternode][h][5]
                    probd[h][i]=-cvx.logistic(a) # logistic concave (max p)
                if distribution==adists[5]:
                    probd[h][i]=2*cvx.log(dc[h][i])-np.log(2*dparams[clusternode][h][10]*dparams[clusternode][h][10]) # Rayleigh concave (max p)
                if distribution==adists[6]:
                    probd[h][i]=cvx.log(dc[h][i])+np.log(dparams[clusternode][h][12]) # Weibull concave (max p)   
        
        if bestfitsi[h]!=10:
            distribution=adists[bestfitsi[h]]
            if distribution==adists[0]:               
                probi[h]=1-cvx.exp(iparams[h][0]*ic[h]) # exponential convex (min p)--> concave (max -p)
            if distribution==adists[1]:
                a=ic[h]/iparams[h][2]
                probi[h]=-cvx.power(a,iparams[h][3]) # fisk convex (min p)--> concave (max -p)   
            if  distribution==adists[2]:
                a=(-ic[h]+iparams[h][4])/iparams[h][5]
                probi[h]=-cvx.logistic(a)   # Logistic concave (max p)   
            if  distribution==adists[5]:
                probi[h]=(-cvx.square(ic[h]))/(2*iparams[h][10]*iparams[h][10]) # Rayleigh concave (max 1-p)
            if  distribution==adists[6]:
                a=cvx.power(ic[h]*iparams[h][12],iparams[h][13])
                probi[h]=1-cvx.exp(a) # Weibull convex (min p)--> concave (max -p)
        else:
            res += [probi[h]==0]
        
    for h in range(H):
        hprobd[h]=cvx.sum(probd[h])
        pl[h]=cvx.sum(EqNp[h])
        ql[h]=cvx.sum(EqNq[h])
    "-------Objective definition--------"
    #obj = cvx.Minimize(cvx.real(cvx.sum(sref+sgen-sd-cvx.hstack(EqN)))+cvx.imag(cvx.sum(sref+sgen-sd-cvx.hstack(EqN))))
    #obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
    obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql)+cvx.sum(dc)-cvx.sum(ic)-cvx.sum(hprobd)-cvx.sum(probi))
    #obj = cvx.Minimize(cvx.abs(sref[0])+cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
    #obj = cvx.Minimize(cvx.abs(sref[0]))
    #obj = cvx.Minimize(1)
    "-------Problem/solver Setup--------"
    OPFSOC = cvx.Problem(obj,res)
    OPFSOC.solve(solver=cvx.MOSEK,save_file='probmosek.ptf',verbose=True)
    #OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_FREE'},verbose=True)    
    #print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)

    

