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
import scipy.stats as ss



case='IEEE33'
city='Bog'
city1='BOG'
problem='MPF'

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
    
okfitsdem=[1,1,1,1,1,1,1]
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
            if bestfitsi1[h]==1 and iparams[h][3]>1:
                bestfitsi[h]=bestfitsi1[h]
            elif bestfitsi2[h]==1 and iparams[h][3]>1:
                bestfitsi[h]=bestfitsi2[h]
            elif bestfitsi3[h]==1 and iparams[h][3]>1:
                bestfitsi[h]=bestfitsi3[h]
            else:
                bestfitsi[h]=10
            
if np.sum(bestfitsd==10)>0 or np.sum(bestfitsi==10)>nten:
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
    
    prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]+10
    prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
    qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]+10
    qrefmin[iref]=np.max([0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]])
    
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
    npv=1
    pvcmax=0.5*np.sum(pdm)
    pveff=0.8
        
    mindc=np.zeros([H,ncluster])
    maxdc=np.zeros([H,ncluster])

    minic=np.zeros(H)
    maxic=np.zeros(H)

    for h in range(H):
        for c in range(ncluster):
            dist=bestfitsd[h,c]
            mindc[h][c]=np.min(xdempwl[h][c][dist])
            maxdc[h][c]=np.max(xdempwl[h][c][dist])
        dist=bestfitsi[h]
        if dist!=10:
            minic[h]=np.min(xirrpwl[h][dist])
            maxic[h]=np.max(xirrpwl[h][dist]) 
            
    npwl=3
    icxg=np.zeros([H,npwl])
    for h in range(H):
        icxg[h]=np.linspace(minic[h],maxic[h],npwl)
    
    icxm=np.zeros([H,npwl,npwl])
    for h in range(H):
        icxm[h]=icxg[h]*np.ones(npwl)
        
    
    pvxg=np.linspace(0,pvcmax,npwl)
    
    pvxm=np.zeros([npwl,npwl])
    
    for i in range(npwl):
        pvxm[i]=pvxg[i]*np.ones(npwl)
        
    pvhx=np.zeros([H,npwl,npwl])
    
    for h in range(H):
        pvhx[h]=np.multiply(icxm[h],pvxm)
    
    pvhx=np.multiply(pvxm,icxm)
    
           
            
    "----- Optimization model -----"
    
    pgref = cvx.Variable((H,num_nodes),name='pgref')
    qgref = cvx.Variable((H,num_nodes),name='qgref')

    qvn= cvx.Variable((H,num_nodes),name='qvn')
    # qik= cvx.Variable((H,num_lines),nonneg=True,name='qik')
    # pk= cvx.Variable((H,num_lines),nonneg=True,name='pk')
    # qk= cvx.Variable((H,num_lines),nonneg=True,name='qk')
    
    qik= cvx.Variable((H,num_lines),name='qik')
    pk= cvx.Variable((H,num_lines),name='pk')
    qk= cvx.Variable((H,num_lines),name='qk')
    
    pv= cvx.Variable(num_nodes,name='pv')
    ppv= cvx.Variable([H,num_nodes],name='ppv')
    pvh= cvx.Variable([H,num_nodes],name='pvh')
    zpv= cvx.Variable(num_nodes,boolean=True,name='zpv')
    

    dc= cvx.Variable((H,ncluster),name='dc')
    ic= cvx.Variable(H,name='ic')

       
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
    
    probd=  [[0]*ncluster for i in range(H)]
    hprobd= [0]*H 
    probi=  [0]*H 
    
    pl=  [0]*H
    ql=  [0]*H
    
    res=[]
    
    "Variable Bounds"
    res += [qvn<=qvnmax]
    res += [qvn>=qvnmin]
    res += [pgref<=prefmax]
    res += [pgref>=prefmin]
    res += [qgref<=qrefmax]
    res += [qgref>=qrefmin]
    #res += [qik<=100]
    #res += [pk<=10]
    #res += [qk<=10]
    res += [ic<=maxic]
    res += [ic>=minic]
    res += [dc<=maxdc]
    res += [dc>=mindc]
    res += [cvx.sum(zpv)==npv]
    res += [cvx.sum(pv)==pvcmax]

    for h in range(H):
        for k in range(num_lines):
            EqNp[h][fr[k]]+=pk[h][k]
            EqNp[h][to[k]]+=(np.real(zk[k])*qik[h][k])-pk[h][k]
            EqNq[h][fr[k]]+=qk[h][k]
            EqNq[h][to[k]]+=(np.imag(zk[k])*qik[h][k])-qk[h][k]
            res += [qvn[h][fr[k]]-qvn[h][to[k]]==2*(pk[h][k]*np.real(zk[k])+qk[h][k]*np.imag(zk[k]))-qik[h][k]*(np.square(np.abs(zk[k])))]            
            
            up = qik[h][k]+qvn[h][fr[k]]
            um = qik[h][k]-qvn[h][fr[k]]
            st = cvx.vstack([2*pk[h][k],2*qk[h][k],um])
            res += [cvx.SOC(up,st)]   
              
        for i in range(num_nodes): 
            "Power Flow constraints"
            #res +=  [pgref[h][i]+pgen[i]-(pdm[i]*dc[h][cnode[i]])==EqNp[h][i]]
            res +=  [pgref[h][i]+(ppv[h][i]*pveff)+pgen[i]-(pdm[i]*dc[h][cnode[i]])==EqNp[h][i]]
            res +=  [qgref[h][i]+qgen[i]-(qdm[i]*dc[h][cnode[i]])==EqNq[h][i]]
            "Pv integer linearization"
            res +=  [ppv[h][i]<=pvcmax*zpv[i]]
            res +=  [ppv[h][i]<=pvh[h][i]]
            res +=  [ppv[h][i]>=pvh[h][i]-(pvcmax*(1-zpv[i]))]          
            "Piecewise linear constraints"
            res +=  [pv[i]==cvx.sum(cvx.multiply(pvxm,lambdas[h][i]))]                               
            res +=  [ic[h]==cvx.sum(cvx.multiply(icxm[h],lambdas[h][i]))]
            res +=  [pvh[h][i]==cvx.sum(cvx.multiply(pvhx[h],lambdas[h][i]))]
           
            res += [cvx.sum(lambdas[h][i])==1]
            res += [xis[h][i]==cvx.sum(lambdas[h][i],axis=0)]
            res += [etas[h][i]==cvx.sum(lambdas[h][i],axis=1)]
            res += [cvx.sum(deltax[h][i])==1]
            res += [cvx.sum(deltae[h][i])==1]            
            for k in range(npwl):
                if k==0:
                    res += [xis[h][i][k]-deltax[h][i][k]<=0]
                    res += [etas[h][i][k]-deltae[h][i][k]<=0]
                elif k==npwl-1:
                    res += [xis[h][i][k]-deltax[h][i][k-1]<=0]
                    res += [etas[h][i][k]-deltae[h][i][k-1]<=0]
                else:
                    res += [xis[h][i][k]-deltax[h][i][k]-deltax[h][i][k-1]<=0]
                    res += [etas[h][i][k]-deltae[h][i][k]-deltae[h][i][k-1]<=0]
            
        "Probabilities calculation"
        for c in range(ncluster):        
            if bestfitsd[h][c]==0:                    
                probd[h][c]=1-cvx.exp(-dparams[c][h][0]*dc[h][c])  #expon concave (max p)
            elif bestfitsd[h][c]==1:
                if dparams[c][h][3]<=1:
                    probd[h][c]=cvx.power(dc[h][c]/dparams[c][h][2],dparams[c][h][3]) # Fisk concave (max p)
                else:
                    probd[h][c]=dparams[c][h][3]*(cvx.log(dc[h][c])-np.log(dparams[c][h][2]))  # Fisk concave (max p)                
            elif bestfitsd[h][c]==2:                
                probd[h][c]=-cvx.logistic(((-dc[h][c]+dparams[c][h][4])/dparams[c][h][5])) # logistic concave (max p)
            elif bestfitsd[h][c]==3:
                probd[h][c]=cvx.log_normcdf((cvx.log(dc[h][c])-dparams[c][h][6])/dparams[c][h][7])
            elif bestfitsd[h][c]==4:
                probd[h][c]=cvx.log_normcdf((dc[h][c]-dparams[c][h][8])/dparams[c][h][9])
            elif bestfitsd[h][c]==5:
                probd[h][c]=2*cvx.log(dc[h][c])-np.log(2*np.square(dparams[c][h][10])) # Rayleigh concave (max p)
            elif bestfitsd[h][c]==6:
                if dparams[c][h][13]>1:
                    probd[h][c]=dparams[c][h][13]*(cvx.log(dc[h][c])+np.log(dparams[c][h][12]))   # Weibull concave (max p) 
                else:
                    probd[h][c]=-cvx.exp(-cvx.power(dparams[c][h][12]*dc[h][c],dparams[c][h][13])) # Weibull concave (max p)
        
        if bestfitsi[h]!=10:
            if  bestfitsi[h]==0:               
                probi[h]=cvx.exp(iparams[h][0]*ic[h])  #expon convex (min p)            
            elif  bestfitsi[h]==2:
                probi[h]=((ic[h]-iparams[h][4])/iparams[h][5])+cvx.logistic(((-ic[h]+iparams[h][4])/iparams[h][5])) # logistic convex (min p)            
            elif  bestfitsi[h]==5:
                probi[h]=(cvx.square(ic[h]))/(2*np.square(iparams[h][10])) # Rayleigh convex (max 1-p) (min p)
            elif  bestfitsi[h]==6:
                if iparams[h][13]>=1:
                    probi[h]=cvx.power(iparams[h][12]*ic[h],iparams[h][13])                
        else:
            res += [probi[h]==0]
        
        hprobd[h]=cvx.sum(probd[h])
        pl[h]=cvx.sum(EqNp[h])
        ql[h]=cvx.sum(EqNq[h])
    "-------Objective definition--------"
    #obj = cvx.Minimize(cvx.real(cvx.sum(sref+sgen-sd-cvx.hstack(EqN)))+cvx.imag(cvx.sum(sref+sgen-sd-cvx.hstack(EqN))))
    #obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
    #obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))
    obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql)-cvx.sum(hprobd)+cvx.sum(probi))
    #obj = cvx.Minimize(cvx.abs(sref[0])+cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
    #obj = cvx.Minimize(cvx.abs(sref[0]))
    #obj = cvx.Minimize(1)
    "-------Problem/solver Setup--------"
    OPFSOC = cvx.Problem(obj,res)
    OPFSOC.solve(solver=cvx.MOSEK,mosek_params={'MSK_DPAR_MIO_TOL_REL_GAP':1e-6} ,verbose=True)
    #OPFSOC.solve(solver=cvx.MOSEK,save_file='probmosek1.ptf',verbose=True)
    #OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_DPAR_MIO_TOL_REL_GAP':1e-6,'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_FREE'},verbose=True)    
    #print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)
    
    pgo=pgref.value
    qgo=qgref.value
    qvno=qvn.value
    qiko=qik.value
    pko=pk.value
    qko=qk.value

    pvo=pv.value
    pvho=pvh.value
    ppvo=ppv.value

    zpvo=zpv.value

    ico=ic.value
    dco=dc.value      

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
    t[0]=OPFSOC.solver_stats.solve_time

    npwlout=np.zeros(H)
    npwlout[0]=npwl

    ppvout=np.zeros([H,num_nodes])
    zpvout=np.zeros([H,num_nodes])    


    for h in range(H):
        ppvout[h]=pvo
        zpvout[h]=zpvo

    probdco= np.zeros([H,ncluster])
    probico=  np.zeros(H)

    for h in range(H):        
        for i in range(ncluster):            
            distribution=adists[bestfitsd[h][i]]
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


    out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,npwlout,ico,probico)).T
    output=np.hstack((vo,pho,Equp,Equq,out1,dco,probdco,ppvout,zpvout))
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
    columns.append('prob_ic_cal')
    for i in range(ncluster):    
        columns.append('dc_c'+str(i+1))    
    for i in range(ncluster):    
        columns.append('prob_dc_cal'+str(i+1))
    for i in range(num_nodes):    
        columns.append('ppv'+str(i+1))
    for i in range(num_nodes):    
        columns.append('zpv'+str(i+1))
        
    df.columns=columns

    solvlist=[0]*H
    solvlist[0]='CM'


    df.insert(len(df.columns),'Solver',solvlist)
    df.to_excel("Results.xlsx")
    
    

