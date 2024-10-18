# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 07:49:52 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import cvxpy as cvx


case='IEEE33'
city='Bog'
city1='BOG'
problem='OPF_PV_S'

"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')


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

k1=np.zeros(H)
k2=np.zeros(H)
k3=np.zeros(H)
k11=np.zeros(H)
k21=np.zeros(H)
k31=np.zeros(H)
for h in range(H):
    if bestfitsi1[h]==1:
        if iparams[h][3]<1:
            k1[h]=1
    if bestfitsi2[h]==1:
        if iparams[h][3]<1:
            k2[h]=1
    if bestfitsi3[h]==1:
        if iparams[h][3]<1:
            k3[h]=1
    if bestfitsi1[h]==6:
        if iparams[h][12]<1:
            k11[h]=1
    if bestfitsi2[h]==6:
        if iparams[h][12]<1:
            k21[h]=1
    if bestfitsi3[h]==6:
        if iparams[h][12]<1:
            k31[h]=1

                

for h in range(H):
    for i in range(ncluster):
        bestfitsd[h][i]=bestfitsd1[h][i]
           
    if bestfitsi1[h]!=10:
        if bestfitsi1[h]==1 and k1[h]==0:
            bestfitsi[h]=1
        elif bestfitsi1[h]==6 and k11[h]==0:
            bestfitsi[h]=6
        elif bestfitsi1[h]!=3 and bestfitsi1[h]!=4:
            bestfitsi[h]=bestfitsi1[h]
        elif bestfitsi2[h]==1 and k2[h]==0:
            bestfitsi[h]=1
        elif bestfitsi2[h]==6 and k21[h]==0:
            bestfitsi[h]=6
        elif bestfitsi2[h]!=3 and bestfitsi2[h]!=4:
            bestfitsi[h]=bestfitsi2[h]
        elif bestfitsi3[h]==1 and k3[h]==0:
            bestfitsi[h]=1
        elif bestfitsi3[h]==6 and k31[h]==0:
            bestfitsi[h]=6
        elif bestfitsi3[h]!=3 and bestfitsi3[h]!=4:
            bestfitsi[h]=bestfitsi3[h]
        
            
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
            
    vmax=vmax+0.2
    vmin=vmin-0.2
    
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
        
    npar=5
    icxg=np.linspace(0,2,num=npar+1)
    pvxg=np.linspace(0,pvcmax,num=npar+1) 
    
    iclp=icxg[:npar]
    pvlp=pvxg[:npar]
    icup=icxg[1:]
    pvup=pvxg[1:]
    
    pvhup=np.multiply(pvup,icup)
    pvhlp=np.multiply(pvlp,iclp)
    
    "----- Optimization model -----"
    
    pgref = cvx.Variable((H,num_nodes),name='pgref')
    qgref = cvx.Variable((H,num_nodes),name='qgref')

    qvn= cvx.Variable((H,num_nodes),name='qvn')
    qik= cvx.Variable((H,num_lines),nonneg=True,name='qik')
    pk= cvx.Variable((H,num_lines),nonneg=True,name='pk')
    qk= cvx.Variable((H,num_lines),nonneg=True,name='qk')
    
    pv= cvx.Variable(num_nodes,nonneg=True,name='pv')
    ppv= cvx.Variable([H,num_nodes],nonneg=True,name='ppv')
    pvh= cvx.Variable([H,num_nodes],nonneg=True,name='pvh')
    zpv= cvx.Variable(num_nodes,boolean=True,name='zpv')
    

    dc= cvx.Variable((H,ncluster),nonneg=True,name='dc')    
    ic= cvx.Variable(H,nonneg=True,name='ic')

    pl=cvx.Variable(H,nonneg=True)
    ql=cvx.Variable(H,nonneg=True)
    
    pvp= cvx.Variable((num_nodes,npar),nonneg=True,name='pvp')
    pv1= cvx.Variable((num_nodes,npar),nonneg=True,name='pv1')
    
    icp= cvx.Variable((nihours,npar),nonneg=True,name='icp')
    ic1= cvx.Variable((nihours,npar),nonneg=True,name='ic1')
    
    pvhp=[0]*nihours
    pvh1=[0]*nihours
    zpwl=[0]*nihours
    
    for h in range(nihours):
        pvhp[h]=cvx.Variable((num_nodes,npar),nonneg=True,name='pvhp')
        pvh1[h]=cvx.Variable((num_nodes,npar),nonneg=True,name='pvh1')
        zpwl[h]=cvx.Variable((num_nodes,npar),boolean=True,name='zpwl')
        
        
    "-------Constraint Construction-------- "

    EqNp = [num_nodes*[0] for h in range(H)]
    EqNq = [num_nodes*[0] for h in range(H)]
    
    probd=  [[0]*ncluster for h in range(H)]
    hprobd= [0]*H    
    probi=  [0]*nihours
    
    
    
    res=[]

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
            
    
    
    
    "Variable Bounds"
    res += [qvn<=qvnmax]
    res += [qvn>=qvnmin]        
    res += [pgref<=prefmax]
    res += [pgref>=prefmin]
    res += [qgref<=qrefmax]
    res += [qgref>=qrefmin]
    res += [cvx.sum(zpv)==npv]
    res += [cvx.sum(pv)==pvcmax]
    
    for h in range(H):        
        for i in range(num_nodes): 
            "Power Flow constraints"
            if i>0:
                res +=  [pgref[h][i]+(ppv[h][i]*pveff)+pgen[i]-(pdm[i]*dc[h][cnode[i][0]-1])==EqNp[h][i]]
                res +=  [qgref[h][i]+qgen[i]-(qdm[i]*dc[h][cnode[i][0]-1])==EqNq[h][i]]
                # res +=  [pgref[h][i]+(ppv[h][i]*pveff)+pgen[i]-(pdm[i]*dc[h])==EqNp[h][i]]
                # res +=  [qgref[h][i]+qgen[i]-(qdm[i]*dc[h])==EqNq[h][i]]
            else:
                res +=  [pgref[h][i]+(ppv[h][i]*pveff)+pgen[i]-pdm[i]==EqNp[h][i]]
                res +=  [qgref[h][i]+qgen[i]-qdm[i]==EqNq[h][i]]
            "Pv integer linearization"
            res +=  [ppv[h][i]<=pvcmax*zpv[i]]
            res +=  [ppv[h][i]<=pvh[h][i]]
            res +=  [ppv[h][i]>=pvh[h][i]-(pvcmax*(1-zpv[i]))]      
                
        "Probabilities calculation"
        for i in range(ncluster):
            res += [dc[h][i]<=2]
            distribution=adists[bestfitsd[h][i]]
            "Exponential"
            if distribution==adists[0]:                    
                a=-dparams[i][h][0]*dc[h][i]
                probd[h][i]=cvx.exp(a)  #expon convex (min 1-p)            
            "Fisk  (Beta is always positive)"
            if distribution==adists[1]:
                a=dc[h][i]/dparams[i][h][2]
                if dparams[i][h][3]>=1:
                    probd[h][i]=cvx.power(a,-dparams[i][h][3]) #fisk convex min(1-p)
                else:
                    probd[h][i]=cvx.power(a,-dparams[i][h][3]) #fisk convex min(-p) --> max (p)
            "Logistic"                        
            if distribution==adists[2]:
                a=(-dc[h][i]+dparams[i][h][4])/dparams[i][h][5]
                probd[h][i]=cvx.logistic(a) # logistic convex (min 1-p)
            "Lognormal"
            if distribution==adists[3]:
               probd[h][i]=-cvx.log_normcdf((cvx.log(dc[h][i])-dparams[i][h][6])/dparams[i][h][7]) #log norm convex(min 1-p)
            "Normal"
            if distribution==adists[4]:
               probd[h][i]=-cvx.log_normcdf((dc[h][i]-dparams[i][h][8])/dparams[i][h][9]) #norm convex(min 1-p)
            "Rayleigh"
            if distribution==adists[5]:
                probd[h][i]=cvx.exp(-cvx.square(dc[h][i])/(2*np.square(dparams[i][h][10]))) #Quasilinear (min 1-p)
            "Weibull"
            if distribution==adists[6]:
                if dparams[i][h][13]>1:
                    probd[h][i]=cvx.power(dc[h][i]*dparams[i][h][12],dparams[i][h][13]) # min (1-p)
                else:
                    probd[h][i]=-dparams[i][h][13]*(cvx.log(dparams[i][h][12])+np.log(dc[h][i])) # convex min(-p)-->max p 
                    
        
        if bestfitsi[h]!=10:
            distribution=adists[bestfitsi[h]]
            "Exponential"
            if distribution==adists[0]:               
                probi[h-7]=cvx.exp(iparams[h][0]*ic[h])-1 # exponential convex (min p)
            "Fisk  (Beta is always positive)"
            if distribution==adists[1]:
                a=ic[h]/iparams[h][2]
                if iparams[h][3]>=1:
                    probi[h-7]=cvx.power(a,iparams[h][3]) #fisk convex min(p)                               
            "Logistic"                     
            if  distribution==adists[2]:
                a=(-ic[h]+iparams[h][4])/iparams[h][5]
                probi[h-7]=-a+cvx.logistic(a)   # Logistic concave max (1-p)-->convex min(p-1)--->min p
            "Rayleigh"
            if  distribution==adists[5]:
                probi[h-7]=(cvx.square(ic[h]))/(2*iparams[h][10]*iparams[h][10]) # Rayleigh concave max (1-p)-->convex min (p-1)
            "Weibull"
            if  distribution==adists[6]:                
                if iparams[h][13]>1:
                    probi[h-7]=cvx.exp(cvx.power(ic[h]*iparams[h][12],iparams[h][13])) #convex min p
                    
    "Piecewise linear constraints"       
    for h in range(H):               
        if bestfitsi[h]!=10:
            for i in range(num_nodes):
                for k in range (npar):                    
                    res += [pvhp[h-7][i][k]>=icp[h-7][k]*pvlp[k]+pvp[i][k]*iclp[k]-pvlp[k]*iclp[k]]
                    res += [pvhp[h-7][i][k]>=icp[h-7][k]*pvup[k]+pvp[i][k]*icup[k]-pvup[k]*icup[k]]
                    res += [pvhp[h-7][i][k]<=icp[h-7][k]*pvup[k]+pvp[i][k]*iclp[k]-pvup[k]*iclp[k]]
                    res += [pvhp[h-7][i][k]<=icp[h-7][k]*pvlp[k]+pvp[i][k]*icup[k]-pvlp[k]*icup[k]]
                    
                    
                    res += [pv1[i][k]<=zpwl[h-7][i][k]*pvup[k]]
                    res += [pv1[i][k]<=pvp[i][k]]
                    res += [pv1[i][k]>=pvp[i][k]-pvup[k]*(1-zpwl[h-7][i][k])]
                    
                    res += [ic1[h-7][k]<=zpwl[h-7][i][k]*icup[k]]
                    res += [ic1[h-7][k]<=icp[h-7][k]]
                    res += [ic1[h-7][k]>=icp[h-7][k]-icup[k]*(1-zpwl[h-7][i][k])]
                    
                    res += [pvh1[h-7][i][k]<=zpwl[h-7][i][k]*pvhup[k]]
                    res += [pvh1[h-7][i][k]<=pvhp[h-7][i][k]]
                    res += [pvh1[h-7][i][k]>=pvhp[h-7][i][k]-pvhup[k]*(1-zpwl[h-7][i][k])]
                
                res += [cvx.sum(pvh1[h-7][i])==pvh[h][i]]
                res += [cvx.sum(zpwl[h-7][i])==1]
        
            res += [cvx.sum(ic1[h-7])==ic[h]]
        else:
            res += [ic[h]==0]
            res += [pvh[h]==0]
        
        hprobd[h]=cvx.sum(probd[h])
        res += [pl[h]==cvx.sum(EqNp[h])]
        res += [ql[h]==cvx.sum(EqNq[h])]
        
               
    
    res += [cvx.sum(pv1,axis=1)==pv]            
                
        
        
    "-------Objective definition--------"
    #obj = cvx.Minimize(cvx.real(cvx.sum(sref+sgen-sd-cvx.hstack(EqN)))+cvx.imag(cvx.sum(sref+sgen-sd-cvx.hstack(EqN))))
    #obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
    obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql)+cvx.sum(probi)+cvx.sum(hprobd))
    #obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql)+cvx.sum(hprobd)+cvx.sum(probi))
    #obj = cvx.Minimize(cvx.abs(sref[0])+cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
    #obj = cvx.Minimize(cvx.abs(sref[0]))
    #obj = cvx.Minimize(1)
    "-------Problem/solver Setup--------"
    OPFSOC = cvx.Problem(obj,res)
    OPFSOC.solve(qcp=True,solver=cvx.MOSEK,save_file='probmosek.ptf',verbose=True)
    #OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_FREE'},verbose=True)    
    #print(OPFSOC.status,obj.value,OPFSOC.solver_stats.solve_time)
    
    #mosek_params={}
    #'MSK_IPAR_MIO_CONIC_OUTER_APPROXIMATION':'MSK_ON'
    #'MSK_IPAR_MIO_NODE_OPTIMIZER':'MSK_OPTIMIZER_DUAL_SIMPLEX'
    #'MSK_IPAR_MIO_ROOT_OPTIMIZER':'MSK_OPTIMIZER_CONIC'
    #'MSK_IPAR_MIO_HEURISTIC_LEVEL':-1
    #'MSK_IPAR_PRESOLVE_USE', 'MSK_PRESOLVE_MODE_OFF'
    #'MSK_IPAR_MIO_QCQO_REFORMULATION_METHOD','MSK_MIO_QCQO_REFORMULATION_METHOD_LINEARIZATION'
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

    plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

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
    npwlout[0]=npar

    ppvout=np.zeros([H,num_nodes])
    zpvout=np.zeros([H,num_nodes])

    for h in range(H):
        ppvout[h]=ppvo
        zpvout[h]=zpvo
    
    
    probdo= np.zeros([H,ncluster])
    probio=  np.zeros(H)
    
    for h in range(H):        
        for i in range(ncluster):            
            distribution=adists[bestfitsd[h][i]]
            "Exponential"
            if distribution==adists[0]:                    
                probdo[h][i]=1-np.exp(-dparams[i][h][0]*dco[h][i])            
            "Fisk  (Beta is always positive)"
            if distribution==adists[1]:
                probdo[h][i]=1/(1+np.power(dco[h][i]/dparams[i][h][2],-dparams[i][h][3]))                          
            "Logistic"                        
            if distribution==adists[2]:                
                probdo[h][i]=1/(1+np.exp((-dco[h][i]+dparams[i][h][4])/dparams[i][h][5])) 
            "Rayleigh"
            if distribution==adists[5]:                
                probdo[h][i]=1-np.exp(-np.square(dco[h][i])/(2*np.square(dparams[i][h][10])))
            "Weibull"
            if distribution==adists[6]:                
                probdo[h][i]=1-np.exp(-np.power(dco[h][i]*dparams[i][h][12],dparams[i][h][13]))
                
        if bestfitsi[h]!=10:
            distribution=adists[bestfitsi[h]]
            "Exponential"
            if distribution==adists[0]:               
                probio[h]=1-np.exp(-iparams[h][0]*ico[h])            
            "Fisk  (Beta is always positive)"
            if distribution==adists[1]:
                probio[h]=1/(1+np.power(ico[h]/iparams[h][2],-iparams[h][3]))                          
            "Logistic"                        
            if distribution==adists[2]:                
                probio[h]=1/(1+np.exp((-ico[h]+iparams[h][4])/iparams[h][5])) 
            "Rayleigh"
            if distribution==adists[5]:                
                probio[h]=1-np.exp(-np.square(ico[h])/(2*np.square(iparams[h][10])))
            "Weibull"
            if distribution==adists[6]:                
                probio[h]=1-np.exp(-np.power(ico[h]*iparams[h][12],iparams[h][13])) 
    
    out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,npwlout,ico,probio)).T
    output=np.hstack((vo,pho,Equp,Equq,out1,dco,probdo,ppvout,zpvout))
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
    columns.append('npwl_pv')
    columns.append('ic')    
    columns.append('prob_ic_cal')
    for i in range(ncluster):    
        columns.append('dc'+str(i+1))    
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