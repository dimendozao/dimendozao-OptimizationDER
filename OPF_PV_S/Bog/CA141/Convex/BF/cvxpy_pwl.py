# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:19:59 2024

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
problem='OPF_PV_S'


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
                bestfitsd[h][i]=j
            
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
    
npwl=3
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
#dc=np.ones(H)    

nppwl=np.size(xdempwl,axis=3)    
        
"----- Optimization model -----"

pgref = cvx.Variable((H,num_nodes),name='pgref')
qgref = cvx.Variable((H,num_nodes),name='qgref')

qvn= cvx.Variable((H,num_nodes),name='qvn')
qik= cvx.Variable((H,num_lines),name='qik')
pk= cvx.Variable((H,num_lines),name='pk')
qk= cvx.Variable((H,num_lines),name='qk')

pv= cvx.Variable(num_nodes,nonneg=True,name='pv')
ppv= cvx.Variable([H,num_nodes],nonneg=True,name='ppv')
pvh= cvx.Variable([H,num_nodes],nonneg=True,name='pvh')
zpv= cvx.Variable(num_nodes,boolean=True,name='zpv')


dc= cvx.Variable((H,ncluster),nonneg=True,name='dc')
#dc= cvx.Variable(H,nonneg=True,name='dc')
ic= cvx.Variable(H,nonneg=True,name='ic')

   
lambdas=[[0]*num_nodes for i in range(nihours)]
deltax=[0]*nihours
deltae=[0]*nihours
xis=[0]*nihours
etas=[0]*nihours

for h in range(nihours):
    for i in range(num_nodes):
        lambdas[h][i]=  cvx.Variable([npwl,npwl],nonneg=True)
    
    deltax[h]= cvx.Variable((num_nodes,npwl-1),boolean=True)
    deltae[h]= cvx.Variable((num_nodes,npwl-1),boolean=True)
    xis[h]=    cvx.Variable((num_nodes,npwl),nonneg=True)
    etas[h]=   cvx.Variable((num_nodes,npwl),nonneg=True)
    

probd=cvx.Variable([H,ncluster],nonneg=True)
zprobd=[0]*H
lambdad=[0]*H
lambdad1=[0]*H

for h in range(H):
    zprobd[h]=cvx.Variable([ncluster,nppwl],boolean=True)
    lambdad[h]=cvx.Variable([ncluster,nppwl],nonneg=True)
    lambdad1[h]=cvx.Variable([ncluster,nppwl],nonneg=True)


probi=cvx.Variable(H,nonneg=True)
zprobi=cvx.Variable([H,nppwl],boolean=True)
lambdai=cvx.Variable([H,nppwl],nonneg=True)
lambdai1=cvx.Variable([H,nppwl],nonneg=True)
    
pl=cvx.Variable(H,nonneg=True)
ql=cvx.Variable(H,nonneg=True)
    
"-------Constraint Construction-------- "

EqNp = [num_nodes*[0] for h in range(H)]
EqNq = [num_nodes*[0] for h in range(H)]


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
        
        "Piecewise linear constraints (pv*ic)"
        
        if bestfitsi[h]!=10:            
            res += [pv[i]==cvx.sum(cvx.multiply(pvxm,lambdas[h-7][i]))]            
            res += [ic[h]==cvx.sum(cvx.multiply(icxm,lambdas[h-7][i]))]
            res += [pvh[h][i]==cvx.sum(cvx.multiply(pvhx,lambdas[h-7][i]))]
            
        
            res += [cvx.sum(lambdas[h-7][i])==1]
            res += [xis[h-7][i]==cvx.sum(lambdas[h-7][i],axis=0)]
            res += [etas[h-7][i]==cvx.sum(lambdas[h-7][i],axis=1)]
            res += [cvx.sum(deltax[h-7][i])==1]
            res += [cvx.sum(deltae[h-7][i])==1]           
            for k in range(npwl):
                if k==0:
                    res += [xis[h-7][i][k]-deltax[h-7][i][k]<=0]
                    res += [etas[h-7][i][k]-deltae[h-7][i][k]<=0]
                elif k==npwl-1:
                    res += [xis[h-7][i][k]-deltax[h-7][i][k-1]<=0]
                    res += [etas[h-7][i][k]-deltae[h-7][i][k-1]<=0]
                else:
                    res += [xis[h-7][i][k]-deltax[h-7][i][k]-deltax[h-7][i][k-1]<=0]
                    res += [etas[h-7][i][k]-deltae[h-7][i][k]-deltae[h-7][i][k-1]<=0]        
    
    "Piecewise linear constraints (prob)"
    for i in range(ncluster):
        res +=[probd[h][i]==cvx.sum(cvx.multiply(lambdad[h][i],ydempwl[h][i][bestfitsd[h][i]]))]
        res +=[dc[h][i]==cvx.sum(cvx.multiply(lambdad[h][i],xdempwl[h][i][bestfitsd[h][i]]))]
        res +=[cvx.sum(zprobd[h][i])==1]
        for k in range(nppwl):
            res +=  [lambdad1[h][i][k]<=1]
            res +=  [lambdad[h][i][k]<=zprobd[h][i][k]]
            res +=  [lambdad[h][i][k]<=lambdad1[h][i][k]]
            res +=  [lambdad[h][i][k]>=lambdad1[h][i][k]-1+zprobd[h][i][k]]        
    
    if bestfitsi[h]!=10:
        res +=[probi[h]==cvx.sum(cvx.multiply(lambdai[h],yirrpwl[h][bestfitsi[h]]))]
        res +=[ic[h]==cvx.sum(cvx.multiply(lambdai[h],xirrpwl[h][bestfitsi[h]]))]
        res +=[cvx.sum(zprobi[h])==1]
        for k in range(nppwl):
            res +=  [lambdai1[h][k]<=1]
            res +=  [lambdai[h][k]<=zprobi[h][k]]
            res +=  [lambdai[h][k]<=lambdai1[h][k]]
            res +=  [lambdai[h][k]>=lambdai1[h][k]-1+zprobi[h][k]]   
    else:
        res +=[probi[h]==0]
        res +=[ic[h]==0]
        res +=[zprobi[h]==0]
        res +=[lambdai[h]==0]
        res +=[lambdai1[h]==0]
    
for h in range(H):       
    res += [pl[h]==cvx.sum(EqNp[h])]
    res += [ql[h]==cvx.sum(EqNq[h])]
    
"-------Objective definition--------"
#obj = cvx.Minimize(cvx.real(cvx.sum(sref+sgen-sd-cvx.hstack(EqN)))+cvx.imag(cvx.sum(sref+sgen-sd-cvx.hstack(EqN))))
#obj = cvx.Minimize(cvx.sum(pgref)+cvx.sum(qgref))
obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql)+cvx.sum(probi)-cvx.sum(probd))
#obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql)+cvx.sum(hprobd)+cvx.sum(probi))
#obj = cvx.Minimize(cvx.abs(sref[0])+cvx.real(cvx.sum(EqN))+cvx.imag(cvx.sum(EqN)))
#obj = cvx.Minimize(cvx.abs(sref[0]))
#obj = cvx.Minimize(1)
"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,save_file='probmosek.ptf',verbose=True)
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
npwlout[0]=npwl

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