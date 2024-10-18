# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:18:03 2024

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
pgref = cvx.Variable((H,num_nodes),nonneg=True,name='pgref')
qgref = cvx.Variable((H,num_nodes),nonneg=True,name='qgref')

qvn= cvx.Variable((H,num_nodes),nonneg=True,name='qvn')
qik= cvx.Variable((H,num_lines),name='qik')
pk= cvx.Variable((H,num_lines),name='pk')
qk= cvx.Variable((H,num_lines),name='qk')


pbc= cvx.Variable(nonneg=True,name='pbc')
pb=  cvx.Variable(H,name='pb')
ppb= cvx.Variable((H,num_nodes),name='ppb')
zb= cvx.Variable((H,num_nodes),boolean=True,name='zb')
cap= cvx.Variable(H,name='cap')
cap0= cvx.Variable(name='cap0')

pl= cvx.Variable(H,nonneg=True,name='pl')
ql= cvx.Variable(H,nonneg=True,name='ql')

"-------Constraint Construction-------- "

EqNp = [num_nodes*[0] for h in range(H)]
EqNq = [num_nodes*[0] for h in range(H)]

res=[]

"Variable Bounds"
res += [qvn<=qvnmax]
res += [qvn>=qvnmin]
res += [pgref<=prefmax]
res += [pgref>=prefmin]
res += [qgref<=qrefmax]
res += [qgref>=qrefmin]
res += [cap[0]==cap0-pb[0]]
res += [cap[H-1]==cap0]
res += [cap0<=pbc]

for h in range(H):
    res += [cvx.sum(zb[h])==1]    
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
    
for h in range(H):        
    for i in range(num_nodes): 
        "Power Flow constraints"
        res +=  [pgref[h][i]+(ppv[i]*pveff*imeans[h])+ppb[h][i]+pgen[i]-(pdm[i]*dmeans[h][cnode[i]])==EqNp[h][i]]
        res +=  [qgref[h][i]+qgen[i]-(qdm[i]*dmeans[h][cnode[i]])==EqNq[h][i]]    
            
        res += [ppb[h][i]<=zb[h][i]*pbcmax]
        res += [ppb[h][i]>=zb[h][i]*-pbcmax]
        res += [ppb[h][i]<=pb[h]+pbcmax*(1-zb[h][i])]
        res += [ppb[h][i]>=pb[h]-pbcmax*(1-zb[h][i])]
    res += [pb[h]>=-pbc]
    res += [pb[h]<=pbc]
    res += [cap[h]<=pbc]
    if h<H-1:
        res += [cap[h+1]==cap[h]-pb[h+1]]
    
    res += [pl[h]==cvx.sum(EqNp[h])]
    res += [ql[h]==cvx.sum(EqNq[h])]
    
"-------Objective definition--------"

obj = cvx.Minimize(cvx.sum(pl)+cvx.sum(ql))

"-------Problem/solver Setup--------"
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.MOSEK,mosek_params={'MSK_DPAR_MIO_TOL_REL_GAP':1e-6} ,verbose=True)
#OPFSOC.solve(solver=cvx.MOSEK,mosek_params = {'MSK_DPAR_MIO_TOL_REL_GAP':1e-6,'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL','MSK_IPAR_PRESOLVE_USE':'MSK_PRESOLVE_MODE_FREE'},verbose=True)


"-------------Print solution------------"
pgo=pgref.value
qgo=qgref.value
qvno=qvn.value
qiko=qik.value
pko=pk.value
qko=qk.value
pbco=pbc.value
pbo=pb.value
ppbo=ppb.value
zbo=zb.value
cap0o=cap0.value
capo=cap.value

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

pbcout=np.zeros(H)
pbcout[0]=pbco

ppvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=ppv
   
cap0out=np.zeros(H)
cap0out[0]=cap0o

out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,imeans,pbcout,pbo,cap0out,capo)).T
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
solvlist[0]='CM'


df.insert(len(df.columns),'Solver',solvlist)
#df.to_excel("Results.xlsx")