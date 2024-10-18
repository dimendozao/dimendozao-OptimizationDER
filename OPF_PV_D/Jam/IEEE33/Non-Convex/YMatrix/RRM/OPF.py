# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:01:30 2024

@author: diego
"""

from amplpy import AMPL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

case='IEEE33'
city='Jam'
city1='JAM'
problem='OPF_PV_D'

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

idx1=(np.abs(ym)!=0)*1
idx2=idx1-np.eye(num_nodes)

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx1[i][j]:
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

ampl = AMPL()


ampl.eval(
    r"""
    param nn;
    param nh;       
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nh").set(H)

ampl.eval(
    r"""
    set N :=1..nn;
    set H=1..nh;
    param yr {N,N};
    param yi {N,N};
    param pd {H,N};
    param qd {H,N};
    param cnmax{N,N};
    param cnmin{N,N};
    param snmax{N,N};
    param snmin{N,N};
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N}; 
    param imeans{H};    
    param npv;
    param pvcmax;
    param pveff;
    param idx1{N,N};
    param idx2{N,N};
    
"""
)

ampl.get_parameter("yr").set_values(ymr)
ampl.get_parameter("yi").set_values(ymi)
ampl.get_parameter("pd").set_values(pdh)
ampl.get_parameter("qd").set_values(qdh)
ampl.get_parameter("cnmax").set_values(cnmax)
ampl.get_parameter("cnmin").set_values(cnmin)
ampl.get_parameter("snmax").set_values(snmax)
ampl.get_parameter("snmin").set_values(snmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin) 
ampl.get_parameter("imeans").set_values(imeans)  
ampl.get_parameter("npv").set(npv)  
ampl.get_parameter("pvcmax").set(pvcmax)
ampl.get_parameter("idx1").set_values(idx1)
ampl.get_parameter("idx2").set_values(idx2)
ampl.get_parameter("pveff").set(pveff)  



"-------Constraint Construction-------- "

ampl.eval(
    r"""
    var  cn{h in H,i in N, j in N} >= cnmin[i,j], <= cnmax[i,j];
    var  sn{h in H,i in N, j in N} >= snmin[i,j], <= snmax[i,j];
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    var  pv{i in N} >=0, <=pvcmax;
    var  ppv{i in N} >=0, <=pvcmax;
    var  zpv{i in N} binary;
    
    minimize Losses:
         sum{h in H,i in N, j in N: idx1[i,j]=1} cn[h,i,j]*yr[i,j]
       + sum{h in H,i in N, j in N: idx1[i,j]=1} sn[h,i,j]*yi[i,j] 
       + sum{h in H,i in N, j in N: idx1[i,j]=1} sn[h,i,j]*yr[i,j]
       - sum{h in H,i in N, j in N: idx1[i,j]=1} cn[h,i,j]*yi[i,j];
    
    subject to PB {h in H,i in N}: 
        pg[h,i]+(ppv[i]*imeans[h]*pveff)-pd[h,i] = sum {j in N: idx1[i,j]=1} cn[h,i,j]*yr[i,j] + sum {j in N: idx1[i,j]=1} sn[h,i,j]*yi[i,j];  
        
    subject to QB {h in H,i in N}: 
        qg[h,i]-qd[h,i] = sum {j in N: idx1[i,j]=1} sn[h,i,j]*yr[i,j] - sum {j in N: idx1[i,j]=1} cn[h,i,j]*yi[i,j];    
    
    subject to SOC {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        (cn[h,i,j]*cn[h,i,j])+(sn[h,i,j]*sn[h,i,j])=cn[h,i,i]*cn[h,j,j];
        
    subject to Hermit1 {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        cn[h,i,j]=cn[h,j,i];
        
    subject to Hermit2 {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        sn[h,i,j]=-sn[h,j,i];
    
    subject to PVLIN1 {i in N}:
        ppv[i]<=pvcmax*zpv[i];
    
    subject to PVLIN2 {i in N}:
        ppv[i]<=pv[i];
    
    subject to PVLIN3 {i in N}:
        ppv[i]>=pv[i]-pvcmax*(1-zpv[i]);
    
    subject to MAXPV:
        sum{i in N} ppv[i]=pvcmax;
    
    subject to MAXZPV:
        sum{i in N} zpv[i]=npv;
        
"""
)

"-------Problem/solver Setup--------"

#ampl.solve(solver="ipopt", ipopt_options='mumps_mem_percent=64000')
#ampl.solve(solver="scip", scip_options='maxnthreads=12 minnthreads=6 outlev=1')
ampl.solve(solver="bonmin")

pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
cno= np.zeros([H,num_nodes,num_nodes])
sno= np.zeros([H,num_nodes,num_nodes])
pvo= np.zeros(num_nodes)
ppvo= np.zeros(num_nodes)
zpvo= np.zeros(num_nodes)

for h in range(H):
    for i in range (num_nodes):
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()           
        for j in range(num_nodes):
            if idx1[i][j]==1:
                cno[h][i][j]=ampl.get_variable('cn')[h+1,i+1,j+1].value()
                sno[h][i][j]=ampl.get_variable('sn')[h+1,i+1,j+1].value()

for i in range(num_nodes):
    pvo[i]=ampl.get_variable('pv')[i+1].value() 
    ppvo[i]=ampl.get_variable('ppv')[i+1].value()
    zpvo[i]=ampl.get_variable('zpv')[i+1].value()        

vo=np.zeros([H,num_nodes])
for h in range(H):        
    vo[h]=np.sqrt(np.diag(cno[h]))
plt.plot(vo)


Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            Equp[h][i]+=(cno[h][i][j]*ymr[i][j])+(sno[h][i][j]*ymi[i][j])
            Equq[h][i]+=(sno[h][i][j]*ymr[i][j])-(cno[h][i][j]*ymi[i][j])
    

ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])

pho=np.zeros([H,num_nodes])

for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j>i and idx2[i][j]:
                pho[h][j]=pho[h][i]-np.arctan(sno[h][i][j]/cno[h][i][j])

t=np.zeros(H)

t[0]=ampl.getValue('_solve_elapsed_time')
    
   
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
solvlist[0]='AB'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")



