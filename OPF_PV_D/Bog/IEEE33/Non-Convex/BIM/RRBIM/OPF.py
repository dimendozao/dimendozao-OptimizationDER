# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:16:37 2024

@author: diego
"""
from amplpy import AMPL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat


case='IEEE33'
city='Bog'
city1='BOG'
problem='OPF_PV'


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
yrl=np.zeros(num_lines)
yil=np.zeros(num_lines)
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]
    to[k]=branch['j'][k]
    yrl[k]=np.real(1/(branch['r'][k] + 1j*branch['x'][k]))
    yil[k]=np.imag(1/(branch['r'][k] + 1j*branch['x'][k]))

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


ciimax=np.zeros(num_nodes)
ciimin=np.zeros(num_nodes)
cnlmax=np.zeros(num_lines)
cnlmin=np.zeros(num_lines)
snlmax=np.zeros(num_lines)
snlmin=np.zeros(num_lines)


for i in range(num_nodes):
    ciimax[i]=vmax[i]*vmax[i]
    ciimin[i]=vmin[i]*vmin[i]

for k in range(num_lines):
    cnlmax[k]=vmax[fr[k]-1]*vmax[to[k]-1]
    cnlmin[k]=vmin[fr[k]-1]*vmin[to[k]-1]
    snlmax[k]=vmax[fr[k]-1]*vmax[to[k]-1]
    snlmin[k]=-vmax[fr[k]-1]*vmax[to[k]-1]

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))

pveff=0.8
            
"----- Optimization model -----"

ampl = AMPL()

ampl.eval(
    r"""
    param nn;
    param nl;
    param nh;     
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nl").set(num_lines)
ampl.get_parameter("nh").set(H)

ampl.eval(
    r"""
    set N=1..nn;
    set L=1..nl;
    set H=1..nh;              
"""
)

ampl.eval(
    r"""   
    param fr {L};
    param to {L};
    param yr {L};
    param yi {L};
    param pd {H,N};
    param qd {H,N};
    param ciimax{N};
    param ciimin{N};
    param cnlmax{L};
    param cnlmin{L};
    param snlmax{L};
    param snlmin{L};
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};
    param imeans{H};    
    param npv;
    param pvcmax;
    param pveff;         
"""
)
ampl.get_parameter("fr").set_values(fr)
ampl.get_parameter("to").set_values(to)
ampl.get_parameter("yr").set_values(yrl)
ampl.get_parameter("yi").set_values(yil)
ampl.get_parameter("pd").set_values(pdh)
ampl.get_parameter("qd").set_values(qdh)
ampl.get_parameter("ciimax").set_values(ciimax)
ampl.get_parameter("ciimin").set_values(ciimin)
ampl.get_parameter("cnlmax").set_values(cnlmax)
ampl.get_parameter("cnlmin").set_values(cnlmin)
ampl.get_parameter("snlmax").set_values(snlmax)
ampl.get_parameter("snlmin").set_values(snlmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
ampl.get_parameter("imeans").set_values(imeans)  
ampl.get_parameter("npv").set(npv)  
ampl.get_parameter("pvcmax").set(pvcmax)
ampl.get_parameter("pveff").set(pveff)
               

ampl.eval(
    r"""
    var  cii{h in H,i in N} >= ciimin[i], <= ciimax[i];
    var  cnl{h in H,k in L} >= cnlmin[k], <= cnlmax[k];
    var  snl{h in H,k in L} >= snlmin[k], <= snlmax[k];
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    var  pv{i in N} >=0, <=pvcmax;
    var  ppv{i in N} >=0, <=pvcmax;
    var  zpv{i in N} binary;
    
    minimize Losses:
          sum{h in H,i in N,k in L: fr[k]=i} cii[h,i]*yr[k]
        - sum{h in H,i in N,k in L: fr[k]=i} cnl[h,k]*yr[k]
        - sum{h in H,i in N,k in L: fr[k]=i} snl[h,k]*yi[k]
        + sum{h in H,i in N,k in L: to[k]=i} cii[h,i]*yr[k]
        - sum{h in H,i in N,k in L: to[k]=i} cnl[h,k]*yr[k]
        + sum{h in H,i in N,k in L: to[k]=i} snl[h,k]*yi[k]
        - sum{h in H,i in N,k in L: fr[k]=i} cii[h,i]*yi[k]
        + sum{h in H,i in N,k in L: fr[k]=i} cnl[h,k]*yi[k]
        - sum{h in H,i in N,k in L: fr[k]=i} snl[h,k]*yr[k]
        - sum{h in H,i in N,k in L: to[k]=i} cii[h,i]*yi[k]
        + sum{h in H,i in N,k in L: to[k]=i} cnl[h,k]*yi[k]
        + sum{h in H,i in N,k in L: to[k]=i} snl[h,k]*yr[k];
    
    subject to PB {h in H,i in N}: 
        pg[h,i]+(ppv[i]*imeans[h]*pveff)-pd[h,i] = sum {k in L: fr[k]=i} cii[h,i]*yr[k]
        - sum{k in L: fr[k]=i} cnl[h,k]*yr[k]
        - sum{k in L: fr[k]=i} snl[h,k]*yi[k]
        + sum{k in L: to[k]=i} cii[h,i]*yr[k]
        - sum{k in L: to[k]=i} cnl[h,k]*yr[k]
        + sum{k in L: to[k]=i} snl[h,k]*yi[k];
    subject to QB {h in H,i in N}: 
        qg[h,i]-qd[h,i] = -sum {k in L: fr[k]=i} cii[h,i]*yi[k]
        + sum{k in L: fr[k]=i} cnl[h,k]*yi[k]
        - sum{k in L: fr[k]=i} snl[h,k]*yr[k]
        - sum{k in L: to[k]=i} cii[h,i]*yi[k]
        + sum{k in L: to[k]=i} cnl[h,k]*yi[k]
        + sum{k in L: to[k]=i} snl[h,k]*yr[k];
    subject to SOC {h in H,k in L}:
        cii[h,fr[k]]*cii[h,to[k]]=(cnl[h,k]*cnl[h,k])+(snl[h,k]*snl[h,k]); 
    
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

ampl.option["solver"] = "bonmin"
ampl.solve()

"-------------Print solution------------"

pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
cio= np.zeros([H,num_nodes])
clo= np.zeros([H,num_lines])
slo= np.zeros([H,num_lines])
pvo= np.zeros(num_nodes)
ppvo= np.zeros(num_nodes)
zpvo= np.zeros(num_nodes)

for h in range(H):
    for i in range (num_nodes):
        cio[h][i]=ampl.get_variable('cii')[h+1,i+1].value()
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()

for i in range(num_nodes):
    pvo[i]=ampl.get_variable('pv')[i+1].value()
    ppvo[i]=ampl.get_variable('ppv')[i+1].value()
    zpvo[i]=ampl.get_variable('zpv')[i+1].value()

for h in range(H):
    for k in range(num_lines):
        clo[h][k]=ampl.get_variable('cnl')[h+1,k+1].value()
        slo[h][k]=ampl.get_variable('snl')[h+1,k+1].value()


vo=np.sqrt(cio)
plt.plot(vo)


Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for i in range(num_nodes):
        for k in range(num_lines):
            if i==fr[k]-1:
                Equp[h][i]+=(cio[h][i]*yrl[k])-(clo[h][k]*yrl[k])-(slo[h][k]*yil[k])
                Equq[h][i]+=(-cio[h][i]*yil[k])+(clo[h][k]*yil[k])-(slo[h][k]*yrl[k])
            if i==to[k]-1:
                Equp[h][i]+=(cio[h][i]*yrl[k])-(clo[h][k]*yrl[k])+(slo[h][k]*yil[k])
                Equq[h][i]+=(-cio[h][i]*yil[k])+(clo[h][k]*yil[k])+(slo[h][k]*yrl[k])

ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])

pho=np.zeros([H,num_nodes])

for h in range(H):
    for k in range(num_lines):
        pho[h][to[k]-1]=pho[h][fr[k]-1]-np.angle(clo[h][k]+1j*slo[h][k])

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