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

rad=np.mean(imeans)
dem=np.mean(dmeans,axis=0)

num_lines = len(branch)
num_nodes=len(bus)
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
    
idem=np.zeros(num_nodes)

for i in range(num_nodes):
    idem[i]=dem[cnode[i]]
    
pdm=np.real(sd)
qdm=np.imag(sd)

pdem=np.multiply(pdm,idem)
qdem=np.multiply(qdm,idem)

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
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nl").set(num_lines)

ampl.eval(
    r"""
    set N=1..nn;
    set L=1..nl;            
"""
)

ampl.eval(
    r"""   
    param fr {L};
    param to {L};
    param yr {L};
    param yi {L};
    param pd {N};
    param qd {N};
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
    param rad;
    param npv;
    param pvcmax;
    param pveff;       
"""
)
ampl.get_parameter("fr").set_values(fr)
ampl.get_parameter("to").set_values(to)
ampl.get_parameter("yr").set_values(yrl)
ampl.get_parameter("yi").set_values(yil)
ampl.get_parameter("pd").set_values(pdem)
ampl.get_parameter("qd").set_values(qdem)
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
ampl.get_parameter("rad").set(rad)  
ampl.get_parameter("npv").set(npv)  
ampl.get_parameter("pvcmax").set(pvcmax)
ampl.get_parameter("pveff").set(pveff)
               

ampl.eval(
    r"""
    var  cii{i in N} >= ciimin[i], <= ciimax[i];
    var  cnl{k in L} >= cnlmin[k], <= cnlmax[k];
    var  snl{k in L} >= snlmin[k], <= snlmax[k];
    var  pg{i in N} >= prefmin[i], <= prefmax[i];
    var  qg{i in N} >= qrefmin[i], <= qrefmax[i];
    var  pv{i in N} >=0, <=pvcmax;
    var  ppv{i in N} >=0, <=pvcmax;
    var  zpv{i in N} binary;
    
    minimize Losses:
          sum{i in N,k in L: fr[k]=i} cii[i]*yr[k]
        - sum{i in N,k in L: fr[k]=i} cnl[k]*yr[k]
        - sum{i in N,k in L: fr[k]=i} snl[k]*yi[k]
        + sum{i in N,k in L: to[k]=i} cii[i]*yr[k]
        - sum{i in N,k in L: to[k]=i} cnl[k]*yr[k]
        + sum{i in N,k in L: to[k]=i} snl[k]*yi[k]
        - sum{i in N,k in L: fr[k]=i} cii[i]*yi[k]
        + sum{i in N,k in L: fr[k]=i} cnl[k]*yi[k]
        - sum{i in N,k in L: fr[k]=i} snl[k]*yr[k]
        - sum{i in N,k in L: to[k]=i} cii[i]*yi[k]
        + sum{i in N,k in L: to[k]=i} cnl[k]*yi[k]
        + sum{i in N,k in L: to[k]=i} snl[k]*yr[k];
    
    subject to PB {i in N}: 
        pg[i]+(ppv[i]*rad*pveff)-pd[i] = sum {k in L: fr[k]=i} cii[i]*yr[k]
        - sum{k in L: fr[k]=i} cnl[k]*yr[k]
        - sum{k in L: fr[k]=i} snl[k]*yi[k]
        + sum{k in L: to[k]=i} cii[i]*yr[k]
        - sum{k in L: to[k]=i} cnl[k]*yr[k]
        + sum{k in L: to[k]=i} snl[k]*yi[k];
    subject to QB {i in N}: 
        qg[i]-qd[i] = -sum {k in L: fr[k]=i} cii[i]*yi[k]
        + sum{k in L: fr[k]=i} cnl[k]*yi[k]
        - sum{k in L: fr[k]=i} snl[k]*yr[k]
        - sum{k in L: to[k]=i} cii[i]*yi[k]
        + sum{k in L: to[k]=i} cnl[k]*yi[k]
        + sum{k in L: to[k]=i} snl[k]*yr[k];
    subject to SOC {k in L}:
        cii[fr[k]]*cii[to[k]]=(cnl[k]*cnl[k])+(snl[k]*snl[k]); 
    
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

pgo=np.zeros(num_nodes)
qgo=np.zeros(num_nodes)
cio= np.zeros(num_nodes)
clo= np.zeros(num_lines)
slo= np.zeros(num_lines)
pvo= np.zeros(num_nodes)
ppvo= np.zeros(num_nodes)
zpvo= np.zeros(num_nodes)

for i in range (num_nodes):
    cio[i]=ampl.get_variable('cii')[i+1].value()
    pgo[i]=ampl.get_variable('pg')[i+1].value()
    qgo[i]=ampl.get_variable('qg')[i+1].value()
    pvo[i]=ampl.get_variable('pv')[i+1].value()
    ppvo[i]=ampl.get_variable('ppv')[i+1].value()
    zpvo[i]=ampl.get_variable('zpv')[i+1].value()

for k in range(num_lines):
    clo[k]=ampl.get_variable('cnl')[k+1].value()
    slo[k]=ampl.get_variable('snl')[k+1].value()


vo=np.sqrt(cio)
plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = num_nodes*[0]
Equq = num_nodes*[0]

for i in range(num_nodes):
    for k in range(num_lines):
        if i==fr[k]-1:
            Equp[i]+=(cio[i]*yrl[k])-(clo[k]*yrl[k])-(slo[k]*yil[k])
            Equq[i]+=(-cio[i]*yil[k])+(clo[k]*yil[k])-(slo[k]*yrl[k])
        if i==to[k]-1:
            Equp[i]+=(cio[i]*yrl[k])-(clo[k]*yrl[k])+(slo[k]*yil[k])
            Equq[i]+=(-cio[i]*yil[k])+(clo[k]*yil[k])+(slo[k]*yrl[k])

ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

ph=np.zeros(num_nodes)

for k in range(num_lines):
    ph[to[k]-1]=ph[fr[k]-1]-np.angle(clo[k]+1j*slo[k])

t=np.zeros(num_nodes)

t[0]=ampl.getValue('_solve_elapsed_time')
    
   
output=np.vstack((vo,ph,Equp,Equq,ploss,qloss,pgo,qgo,pvo,ppvo,zpvo,t)).T
df=pd.DataFrame(output)
columns=[]
columns.append('v')
columns.append('ph')
columns.append('eqp')
columns.append('eqq')
columns.append('pl')
columns.append('ql')
columns.append('pg')
columns.append('qg')
columns.append('pv')
columns.append('ppv')
columns.append('zpv')
columns.append('t')
    
df.columns=columns
solvlist=[0]*num_nodes
solvlist[0]='AB'


df.insert(len(df.columns),'Solver',solvlist)

df.to_excel("Results.xlsx")