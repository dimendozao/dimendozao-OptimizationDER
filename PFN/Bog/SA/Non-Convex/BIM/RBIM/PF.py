# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:33:24 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL
from scipy.io import loadmat

case='SA'
city='Bog'
city1='BOG'
problem='PFN'
case1='SA_J23'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Gen.csv')

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
dmeans=np.squeeze(mat['clustermeans']).T

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=np.squeeze(mat['clusternode'])

cnode[0]=1
cnode=cnode-1

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

for i in range(num_nodes):
    sd[i]=sd[i]*dem[cnode[i]]
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
    param vrmax{N};
    param vrmin{N};
    param vimax{N};
    param vimin{N};
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};       
"""
)
ampl.get_parameter("fr").set_values(fr)
ampl.get_parameter("to").set_values(to)
ampl.get_parameter("yr").set_values(yrl)
ampl.get_parameter("yi").set_values(yil)
ampl.get_parameter("pd").set_values(pdm)
ampl.get_parameter("qd").set_values(qdm)
ampl.get_parameter("vrmax").set_values(vrmax)
ampl.get_parameter("vrmin").set_values(vrmin)
ampl.get_parameter("vimax").set_values(vimax)
ampl.get_parameter("vimin").set_values(vimin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
               

ampl.eval(
    r"""
    var  vr{i in N} >= vrmin[i], <= vrmax[i];
    var  vi{i in N} >= vimin[i], <= vimax[i];
    var  pg{i in N} >= prefmin[i], <= prefmax[i];
    var  qg{i in N} >= qrefmin[i], <= qrefmax[i];
    
    
    minimize Losses:
       sum{i in N} pg[i]+sum{i in N} qg[i];       
    
    subject to PB {i in N}: 
        pg[i]-pd[i] = sum {k in L: fr[k]=i} yr[k]*(vr[i]*vr[i]+vi[i]*vi[i])
        - sum{k in L: fr[k]=i} yr[k]*(vr[i]*vr[to[k]]+vi[i]*vi[to[k]])
        + sum{k in L: fr[k]=i} yi[k]*(vr[i]*vi[to[k]]-vi[i]*vr[to[k]])
        + sum{k in L: to[k]=i} yr[k]*(vr[i]*vr[i]+vi[i]*vi[i])
        - sum{k in L: to[k]=i} yr[k]*(vr[i]*vr[fr[k]]+vi[i]*vi[fr[k]])
        + sum{k in L: to[k]=i} yi[k]*(vr[i]*vi[fr[k]]-vi[i]*vr[fr[k]]);
    subject to QB {i in N}: 
        qg[i]-qd[i] = -sum {k in L: fr[k]=i} yi[k]*(vr[i]*vr[i]+vi[i]*vi[i])
        + sum{k in L: fr[k]=i} yi[k]*(vr[i]*vr[to[k]]+vi[i]*vi[to[k]])
        + sum{k in L: fr[k]=i} yr[k]*(vr[i]*vi[to[k]]-vi[i]*vr[to[k]])
        - sum{k in L: to[k]=i} yi[k]*(vr[i]*vr[i]+vi[i]*vi[i])
        + sum{k in L: to[k]=i} yi[k]*(vr[i]*vr[fr[k]]+vi[i]*vi[fr[k]])
        + sum{k in L: to[k]=i} yr[k]*(vr[i]*vi[fr[k]]-vi[i]*vr[fr[k]]);    
    
"""
)

    
"-------Problem/solver Setup--------"

ampl.option["solver"] = "ipopt"
ampl.solve()

"-------------Print solution------------"
pgo=np.zeros(num_nodes)
qgo=np.zeros(num_nodes)
vro= np.zeros(num_nodes)
vio= np.zeros(num_nodes)


for i in range (num_nodes):
    vro[i]=ampl.get_variable('vr')[i+1].value()
    vio[i]=ampl.get_variable('vi')[i+1].value()
    pgo[i]=ampl.get_variable('pg')[i+1].value()
    qgo[i]=ampl.get_variable('qg')[i+1].value()

vo=np.sqrt(vro**2 + vio**2)

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = num_nodes*[0]
Equq = num_nodes*[0]

for i in range(num_nodes):
    for k in range(num_lines):       
        if i==fr[k]-1:
            Equp[i]+=(yrl[k]*(vro[i]*vro[i]+vio[i]*vio[i]))-(yrl[k]*(vro[i]*vro[to[k]-1]+vio[i]*vio[to[k]-1]))+(yil[k]*(vro[i]*vio[to[k]-1]-vio[i]*vro[to[k]-1]))
            Equq[i]+=(-yil[k]*(vro[i]*vro[i]+vio[i]*vio[i]))+(yil[k]*(vro[i]*vro[to[k]-1]+vio[i]*vio[to[k]-1]))+(yrl[k]*(vro[i]*vio[to[k]-1]-vio[i]*vro[to[k]-1]))
        if i==to[k]-1:
            Equp[i]+=(yrl[k]*(vro[i]*vro[i]+vio[i]*vio[i]))-(yrl[k]*(vro[i]*vro[fr[k]-1]+vio[i]*vio[fr[k]-1]))+(yil[k]*(vro[i]*vio[fr[k]-1]-vio[i]*vro[fr[k]-1]))
            Equq[i]+=(-yil[k]*(vro[i]*vro[i]+vio[i]*vio[i]))+(yil[k]*(vro[i]*vro[fr[k]-1]+vio[i]*vio[fr[k]-1]))+(yrl[k]*(vro[i]*vio[fr[k]-1]-vio[i]*vro[fr[k]-1]))

pho=np.angle(vro+1j*vio)            
        
ploss=np.zeros(num_nodes)
qloss=np.zeros(num_nodes)
ploss[0]=np.sum(Equp)
qloss[0]=np.sum(Equq)

t=np.zeros(num_nodes)
t[0]=ampl.getValue('_solve_elapsed_time')

output=np.vstack((vo,pho,Equp,Equq,ploss,qloss,pgo,qgo,t)).T
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
columns.append('t')

    
df.columns=columns
solvlist=[0]*num_nodes
solvlist[0]='AI'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")  