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


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE69Gen.csv")

case='IEEE69'
city='Pop'
city1='POP'
problem='MPF'

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
means=mat['clustermeans']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=mat['clusternode']


H=np.size(means,axis=1)

num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for k in range(num_lines):
        sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*means[cnode[branch['j'][k]-1][0]-1][h]
        
yrl=np.zeros(num_lines)
yil=np.zeros(num_lines)
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):        
    fr[k]=branch['i'][k]
    to[k]=branch['j'][k]
    yrl[k]=np.real(1/(branch['r'][k] + 1j*branch['x'][k]))
    yil[k]=np.imag(1/(branch['r'][k] + 1j*branch['x'][k]))


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
    var  vr{h in H,i in N} >= vrmin[i], <= vrmax[i];
    var  vi{h in H,i in N} >= vimin[i], <= vimax[i];
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    
    
    minimize Losses:
          sum{h in H, i in N,k in L: fr[k]=i} yr[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        - sum{h in H, i in N,k in L: fr[k]=i} yr[k]*(vr[h,i]*vr[h,to[k]]+vi[h,i]*vi[h,to[k]])
        + sum{h in H, i in N,k in L: fr[k]=i} yi[k]*(vr[h,i]*vi[h,to[k]]-vi[h,i]*vr[h,to[k]])
        + sum{h in H, i in N,k in L: to[k]=i} yr[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        - sum{h in H, i in N,k in L: to[k]=i} yr[k]*(vr[h,i]*vr[h,fr[k]]+vi[h,i]*vi[h,fr[k]])
        + sum{h in H, i in N,k in L: to[k]=i} yi[k]*(vr[h,i]*vi[h,fr[k]]-vi[h,i]*vr[h,fr[k]])
        - sum{h in H, i in N,k in L: fr[k]=i} yi[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        + sum{h in H, i in N,k in L: fr[k]=i} yi[k]*(vr[h,i]*vr[h,to[k]]+vi[h,i]*vi[h,to[k]])
        + sum{h in H, i in N,k in L: fr[k]=i} yr[k]*(vr[h,i]*vi[h,to[k]]-vi[h,i]*vr[h,to[k]])
        - sum{h in H, i in N,k in L: to[k]=i} yi[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        + sum{h in H, i in N,k in L: to[k]=i} yi[k]*(vr[h,i]*vr[h,fr[k]]+vi[h,i]*vi[h,fr[k]])
        + sum{h in H, i in N,k in L: to[k]=i} yr[k]*(vr[h,i]*vi[h,fr[k]]-vi[h,i]*vr[h,fr[k]]);
    
    subject to PB {h in H,i in N}: 
        pg[h,i]-pd[h,i] = sum{k in L: fr[k]=i} yr[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        - sum{k in L: fr[k]=i} yr[k]*(vr[h,i]*vr[h,to[k]]+vi[h,i]*vi[h,to[k]])
        + sum{k in L: fr[k]=i} yi[k]*(vr[h,i]*vi[h,to[k]]-vi[h,i]*vr[h,to[k]])
        + sum{k in L: to[k]=i} yr[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        - sum{k in L: to[k]=i} yr[k]*(vr[h,i]*vr[h,fr[k]]+vi[h,i]*vi[h,fr[k]])
        + sum{k in L: to[k]=i} yi[k]*(vr[h,i]*vi[h,fr[k]]-vi[h,i]*vr[h,fr[k]]);
    subject to QB {h in H,i in N}: 
        qg[h,i]-qd[h,i] = -sum{k in L: fr[k]=i} yi[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        + sum{k in L: fr[k]=i} yi[k]*(vr[h,i]*vr[h,to[k]]+vi[h,i]*vi[h,to[k]])
        + sum{k in L: fr[k]=i} yr[k]*(vr[h,i]*vi[h,to[k]]-vi[h,i]*vr[h,to[k]])
        - sum{k in L: to[k]=i} yi[k]*(vr[h,i]*vr[h,i]+vi[h,i]*vi[h,i])
        + sum{k in L: to[k]=i} yi[k]*(vr[h,i]*vr[h,fr[k]]+vi[h,i]*vi[h,fr[k]])
        + sum{k in L: to[k]=i} yr[k]*(vr[h,i]*vi[h,fr[k]]-vi[h,i]*vr[h,fr[k]]);    
    
"""
)

    
"-------Problem/solver Setup--------"

ampl.option["solver"] = "ipopt"
ampl.solve()

"-------------Print solution------------"
pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
vro= np.zeros([H,num_nodes])
vio= np.zeros([H,num_nodes])

for h in range(H):
    for i in range (num_nodes):
        vro[h][i]=ampl.get_variable('vr')[h+1,i+1].value()
        vio[h][i]=ampl.get_variable('vi')[h+1,i+1].value()
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()

vo=np.sqrt(vro**2 + vio**2)

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = [[0] * num_nodes for h in range(H)]
Equq = [[0] * num_nodes for h in range(H)]


for h in range(H):
    for i in range(num_nodes):
        for k in range(num_lines):       
            if i==fr[k]-1:
                Equp[h][i]+=(yrl[k]*(vro[h,i]*vro[h,i]+vio[h,i]*vio[h,i]))-(yrl[k]*(vro[h,i]*vro[h,to[k]-1]+vio[h,i]*vio[h,to[k]-1]))+(yil[k]*(vro[h,i]*vio[h,to[k]-1]-vio[h,i]*vro[h,to[k]-1]))
                Equq[h][i]+=(-yil[k]*(vro[h,i]*vro[h,i]+vio[h,i]*vio[h,i]))+(yil[k]*(vro[h,i]*vro[h,to[k]-1]+vio[h,i]*vio[h,to[k]-1]))+(yrl[k]*(vro[h,i]*vio[h,to[k]-1]-vio[h,i]*vro[h,to[k]-1]))
            if i==to[k]-1:
                Equp[h][i]+=(yrl[k]*(vro[h,i]*vro[h,i]+vio[h,i]*vio[h,i]))-(yrl[k]*(vro[h,i]*vro[h,fr[k]-1]+vio[h,i]*vio[h,fr[k]-1]))+(yil[k]*(vro[h,i]*vio[h,fr[k]-1]-vio[h,i]*vro[h,fr[k]-1]))
                Equq[h][i]+=(-yil[k]*(vro[h,i]*vro[h,i]+vio[h,i]*vio[h,i]))+(yil[k]*(vro[h,i]*vro[h,fr[k]-1]+vio[h,i]*vio[h,fr[k]-1]))+(yrl[k]*(vro[h,i]*vio[h,fr[k]-1]-vio[h,i]*vro[h,fr[k]-1]))


pho=np.angle(vro+1j*vio)            
        
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])

t=np.zeros(H)
t[0]=ampl.getValue('_solve_elapsed_time')

out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t)).T
output=np.hstack((vo,pho,Equp,Equq,out1))
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
    
df.columns=columns

solvlist=[0]*H
solvlist[0]='AI'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")