# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:26:26 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL
from scipy.io import loadmat

case='IEEE33'
city='Pop'
city1='POP'
problem='PFN'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')

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

yr=np.zeros(num_lines)
yi=np.zeros(num_lines)
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]
    to[k]=branch['j'][k]
    yr[k]=np.real(1/(branch['r'][k] + 1j*branch['x'][k]))
    yi[k]=np.imag(1/(branch['r'][k] + 1j*branch['x'][k]))
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

phmin=np.ones(num_nodes)*(-np.pi)
phmax=np.ones(num_nodes)*(np.pi)

phmin[iref]=0
phmax[iref]=0

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
    param vmax{N};
    param vmin{N};
    param phmax{N};
    param phmin{N};
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};       
"""
)
ampl.get_parameter("fr").set_values(fr)
ampl.get_parameter("to").set_values(to)
ampl.get_parameter("yr").set_values(yr)
ampl.get_parameter("yi").set_values(yi)
ampl.get_parameter("pd").set_values(pdm)
ampl.get_parameter("qd").set_values(qdm)
ampl.get_parameter("vmax").set_values(vmax)
ampl.get_parameter("vmin").set_values(vmin)
ampl.get_parameter("phmax").set_values(phmax)
ampl.get_parameter("phmin").set_values(phmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
               

ampl.eval(
    r"""
    var  v{i in N} >= vmin[i], <= vmax[i];
    var  ph{i in N} >= phmin[i], <= phmax[i];
    var  pg{i in N} >= prefmin[i], <= prefmax[i];
    var  qg{i in N} >= qrefmin[i], <= qrefmax[i];
    
    
    minimize Losses:
       sum{i in N} pg[i]+sum{i in N} qg[i];       
    
    subject to PB {i in N}: 
        pg[i]-pd[i] = sum {k in L: fr[k]=i} v[i]*v[i]*yr[k]
        - sum{k in L: fr[k]=i} v[i]*v[to[k]]*yr[k]*cos(ph[i]-ph[to[k]])
        - sum{k in L: fr[k]=i} v[i]*v[to[k]]*yi[k]*sin(ph[i]-ph[to[k]])
        + sum{k in L: to[k]=i} v[i]*v[i]*yr[k]
        - sum{k in L: to[k]=i} v[i]*v[fr[k]]*yr[k]*cos(ph[i]-ph[fr[k]])
        - sum{k in L: to[k]=i} v[i]*v[fr[k]]*yi[k]*sin(ph[i]-ph[fr[k]]);
    subject to QB {i in N}: 
        qg[i]-qd[i] = sum {k in L: fr[k]=i} -v[i]*v[i]*yi[k]
        + sum{k in L: fr[k]=i} v[i]*v[to[k]]*yi[k]*cos(ph[i]-ph[to[k]])
        - sum{k in L: fr[k]=i} v[i]*v[to[k]]*yr[k]*sin(ph[i]-ph[to[k]])
        - sum{k in L: to[k]=i} v[i]*v[i]*yi[k]
        + sum{k in L: to[k]=i} v[i]*v[fr[k]]*yi[k]*cos(ph[i]-ph[fr[k]])
        - sum{k in L: to[k]=i} v[i]*v[fr[k]]*yr[k]*sin(ph[i]-ph[fr[k]]);    
    
"""
)

    
"-------Problem/solver Setup--------"

ampl.option["solver"] = "ipopt"
ampl.solve()

"-------------Print solution------------"
pgo=np.zeros(num_nodes)
qgo=np.zeros(num_nodes)
vo= np.zeros(num_nodes)
pho= np.zeros(num_nodes)


for i in range (num_nodes):
    vo[i]=ampl.get_variable('v')[i+1].value()
    pho[i]=ampl.get_variable('ph')[i+1].value()
    pgo[i]=ampl.get_variable('pg')[i+1].value()
    qgo[i]=ampl.get_variable('qg')[i+1].value()

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = num_nodes*[0]
Equq = num_nodes*[0]

for i in range(num_nodes):
    for k in range(num_lines):
        if i==fr[k]-1:
            Equp[i]+=(vo[i]*vo[i]*yr[k])-(vo[i]*vo[to[k]-1]*yr[k]*np.cos(pho[i]-pho[to[k]-1]))-(vo[i]*vo[to[k]-1]*yi[k]*np.sin(pho[i]-pho[to[k]-1]))
            Equq[i]+=(-vo[i]*vo[i]*yi[k])+(vo[i]*vo[to[k]-1]*yi[k]*np.cos(pho[i]-pho[to[k]-1]))-(vo[i]*vo[to[k]-1]*yr[k]*np.sin(pho[i]-pho[to[k]-1]))
        if i==to[k]-1:
            Equp[i]+=(vo[i]*vo[i]*yr[k])-(vo[i]*vo[fr[k]-1]*yr[k]*np.cos(pho[i]-pho[fr[k]-1]))-(vo[i]*vo[fr[k]-1]*yi[k]*np.sin(pho[i]-pho[fr[k]-1]))
            Equq[i]+=(-vo[i]*vo[i]*yi[k])+(vo[i]*vo[fr[k]-1]*yi[k]*np.cos(pho[i]-pho[fr[k]-1]))-(vo[i]*vo[fr[k]-1]*yr[k]*np.sin(pho[i]-pho[fr[k]-1]))
            
        
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