# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:29:05 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL
from scipy.io import loadmat

case='CA141'
city='Jam'
city1='JAM'
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

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]

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

ym=np.zeros([num_nodes,num_nodes],dtype='complex')

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])
    
for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])

yr=np.real(ym)
yi=np.imag(ym)

"----- Optimization model -----"

ampl = AMPL()

ampl.eval(
    r"""
    param nn;       
"""
)

ampl.get_parameter("nn").set(num_nodes)

ampl.eval(
    r"""
    set N :=1..nn;
    
    param Yr {N,N};
    param Yi {N,N};
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

ampl.get_parameter("Yr").set_values(yr)
ampl.get_parameter("Yi").set_values(yi)
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
       sum{i in N, j in N} v[i]*v[j]*Yr[i,j]*cos(ph[i]-ph[j])
       +sum{i in N, j in N} v[i]*v[j]*Yi[i,j]*sin(ph[i]-ph[j])
       +sum{i in N, j in N} v[i]*v[j]*Yr[i,j]*sin(ph[i]-ph[j])
       -sum{i in N, j in N} v[i]*v[j]*Yi[i,j]*cos(ph[i]-ph[j]);
       
   subject to PB {i in N}: 
       pg[i]-pd[i] = sum {j in N} (v[i]*v[j]*Yr[i,j]*cos(ph[i]-ph[j]))+sum {j in N} (v[i]*v[j]*Yi[i,j]*sin(ph[i]-ph[j]));
       
   subject to QB {i in N}: 
       qg[i]-qd[i] = sum {j in N} (v[i]*v[j]*Yr[i,j]*sin(ph[i]-ph[j]))-sum {j in N} (v[i]*v[j]*Yi[i,j]*cos(ph[i]-ph[j]));   
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
    for j in range(num_nodes):
        Equp[i]+=(vo[i]*vo[j]*yr[i][j]*np.cos(pho[i]-pho[j]))+(vo[i]*vo[j]*yi[i][j]*np.sin(pho[i]-pho[j]))
        Equq[i]+=(vo[i]*vo[j]*yr[i][j]*np.sin(pho[i]-pho[j]))-(vo[i]*vo[j]*yi[i][j]*np.cos(pho[i]-pho[j]))
        
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