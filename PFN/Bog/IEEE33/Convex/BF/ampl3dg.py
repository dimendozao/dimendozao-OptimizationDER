# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:09:07 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL
from scipy.io import loadmat


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Gen.csv")
mat=loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\Bog\\ClusterMeans_BOG.mat')


clusters=mat['clustermeans']

mat=loadmat("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\Bog\\MeansRAD_BOG.mat")

radmean=mat['means'].T

# Calcular L
num_lines = len(branch)
# Calcular N
num_nodes=len(bus)
# Nodo de referencia
iref=np.where(bus['type']==3)[0][0]

# numero de periodos

H=clusters.shape[1]

# Vector de demandas
sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]

#Demandas activas y reactivas (nodos PQ)   
pdm=np.real(sd)
qdm=np.imag(sd)

#Definir el numero de generadores (Nodos PV)
ngen=np.sum(bus['type']==2)
pgen=np.zeros(num_nodes)
qgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

# Definir limites de voltaje
vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

# Definir potencias y voltajes en nodos PV
if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]

# Definir limites de voltaje en nodo de referencia       
vmax[iref]=1
vmin[iref]=1

#Definir limites de potencia en el generador Slack
prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

# Inicializar impedancias de linea
zkr=np.zeros(num_lines)
zki=np.zeros(num_lines)
# Inicializar nodos de origen y destino de cada linea
fr=np.zeros(num_lines,dtype='int')
to=np.zeros(num_lines,dtype='int')

#Definir impedancias y nodos de destino y origen
#Python trabaja con indices cero, AMPLPY no.
for k in range(num_lines):
    fr[k]=branch['i'][k]
    to[k]=branch['j'][k]
    zkr[k]=branch['r'][k]
    zki[k]=branch['x'][k]

#Definir los limites de la variable cuadratica del voltage ($\upsilon$)

qvnmax=vmax**2
qvnmin=vmin**2

# Si hay limites en la corriente de las lineas, se pueden definir para ($\iota$)

cpv=2.5
npv=3

"----- Optimization model -----"

ampl = AMPL()

ampl.set_option("presolve", 1)

ampl.eval(
    r"""
    param nn;
    param nl;
    param nh;
    param cpv;
    param npv;    
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nl").set(num_lines)
ampl.get_parameter("nh").set(H)
ampl.get_parameter("cpv").set(cpv)
ampl.get_parameter("npv").set(npv)

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
    param zkr {L};
    param zki {L};
    param pd {N};
    param qd {N};
    param qvnmax{N};
    param qvnmin{N};    
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};
    param cluster{H};
    param rad{H};
      
"""
)
ampl.get_parameter("fr").set_values(fr)
ampl.get_parameter("to").set_values(to)
ampl.get_parameter("zkr").set_values(zkr)
ampl.get_parameter("zki").set_values(zki)
ampl.get_parameter("pd").set_values(pdm)
ampl.get_parameter("qd").set_values(qdm)
ampl.get_parameter("qvnmax").set_values(qvnmax)
ampl.get_parameter("qvnmin").set_values(qvnmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
ampl.get_parameter("cluster").set_values(clusters[1])
ampl.get_parameter("rad").set_values(radmean)

ampl.eval(
    r"""
    var  qvn{h in H,i in N} >= qvnmin[i], <= qvnmax[i];
    var  qik{h in H,k in L}>= 0;
    var  pk{h in H,k in L}>=0;
    var  qk{h in H,k in L}>=0;
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    var  z{i in N} binary;
    var  pv{i in N} >=0, <=cpv;
    var  ppv{i in N} >=0,<= cpv;

    minimize Losses:
       sum{h in H,i in N,k in L: fr[k]=i} pk[h,k]
        -sum{h in H,i in N, k in L: to[k]=i} pk[h,k]
        +sum{h in H,i in N, k in L: to[k]=i} zkr[k]*qik[h,k]
        +sum{h in H,i in N,k in L: fr[k]=i} qk[h,k]
        -sum{h in H,i in N,k in L: to[k]=i} qk[h,k]
        +sum{h in H,i in N,k in L: to[k]=i} zki[k]*qik[h,k];

    subject to PB {h in H,i in N}:
        pg[h,i]+(ppv[i]*rad[h])-(pd[i]*cluster[h]) = sum{k in L: fr[k]=i} pk[h,k]
        -sum{k in L: to[k]=i} pk[h,k]
        +sum{k in L: to[k]=i} zkr[k]*qik[h,k];

    subject to QB {h in H,i in N}:
        qg[h,i]-(qd[i]*cluster[h]) = sum{k in L: fr[k]=i} qk[h,k]
        -sum{k in L: to[k]=i} qk[h,k]
        +sum{k in L: to[k]=i} zki[k]*qik[h,k];

    subject to NCSOC{h in H,k in L}:
        qik[h,k]*qvn[h,fr[k]]>=pk[h,k]*pk[h,k]+qk[h,k]*qk[h,k];

    subject to VDIFF{h in H,k in L}:
        qvn[h,fr[k]]-qvn[h,to[k]]==2*((pk[h,k]*zkr[k])+(qk[h,k]*zki[k]))-(qik[h,k]*((zkr[k]*zkr[k])+(zki[k]*zki[k])));

    subject to PV1{i in N}:
        ppv[i]<=z[i]*cpv;
    subject to PV2{i in N}:
        ppv[i]<=pv[i];
    subject to PV3{i in N}:
        ppv[i]>=pv[i]-cpv*(1-z[i]);
    subject to zmax:
        sum{i in N} z[i]=npv;
"""
)

ampl.solve(solver='scip',verbose=True)

pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
vo= np.zeros([H,num_nodes])
ppvo= np.zeros(num_nodes)
zo=np.zeros(num_nodes)

for h in range(H):
  for i in range (num_nodes):
      vo[h][i]=np.sqrt(ampl.get_variable('qvn')[h+1,i+1].value())
      pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
      qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()
  

plt.plot(vo)

pgor=pgo[:,0]
qgor=qgo[:,0]