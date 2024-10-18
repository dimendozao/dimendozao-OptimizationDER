# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:31:36 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL


"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Gen.csv")


# Calcular L
num_lines = len(branch)
# Calcular N
num_nodes=len(bus)
# Nodo de referencia
iref=np.where(bus['type']==3)[0][0]

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

ampl.eval(
    r"""
    var  qvn{i in N} >= qvnmin[i], <= qvnmax[i];
    var  qik{k in L}>=0;
    var  pk{k in L}>=0;
    var  qk{k in L}>=0;
    var  pg{i in N} >= prefmin[i], <= prefmax[i];
    var  qg{i in N} >= qrefmin[i], <= qrefmax[i];
    
    minimize Losses:
       sum{i in N} pg[i]+sum{i in N} qg[i];       
    
   subject to PB {i in N}:
        pg[i]-pd[i] = sum{k in L: fr[k]=i} pk[k]
        -sum{k in L: to[k]=i} pk[k]
        +sum{k in L: to[k]=i} zkr[k]*qik[k];

    subject to QB {i in N}: 
        qg[i]-qd[i] = sum{k in L: fr[k]=i} qk[k]
        -sum{k in L: to[k]=i} qk[k]
        +sum{k in L: to[k]=i} zki[k]*qik[k];

    subject to SOC{k in L}:
        qik[k]*qvn[fr[k]]>=pk[k]*pk[k]+qk[k]*qk[k];

    subject to VDIFF{k in L}:
        qvn[fr[k]]-qvn[to[k]]==2*((pk[k]*zkr[k])+(qk[k]*zki[k]))-(qik[k]*((zkr[k]*zkr[k])+(zki[k]*zki[k])));
       
"""
)

ampl.solve(solver='scip',verbose=True)
#ampl.solve(solver='scip',verbose=True,highs_options="socp=2 threads=12 timing=1 parallel='on' miploglev=2")

pgo=np.zeros(num_nodes)
qgo=np.zeros(num_nodes)
vo= np.zeros(num_nodes)



for i in range (num_nodes):
    vo[i]=np.sqrt(ampl.get_variable('qvn')[i+1].value())
    pgo[i]=ampl.get_variable('pg')[i+1].value()
    qgo[i]=ampl.get_variable('qg')[i+1].value()

plt.plot(vo)