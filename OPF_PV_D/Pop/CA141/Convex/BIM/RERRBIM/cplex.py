# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:43:34 2024

@author: diego
"""

from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

mat= loadmat("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\Bog\\MeansRAD_BOG.mat")
means=np.squeeze(mat['means'])

idx=means>0.01

rad=np.mean(means[idx])

num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')
ylr=np.zeros(num_lines)
yli=np.zeros(num_lines)
fr=np.zeros(num_lines,dtype=int)
to=np.zeros(num_lines,dtype=int)

for k in range(num_lines):
    sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
    fr[k]=branch['i'][k]-1
    to[k]=branch['j'][k]-1
    ylr[k]=np.real(1/(branch['r'][k] + 1j*branch['x'][k]))
    yli[k]=np.imag(1/(branch['r'][k] + 1j*branch['x'][k]))
    
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

umax=vmax**2
umin=vmin**2        

umax[iref]=1
umin[iref]=1

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

wrmax=np.zeros(num_lines)
wrmin=np.zeros(num_lines)
wimax=np.zeros(num_lines)
wimin=np.zeros(num_lines)
    
for k in range(num_lines):        
    wrmax[k]=vmax[fr[k]]*vmax[to[k]]
    wrmin[k]=vmin[fr[k]]*vmin[to[k]]
    wimax[k]=vmax[fr[k]]*vmax[to[k]]
    wimin[k]=-vmax[fr[k]]*vmax[to[k]]

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))

"----- Optimization model -----"
m = Model(name='OPF')

pgref = m.continuous_var_list(num_nodes,lb=prefmin,ub=prefmax,name='pgref')
qgref = m.continuous_var_list(num_nodes,lb=qrefmin,ub=qrefmax,name='qgref')

u= m.continuous_var_list(num_nodes,lb=umin,ub=umax,name='u')
wr= m.continuous_var_list(num_lines,lb=wrmin,ub=wrmax,name='wr')
wi= m.continuous_var_list(num_lines,lb=wimin,ub=wimax,name='wi')


pv= m.continuous_var_list(num_nodes,lb=0,ub=pvcmax,name='pv')
ppv= m.continuous_var_list(num_nodes,lb=0,ub=pvcmax,name='pv')
zpv= m.binary_var_list(num_nodes,name='zpv')

"-------Constraint Construction-------- "

EqNp = [0] * num_nodes
EqNq = [0] * num_nodes 

for i in range(num_nodes):        
    for k in range(num_lines):
        if i==fr[k]:
            EqNp[i]+=(u[i]*ylr[k])-(wr[k]*ylr[k])-(wi[k]*yli[k])
            EqNq[i]+=(-u[i]*yli[k])+(wr[k]*yli[k])-(wi[k]*ylr[k])
        if i==to[k]:            
            EqNp[i]+=(u[i]*ylr[k])-(wr[k]*ylr[k])+(wi[k]*yli[k])
            EqNq[i]+=(-u[i]*yli[k])+(wr[k]*yli[k])+(wi[k]*ylr[k])    

m.add_constraints((EqNp[i]==pgref[i]+(ppv[i]*rad)+pgen[i]-pdm[i] for i in range(num_nodes)))
m.add_constraints((EqNq[i]==qgref[i]+qgen[i]-qdm[i] for i in range(num_nodes)))

m.add_quadratic_constraints((u[fr[k]]*u[to[k]]>=wr[k]*wr[k]+wi[k]*wi[k] for k in range(num_lines))) 

m.add_constraints((ppv[i]<=zpv[i]*pvcmax for i in range(num_nodes)))
m.add_constraints((ppv[i]<=pv[i] for i in range(num_nodes)))
m.add_constraints((ppv[i]>=pv[i]-pvcmax*(1-zpv[i]) for i in range(num_nodes)))
m.add_constraint(m.sum_vars(zpv)==npv)
m.add_constraint(m.sum_vars(ppv)==pvcmax)   
    
"-------Objective definition--------"
m.minimize(m.sum(EqNp[i] for i in range(num_nodes))+m.sum(EqNq[i] for i in range(num_nodes)))
m.solve(log_output=True)

"----- Print results -----"
