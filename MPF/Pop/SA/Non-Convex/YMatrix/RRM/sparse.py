# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:02:40 2024

@author: diego
"""

from amplpy import AMPL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\CA141Gen.csv")

case='CA141'
city='Bog'
city1='BOG'
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

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if i==j:
            cnmax[i][i]=vmax[i]*vmax[i]
            cnmin[i][i]=vmin[i]*vmin[i]
        else:
            cnmax[i][j]=vmax[i]*vmax[j]
            cnmin[i][j]=vmin[i]*vmin[j]
            snmax[i][j]=vmax[i]*vmax[j]
            snmin[i][j]=-vmax[i]*vmax[j]
            
idx=np.abs(ym)!=0

cnmax=np.multiply(cnmax,idx)
cnmin=np.multiply(cnmin,idx)

snmax=np.multiply(snmax,idx)
snmin=np.multiply(snmin,idx)

nz=np.sum(idx)

ix=np.zeros(nz,dtype=int)
jx=np.zeros(nz,dtype=int)
yrx=np.zeros(nz)
yix=np.zeros(nz)

k=0

for i in range(num_nodes):
    for j in range(num_nodes):
        if not np.abs(ym[i][j])==0:
            ix[k]=i
            jx[k]=j
            yrx[k]=np.real(ym[i][j])
            yix[k]=np.imag(ym[i][j])
            k+=1

idx=np.abs(ym)!=0

cnmax=np.multiply(cnmax,idx)
cnmin=np.multiply(cnmin,idx)

snmax=np.multiply(snmax,idx)
snmin=np.multiply(snmin,idx)

"----- Optimization model -----"

ampl = AMPL()


ampl.eval(
    r"""
    param nn;
    param nh;
    param nz;
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nh").set(H)
ampl.get_parameter("nz").set(nz)

ampl.eval(
    r"""
    set N=1..nn;
    set H=1..nh;
    set M=1..nz;
    
    param ixx {M};
    param jxx {M};
"""
)
    
ampl.get_parameter("ixx").set_values(ix)
ampl.get_parameter("jxx").set_values(jx)    
    
ampl.eval(
    r"""    
    set ix=ixx
    set jx=jxx
      
    param yr {ix,ix};
    param yi {ix,jx};
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
"""
)

ampl.get_parameter("yr").set_values(yrx)
ampl.get_parameter("yi").set_values(yix)
ampl.get_parameter("pd").set_values(pdm)
ampl.get_parameter("qd").set_values(qdm)
ampl.get_parameter("cnmax").set_values(cnmax)
ampl.get_parameter("cnmin").set_values(cnmin)
ampl.get_parameter("snmax").set_values(snmax)
ampl.get_parameter("snmin").set_values(snmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)       

"-------Constraint Construction-------- "

ampl.eval(
    r"""
    var  cn{h in H,i in N, j in N} >= cnmin[i,j], <= cnmax[i,j];
    var  sn{h in H,i in N, j in N} >= snmin[i,j], <= snmax[i,j];
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    
    minimize Losses:
       sum{h in H,i in N, j in N} cn[h,i,j]*yr[i,j]
       + sum{h in H,i in N, j in N} sn[h,i,j]*yi[i,j] 
       + sum{h in H,i in N, j in N} sn[h,i,j]*yr[i,j]
       - sum{h in H,i in N, j in N} cn[h,i,j]*yi[i,j];
    
    subject to PB {h in H,i in N}: 
        pg[h,i]-pd[h,i] = sum {j in N} cn[h,i,j]*yr[i,j]
        + sum {j in N} sn[h,i,j]*yi[i,j];  
        
    subject to QB {h in H,i in N}: 
        qg[h,i]-qd[h,i] = sum {j in N} sn[h,i,j]*yr[i,j]
        - sum {j in N} cn[h,i,j]*yi[i,j];    
    
    subject to SOC {h in H,i in N, j in N: j>i}:
        (cn[h,i,j]*cn[h,i,j])+(sn[h,i,j]*sn[h,i,j])=cn[h,i,i]*cn[h,j,j];
        
    subject to Hermit1 {h in H,i in N, j in N: j>i}:
        cn[h,i,j]=cn[h,j,i];
        
    subject to Hermit2 {h in H,i in N, j in N: j>i}:
        sn[h,i,j]=-sn[h,j,i];        
        
"""
)