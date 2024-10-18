# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:01:30 2024

@author: diego
"""

from amplpy import AMPL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Bus.csv")
gen= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\SA_J23Gen.csv")

case='SA'
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
qrefmin[iref]=np.max([0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]])


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
idx=idx-np.eye(num_nodes)

# cnmax=np.multiply(cnmax,idx)
# cnmin=np.multiply(cnmin,idx)

# snmax=np.multiply(snmax,idx)
# snmin=np.multiply(snmin,idx)

"----- Optimization model -----"

ampl = AMPL()


ampl.eval(
    r"""
    param nn;
    param nh;
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nh").set(H)

ampl.eval(
    r"""
    set N=1..nn;
    set H=1..nh;
    
    param yr {N,N};
    param yi {N,N};
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
    param idx{N,N};   
"""
)

ampl.get_parameter("yr").set_values(ymr)
ampl.get_parameter("yi").set_values(ymi)
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
ampl.get_parameter("idx").set_values(idx)         

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
    
    subject to Hermit1 {h in H,i in N, j in N: i<j}:
        cn[h,i,j]=cn[h,j,i];
        
    subject to Hermit2 {h in H,i in N, j in N:i<j}:
        sn[h,i,j]=-sn[h,j,i];
        
    subject to SOC {h in H,i in N, j in N: idx[i,j]=1}:
        (cn[h,i,j]*cn[h,i,j])+(sn[h,i,j]*sn[h,i,j])=cn[h,i,i]*cn[h,j,j];
    
    subject to POSP{h in H}:
        sum{i in N} pg[h,i]-sum{i in N}pd[h,i]>=0;
    subject to POSQ{h in H}:
        sum{i in N} qg[h,i]-sum{i in N}qd[h,i]>=0;
        
"""
)

"-------Problem/solver Setup--------"

#ampl.option["solver"] = "Couenne"
#ampl.option["couenne_options"] = "use_quadratic=yes"


# ampl.option["solver"] = "kestrel"
# ampl.option["kestrel_options"] = "solver=knitro priority=long"
#ampl.option["gurobi_options"] = "nonconvex=2"
# ampl.option["email"] = "dimendozao@unal.edu.co"

ampl.option["solver"] = "ipopt"
ampl.option["show_stats"] = 1
#ampl.option["presolve"] = 0
ampl.option["ipopt_options"] = "max_cpu_time=480"
ampl.solve()

"-------------Print solution------------"
pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
cno= np.zeros([H,num_nodes,num_nodes])
sno= np.zeros([H,num_nodes,num_nodes])


for h in range(H):
    for i in range (num_nodes):
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()
        for j in range(num_nodes):
            cno[h][i][j]=ampl.get_variable('cn')[h+1,i+1,j+1].value()
            sno[h][i][j]=ampl.get_variable('sn')[h+1,i+1,j+1].value()

vo=np.zeros([H,num_nodes])       

for h in range(H):        
    vo[h]=np.sqrt(np.diag(cno[h]))

plt.plot(vo)

plo=np.sum(pgo)+np.sum(pgen)-np.sum(pdm)

Equp = [[0] * num_nodes for h in range(H)]
Equq = [[0] * num_nodes for h in range(H)]


for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            Equp[h][i]+=(cno[h][i][j]*ymr[i][j])+(sno[h][i][j]*ymi[i][j])
            Equq[h][i]+=(sno[h][i][j]*ymr[i][j])-(cno[h][i][j]*ymi[i][j])
    

ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])

pho=np.zeros([H,num_nodes])

for h in range(H):
    for k in range(num_lines):
        fr=branch['i'][k]-1
        to=branch['j'][k]-1
        pho[h][to]=pho[h][fr]-np.angle(cno[h][fr][to]+1j*sno[h][fr][to])

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


