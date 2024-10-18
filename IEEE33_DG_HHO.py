"""
Created on Wed May 31 17:57:48 2023

@author: diego

HHO Implementation for the 3DG OPF
"""
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
from IEEE33_DG import IEEE33_DG as pf
import copy

#%% 
"HHO parameters"

Nh=100
Ni=100
Nx=6 #0-2 Zpv; 3-5 Pvcap 

E=0
B=1.5
PGMax=2.5;
Ngen=3;
#%% 
"HHO"
x=np.zeros([Ni+1,Nh,Nx])
lx=np.zeros(Nx)
ux=np.ones(Nx)

lx[:Ngen]=1
ux[:Ngen]=32
ux[Ngen:]=PGMax
fx=np.zeros([Ni+1,Nh])
fy=0
fz=0

net=pp.from_pickle("IEEE33.p")
for i in range(Nh):
    x[0][i][:Ngen]=np.round(lx[:Ngen]+(np.random.rand(Ngen)*(ux[:Ngen]-lx[:Ngen])))
    x[0][i][Ngen:]=lx[Ngen:]+(np.random.rand(Ngen)*(ux[Ngen:]-lx[Ngen:]))

for i in range(Ni):
    print("Progress= ",str(i*100/(Ni-1)),"%")
    for k in range(Nh):
        net1 = copy.deepcopy(net)
        for kk in range(Ngen):
           net1.load.p_mw[x[i][k][kk]-1]=net1.load.p_mw[x[i][k][kk]-1]-x[i][k][kk+3]
        [pg,qg,fx[i][k]]=pf(net1)
    
    for k in range(Nh):
        E0=(2*np.random.rand())-1
        E=2*E0*(1-(i/(Ni-1)))
        ir=np.argmin(fx[i])       
        r1=np.random.rand()
        r2=np.random.rand()
        r3=np.random.rand()
        r4=np.random.rand()
        r=np.random.rand()
        r5=np.random.rand()
        J=2*(1-r5)
        xm=np.mean(x[i],axis=0)
        s=np.random.rand(Nx)
        sig=np.power(((gamma(1+B)*np.sin(np.pi*B/2))/(gamma((1+B)/2)*B*np.power(2,(B-1)/2))),(1/B))
        u=np.random.normal(0,np.square(sig),Nx)
        v=np.random.normal(0,1,Nx)            
        lf=u/(np.power(np.absolute(v),(1/B)))
        dx=x[i][ir]-x[i][k]
        if np.absolute(E)>=1:            
            q=np.random.rand()
            if q>=0.5:                    
                ir=np.random.randint(0,Nh)
                x[i+1][k]=x[i][ir]-(r1*np.absolute(x[i][ir]-(2*r2*x[i][k])))
                x[i+1][k][:Ngen]=np.round(x[i+1][k][:Ngen])
                a=x[i+1][k]<lx
                b=x[i+1][k]>ux
                x[i+1][k][a]=lx[a]
                x[i+1][k][b]=ux[b]
            else:
                x[i+1][k]=(x[i][ir]-xm)-(r3*(lx+(r4*(ux-lx))))
                x[i+1][k][:Ngen]=np.round(x[i+1][k][:Ngen])
                a=x[i+1][k]<lx
                b=x[i+1][k]>ux
                x[i+1][k][a]=lx[a]
                x[i+1][k][b]=ux[b]
        elif np.absolute(E)>=0.5:
            if r>=0.5:                   
                x[i+1][k]=dx-(E*np.absolute(J*x[i][ir]-x[i][k]))
                x[i+1][k][:Ngen]=np.round(x[i+1][k][:Ngen])
                a=x[i+1][k]<lx
                b=x[i+1][k]>ux
                x[i+1][k][a]=lx[a]
                x[i+1][k][b]=ux[b]
            else:                           
                y=x[i][ir]-(E*np.absolute(J*x[i][ir]-x[i][k]))
                y[:Ngen]=np.round(y[:Ngen])
                z=y+(s*lf)
                z[:Ngen]=np.round(z[:Ngen])
                a=y<lx
                b=y>ux
                y[a]=lx[a]
                y[b]=ux[b]
                a=z<lx
                b=z>ux
                z[a]=lx[a]
                z[b]=ux[b]
                net1 = copy.deepcopy(net)
                for kk in range(Ngen):
                   net1.load.p_mw[y[kk]-1]=net1.load.p_mw[y[kk]-1]-y[kk+3]
                [pg,qg,fy]=pf(net1)
                net1 = copy.deepcopy(net)
                for kk in range(Ngen):
                   net1.load.p_mw[z[kk]-1]=net1.load.p_mw[z[kk]-1]-y[kk+3]
                [pg,qg,fz]=pf(net1)                
                if fx[i][k]>fy: 
                    x[i+1][k]=y                    
                elif fx[i][k]>fz:
                    x[i+1][k]=z                    
                else:
                    x[i+1][k]=x[i][k]                    
        else:                
            if r>=0.5:                   
                x[i+1][k]=x[i][ir]-(E*np.absolute(dx))
                x[i+1][k][:Ngen]=np.round(x[i+1][k][:Ngen])
                a=x[i+1][k]<lx
                b=x[i+1][k]>ux
                x[i+1][k][a]=lx[a]
                x[i+1][k][b]=ux[b]
            else:                           
                y=x[i][ir]-E*np.absolute(J*x[i][ir]-xm)
                y[:Ngen]=np.round(y[:Ngen])
                z=y+(s*lf)
                z[:Ngen]=np.round(z[:Ngen])
                a=y<lx
                b=y>ux
                y[a]=lx[a]
                y[b]=ux[b]
                a=z<lx
                b=z>ux
                z[a]=lx[a]
                z[b]=ux[b]
                net1 = copy.deepcopy(net)
                for kk in range(Ngen):
                   net1.load.p_mw[y[kk]-1]=net1.load.p_mw[y[kk]-1]-y[kk+3]
                [pg,qg,fy]=pf(net1)
                net1 = copy.deepcopy(net)
                for kk in range(Ngen):
                   net1.load.p_mw[z[kk]-1]=net1.load.p_mw[z[kk]-1]-y[kk+3]
                [pg,qg,fz]=pf(net1)
                if fx[i][k]>fy:
                    x[i+1][k]=y                    
                elif fx[i][k]>fz:
                    x[i+1][k]=z                    
                else:
                    x[i+1][k]=x[i][k]
                  
for k in range(Nh):
    net1 = copy.deepcopy(net)
    for kk in range(Ngen):
       net1.load.p_mw[x[Ni][k][kk]-1]=net1.load.p_mw[x[Ni][k][kk]-1]-x[Ni][k][kk+3]
    [pg,qg,fx[Ni][k]]=pf(net1)

g=np.min(fx[Ni])
i=np.argmin(fx[Ni])
xg=x[Ni][i]

print('The solution is ')
for i in range(Nx):
      print('x'+str(i+1),'= ',str(xg[i]))
print('The objective function takes the value fo=',str(g))    