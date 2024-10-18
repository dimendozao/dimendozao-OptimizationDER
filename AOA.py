# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:19:23 2023

@author: diego
"""

import numpy as np
#from ackley import ackley as fo
from beale import beale as fo
#from easom import easom as fo

Nh=100
Ni=100
Nx=2
alpha=3
mu=0.9
eps=1

x=np.zeros([Ni+1,Nh,Nx])
lx=np.ones(Nx)*-4.5
ux=np.ones(Nx)*4.5
fx=np.zeros([Ni+1,Nh])
moamin=0.2
moamax=0.9
moa=0
mop=0
xbest=np.zeros(Nx)
fxbest=0

for i in range(Nh):
    x[0][i]=lx+(np.random.rand(Nx)*(ux-lx))

for i in range(Ni):
    print("Progress= ",str(i*100/(Ni-1)),"%")
    for k in range(Nh):
        fx[i][k]=fo(x[i][k],lx,ux)
    ibest=np.argmin(fx[i])
    xbest=x[i][ibest]
    gbest=np.min(fx[i])
    moa=moamin+((i+1)*(moamax-moamin)/Ni)
    mop=1-(np.power((i+1),(1/alpha))/np.power(Ni,(1/alpha)))
    for k in range(Nh):
        for l in range(Nx):
            r1=np.random.uniform(0,1)
            r2=np.random.uniform(0,1)
            r3=np.random.uniform(0,1)
            if r1>moa:
                if r2>0.5:
                    x[i+1][k][l]=xbest[l]*mop*(lx[l]+(ux[l]-lx[l])*mu)
                else:
                    x[i+1][k][l]=xbest[l]/((mop+eps)*(lx[l]+(ux[l]-lx[l])*mu))
                    if np.isinf(x[i+1][k][l]):
                        x[i+1][k][l]=x[i][k][l]
            else:
                if r3>0.5:
                    x[i+1][k][l]=xbest[l]+mop*(lx[l]+(ux[l]-lx[l])*mu)
                else:
                    x[i+1][k][l]=xbest[l]-mop*(lx[l]+(ux[l]-lx[l])*mu)

for k in range(Nh):
    fx[Ni]=fo(x[Ni][k],lx,ux)

g=np.min(fx[Ni])
i=np.argmin(fx[Ni])
xg=x[Ni][i] 
print('The solution is ')
for i in range(Nx):
      print('x'+str(i+1),'= ',str(xg[i]))
print('The objective function takes the value fo=',str(g))