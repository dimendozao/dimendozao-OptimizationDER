# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:09:39 2023

@author: diego
"""

import numpy as np
#from ackley import ackley as fo
from beale import beale as fo
#from easom import easom as fo

Nw=100
Ni=100
Nx=2

x=np.zeros([Ni+1,Nw,Nx])
lx=np.ones(Nx)*-4.5
ux=np.ones(Nx)*4.5
fx=np.zeros([Ni+1,Nw])

ifs=np.zeros(Nw)
xa=np.zeros(Nw)
xb=np.zeros(Nw)
xc=np.zeros(Nw)
x1=np.zeros(Nw)
x2=np.zeros(Nw)
x3=np.zeros(Nw)
da=np.zeros(Nw)
db=np.zeros(Nw)
dc=np.zeros(Nw)
a1=np.zeros(Nw)
a2=np.zeros(Nw)
a3=np.zeros(Nw)

for i in range(Nw):
    x[0][i]=lx+(np.random.rand(Nx)*(ux-lx))

for i in range(Ni):
    print("Progress= ",str(i*100/(Ni-1)),"%")
    for k in range(Nw):
        fx[i][k]=fo(x[i][k],lx,ux)
    
    ifs=np.argsort(fx[i])
    xa=x[i][ifs[0]]
    xb=x[i][ifs[1]]
    xc=x[i][ifs[2]]
    
    a=2-(i*2/Ni)
    
    for k in range(Nw):
        if sum(k==ifs[:3])==0:
            r11=np.random.uniform(0,1,Nx)
            r12=np.random.uniform(0,1,Nx)
            r13=np.random.uniform(0,1,Nx)
            a1=(2*a*r11)-a
            a2=(2*a*r12)-a
            a3=(2*a*r13)-a
            r21=np.random.uniform(0,1,Nx)
            r22=np.random.uniform(0,1,Nx)
            r23=np.random.uniform(0,1,Nx)
            c1=2*r21
            c2=2*r22
            c3=2*r23   
            da=np.absolute((c1*xa)-x[i][k])
            db=np.absolute((c2*xb)-x[i][k])
            dc=np.absolute((c1*xc)-x[i][k])
            x1=xa-(a1*da)
            x2=xb-(a2*db)
            x3=xc-(a3*dc)
            x[i+1][k]=(x1+x2+x3)/3
            idx=x[i+1][k]<lx
            x[i+1][k][idx]=lx[idx]
            idx=x[i+1][k]>ux
            x[i+1][k][idx]=ux[idx]
        else:
            x[i+1][k]=x[i][k]

for k in range(Nw):
    fx[Ni]=fo(x[Ni][k],lx,ux)

g=np.min(fx[Ni])
i=np.argmin(fx[Ni])
xg=x[Ni][i] 
print('The solution is ')
for i in range(Nx):
      print('x'+str(i+1),'= ',str(xg[i]))
print('The objective function takes the value fo=',str(g))
        