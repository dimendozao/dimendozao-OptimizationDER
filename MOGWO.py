# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:07:43 2023

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
from pareto import pareto
from zdt2 import ZDT2 as fo

Nw=100
Ni=100
Nx=2
No=2
Nwp=10

x=np.zeros([Ni+1,Nw,Nx])
lx=np.zeros(Nx)
ux=np.ones(Nx)
fx=np.zeros([Ni+1,Nw,No])

idxabc=np.zeros(3,dtype=int)
kk=0

for i in range(Nw):
    x[0][i]=lx+(np.random.rand(Nx)*(ux-lx))

for i in range(Ni):
    print("Progress= ",str(i*100/(Ni-1)),"%")
    for k in range(Nw):
        fx[i][k]=fo(x[i][k],lx,ux)
    
    ax=plt.scatter(fx[i,:,0],fx[i,:,1]) 
    plt.show()
    fp=pareto(fx[i])
    Nfp=np.size(fp)
    ffp=np.zeros([Nfp,No])
    a=2-(i*2/Ni)
    for k in range(Nfp):
        ffp[k]=fx[i][fp[k]]
       
    ffpa=np.argsort(ffp,axis=0)
    if Nfp<Nw:
        idxabc[0]=fp[ffpa[0][0]]
        idxabc[1]=fp[ffpa[0][1]]
        minf=np.min(ffp,axis=0)
        df=np.linalg.norm(ffp-minf,axis=1)
        idx=np.argmin(df)
        idxabc[2]=fp[idx]
        xa=x[i][idxabc[0]]
        xb=x[i][idxabc[1]]
        xc=x[i][idxabc[2]]
        for k in range(Nw):
            if sum(k==idxabc)==0:
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
                dc=np.absolute((c3*xc)-x[i][k])
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
    else:
        kk=1
        break
 
    
if kk==0:
    for k in range(Nw):
        fx[Ni][k]=fo(x[Ni][k],lx,ux)        
    fp=pareto(fx[Ni])
    fs1=np.zeros(np.size(fp))
    fs2=np.zeros(np.size(fp))
    for i in range(np.size(fp)):
        fs1[i]=fx[Ni][fp[i]][0]
        fs2[i]=fx[Ni][fp[i]][1]
    plt.scatter(fs1,fs2) 
else:
    fs1=np.zeros(Nfp)
    fs2=np.zeros(Nfp)    
    for k in range(Nfp):
        fs1[k]=fx[i][fp[k]][0]
        fs2[k]=fx[i][fp[k]][1]
    plt.scatter(fs1,fs2) 
    