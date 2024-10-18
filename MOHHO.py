# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:36:57 2023

@author: diego


MOHHO Implementation for the zdt2
"""
import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
from pareto import pareto
from zdt2 import ZDT2 as fo
import os


Nh=200
Ni=100
Nx=30
No=2


E=0
B=1.5

x=np.zeros([Ni+1,Nh,Nx])
lx=np.zeros(Nx)
ux=np.ones(Nx)
fx=np.zeros([Ni+1,Nh,No])
fy=np.zeros(2)
fz=np.zeros(2)

llx=np.tile(lx,(Nh,1))
uux=np.tile(ux,(Nh,1))

stop=0
for i in range(Nh):
    x[0][i]=lx+(np.random.uniform(0,1,Nx)*(ux-lx))

for i in range(Ni):
    print("Progress= ",str(i*100/(Ni-1)),"%")
    for k in range(Nh):
        fx[i][k]=fo(x[i][k],lx,ux)
    ax=plt.scatter(fx[i,:,0],fx[i,:,1]) 
    plt.show()
    fp=pareto(fx[i])
    Nfp=np.size(fp)
    if Nfp==Nh:
        stop=1
        break
    for k in range(Nh):
        if sum(k==fp)==0:
            E0=(2*np.random.rand())-1
            E=2*E0*(1-(i/(Ni-1)))
            r1=np.random.rand(Nx)
            r2=np.random.rand(Nx)
            r3=np.random.rand(Nx)
            r4=np.random.rand(Nx)
            r=np.random.rand()
            r5=np.random.rand(Nx)
            J=2*(1-r5)
            xm=np.mean(x[i],axis=0)
            s=np.random.rand(Nx)
            sig=np.power(((gamma(1+B)*np.sin(np.pi*B/2))/(gamma((1+B)/2)*B*np.power(2,(B-1)/2))),(1/B))
            u=np.random.rand(Nx)
            v=np.random.rand(Nx)            
            lf=0.01*u*sig/(np.power(np.absolute(v),(1/B)))
            dxfp=np.linalg.norm(fx[i][fp]-fx[i][k],axis=1)
            ir=fp[np.argmin(dxfp)]
            dx=x[i][ir]-x[i][k]        
            if np.absolute(E)>=1:            
                q=np.random.rand()
                if q>=0.5:                    
                    ir=np.random.randint(0,Nh)
                    x[i+1][k]=x[i][ir]-(r1*np.absolute(x[i][ir]-(2*r2*x[i][k])))                    
                else:
                    x[i+1][k]=(x[i][ir]-xm)-(r3*(lx+(r4*(ux-lx))))                   
            elif np.absolute(E)>=0.5:
                if r>=0.5:                   
                    x[i+1][k]=dx-(E*np.absolute(J*x[i][ir]-x[i][k]))                    
                else:                           
                    y=x[i][ir]-(E*np.absolute(J*x[i][ir]-x[i][k]))
                    z=y+(s*lf)
                    fy=fo(y,lx,ux)
                    fz=fo(z,lx,ux)
                    if fx[i][k][0]>fy[0] and fx[i][k][1]>fy[1]:
                        x[i+1][k]=y                        
                    elif fx[i][k][0]>fz[0] and fx[i][k][1]>fz[1]:
                        x[i+1][k]=z                        
                    else:
                        x[i+1][k]=x[i][k]                        
            else:                
                if r>=0.5:                   
                    x[i+1][k]=x[i][ir]-(E*np.absolute(dx))                    
                else:                           
                    y=x[i][ir]-E*np.absolute(J*x[i][ir]-xm)
                    z=y+(s*lf)
                    fy=fo(y,lx,ux)
                    fz=fo(z,lx,ux)
                    if (fx[i][k][0]>=fy[0] and fx[i][k][1]>fy[1]) or (fx[i][k][0]>fy[0] and fx[i][k][1]>=fy[1]):
                        x[i+1][k]=y                        
                    elif (fx[i][k][0]>=fz[0] and fx[i][k][1]>fz[1]) or (fx[i][k][0]>fz[0] and fx[i][k][1]>=fz[1]):
                        x[i+1][k]=z                        
                    else:
                        x[i+1][k]=x[i][k]                        
        else:
            x[i+1][k]=x[i][k]
        
    a=x[i+1]<llx
    b=x[i+1]>uux
    x[i+1][a]=llx[a]
    x[i+1][b]=uux[b]
    
                     
           
            
                        
if stop==0:
    for k in range(Nh):
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
                    
                    
            
        

