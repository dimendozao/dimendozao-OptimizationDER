# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:57:48 2023

@author: diego

HHO Implementation for the zdt2
"""
import numpy as np
from scipy.special import gamma
#from ackley import ackley as fo
#from beale import beale as fo
from easom import easom as fo

Nh=100
Ni=100
Nx=2

E=0
B=1.5

x=np.zeros([Ni+1,Nh,Nx])
lx=np.ones(Nx)*-100
ux=np.ones(Nx)*100
fx=np.zeros([Ni+1,Nh])
fy=0
fz=0


for i in range(Nh):
    x[0][i]=lx+(np.random.rand(Nx)*(ux-lx))

for i in range(Ni):
    print("Progress= ",str(i*100/(Ni-1)),"%")
    for k in range(Nh):
        fx[i][k]=fo(x[i][k],lx,ux)
    
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
                # a=x[i+1][k]<lx
                # b=x[i+1][k]>ux
                # x[i+1][k][a]=lx[a]
                # x[i+1][k][b]=ux[b]
            else:
                x[i+1][k]=(x[i][ir]-xm)-(r3*(lx+(r4*(ux-lx))))
                # a=x[i+1][k]<lx
                # b=x[i+1][k]>ux
                # x[i+1][k][a]=lx[a]
                # x[i+1][k][b]=ux[b]
        elif np.absolute(E)>=0.5:
            if r>=0.5:                   
                x[i+1][k]=dx-(E*np.absolute(J*x[i][ir]-x[i][k]))
                # a=x[i+1][k]<lx
                # b=x[i+1][k]>ux
                # x[i+1][k][a]=lx[a]
                # x[i+1][k][b]=ux[b]
            else:                           
                y=x[i][ir]-(E*np.absolute(J*x[i][ir]-x[i][k]))
                z=y+(s*lf)
                fy=fo(y,lx,ux)
                fz=fo(z,lx,ux)
                if fx[i][k]>fy: 
                    x[i+1][k]=y
                    # a=x[i+1][k]<lx
                    # b=x[i+1][k]>ux
                    # x[i+1][k][a]=lx[a]
                    # x[i+1][k][b]=ux[b]
                elif fx[i][k]>fz:
                    x[i+1][k]=z
                    # a=x[i+1][k]<lx
                    # b=x[i+1][k]>ux
                    # x[i+1][k][a]=lx[a]
                    # x[i+1][k][b]=ux[b]
                else:
                    x[i+1][k]=x[i][k]
                    # a=x[i+1][k]<lx
                    # b=x[i+1][k]>ux
                    # x[i+1][k][a]=lx[a]
                    # x[i+1][k][b]=ux[b]
        else:                
            if r>=0.5:                   
                x[i+1][k]=x[i][ir]-(E*np.absolute(dx))
                # a=x[i+1][k]<lx
                # b=x[i+1][k]>ux
                # x[i+1][k][a]=lx[a]
                # x[i+1][k][b]=ux[b]
            else:                           
                y=x[i][ir]-E*np.absolute(J*x[i][ir]-xm)
                z=y+(s*lf)
                fy=fo(y,lx,ux)
                fz=fo(z,lx,ux)
                if fx[i][k]>fy:
                    x[i+1][k]=y
                    # a=x[i+1][k]<lx
                    # b=x[i+1][k]>ux
                    # x[i+1][k][a]=lx[a]
                    # x[i+1][k][b]=ux[b]
                elif fx[i][k]>fz:
                    x[i+1][k]=z
                    # a=x[i+1][k]<lx
                    # b=x[i+1][k]>ux
                    # x[i+1][k][a]=lx[a]
                    # x[i+1][k][b]=ux[b]
                else:
                    x[i+1][k]=x[i][k]
                    # a=x[i+1][k]<lx
                    # b=x[i+1][k]>ux
                    # x[i+1][k][a]=lx[a]
                    # x[i+1][k][b]=ux[b]
        
                        
for k in range(Nh):
    fx[Ni]=fo(x[Ni][k],lx,ux)

g=np.min(fx[Ni])
i=np.argmin(fx[Ni])
xg=x[Ni][i] 
print('The solution is ')
for i in range(Nx):
      print('x'+str(i+1),'= ',str(xg[i]))
print('The objective function takes the value fo=',str(g))
