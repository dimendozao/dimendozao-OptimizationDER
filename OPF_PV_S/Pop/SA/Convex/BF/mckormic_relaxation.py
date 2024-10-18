# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:39:57 2024

@author: diego
"""

import cvxpy as cvx
import numpy as np


npar=20
xl=0
yl=0
xu=8
yu=4

xp=np.linspace(xl,xu,npar+1)
yp=np.linspace(yl,yu,npar+1)
xlp=xp[:npar]
ylp=yp[:npar]
xup=xp[1:]
yup=yp[1:]

zup=np.multiply(xup,yup)
zlp=np.multiply(xlp,ylp)

x=cvx.Variable(nonneg=True)
y=cvx.Variable(nonneg=True)
z=cvx.Variable(nonneg=True)


xp=cvx.Variable(npar,nonneg=True)
xp1=cvx.Variable(npar,nonneg=True)

yp=cvx.Variable(npar,nonneg=True)
yp1=cvx.Variable(npar,nonneg=True)

zp=cvx.Variable(npar,nonneg=True)
zp1=cvx.Variable(npar,nonneg=True)

b=cvx.Variable(npar,boolean=True)

con=[] 
for i in range(npar):       
    con +=[zp[i]>=xlp[i]*yp[i]+xp[i]*ylp[i]-xlp[i]*ylp[i]]
    con +=[zp[i]>=xup[i]*yp[i]+xp[i]*yup[i]-xup[i]*yup[i]]
    con +=[zp[i]<=xup[i]*yp[i]+xp[i]*ylp[i]-xup[i]*ylp[i]]
    con +=[zp[i]<=xp[i]*yup[i]+xlp[i]*yp[i]-xlp[i]*yup[i]]
    
    con += [xp1[i]<=b[i]*xup[i]]
    con += [xp1[i]<=xp[i]]
    con += [xp1[i]>=xp[i]-xup[i]*(1-b[i])]
    
    con += [yp1[i]<=b[i]*yup[i]]
    con += [yp1[i]<=yp[i]]
    con += [yp1[i]>=yp[i]-yup[i]*(1-b[i])]
    
    con += [zp1[i]<=b[i]*zup[i]]
    con += [zp1[i]<=zp[i]]
    con += [zp1[i]>=zp[i]-zup[i]*(1-b[i])]
    
   
    
    
    
con += [cvx.sum(xp1)==x]
con += [cvx.sum(yp1)==y]
con += [cvx.sum(zp1)==z]
con += [cvx.sum(b)==1]

obj=cvx.Minimize(cvx.square(z-10))

prob = cvx.Problem(obj,con)

prob.solve(solver=cvx.MOSEK,verbose=True)
