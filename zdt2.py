# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:22:07 2023

@author: diego
"""
import numpy as np
def ZDT2(x,lx,ux):
    Out=np.zeros(2)
    N=np.size(x)
    a=sum(x<lx)
    b=sum(x>ux)
    Out[0]=x[0]+a+b
    g=1+(9*np.sum(x[1:])/(N-1))
    Out[1]=(g*(1-np.square(x[0]/g)))+a+b    
    return Out