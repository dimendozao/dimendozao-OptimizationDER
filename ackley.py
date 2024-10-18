# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:45:59 2023

@author: diego
"""

import numpy as np
def ackley(x,lx,ux):
    Out=0
    a=np.square(x[0])+np.square(x[1])
    b=np.sqrt(0.5*a)
    c=-20*np.exp(-0.2*b)
    d=np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])
    e=np.exp(0.5*d)
    Out=c-e+np.exp(1)+20
    Out=Out+sum(x<lx)+sum(x>ux)
    return Out