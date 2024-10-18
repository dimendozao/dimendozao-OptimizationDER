# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:27:07 2023

@author: diego
"""

import numpy as np
def easom(x,lx,ux):
    Out=0
    a=sum(x<lx)
    b=sum(x>ux)
    Out=-np.cos(x[0])*np.cos(x[1])*np.exp(-(np.square(x[0]-np.pi)+np.square(x[1]-np.pi)))
    Out=Out+a+b
    return Out