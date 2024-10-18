# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:06:31 2023

@author: diego
"""

import numpy as np
def beale(x,lx,ux):
    Out=0
    a=sum(x<lx)
    b=sum(x>ux)
    Out=np.square(1.5-x[0]+(x[0]*x[1]))+np.square(2.25-x[0]+(x[0]*np.square(x[1])))+np.square(2.625-x[0]+(x[0]*x[1]*np.square(x[1])))+a+b    
    return Out