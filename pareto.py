# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:23:27 2023

@author: diego
"""

import numpy as np

def pareto(F):
    A=[]
    N=np.size(F,0)
    No=np.size(F,1)
    for i in range(N):
        j=0
        while j<N:
            k=0
            if not i==j:
                for l in range (No-1):
                    if F[i][l]>=F[j][l] and F[i][l+1]>F[j][l+1]:
                        k+=1/(No-1)                        
                    elif F[i][l]>F[j][l] and F[i][l+1]>=F[j][l+1]:
                        k+=1/(No-1)                        
                if k<1:
                    j+=1
                else:
                    j=N
            else:
                j+=1
        if k<1:
            A+=[i]
    A=np.asarray(A)        
    return A

# b1=np.array([[0.1,0.5,0.7,0.05,0.3,0.1,0.1],[2,1,3,1,4,2,3],[11,13,15,11,10,11,11]])
# b1=b1.T
# pf1=pareto(b1)

# b2=np.array([[0.04,0.07,0.09,0,0.69,0],[236.33,192.99,182.81,205.54,135.54,192.91]])
# b2=b2.T
# pf2=pareto(b2)


