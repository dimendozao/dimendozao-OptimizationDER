# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:54:27 2023

@author: diego
"""

import numpy as np
import pandas as pd

from pareto import pareto


data = pd.read_excel("datapareto.xlsx", index_col=None, header=None)


a=data.to_numpy()
f1=np.zeros([2,100])
f2=np.zeros([2,100])

f1i=np.arange(0,100)
f1[0]=np.array(a[4][1:])
f1[1]=np.array(a[5][1:])

f2i=np.arange(0,100)
f2[0]=np.array(a[24][1:])
f2[1]=np.array(a[25][1:])

f1=f1.T
f2=f2.T

pf1={}
pf2={}


N1=np.size(f1,0)
N2=np.size(f2,0)

m=0
l=0
while l<N1:
    if m==0:
        pf1[m]=pareto(f1)
    else:
        nf1=len(pf1)
        a=np.ones(N1,dtype=bool)
        for i in range(nf1):
            npf1=np.size(pf1[i])
            for j in range(npf1):
                a[pf1[i][j]]=False
        pf1[m]=pareto(f1[a])
        f1i_alt=f1i[a]
        for i in range(np.size(pf1[m])):            
            pf1[m][i]=f1i_alt[pf1[m][i]]
    l+=np.size(pf1[m])
    m+=1

m=0
l=0
while l<N2:
    if m==0:
        pf2[m]=pareto(f2)
    else:
        nf2=len(pf2)
        a=np.ones(N2,dtype=bool)
        for i in range(nf2):
            npf2=np.size(pf2[i])
            for j in range(npf2):
                a[pf2[i][j]]=False
        pf2[m]=pareto(f2[a])
        f2i_alt=f2i[a]
        for i in range(np.size(pf2[m])):            
            pf2[m][i]=f2i_alt[pf2[m][i]]
    l+=np.size(pf2[m])
    m+=1


