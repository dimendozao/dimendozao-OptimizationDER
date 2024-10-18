# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:04:49 2024

@author: diego
"""

import pandas as pd
import numpy as np


form=['PM','RM','RRM','PBIM','RBIM','RRBIM','BF','RERCM','RERRM','RERCBIM','RERRBIM','REBF','MAT']
conv=['Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Convex','Convex','Convex','Convex','Convex','Matpower']
fold=['YMatrix','YMatrix','YMatrix','BIM','BIM','BIM','BF','YMatrix','YMatrix','BIM','BIM','BF','']
cases=['IEEE33','IEEE69','SA','CA141']
problem='PFN'
city='Pop'


nform=len(form)
ncases=len(cases)

datadict={}

for i in range(ncases):
    if i==0:
        pfdf=pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city+'\\'+cases[i]+'\\best.csv',header=0,index_col=0)
    else:
        pfdf=pd.concat([pfdf,pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city+'\\'+cases[i]+'\\best.csv',header=0,index_col=0)])
        

pfdf.to_csv('best.csv')

latexresults = pfdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c c c|')
 
file = open('MPF_Results_tab.txt','w')
file.write(latexresults)
file.close() 