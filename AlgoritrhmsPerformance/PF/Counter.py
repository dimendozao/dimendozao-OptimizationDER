# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 07:51:47 2024

@author: diego
"""

import pandas as pd
import numpy as np


form=['PM','RM','RRM','PBIM','RBIM','RRBIM','BF','RERCM','RERRM','RERCBIM','RERRBIM','REBF','MAT']
conv=['Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Convex','Convex','Convex','Convex','Convex','MAT']
fold=['YMatrix','YMatrix','YMatrix','BIM','BIM','BIM','BF','YMatrix','YMatrix','BIM','BIM','BF','Matpower']
cases=['IEEE33','IEEE69','SA','CA141']
problems=['PF','PFN','MPF']
case=['IEEE33','IEEE69','SA','CA141']
city=['Bog','Jam','Pop']
city1=['*','Bog','Jam','Pop']


nform=len(form)
ncases=len(cases)
nprob=len(problems)
ncity=len(city)


datadict={}

for i in range(nprob):
    if i==0:
        datadict[(problems[i],city1[0])]=pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problems[i]+'\\best.csv',header=0,index_col=False)    
    else:
        for j in range(ncity):
            datadict[(problems[i],city[j])]=pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problems[i]+'\\'+city[j]+'\\best.csv',header=0,index_col=False)   
            
ncity1=len(city1)
nbest=len(datadict[(problems[-1],city1[1])]['F'])


cols=['$\overline{SSE}$','Time','Inf1','Inf2','$\\Sigma$']
ncols=len(cols)

cols1=datadict[(problems[-1],city[-1])].columns
cols2=cols1.drop(cols1[:4])
cols2=cols2.drop(cols1[-1]).to_list()
ncols2=len(cols2)

mux = pd.MultiIndex.from_arrays([conv,
                                  form], 
                                  names=['Type','F'])
        
nmform=pd.DataFrame(np.zeros([nform,ncols]),index=mux,columns=cols)
sollist=[]
solconv=[]

for i in range(nprob):
    if i==0:
        for k in range(nbest):
            if datadict[(problems[i],city1[0])]['$x$'][k]!='$\\Pi |x|$':
                idxf=form.index(datadict[(problems[i],city1[0])]['F'][k])
                nmform[cols[4]][conv[idxf],form[idxf]]+=1
                idxx=cols2.index(datadict[(problems[i],city1[0])]['$x$'][k])
                nmform[cols[idxx]][conv[idxf],form[idxf]]+=1          
                            
                if datadict[(problems[i],city1[0])]['I/S'][k]=='*':
                    datadict[(problems[i],city1[0])]['I/S'][k]='MAT'
                    
                sollist.append(datadict[(problems[i],city1[0])]['I/S'][k])
                solconv.append(conv[idxf])               
        
    else:    
        for j in range(ncity):
            for k in range(nbest):
                if datadict[(problems[i],city[j])]['$x$'][k]!='$\\Pi |x|$':
                    idxf=form.index(datadict[(problems[i],city[j])]['F'][k])
                    nmform[cols[4]][conv[idxf],form[idxf]]+=1
                    idxx=cols2.index(datadict[(problems[i],city[j])]['$x$'][k])
                    nmform[cols[idxx]][conv[idxf],form[idxf]]+=1
                                        
                    if datadict[(problems[i],city[j])]['I/S'][k]=='*':
                        datadict[(problems[i],city[j])]['I/S'][k]='MAT'
                           
                    sollist.append(datadict[(problems[i],city[j])]['I/S'][k])
                    solconv.append(conv[idxf])
                    
                    
                    
                            

solindex=[]

setsols=set(sollist)
sols=list(setsols)

nsollist=len(sollist)
nsols=len(sols)

solconvindex=['']*nsols


for i in range(nsols):
     idxs1=sollist.index(sols[i])
     solconvindex[i]=solconv[idxs1]

mux = pd.MultiIndex.from_arrays([solconvindex,
                                  sols], 
                                  names=['Type','I/S'])
nmsols=pd.DataFrame(np.zeros([nsols,ncols]),index=mux,columns=cols)

for i in range(nprob):
    if i==0:
        for k in range(nbest):
            if datadict[(problems[i],city1[0])]['$x$'][k]!='$\\Pi |x|$':
                idxs=sols.index(datadict[(problems[i],city1[0])]['I/S'][k])
                nmsols[cols[4]][solconvindex[idxs],sols[idxs]]+=1
                idxx=cols2.index(datadict[(problems[i],city1[0])]['$x$'][k])
                nmsols[cols[idxx]][solconvindex[idxs],sols[idxs]]+=1
    else:        
        for j in range(ncity):
            for k in range(nbest):
                if datadict[(problems[i],city[j])]['$x$'][k]!='$\\Pi |x|$':
                    idxs=sols.index(datadict[(problems[i],city[j])]['I/S'][k])
                    nmsols[cols[4]][solconvindex[idxs],sols[idxs]]+=1
                    idxx=cols2.index(datadict[(problems[i],city[j])]['$x$'][k])
                    nmsols[cols[idxx]][solconvindex[idxs],sols[idxs]]+=1
            



groupedform = nmform.groupby(['Type','F']).sum()
groupedsols = nmsols.groupby(['Type','I/S']).sum()
    
latexnmforms = groupedform.to_latex(index=True, escape=False, column_format='|c| c c c c c c|',float_format="%.0f")
latexnmsols = groupedsols.to_latex(index=True, escape=False, column_format='|c| c c c c c c|',float_format="%.0f")
 
file = open('Forms_Performance_tab.txt','w')
file.write(latexnmforms)
file.close() 

file = open('Solvs_Performance_tab.txt','w')
file.write(latexnmsols)
file.close()


