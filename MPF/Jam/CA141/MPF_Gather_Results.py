# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:04:49 2024

@author: diego
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat


form=['PM','RM','RRM','PBIM','RBIM','RRBIM','BF','RERCM','RERRM','RERCBIM','RERRBIM','REBF','MAT']
conv=['Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Convex','Convex','Convex','Convex','Convex','Matpower']
fold=['YMatrix','YMatrix','YMatrix','BIM','BIM','BIM','BF','YMatrix','YMatrix','BIM','BIM','BF','']
case='CA141'
problem='MPF'
city='Jam'
city1='JAM'


mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\ClusterMeans_'+city1+'.mat')
means=mat['clustermeans']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=mat['clusternode']

branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case+"Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case+"Bus.csv")

num_lines = len(branch)
num_nodes=len(bus)

iref=np.where(bus['type']==3)[0][0]

H=np.size(means,axis=1)

sd=np.zeros([H,num_nodes],dtype='complex')

for h in range(H):
    for k in range(num_lines):
        sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*means[cnode[branch['j'][k]-1][0]-1][h]
    
pdm=np.real(sd)
qdm=np.imag(sd)

ym=np.zeros([num_nodes,num_nodes],dtype='complex')
for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])

nform=len(form)

datadict={}

for i in range(nform):
    if i<nform-1 and not i==6 and not i==11:
        datadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city+'\\'+case+'\\'+conv[i]+'\\'+fold[i]+'\\'+form[i]+'\\Results.xlsx',index_col=0)})
    elif i==6 or i==11:
        datadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city+'\\'+case+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx',index_col=0)})
    else:    
        datadict.update({form[i]:pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city+'\\'+case+'\\'+conv[i]+'\\Results.csv')})

"----- Read the results -----"

diffs=['$\\Delta v$','$\\Delta \\phi$','$\\Delta P^{g}$','$\\Delta Q^{g}$','$\\Delta P_{loss}$','$\\Delta Q_{loss}$']
ov=['$\\overline{SSE}$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']


vs=datadict[form[nform-1]].columns

ndiffs=len(diffs)
nov=len(ov)

results=np.zeros([nform,ndiffs])
times=np.zeros(nform)
solvers=['*']*nform
infeas1=np.zeros(nform)
infeas2=np.zeros(nform)

varsn=datadict['BF'].columns

v=np.zeros([nform,H,num_nodes])
ph=np.zeros([nform,H,num_nodes])
pl=np.zeros([nform,H])
ql=np.zeros([nform,H])
pg=np.zeros([nform,H])
qg=np.zeros([nform,H])
eqp=np.zeros([nform-1,H,num_nodes])
eqq=np.zeros([nform-1,H,num_nodes])

gvect=np.zeros(num_nodes)
gvect[iref]=1

for i in range(nform):
    for h in range(H):
        for j in range(num_nodes):
            v[i][h][j]=datadict[form[i]][varsn[j]][h]
            ph[i][h][j]=datadict[form[i]][varsn[j+num_nodes]][h]
            if i<nform-1:
                eqp[i][h][j]=datadict[form[i]][varsn[j+(2*num_nodes)]][h]
                eqq[i][h][j]=datadict[form[i]][varsn[j+(3*num_nodes)]][h]
        
    pg[i]=datadict[form[i]]['pg'].to_numpy()
    qg[i]=datadict[form[i]]['qg'].to_numpy()
    pl[i]=datadict[form[i]]['pl'].to_numpy()
    ql[i]=datadict[form[i]]['ql'].to_numpy()
             
                
for i in range(nform):
    if i<nform-1:
        results[i][0]=np.sum(np.square(v[i]-v[nform-1]))
        results[i][1]=np.sum(np.square(ph[i]-ph[nform-1]))
        results[i][2]=np.sum(np.square(pg[i]-pg[nform-1]))
        results[i][3]=np.sum(np.square(qg[i]-qg[nform-1]))
        results[i][4]=np.sum(np.square(pl[i]-pl[nform-1]))
        results[i][5]=np.sum(np.square(ql[i]-ql[nform-1]))
    else:
        results[i]=0

for j in range(nform):
    times[j]=datadict[form[j]]['t'][0]
    if not j==nform-1:
        solvers[j]=datadict[form[j]]['Solver'][0]       

resultsdf=pd.DataFrame(index=form,columns=diffs)
overviewdf=pd.DataFrame(index=form,columns=ov)

for h in range(H):
    for i in range(nform):
        if i<nform-1:
            infeas1[i]+=np.sum(np.abs((pg[i][h]*gvect)-eqp[i][h]-pdm[h]))+np.sum(np.abs((qg[i][h]*gvect)-eqq[i][h]-qdm[h]))
            vmat=np.multiply(v[i][h],np.cos(ph[i][h]))+1j*np.multiply(v[i][h],np.sin(ph[i][h]))
            infeas2[i]+=np.sum(np.abs((pg[i][h]*gvect)-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdm[h]))+np.sum(np.abs((qg[i][h]*gvect)-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdm[h]))
        else:
            vmat=np.multiply(v[i][h],np.cos(ph[i][h]))+1j*np.multiply(v[i][h],np.sin(ph[i][h]))
            infeas1[i]+=np.sum(np.abs((pg[i][h]*gvect)-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdm[h]))+np.sum(np.abs((qg[i][h]*gvect)-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdm[h]))
            infeas2[i]=infeas1[i]
            
        
means=np.mean(results,axis=1)
mtt=np.multiply(np.abs(times),np.abs(means))
mtt=np.multiply(mtt,np.abs(infeas1))
mtt=np.multiply(mtt,np.abs(infeas2))
overview=np.zeros([nform,nov])

for i in range(nform):
    for j in range(ndiffs):
        resultsdf[diffs[j]][form[i]]='$'+np.format_float_scientific(results[i][j],precision=2)+'$'

idxi1=infeas1>1e-3
idxi2=infeas2>1e-3

idxd=np.argsort(means[:nform-1])
idxt=np.argsort(times)
idxinf1=np.argsort(infeas1+(100*idxi1)+(100*idxi2))
idxinf2=np.argsort(infeas2+(100*idxi1)+(100*idxi2))
idxpx=np.argsort(mtt[:nform-1])

for i in range(nform):
    for j in range(ndiffs):
        resultsdf[diffs[j]][form[i]]='$'+np.format_float_scientific(results[i][j],precision=2)+'$'


for i in range(nform):
    for j in range(nov):
        if j==0:
            if i==nform-1:
                overview[i][j]=means[i]
                overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(means[i], precision=2)+'$'                    
            else:
                overview[i][j]=means[i]
                ii=np.where(idxd==i)[0][0]
                if ii==0:
                    overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_scientific(means[i], precision=2)+'$'
                elif ii==1:
                    overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_scientific(means[i], precision=2)+'$'
                elif ii==2:
                    overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_scientific(means[i], precision=2)+'$'
                else:
                    overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(means[i], precision=2)+'$'
        elif j==1:
            overview[i][j]=times[i]
            ii=np.where(idxt==i)[0][0]
            if ii==0:
                overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_scientific(times[i], precision=2)+'$'
            elif ii==1:
                overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_scientific(times[i], precision=2)+'$'
            elif ii==2:
                overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_scientific(times[i], precision=2)+'$'
            else:
                overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(times[i], precision=2)+'$'
        elif j==2:
            overview[i][j]=infeas1[i]
            ii=np.where(idxinf1==i)[0][0]
            if ii==0:
                overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_scientific(infeas1[i], precision=2)+'$'
            elif ii==1:
                overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_scientific(infeas1[i], precision=2)+'$'
            elif ii==2:
                overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_scientific(infeas1[i], precision=2)+'$'
            else:
                overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(infeas1[i], precision=2)+'$'
        elif j==3:
            overview[i][j]=infeas2[i]
            ii=np.where(idxinf2==i)[0][0]
            if ii==0:
                overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_scientific(infeas2[i], precision=2)+'$'
            elif ii==1:
                overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_scientific(infeas2[i], precision=2)+'$'
            elif ii==2:
                overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_scientific(infeas2[i], precision=2)+'$'
            else:
                overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(infeas2[i], precision=2)+'$'                
        else:
            if i==nform-1:
                overview[i][j]=mtt[i]
                overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(mtt[i], precision=2)+'$'
            else:
                overview[i][j]=mtt[i]                
                ii=np.where(idxpx==i)[0][0]
                if ii==0:
                    overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_scientific(mtt[i], precision=2)+'$'
                elif ii==1:
                    overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_scientific(mtt[i], precision=2)+'$'
                elif ii==2:
                    overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_scientific(mtt[i], precision=2)+'$'
                else:
                    overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(mtt[i], precision=2)+'$'


overviewdf.insert(0,'I/S',solvers)
        
latexresults = resultsdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|') 
latexoverview = overviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|')           
        
file = open('Results_tab.txt','w')
file.write(latexresults)
file.close() 

file = open('Overview_tab.txt','w')
file.write(latexoverview)
file.close()                 
            

diffs1=['dv','dph','dpg','dqg','dpl','dql']
ov1=['$\\overline{SSE}$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
resultsdf1=pd.DataFrame(results,index=form,columns=diffs1)
overviewdf1=pd.DataFrame(overview,index=form,columns=ov1)
resultsdf1.to_csv('matdifs.csv')
overviewdf1.to_csv('overview.csv')

latexcase=['\\multirow{3}{*}{'+case+'}']

bestsse=pd.DataFrame([[ov1[0],form[idxd[0]],solvers[idxd[0]],'$\\cellcolor{green}'+np.format_float_scientific(overview[idxd[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][4], precision=2)+'$']],index=latexcase,columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
bestt=pd.DataFrame([[ov1[1],form[idxt[0]],solvers[idxt[0]],'$'+np.format_float_scientific(overview[idxt[0]][0], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxt[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
bestinf1=pd.DataFrame([[ov1[2],form[idxinf1[0]],solvers[idxinf1[0]],'$'+np.format_float_scientific(overview[idxinf1[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][1], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf1[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
bestinf2=pd.DataFrame([[ov1[3],form[idxinf2[0]],solvers[idxinf2[0]],'$'+np.format_float_scientific(overview[idxinf2[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][2], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf2[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
bestpx=pd.DataFrame([[ov1[4],form[idxpx[0]],solvers[idxpx[0]],'$'+np.format_float_scientific(overview[idxpx[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][3], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxpx[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])


bestresult=pd.concat([bestsse,bestt,bestinf1,bestinf2,bestpx])
bestresult.to_csv('best.csv')