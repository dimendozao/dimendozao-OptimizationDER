# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:44:56 2024

@author: diego
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat


form=['PM','RM','RRM','PBIM','RBIM','RRBIM','BF','RERCM','RERRM','RERCBIM','RERRBIM','REBF','MAT']
conv=['Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Convex','Convex','Convex','Convex','Convex','Matpower']
fold=['YMatrix','YMatrix','YMatrix','BIM','BIM','BIM','BF','YMatrix','YMatrix','BIM','BIM','BF','']
case=['IEEE33','IEEE69','SA','CA141']
problem='PFN'
case1=['IEEE33','IEEE69','SA_J23','CA141']
city=['Bog','Jam','Pop']
city1=['BOG','JAM','POP']

ncase=len(case)
ncity=len(city)

matsolcols=['$V_{max}$','$V_{min}$','$\\phi_{max}$','$\\phi_{min}$','$P^{g}$','$Q^{g}$','$P_{loss}$','$Q_{loss}$']
nmatsol=len(matsolcols)

for ct in range(ncity):
    matsol=[['']*nmatsol for i in range(ncase)]
    for cs in range(ncase):
        branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1[cs]+'Branch.csv')
        bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1[cs]+'Bus.csv')
        
    
        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\ClusterMeans_'+city1[ct]+'.mat')
        dmeans=np.squeeze(mat['clustermeans']).T
    
        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+case[cs]+'_'+city[ct]+'_'+'ClusterNode.mat')
        cnode=np.squeeze(mat['clusternode'])
    
        cnode[0]=1
        cnode=cnode-1
    
        dem=np.mean(dmeans,axis=0)
        
        num_lines = len(branch)
        num_nodes=len(bus)
        
        iref=np.where(bus['type']==3)[0][0]
        
        sd=np.zeros(num_nodes,dtype='complex')
        
        for k in range(num_lines):
            sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
        for i in range(num_nodes):
            sd[i]=sd[i]*dem[cnode[i]]       
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
                datadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\'+conv[i]+'\\'+fold[i]+'\\'+form[i]+'\\Results.xlsx',index_col=0)})
            elif i==6 or i==11:
                datadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx',index_col=0)})
            else:    
                datadict.update({form[i]:pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\'+conv[i]+'\\Results.csv')})
        
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
        
        resultsdf=pd.DataFrame(index=form,columns=diffs)
        overviewdf=pd.DataFrame(index=form,columns=ov)
        
        for i in range(nform):
            for j in range(ndiffs):
                if i<nform-1:
                    results[i][j]=np.sum(np.square(datadict[form[i]][vs[j]].to_numpy()-datadict[form[nform-1]][vs[j]].to_numpy()))
                else:
                    results[i][j]=0
        
        for j in range(nform):
            times[j]=datadict[form[j]]['t'][0]
            if not j==nform-1:
                solvers[j]=datadict[form[j]]['Solver'][0]
        
        for i in range(nform):
            if i<nform-1:
                infeas1[i]=np.sum(np.abs(datadict[form[i]]['pg'].to_numpy()-datadict[form[i]]['eqp'].to_numpy()-pdm))+np.sum(np.abs(datadict[form[i]]['qg'].to_numpy()-datadict[form[i]]['eqq'].to_numpy()-qdm))
                vmat=np.multiply(datadict[form[i]]['v'].to_numpy(),np.cos(datadict[form[i]]['ph'].to_numpy()))+1j*np.multiply(datadict[form[i]]['v'].to_numpy(),np.sin(datadict[form[i]]['ph'].to_numpy()))
                infeas2[i]=np.sum(np.abs(datadict[form[i]]['pg'].to_numpy()-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdm))+np.sum(np.abs(datadict[form[i]]['qg'].to_numpy()-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdm))        
            else:
                vmat=np.multiply(datadict[form[i]]['v'].to_numpy(),np.cos(datadict[form[i]]['ph'].to_numpy()))+1j*np.multiply(datadict[form[i]]['v'].to_numpy(),np.sin(datadict[form[i]]['ph'].to_numpy()))
                infeas1[i]=np.sum(np.abs(datadict[form[i]]['pg'].to_numpy()-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdm))+np.sum(np.abs(datadict[form[i]]['qg'].to_numpy()-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdm))
                infeas2[i]=infeas1[i]
            for j in range(ndiffs):
                resultsdf[diffs[j]][form[i]]='$'+np.format_float_scientific(results[i][j],precision=2)+'$'    
        
        means=np.mean(results,axis=1)
        
        
        idxi1=infeas1>1e-3
        idxi2=infeas2>1e-3
        
        mtt=np.multiply(np.abs(times),np.abs(means))
        mtt=np.multiply(mtt,np.abs(infeas1))
        mtt=np.multiply(mtt,np.abs(infeas2))
        
        
        idxd=np.argsort(means[:nform-1])
        idxt=np.argsort(times)
        idxinf1=np.argsort(infeas1+(100*idxi1)+(100*idxi2))
        idxinf2=np.argsort(infeas2+(100*idxi1)+(100*idxi2))
        idxpx=np.argsort(mtt[:nform-1])
        
        
        overview=np.zeros([nform,nov])
        
        
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
               
        matsol[cs][0]='$'+np.format_float_scientific(np.max(datadict[form[-1]]['v'].to_numpy()), precision=2)+'$'
        matsol[cs][1]='$'+np.format_float_scientific(np.min(datadict[form[-1]]['v'].to_numpy()), precision=2)+'$'
        matsol[cs][2]='$'+np.format_float_scientific(np.max(np.abs(datadict[form[-1]]['ph'].to_numpy())), precision=2)+'$'
        matsol[cs][3]='$'+np.format_float_scientific(np.min(np.abs(datadict[form[-1]]['ph'].to_numpy())), precision=2)+'$'
        matsol[cs][4]='$'+np.format_float_scientific(datadict[form[-1]]['pg'][0], precision=2)+'$'
        matsol[cs][5]='$'+np.format_float_scientific(datadict[form[-1]]['qg'][0], precision=2)+'$'
        matsol[cs][6]='$'+np.format_float_scientific(datadict[form[-1]]['pl'][0], precision=2)+'$'
        matsol[cs][7]='$'+np.format_float_scientific(datadict[form[-1]]['ql'][0], precision=2)+'$'              
                        
        overviewdf.insert(0,'I/S',solvers)
                
        latexresults = resultsdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|') 
        latexoverview = overviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|')           
                
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\Results_tab.txt','w')
        file.write(latexresults)
        file.close() 
        
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\Overview_tab.txt','w')
        file.write(latexoverview)
        file.close()                 
                    
        
        diffs1=['dv','dph','dpg','dqg','dpl','dql']
        ov1=['$\\overline{SSE}$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
        resultsdf1=pd.DataFrame(results,index=form,columns=diffs1)
        overviewdf1=pd.DataFrame(overview,index=form,columns=ov1)
        resultsdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\matdifs.csv')
        overviewdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\overview.csv')
        
        latexcase=['\\multirow{'+str(nov)+'}{*}{'+case[cs]+'}']
        
        bestsse=pd.DataFrame([[ov1[0],form[idxd[0]],solvers[idxd[0]],'$\\cellcolor{green}'+np.format_float_scientific(overview[idxd[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][4], precision=2)+'$']],index=latexcase,columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestt=pd.DataFrame([[ov1[1],form[idxt[0]],solvers[idxt[0]],'$'+np.format_float_scientific(overview[idxt[0]][0], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxt[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestinf1=pd.DataFrame([[ov1[2],form[idxinf1[0]],solvers[idxinf1[0]],'$'+np.format_float_scientific(overview[idxinf1[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][1], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf1[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestinf2=pd.DataFrame([[ov1[3],form[idxinf2[0]],solvers[idxinf2[0]],'$'+np.format_float_scientific(overview[idxinf2[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][2], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf2[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestpx=pd.DataFrame([[ov1[4],form[idxpx[0]],solvers[idxpx[0]],'$'+np.format_float_scientific(overview[idxpx[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][3], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxpx[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        
        bestresult=pd.concat([bestsse,bestt,bestinf1,bestinf2,bestpx])
        bestresult.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[cs]+'\\best.csv')
    matsoldf=pd.DataFrame(matsol,index=case,columns=matsolcols)
    latexmatsol = matsoldf.to_latex(index=True, escape=False, column_format='|c| c c c c c c c c|')           
            
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\MatSol_tab.txt','w')
    file.write(latexmatsol)
    file.close()