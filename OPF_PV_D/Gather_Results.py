# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:12:04 2024

@author: diego
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
import os

form=['PM','RM','RRM','PBIM','RBIM','RRBIM','BF','RERCM','RERRM','RERCBIM','RERRBIM','REBF','Meta-GA','Meta-GWO','Meta-PSO','Meta-WOA']
conv=['Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Non-Convex','Convex','Convex','Convex','Convex','Convex','Metaheuristic','Metaheuristic','Metaheuristic','Metaheuristic']
fold=['YMatrix','YMatrix','YMatrix','BIM','BIM','BIM','BF','YMatrix','YMatrix','BIM','BIM','BF','GA','GWO','PSO','WOA']
case=['IEEE33','IEEE69','SA','CA141']
city=['Bog','Jam','Pop']
problem='OPF_PV_D'
city1=['BOG','JAM','POP']
case1=['IEEE33','IEEE69','SA_J23','CA141']

nct=len(city)
nca=len(case)
pveff=0.8
for ct in range(nct):
    for ca in range(nca):
        "----- Read the database -----"
        branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case1[ca]+"Branch.csv")
        bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case1[ca]+"Bus.csv")
        
        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\MeansRAD_'+city1[ct]+'.mat')
        imeans=np.squeeze(mat['means'])

        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\ClusterMeans_'+city1[ct]+'.mat')
        dmeans=np.squeeze(mat['clustermeans']).T

        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+case[ca]+'_'+city[ct]+'_'+'ClusterNode.mat')
        cnode=np.squeeze(mat['clusternode'])
 
        cnode[0]=1
        cnode=cnode-1
        
        num_lines = len(branch)
        num_nodes=len(bus)
        
        iref=np.where(bus['type']==3)[0][0]
        
        H=len(imeans)
        
        sd=np.zeros([H,num_nodes],dtype='complex')
        
        for h in range(H):
            for k in range(num_lines):
                sd[h][branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])*dmeans[h][cnode[branch['j'][k]-1]]
            
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
        
        metas=['Meta-GA','Meta-GWO','Meta-PSO','Meta-WOA']
        
        nform=len(form)
        
        
        datadict={}
        metadict={}
        
        for i in range(nform):
            if i<nform-4 and not i==6 and not i==11:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\'+form[i]+'\\Results.xlsx'
                if os.path.isfile(path):
                    datadict.update({form[i]:pd.read_excel(path,index_col=0)})
                else:
                    datadict.update({form[i]:'NR'})
            elif i==6 or i==11:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx'
                if os.path.isfile(path):
                    datadict.update({form[i]:pd.read_excel(path,index_col=0)})
                else:
                    datadict.update({form[i]:'NR'})
            else:
                df=pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx')
                cols0=datadict[form[11]].columns
                # ncols0=len(cols0)
                # cols=df.columns
                # ncols=len(cols)
                # if ncols>ncols0:
                #     datadict.update({form[i]:df.drop(cols[ncols0:],axis=1)})
                # else:
                #     datadict.update({form[i]:df})                
                datadict.update({form[i]:df})                
                datadict[form[i]].columns=cols0
                metadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\MetaResults.xlsx')})
        
        "----- Read the results -----"
        
        diffs=['$\\Delta v$','$\\Delta \\phi$','$\\Delta P_{loss}$','$\\Delta Q_{loss}$','$\\Delta P^{g}$','$\\Delta Q^{g}$']
        ov=['$P_{loss}$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
        metaov=['$N$','Best $f_{o}$','$\\mu (f_{o})$','$\\sigma (f_{o})$','Best $t$','$\\mu t$','$\\sigma (t)$']
        
        cols=datadict[form[nform-1]].columns       
        vs=cols.drop(cols[2*num_nodes:4*num_nodes])
        vs=vs.drop(cols[4*num_nodes+4:])
        vs=vs.to_list()
        
        vcols=cols.drop(cols[num_nodes:]).to_list()
        phcols=cols.drop(cols[:num_nodes])
        phcols=phcols.drop(cols[2*num_nodes:])
        phcols=phcols.to_list()
        eqpcols=cols.drop(cols[:2*num_nodes])
        eqpcols=eqpcols.drop(cols[3*num_nodes:])
        eqpcols=eqpcols.to_list()
        eqqcols=cols.drop(cols[:3*num_nodes])
        eqqcols=eqqcols.drop(cols[4*num_nodes:])
        eqqcols=eqqcols.to_list()
        dccols=cols.drop(cols[:4*num_nodes+6])
        dccols=dccols.drop(cols[4*num_nodes+11:])
        dccols=dccols.to_list()
        ppvcols=cols.drop(cols[:4*num_nodes+11])
        ppvcols=ppvcols.drop(cols[5*num_nodes+11:])
        ppvcols=ppvcols.to_list()
        zpvcols=cols.drop(cols[:5*num_nodes+11])
        zpvcols=zpvcols.drop(cols[6*num_nodes+11:])
        zpvcols=zpvcols.to_list()
        
        
        ndiffs=len(diffs)
        nov=len(ov)
        nmov=len(metaov)
        
        results=np.zeros([nform,ndiffs])
        times=np.zeros(nform)
        solvers=['*']*nform
        losses=np.zeros(nform)
        
        infeas1=np.zeros(nform)
        infeas2=np.zeros(nform)
        idx0=np.zeros(nform)
        
        for j in range(nform):
            if isinstance(datadict[form[j]], pd.DataFrame):
                times[j]=datadict[form[j]]['t'][0]
                solvers[j]=datadict[form[j]]['Solver'][0]
                losses[j]=np.sum(datadict[form[j]]['pl'])
                idx0[j]=np.sum(datadict[form[j]]['pl']<0)                
            else:
                times[j]=-10                
                losses[j]=-10
                idx0[j]=-10
                    
        
        for i in range(nform):
            if not isinstance(datadict[form[i]], pd.DataFrame):
                infeas1[i]=-10
                infeas2[i]=-10
            else:        
                for h in range(H):
                    vmat=np.zeros(num_nodes,dtype='complex')
                    ppvmat=np.zeros(num_nodes)
                    pgmat=np.zeros(num_nodes)
                    qgmat=np.zeros(num_nodes)
                    eqpmat=np.zeros(num_nodes)
                    eqqmat=np.zeros(num_nodes)
                    pgmat[0]=datadict[form[i]]['pg'][h]
                    qgmat[0]=datadict[form[i]]['qg'][h]
                    
                    for j in range(num_nodes):
                        vmat[j]=np.multiply(datadict[form[i]][vcols[j]][h],np.cos(np.abs(datadict[form[i]][phcols[j]][h])))+1j*np.multiply(datadict[form[i]][vcols[j]][h],np.sin(datadict[form[i]][phcols[j]][h]))
                        eqpmat[j]=datadict[form[i]][eqpcols[j]][h]
                        eqqmat[j]=datadict[form[i]][eqqcols[j]][h]
                        ppvmat[j]=datadict[form[i]][ppvcols[j]][0]                    
                    
                    infeas1[i]+=np.abs(np.sum(pgmat+(ppvmat*datadict[form[i]]['ic'][h]*pveff)-eqpmat-pdm[h]))+np.abs(np.sum(qgmat-eqqmat-qdm[h]))                       
                    infeas2[i]+=np.abs(np.sum(pgmat+(ppvmat*datadict[form[i]]['ic'][h]*pveff)-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdm[h]))+np.abs(np.sum(qgmat-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdm[h]))
                
                
        idx=np.logical_or(losses<0,idx0<0)
                
        idxi1=np.logical_or(infeas1>1e-3,infeas1<0)
        idxi2=np.logical_or(infeas2>1e-3,infeas2<0)
        
        mtt=np.multiply(np.abs(times),np.abs(losses))
        mtt=np.multiply(mtt,np.abs(infeas1))
        mtt=np.multiply(mtt,np.abs(infeas2))
        overview=np.zeros([nform,nov])
        
        resultsdf=pd.DataFrame(index=form,columns=diffs)
        overviewdf=pd.DataFrame(index=form,columns=ov)
        
        idxd=np.argsort(losses+(100*idx)+(100*idxi1)+(100*idxi2))
        idxt=np.argsort(times+(100*idx)+(100*idxi1)+(100*idxi2))
        idxpx=np.argsort(mtt+(100*idx)+(100*idxi1)+(100*idxi2))
        idxinf1=np.argsort(infeas1+(100*idx)+(100*idxi1)+(100*idxi2))
        idxinf2=np.argsort(infeas2+(100*idx)+(100*idxi1)+(100*idxi2))
        
        for i in range(nform):
            if not isinstance(datadict[form[i]], pd.DataFrame):
                results[i]=-10
            else:
                for j in range(ndiffs):
                    if j==0:                    
                        for n in range(num_nodes):
                            results[i][j]+=np.sum(np.square(datadict[form[i]][vs[n]].to_numpy()-datadict[form[idxd[0]]][vs[n]].to_numpy()))
                    elif j==1:
                        for n in range(num_nodes):
                            results[i][j]+=np.sum(np.square(datadict[form[i]][vs[num_nodes+n]].to_numpy()-datadict[form[idxd[0]]][vs[num_nodes+n]].to_numpy()))
                    else:
                        results[i][j]=np.sum(np.square(datadict[form[i]][vs[2*num_nodes+j-2]].to_numpy()-datadict[form[idxd[0]]][vs[2*num_nodes+j-2]].to_numpy()))
                    
        for i in range(nform):
            for j in range(ndiffs):
                if not isinstance(datadict[form[i]], pd.DataFrame):
                    resultsdf[diffs[j]][form[i]]='*'
                else:                        
                    resultsdf[diffs[j]][form[i]]='$'+np.format_float_scientific(results[i][j],precision=2)+'$'    
        
        
        for i in range(nform):               
            for j in range(nov):
                if not isinstance(datadict[form[i]], pd.DataFrame):
                    overview[i][j]=-10
                    overviewdf[ov[j]][form[i]]='*'
                else:
                    if j==0:
                        overview[i][j]=losses[i]
                        ii=np.where(idxd==i)[0][0]
                        if ii==0:
                            overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_scientific(losses[i], precision=2)+'$'
                        elif ii==1:
                            overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_scientific(losses[i], precision=2)+'$'
                        elif ii==2:
                            overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_scientific(losses[i], precision=2)+'$'
                        else:
                            overviewdf[ov[j]][form[i]]='$'+np.format_float_scientific(losses[i], precision=2)+'$'
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
                        
                        
                
        
        solutions=[['']*2 for i in range(nform)]
        for i in range(nform):
            k=0
            if not isinstance(datadict[form[i]], pd.DataFrame):
                solutions[i]=['*','*']
            else:                
                for j in range(num_nodes):
                    if datadict[form[i]][zpvcols[j]][0]==1 and datadict[form[i]][ppvcols[j]][0]>1e-3:
                        solutions[i][0]+='$('+str(j+1)+','+np.format_float_scientific(datadict[form[i]][ppvcols[j]][0], precision=2)+')$'
                        k+=1
                solutions[i][1]+=str(k)    
        
        bestzpv=np.zeros(num_nodes)
        bestpvc=np.zeros(num_nodes)
        npv=np.sum(ppvmat!=0)
        nvarssol=npv+4
        varssol=[['']*nvarssol]
        sumcap=0
        sumnpv=0
        bestsolutions=[['']*nvarssol]
        k=0        
        varssol[0][k]='$P_{loss}$'
        bestsolutions[0][k]='$'+np.format_float_scientific(losses[idxd[0]], precision=4)+'$'
        k+=1
        vmax=0
        vmin=1
        for j in range(num_nodes):
            vmax=np.max([vmax,np.max(datadict[form[idxd[0]]][vcols[j]].to_numpy())])
            vmin=np.min([vmin,np.min(datadict[form[idxd[0]]][vcols[j]].to_numpy())])
            if datadict[form[idxd[0]]][zpvcols[j]][0]==1:
                bestzpv[j]=1
                bestpvc[j]=datadict[form[idxd[0]]][ppvcols[j]][0]
                bestsolutions[0][k]+='$('+str(j+1)+','+np.format_float_scientific(datadict[form[idxd[0]]][ppvcols[j]][0], precision=4)+')$'
                varssol[0][k]+='$(Loc'+str(k)+',Cap'+str(k)+')$'                
                sumcap+=datadict[form[idxd[0]]][ppvcols[j]][0]
                sumnpv+=1
                k+=1
        
        bestsoldf=pd.DataFrame(columns=['zpv','ppv'])
        bestsoldf['zpv']=bestzpv
        bestsoldf['ppv']=bestpvc
        
        bestsoldf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\bestsol.csv')
        
        varssol[0][k]='$\\Sigma (z_{i}^{pv},CP_{i})$'
        bestsolutions[0][k]='$('+str(sumnpv)+','+np.format_float_scientific(sumcap, precision=4)+')$'
        k+=1
        varssol[0][k]='$V_{max}$'
        bestsolutions[0][k]+='$'+np.format_float_scientific(vmax, precision=4)+'$'
        k+=1
        varssol[0][k]='$V_{min}$'
        bestsolutions[0][k]+='$'+np.format_float_scientific(vmin, precision=4)+'$'
        
        solutionsdf=pd.DataFrame(solutions,index=form,columns=['Allocation Solution $(z^{pv},PPV)$','N_{pv}'])
        
        bestsolutiondf=pd.DataFrame(np.transpose(np.vstack((varssol,bestsolutions))),columns=['Variables','Values'])
        
        metaoverviewdf=pd.DataFrame(index=metas,columns=metaov)
        
        nmetas=len(metas)
        
        for i in range(nmetas):
            for j in range(nmov):
                if j==0:
                    metaoverviewdf[metaov[j]][metas[i]]='$'+str(len(metadict[metas[i]]))+'$'
                if j==1:
                    ii=np.argsort(metadict[metas[i]]['obj'].to_numpy())[0]
                    metaoverviewdf[metaov[j]][metas[i]]='$'+np.format_float_scientific(metadict[metas[i]]['obj'][ii], precision=4)+'$'
                if j==2:
                    metaoverviewdf[metaov[j]][metas[i]]='$'+np.format_float_scientific(np.mean(metadict[metas[i]]['obj'].to_numpy()), precision=4)+'$'
                if j==3:
                    metaoverviewdf[metaov[j]][metas[i]]='$'+np.format_float_scientific(np.std(metadict[metas[i]]['obj'].to_numpy()), precision=4)+'$'
                if j==4:
                    ii=np.argsort(metadict[metas[i]]['t'])[0]
                    metaoverviewdf[metaov[j]][metas[i]]='$'+np.format_float_scientific(metadict[metas[i]]['t'][ii], precision=4)+'$'
                if j==5:
                    metaoverviewdf[metaov[j]][metas[i]]='$'+np.format_float_scientific(np.mean(metadict[metas[i]]['t'].to_numpy()), precision=4)+'$'
                if j==6:
                    metaoverviewdf[metaov[j]][metas[i]]='$'+np.format_float_scientific(np.std(metadict[metas[i]]['t'].to_numpy()), precision=4)+'$'
                
        
        overviewdf.insert(0,'I/S',solvers)
                
        latexresults = resultsdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|') 
        latexoverview = overviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|')           
                
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\Results_tab.txt','w')
        file.write(latexresults)
        file.close() 
        
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\Overview_tab.txt','w')
        file.write(latexoverview)
        file.close()                 
                    
        
        diffs1=['dv','dph','dpg','dqg','dpl','dql']
        ov1=['$P_{loss}$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
        resultsdf1=pd.DataFrame(results,index=form,columns=diffs1)
        overviewdf1=pd.DataFrame(overview,index=form,columns=ov1)
        resultsdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\matdifs.csv')
        overviewdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\overview.csv')
        
        latexcase=['\\multirow{'+str(nov)+'}{*}{'+case[ca]+'}']
        
        bestloss=pd.DataFrame([[ov1[0],form[idxd[0]],solvers[idxd[0]],'$\\cellcolor{green}'+np.format_float_scientific(overview[idxd[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxd[0]][4], precision=2)+'$']],index=latexcase,columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestt=pd.DataFrame([[ov1[1],form[idxt[0]],solvers[idxt[0]],'$'+np.format_float_scientific(overview[idxt[0]][0], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxt[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxt[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestinf1=pd.DataFrame([[ov1[2],form[idxinf1[0]],solvers[idxinf1[0]],'$'+np.format_float_scientific(overview[idxinf1[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][1], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf1[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf1[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestinf2=pd.DataFrame([[ov1[3],form[idxinf2[0]],solvers[idxinf2[0]],'$'+np.format_float_scientific(overview[idxinf2[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][2], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf2[0]][3], precision=2)+'$','$'+np.format_float_scientific(overview[idxinf2[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        bestpx=pd.DataFrame([[ov1[4],form[idxpx[0]],solvers[idxpx[0]],'$'+np.format_float_scientific(overview[idxpx[0]][0], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][1], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][2], precision=2)+'$','$'+np.format_float_scientific(overview[idxpx[0]][3], precision=2)+'$','$\\cellcolor{green}'+np.format_float_scientific(overview[idxpx[0]][4], precision=2)+'$']],index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        
        bestresult=pd.concat([bestloss,bestt,bestinf1,bestinf2,bestpx])
        bestresult.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\best.csv')
        
        latexsolutions = solutionsdf.to_latex(index=True, escape=False, column_format='|c| c c|') 
        latexmetas = metaoverviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c c|')
        latexbestsolution = bestsolutiondf.to_latex(index=False, escape=False, column_format='|c c|')
        
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\Solutions_tab.txt','w')
        file.write(latexsolutions)
        file.close()
        
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\BestSolution_tab.txt','w')
        file.write(latexbestsolution)
        file.close()
        
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\Metas_tab.txt','w')
        file.write(latexmetas)
        file.close() 