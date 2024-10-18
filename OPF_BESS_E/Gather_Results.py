# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:12:04 2024

@author: diego
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
import scipy.stats as ss
import matplotlib.pyplot as plt


form=['PM','REBF','Meta']
conv=['Non-Convex','Convex','Metaheuristic']
fold=['YMatrix','BF','GWO']
case=['IEEE33','IEEE69','SA','CA141']
city=['Bog','Jam','Pop']
problem='OPF_BESS_E'
prob_sol='OPF_PV_D'
city1=['BOG','JAM','POP']
case1=['IEEE33','IEEE69','SA_J23','CA141']


nct=len(city)
nca=len(case)
pveff=0.8


vmaxca=np.zeros([nct,nca])
vminca=np.ones([nct,nca])
pvca=np.zeros([nct,nca])
pbchca=np.zeros([nct,nca])
pbdhca=np.zeros([nct,nca])
ploca=np.zeros([nct,nca])
plca=np.zeros([nct,nca])

pbca=np.zeros([24,nct,nca])
zbca=np.zeros([24,nct,nca],dtype=int)
socca=np.zeros([24,nct,nca])
soc0ca=np.zeros([nct,nca])
pbcca=np.zeros([nct,nca])
fca=[['']*nca for i in range(nct)]
isca=[['']*nca for i in range(nct)]


for ct in range(nct):
    metaov=['$N$','Best $f_{o}$','$\\mu (f_{o})$','$\\sigma (f_{o})$','Best $t$','$\\mu t$','$\\sigma (t)$']
    metaoverviewdf=pd.DataFrame(index=case,columns=metaov)
    nmov=len(metaov)

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\MeansRAD_'+city1[ct]+'.mat')
    imeans=np.squeeze(mat['means'])

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\ClusterMeans_'+city1[ct]+'.mat')
    dmeans=np.squeeze(mat['clustermeans']).T 
    
    H=len(imeans)
    
    for ca in range(nca):
        "----- Read the database -----"
        branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case1[ca]+"Branch.csv")
        bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case1[ca]+"Bus.csv")        

        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+case[ca]+'_'+city[ct]+'_'+'ClusterNode.mat')
        cnode=np.squeeze(mat['clusternode'])        
       
        pvall=pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+prob_sol+'\\'+city[ct]+'\\'+case[ca]+'\\bestsol.csv')
        
        ppv=pvall['ppv'].to_numpy()
        zpv=pvall['zpv'].to_numpy()
        
        cnode[0]=1
        cnode=cnode-1        
        
        num_lines = len(branch)
        num_nodes=len(bus)
        
        iref=np.where(bus['type']==3)[0][0]
        
        sd=np.zeros(num_nodes,dtype='complex')
        
        for k in range(num_lines):
            sd[branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])
            
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
        
        metas=['Meta-GWO']
        
        nform=len(form)
        
        datadict={}
        metadict={}        
        
        
        for i in range(nform):
            if i==0:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\'+form[i]+'\\Results.xlsx'
                datadict.update({form[i]:pd.read_excel(path,index_col=0)})                             
            elif i==1:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx'
                datadict.update({form[i]:pd.read_excel(path,index_col=0)})                               
            else:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx'                        
                datadict.update({form[i]:pd.read_excel(path)})
                cols0=datadict[form[0]].columns                                           
                datadict[form[i]].columns=cols0
                metadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\MetaResults.xlsx')})
            
            
                
        "----- Read the results -----"
        
        diffs=['$\\Delta v$','$\\Delta \\phi$','$\\Delta P_{loss}$','$\\Delta Q_{loss}$','$\\Delta P^{g}$','$\\Delta Q^{g}$']
        ov=['$P_{loss}$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
       
        
        cols=datadict[form[0]].columns       
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
        zbcols=cols.drop(cols[:4*num_nodes+13])
        zbcols=zbcols.drop(cols[5*num_nodes+13:])
        zbcols=zbcols.to_list()
        dccols=cols.drop(cols[:5*num_nodes+13])
        dccols=dccols.drop(cols[5*num_nodes+18:])
        dccols=dccols.to_list()
        ppvcols=cols.drop(cols[:5*num_nodes+18])
        ppvcols=ppvcols.drop(cols[6*num_nodes+18:])
        ppvcols=ppvcols.to_list()
        
        
        
        ndiffs=len(diffs)
        nov=len(ov)
        
        
        results=np.zeros([nform,ndiffs])
        times=np.zeros(nform)
        solvers=['*']*nform
        losses=np.zeros(nform)
        gaps=np.zeros(nform)
        
        infeas1=np.zeros(nform)
        infeas2=np.zeros(nform)
        idx0=np.zeros(nform)
        
        for i in range(nform):
            times[i]=datadict[form[i]]['t'][0]
            solvers[i]=datadict[form[i]]['Solver'][0]
            losses[i]=np.sum(datadict[form[i]]['pl'])              
            idx0[i]=np.sum(datadict[form[i]]['pl']<0)
            gaps[i]=datadict[form[i]]['Gap'][0]              
            for h in range(H):
                vmat=np.zeros(num_nodes,dtype='complex')
                ppvmat=np.zeros(num_nodes)
                zpvmat=np.zeros(num_nodes)
                pgmat=np.zeros(num_nodes)                
                qgmat=np.zeros(num_nodes) 
                pbmat=np.zeros(num_nodes)
                pdmc=np.zeros(num_nodes)
                qdmc=np.zeros(num_nodes)                
                eqpmat=np.zeros(num_nodes)
                eqqmat=np.zeros(num_nodes)
                pgmat[0]=datadict[form[i]]['pg'][h]
                qgmat[0]=datadict[form[i]]['qg'][h]                                
                for j in range(num_nodes):
                    vmat[j]=np.multiply(datadict[form[i]][vcols[j]][h],np.cos(np.abs(datadict[form[i]][phcols[j]][h])))+1j*np.multiply(datadict[form[i]][vcols[j]][h],np.sin(datadict[form[i]][phcols[j]][h]))
                    eqpmat[j]=datadict[form[i]][eqpcols[j]][h]
                    eqqmat[j]=datadict[form[i]][eqqcols[j]][h]
                    ppvmat[j]=datadict[form[i]][ppvcols[j]][0]                    
                    pdmc[j]=pdm[j]*dmeans[h][cnode[j]]
                    qdmc[j]=qdm[j]*dmeans[h][cnode[j]]
                    if datadict[form[i]][zbcols[j]][h]==1:
                        pbmat[j]=datadict[form[i]]['pbe'][h]                    
                
                infeas1[i]+=np.abs(np.sum(pgmat+(ppvmat*imeans[h]*pveff)+pbmat-eqpmat-pdmc))+np.abs(np.sum(qgmat-eqqmat-qdmc))                       
                infeas2[i]+=np.abs(np.sum(pgmat+(ppvmat*imeans[h]*pveff)+pbmat-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdmc))+np.abs(np.sum(qgmat-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdmc))
                    
                
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
        idxg=np.argsort(gaps)
        
        
        for i in range(nform):
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
                resultsdf[diffs[j]][form[i]]='$'+np.format_float_scientific(results[i][j],precision=2)+'$'    
        
        
        for i in range(nform):               
            for j in range(nov):
                if j==0:
                    overview[i][j]=losses[i]
                    ii=np.where(idxd==i)[0][0]
                    if ii==0:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_positional(losses[i], precision=4)+'$'
                    elif ii==1:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_positional(losses[i], precision=4)+'$'
                    elif ii==2:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_positional(losses[i], precision=4)+'$'
                    else:
                        overviewdf[ov[j]][form[i]]='$'+np.format_float_positional(losses[i], precision=4)+'$'
                elif j==1:                   
                    overview[i][j]=times[i]
                    ii=np.where(idxt==i)[0][0]
                    if ii==0:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_positional(times[i], precision=4)+'$'
                    elif ii==1:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_positional(times[i], precision=4)+'$'
                    elif ii==2:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_positional(times[i], precision=4)+'$'
                    else:
                        overviewdf[ov[j]][form[i]]='$'+np.format_float_positional(times[i], precision=4)+'$'
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
        gapstr=['*']*nform
        
        for i in range(nform-1):
            gapstr[i]='$'+np.format_float_scientific(gaps[i], precision=2)+'$'
        
        overviewdf.insert(0,'I/S',solvers)
        overviewdf.insert(1,'Gap',gapstr)
                        
        ntmeta=len(metadict['Meta'])
        
        
        mlosses=np.zeros(ntmeta)
        
        for i in range(ntmeta):
            mlosses[i]=metadict['Meta']['obj'][i]
            
        for j in range(nmov):
            if j==0:
                metaoverviewdf[metaov[j]][case[ca]]='$'+str(ntmeta)+'$'
            if j==1:
                ii=np.argsort(metadict['Meta']['obj'])[0]
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(metadict['Meta']['obj'][ii], precision=4)+'$'
            if j==2:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.mean(mlosses), precision=4)+'$'
            if j==3:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.std(mlosses), precision=4)+'$'
            if j==4:
                ii=np.argsort(metadict['Meta']['t'])[0]
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(metadict['Meta']['t'][ii], precision=4)+'$'
            if j==5:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.mean(metadict['Meta']['t'].to_numpy()), precision=4)+'$'
            if j==6:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.std(metadict['Meta']['t'].to_numpy()), precision=4)+'$'
        
        
        ploca[ct][ca]=losses[idxd[0]]
        
        for h in range(H):
            if datadict[form[idxd[0]]]['pbe'][h]>=0:
                pbdhca[ct][ca]+=datadict[form[idxd[0]]]['pbe'][h]
            else:
                pbchca[ct][ca]+=datadict[form[idxd[0]]]['pbe'][h]            
            for i in range(num_nodes):
                plca[ct][ca]+=pdm[i]*dmeans[h][cnode[i]]
                pvca[ct][ca]+=ppv[i]*imeans[h]*pveff
                if datadict[form[idxd[0]]][vcols[i]][h]<vminca[ct][ca]:
                    vminca[ct][ca]=datadict[form[idxd[0]]][vcols[i]][h]
                if datadict[form[idxd[0]]][vcols[i]][h]>vmaxca[ct][ca]:
                    vmaxca[ct][ca]=datadict[form[idxd[0]]][vcols[i]][h]                
                if datadict[form[idxd[0]]][zbcols[i]][h]==1:
                    zbca[h][ct][ca]=i+1
        
        pbca[:,ct,ca]=datadict[form[idxd[0]]]['pbe'].to_numpy()
        pbcca[ct][ca]=datadict[form[idxd[0]]]['pbc'][0]
        soc0ca[ct][ca]=datadict[form[idxd[0]]]['soc0'][0]
        socca[:,ct,ca]=datadict[form[idxd[0]]]['soc'].to_numpy()
        fca[ct][ca]=form[idxd[0]]
        isca[ct][ca]=solvers[idxd[0]]  
        

        for h in range(H):
            for i in range(num_nodes):
                if datadict[form[idxd[0]]][zbcols[i]][h]==1:
                    zbca[h][ct][ca]=i+1
        
        outsoldf=pd.DataFrame(columns=['CB','SOC0','ZB','PB','SOC'])
        pbcout=np.zeros(H)
        pbcout[0]=pbcca[ct][ca]
        soc0out=np.zeros(H)
        soc0out[0]=soc0ca[ct][ca]
        outsoldf['CB']=pbcout
        outsoldf['SOC0']=soc0out
        outsoldf['ZB']=zbca[:,ct,ca]
        outsoldf['PB']=pbca[:,ct,ca]
        outsoldf['SOC']=socca[:,ct,ca]
        
        outsoldf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\bestsol.csv')
                   
               
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
        
        bestloss=pd.DataFrame([[ov1[0],form[idxd[0]],solvers[idxd[0]]
                                ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxd[0]][0], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxd[0]][1], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxd[0]][2], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxd[0]][3], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxd[0]][4], precision=2)+'$']]
                              ,index=latexcase,columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        bestloss.insert(3,'Gap',gapstr[idxd[0]])
        
        bestt=pd.DataFrame([[ov1[1],form[idxt[0]],solvers[idxt[0]]
                             ,'$'+np.format_float_scientific(overview[idxt[0]][0], precision=2)+'$'
                             ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxt[0]][1], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][2], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][3], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][4], precision=2)+'$']]
                           ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        bestt.insert(3,'Gap',gapstr[idxt[0]])
        
        bestinf1=pd.DataFrame([[ov1[2],form[idxinf1[0]],solvers[idxinf1[0]]
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][0], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][1], precision=2)+'$'
                                ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf1[0]][2], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][3], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][4], precision=2)+'$']]
                              ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        bestinf1.insert(3,'Gap',gapstr[idxinf1[0]])
        
        bestinf2=pd.DataFrame([[ov1[3],form[idxinf2[0]],solvers[idxinf2[0]]
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][0], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][1], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][2], precision=2)+'$'
                                ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf2[0]][3], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][4], precision=2)+'$']]
                              ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        bestinf2.insert(3,'Gap',gapstr[idxinf2[0]])
        
        bestpx=pd.DataFrame([[ov1[4],form[idxpx[0]],solvers[idxpx[0]]
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][0], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][1], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][2], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][3], precision=2)+'$'
                              ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxpx[0]][4], precision=2)+'$']]
                            ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4]])
        
        bestpx.insert(3,'Gap',gapstr[idxpx[0]])
        
        bestresult=pd.concat([bestloss,bestt,bestinf1,bestinf2,bestpx])
        bestresult.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\best.csv')
            
    latexmetas = metaoverviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\Metas_tab.txt','w')
    file.write(latexmetas)
    file.close()

solcols=[]
solrows=[]

solrows1=['F','I/S','CB','SOC0','$V_{max}$','$V_{min}$','$P_{loss}$','$\Sigma P^{d}$','$\Sigma PPV$','$\Sigma PB^{+}$','$\Sigma PB^{-}$']


for ct in range(nct):
    solcols.append((city1[ct],'$z^{B}$'))
    solcols.append((city1[ct],'$p^{B}$'))
    solcols.append((city1[ct],'$SOC$'))

for h in range(H):
    solrows.append('h'+str(h+1))

solcols=pd.MultiIndex.from_tuples(solcols)    
for ca in range(nca):
    solutiondf=pd.DataFrame(index=solrows,columns=solcols)
    solution1df=pd.DataFrame(index=solrows1,columns=city1)
    for ct in range(nct):
        solution1df[city1[ct]][solrows1[0]]=fca[ct][ca]
        solution1df[city1[ct]][solrows1[1]]=isca[ct][ca]
        solution1df[city1[ct]][solrows1[2]]='$'+np.format_float_scientific(pbcca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[3]]='$'+np.format_float_scientific(soc0ca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[4]]='$'+np.format_float_scientific(vmaxca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[5]]='$'+np.format_float_scientific(vminca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[6]]='$'+np.format_float_scientific(ploca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[7]]='$'+np.format_float_scientific(plca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[8]]='$'+np.format_float_scientific(pvca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[9]]='$'+np.format_float_scientific(pbdhca[ct][ca], precision=2)+'$'
        solution1df[city1[ct]][solrows1[10]]='$'+np.format_float_scientific(pbchca[ct][ca], precision=2)+'$'
        
        for h in range(H):        
            solutiondf[(city1[ct],'$z^{B}$')][solrows[h]]='$'+str(zbca[h][ct][ca])+'$'
            solutiondf[(city1[ct],'$p^{B}$')][solrows[h]]='$'+np.format_float_scientific(pbca[h][ct][ca], precision=2)+'$'
            solutiondf[(city1[ct],'$SOC$')][solrows[h]]='$'+np.format_float_scientific(socca[h][ct][ca], precision=2)+'$'
        

    latexsolution = solutiondf.to_latex(index=True, escape=False, multicolumn=True,multicolumn_format='c', column_format='|c c c|c c c|c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+case[ca]+'_sol_tab.txt','w')
    file.write(latexsolution)
    file.close()
    
    latexsolution1 = solution1df.to_latex(index=True, escape=False, column_format='|c c c')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+case[ca]+'_sol1_tab.txt','w')
    file.write(latexsolution1)
    file.close()
    
    
    