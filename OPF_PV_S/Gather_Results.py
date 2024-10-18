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
problem='OPF_PV_S'
city1=['BOG','JAM','POP']
case1=['IEEE33','IEEE69','SA_J23','CA141']
adists=['Exponential','Fisk','Logistic','Log-N','Normal','Rayleigh','Weibull']
aclust=['c1','c2','c3','c4','c5']

nct=len(city)
nca=len(case)
pveff=0.8
for ct in range(nct):
    metaov=['$N$','Best $f_{o}$','$\\mu (f_{o})$','$\\sigma (f_{o})$','Best $t$','$\\mu t$','$\\sigma (t)$']
    metaoverviewdf=pd.DataFrame(index=case,columns=metaov)
    nmov=len(metaov)

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\MeansRAD_'+city1[ct]+'.mat')
    imeans=np.squeeze(mat['means'])

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\ClusterMeans_'+city1[ct]+'.mat')
    dmeans=np.squeeze(mat['clustermeans']).T 

    irr = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\'+'ParamTable.csv')

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\NSTparamDem_'+city1[ct]+'.mat')
    dparams=mat['params']

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\NSTparamRAD_'+city1[ct]+'.mat')
    iparams=mat['params']
    
    c1 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+'ParamTableC1.csv')
    c2 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+'ParamTableC2.csv')
    c3 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+'ParamTableC3.csv')
    c4 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+'ParamTableC4.csv')
    c5 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+'ParamTableC5.csv')
    
    ndists=len(adists)
    ncluster=len(dparams)
    H=np.size(dparams,axis=1)
    
    cparameters=pd.concat([c1,c2,c3,c4,c5],keys=aclust)

    bestfitsd=[[0]*H for i in range(ncluster)]

    bestfitsi=[10]*H 
    nihours=int(len(irr)/2)
    ihours=np.zeros(H)
    
    
    for h in range(H):
        for hh in range(nihours):
            if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
                ihours[h]=1
        
        
        
    for h in range(H):
        for hh in range(nihours):
            if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
                ihours[h]=1
                for j in range(ndists):
                    if adists[j] in irr['bestparams1'][2*hh] and irr['bestparams1'][2*hh].find(adists[j])==0:
                       bestfitsi[h]=j 
                       
        for i in range(ncluster):
            for j in range(ndists):
                if adists[j] in cparameters['bestparams1'][aclust[i],2*h] and cparameters['bestparams1'][aclust[i],2*h].find(adists[j])==0:
                    bestfitsd[i][h]=j
                    
    probim=np.zeros(H)
    probdm=np.zeros([H,ncluster])
    
    for h in range(H):
        for c in range(ncluster):
            distribution=adists[bestfitsd[c][h]]
            "Exponential"
            if distribution==adists[0]:
                probdm[h][c]=1-np.exp(-dparams[c][h][0]*dmeans[h][c])            
            "Fisk  (Beta is always positive)"
            if distribution==adists[1]:
                probdm[h][c]=1/(1+np.power(dmeans[h][c]/dparams[c][h][2],-dparams[c][h][3]))                          
            "Logistic"                        
            if distribution==adists[2]:                
                probdm[h][c]=1/(1+np.exp((-dmeans[h][c]+dparams[c][h][4])/dparams[c][h][5]))
            "LogNorm"                        
            if distribution==adists[3]:                
                probdm[h][c]=ss.norm.cdf(np.log(dmeans[h][c]),dparams[c][h][6],dparams[c][h][7])
            "Normal"
            if distribution==adists[4]:                
                probdm[h][c]=ss.norm.cdf(dmeans[h][c],dparams[c][h][8],dparams[c][h][9])
            "Rayleigh"
            if distribution==adists[5]:                
                probdm[h][c]=1-np.exp(-np.square(dmeans[h][c])/(2*np.square(dparams[c][h][10])))
            "Weibull"
            if distribution==adists[6]:                
                probdm[h][c]=1-np.exp(-np.power(dmeans[h][c]*dparams[c][h][12],dparams[c][h][13]))
        if bestfitsi[h]!=10:
            distribution=adists[bestfitsi[h]]
            "Exponential"
            if distribution==adists[0]:               
                probim[h]=1-np.exp(-iparams[h][0]*imeans[h])            
            "Fisk  (Beta is always positive)"
            if distribution==adists[1]:
                probim[h]=1/(1+np.power(imeans[h]/iparams[h][2],-iparams[h][3]))                          
            "Logistic"                        
            if distribution==adists[2]:                
                probim[h]=1/(1+np.exp((-imeans[h]+iparams[h][4])/iparams[h][5]))
            "LogNorm"                        
            if distribution==adists[3]:                
                probim[h]=ss.norm.cdf(np.log(imeans[h]),iparams[h][6],iparams[h][7])
            "Normal"
            if distribution==adists[4]:                
                probim[h]=ss.norm.cdf(imeans[h],iparams[h][8],iparams[h][9])
            "Rayleigh"
            if distribution==adists[5]:                
                probim[h]=1-np.exp(-np.square(imeans[h])/(2*np.square(iparams[h][10])))
            "Weibull"
            if distribution==adists[6]:                
                probim[h]=1-np.exp(-np.power(imeans[h]*iparams[h][12],iparams[h][13]))
                
    hindex=['']*H
    
    for h in range(H):
        hindex[h]='$h_{'+str(h)+'}$'
    
    dicols=['$\\overline{I_{pv}}$']
    dpicols=['$p(\\overline{I_{pv}})$']
    ddcols1=['$\\overline{DC_{1}}$']
    ddcols2=['$\\overline{DC_{2}}$']
    ddcols3=['$\\overline{DC_{3}}$']
    ddcols4=['$\\overline{DC_{4}}$']
    ddcols5=['$\\overline{DC_{5}}$']
    dpdcols1=['$p(\\overline{DC_{1}})$']
    dpdcols2=['$p(\\overline{DC_{2}})$']
    dpdcols3=['$p(\\overline{DC_{3}})$']
    dpdcols4=['$p(\\overline{DC_{4}})$']
    dpdcols5=['$p(\\overline{DC_{5}})$']
    
    for i in range(nca):
        dicols.append(('$I_{pv}$',case[i]))
        dpicols.append(('$p(I_{pv})$',case[i]))
        ddcols1.append(('$DC_{1}$',case[i]))
        dpdcols1.append(('$p(DC_{1})$',case[i]))
        ddcols2.append(('$DC_{2}$',case[i]))
        dpdcols2.append(('$p(DC_{2})$',case[i]))
        ddcols3.append(('$DC_{3}$',case[i]))
        dpdcols3.append(('$p(DC_{3})$',case[i]))
        ddcols4.append(('$DC_{4}$',case[i]))
        dpdcols4.append(('$p(DC_{4})$',case[i]))
        ddcols5.append(('$DC_{5}$',case[i]))
        dpdcols5.append(('$p(DC_{5})$',case[i]))
    
           
   
    dfi = pd.DataFrame(index=hindex, columns=dicols)
    dfpi = pd.DataFrame(index=hindex, columns=dpicols)
    dfd1 = pd.DataFrame(index=hindex, columns=ddcols1)
    dfd2 = pd.DataFrame(index=hindex, columns=ddcols2)
    dfd3 = pd.DataFrame(index=hindex, columns=ddcols3)
    dfd4 = pd.DataFrame(index=hindex, columns=ddcols4)
    dfd5 = pd.DataFrame(index=hindex, columns=ddcols5)
    
    dfpd1 = pd.DataFrame(index=hindex, columns=dpdcols1)
    dfpd2 = pd.DataFrame(index=hindex, columns=dpdcols2)
    dfpd3 = pd.DataFrame(index=hindex, columns=dpdcols3)
    dfpd4 = pd.DataFrame(index=hindex, columns=dpdcols4)
    dfpd5 = pd.DataFrame(index=hindex, columns=dpdcols5)
    
    for h in range(H):
        dfi[dicols[0]][hindex[h]]='$'+np.format_float_scientific(imeans[h],precision=2)+'$'
        dfpi[dpicols[0]][hindex[h]]='$'+np.format_float_scientific(probim[h],precision=2)+'$'
        
        dfd1[ddcols1[0]][hindex[h]]='$'+np.format_float_scientific(dmeans[h][0],precision=2)+'$'
        dfd2[ddcols2[0]][hindex[h]]='$'+np.format_float_scientific(dmeans[h][1],precision=2)+'$'
        dfd3[ddcols3[0]][hindex[h]]='$'+np.format_float_scientific(dmeans[h][2],precision=2)+'$'
        dfd4[ddcols4[0]][hindex[h]]='$'+np.format_float_scientific(dmeans[h][3],precision=2)+'$'
        dfd5[ddcols5[0]][hindex[h]]='$'+np.format_float_scientific(dmeans[h][4],precision=2)+'$'
               
        dfpd1[dpdcols1[0]][hindex[h]]='$'+np.format_float_scientific(probdm[h][0],precision=2)+'$'
        dfpd2[dpdcols2[0]][hindex[h]]='$'+np.format_float_scientific(probdm[h][1],precision=2)+'$'
        dfpd3[dpdcols3[0]][hindex[h]]='$'+np.format_float_scientific(probdm[h][2],precision=2)+'$'
        dfpd4[dpdcols4[0]][hindex[h]]='$'+np.format_float_scientific(probdm[h][3],precision=2)+'$'
        dfpd5[dpdcols5[0]][hindex[h]]='$'+np.format_float_scientific(probdm[h][4],precision=2)+'$'
    
    bestic=np.zeros([H,nca+1])
    bestpic=np.zeros([H,nca+1])
    bestdc=np.zeros([H,nca+1,ncluster])
    bestpdc=np.zeros([H,nca+1,ncluster])
    bestform=np.zeros(nca,dtype=int)
    
    bestic[:,0]=imeans
    bestpic[:,0]=probim
    bestdc[:,0,0]=dmeans[:,0]
    bestdc[:,0,1]=dmeans[:,1]
    bestdc[:,0,2]=dmeans[:,2]
    bestdc[:,0,3]=dmeans[:,3]
    bestdc[:,0,4]=dmeans[:,4]
    bestpdc[:,0,0]=probdm[:,0]
    bestpdc[:,0,1]=probdm[:,1]
    bestpdc[:,0,2]=probdm[:,2]
    bestpdc[:,0,3]=probdm[:,3]
    bestpdc[:,0,4]=probdm[:,4]
        
    
         
    for ca in range(nca):
        "----- Read the database -----"
        branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case1[ca]+"Branch.csv")
        bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\"+case1[ca]+"Bus.csv")        

        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+case[ca]+'_'+city[ct]+'_'+'ClusterNode.mat')
        cnode=np.squeeze(mat['clusternode'])
 
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
        
        probii=np.zeros([nform,H])
        probdi=np.zeros([nform,H,ncluster])
        ici=np.zeros([nform,H])
        dci=np.zeros([nform,H,ncluster])
        
        for i in range(nform):
            if i==0:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\'+form[i]+'\\Results.xlsx'
                datadict.update({form[i]:pd.read_excel(path,index_col=0)})               
                probi=datadict[form[i]]['prob_ic_cal'].to_numpy()
                probd=np.zeros([H,ncluster])
                for h in range(H):
                    probd[h][0]=datadict[form[i]]['prob_dc_cal1'][h]
                    probd[h][1]=datadict[form[i]]['prob_dc_cal2'][h]
                    probd[h][2]=datadict[form[i]]['prob_dc_cal3'][h]
                    probd[h][3]=datadict[form[i]]['prob_dc_cal4'][h]
                    probd[h][4]=datadict[form[i]]['prob_dc_cal5'][h]
                
            elif i==1:
                path='C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\ResultsPWL.xlsx'
                datadict.update({form[i]:pd.read_excel(path,index_col=0)})
                probi=datadict[form[i]]['prob_ic_cal'].to_numpy()
                probd=np.zeros([H,ncluster])
                for h in range(H):
                    probd[h][0]=datadict[form[i]]['prob_dc_cal1'][h]
                    probd[h][1]=datadict[form[i]]['prob_dc_cal2'][h]
                    probd[h][2]=datadict[form[i]]['prob_dc_cal3'][h]
                    probd[h][3]=datadict[form[i]]['prob_dc_cal4'][h]
                    probd[h][4]=datadict[form[i]]['prob_dc_cal5'][h]                
            else:
                df=pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\Results.xlsx')
                probi=np.zeros(H)
                probd=np.zeros([H,ncluster])
                for h in range(H):
                    for c in range(ncluster):
                        keyc='dco_'+str(c+1)
                        distribution=adists[bestfitsd[c][h]]
                        "Exponential"
                        if distribution==adists[0]:
                            probd[h][c]=1-np.exp(-dparams[c][h][0]*df[keyc][h])            
                        "Fisk  (Beta is always positive)"
                        if distribution==adists[1]:
                            probd[h][c]=1/(1+np.power(df[keyc][h]/dparams[c][h][2],-dparams[c][h][3]))                          
                        "Logistic"                        
                        if distribution==adists[2]:                
                            probd[h][c]=1/(1+np.exp((-df[keyc][h]+dparams[c][h][4])/dparams[c][h][5]))
                        "LogNorm"                        
                        if distribution==adists[3]:                
                            probd[h][c]=ss.norm.cdf(np.log(df[keyc][h]),dparams[c][h][6],dparams[c][h][7])
                        "Normal"
                        if distribution==adists[4]:                
                            probd[h][c]=ss.norm.cdf(df[keyc][h],dparams[c][h][8],dparams[c][h][9])
                        "Rayleigh"
                        if distribution==adists[5]:                
                            probd[h][c]=1-np.exp(-np.square(df[keyc][h])/(2*np.square(dparams[c][h][10])))
                        "Weibull"
                        if distribution==adists[6]:                
                            probd[h][c]=1-np.exp(-np.power(df[keyc][h]*dparams[c][h][12],dparams[c][h][13]))
                    if bestfitsi[h]!=10:
                        distribution=adists[bestfitsi[h]]
                        "Exponential"
                        if distribution==adists[0]:               
                            probi[h]=1-np.exp(-iparams[h][0]*df['ico'][h])            
                        "Fisk  (Beta is always positive)"
                        if distribution==adists[1]:
                            probi[h]=1/(1+np.power(df['ico'][h]/iparams[h][2],-iparams[h][3]))                          
                        "Logistic"                        
                        if distribution==adists[2]:                
                            probi[h]=1/(1+np.exp((-df['ico'][h]+iparams[h][4])/iparams[h][5]))
                        "LogNorm"                        
                        if distribution==adists[3]:                
                            probi[h]=ss.norm.cdf(np.log(df['ico'][h]),iparams[h][6],iparams[h][7])
                        "Normal"
                        if distribution==adists[4]:                
                            probi[h]=ss.norm.cdf(df['ico'][h],iparams[h][8],iparams[h][9])
                        "Rayleigh"
                        if distribution==adists[5]:                
                            probi[h]=1-np.exp(-np.square(df['ico'][h])/(2*np.square(iparams[h][10])))
                        "Weibull"
                        if distribution==adists[6]:                
                            probi[h]=1-np.exp(-np.power(df['ico'][h]*iparams[h][12],iparams[h][13]))
            
                        
                cols0=datadict[form[0]].columns                
                datadict.update({form[i]:df.copy()})
                datadict[form[i]].insert((4*num_nodes)+6,'probi_opt',probi)
                datadict[form[i]].insert((4*num_nodes)+7,'probi_cal',probi)
                datadict[form[i]].insert((4*num_nodes)+13,'prob_dc_opt1',probd[:,0])
                datadict[form[i]].insert((4*num_nodes)+14,'prob_dc_opt2',probd[:,1])  
                datadict[form[i]].insert((4*num_nodes)+15,'prob_dc_opt3',probd[:,2])  
                datadict[form[i]].insert((4*num_nodes)+16,'prob_dc_opt4',probd[:,3])  
                datadict[form[i]].insert((4*num_nodes)+17,'prob_dc_opt5',probd[:,4])
                datadict[form[i]].insert((4*num_nodes)+18,'prob_dc_cal1',probd[:,0])
                datadict[form[i]].insert((4*num_nodes)+19,'prob_dc_cal2',probd[:,1])  
                datadict[form[i]].insert((4*num_nodes)+20,'prob_dc_cal3',probd[:,2])  
                datadict[form[i]].insert((4*num_nodes)+21,'prob_dc_cal4',probd[:,3])  
                datadict[form[i]].insert((4*num_nodes)+22,'prob_dc_cal5',probd[:,4])                     
                datadict[form[i]].columns=cols0
                metadict.update({form[i]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv[i]+'\\'+fold[i]+'\\MetaResults.xlsx')})
            
            probii[i]=probi
            probdi[i]=probd
            ici[i]=datadict[form[i]]['ic'].to_numpy()
            for h in range(H):
                dci[i][h][0]=datadict[form[i]]['dc_c1'][h]
                dci[i][h][1]=datadict[form[i]]['dc_c2'][h]
                dci[i][h][2]=datadict[form[i]]['dc_c3'][h]
                dci[i][h][3]=datadict[form[i]]['dc_c4'][h]
                dci[i][h][4]=datadict[form[i]]['dc_c5'][h]
                
        "----- Read the results -----"
        
        diffs=['$\\Delta v$','$\\Delta \\phi$','$\\Delta P_{loss}$','$\\Delta Q_{loss}$','$\\Delta P^{g}$','$\\Delta Q^{g}$']
        ov=['$P_{loss}$','$\\Sigma p(I_{pv})$','$1/\\Sigma p(DC)$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
       
        
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
        dccols=cols.drop(cols[:4*num_nodes+8])
        dccols=dccols.drop(cols[4*num_nodes+13:])
        dccols=dccols.to_list()
        pdcocols=cols.drop(cols[:4*num_nodes+13])
        pdcocols=pdcocols.drop(cols[4*num_nodes+18:])
        pdcocols=pdcocols.to_list()
        pdcccols=cols.drop(cols[:4*num_nodes+18])
        pdcccols=pdcccols.drop(cols[4*num_nodes+23:])
        pdcccols=pdcccols.to_list()
        ppvcols=cols.drop(cols[:4*num_nodes+23])
        ppvcols=ppvcols.drop(cols[5*num_nodes+23:])
        ppvcols=ppvcols.to_list()
        zpvcols=cols.drop(cols[:5*num_nodes+23])
        zpvcols=zpvcols.drop(cols[6*num_nodes+23:])
        zpvcols=zpvcols.to_list()
        
        
        ndiffs=len(diffs)
        nov=len(ov)
        
        
        results=np.zeros([nform,ndiffs])
        times=np.zeros(nform)
        solvers=['*']*nform
        losses=np.zeros(nform)
        sigpi=np.zeros(nform)
        sigpd=np.zeros(nform)
        
        infeas1=np.zeros(nform)
        infeas2=np.zeros(nform)
        idx0=np.zeros(nform)
        
        for i in range(nform):
            times[i]=datadict[form[i]]['t'][0]
            solvers[i]=datadict[form[i]]['Solver'][0]
            losses[i]=np.sum(datadict[form[i]]['pl'])
            sigpi[i]=np.sum(probii[i])
            sigpd[i]=1/np.sum(probdi[i])                 
            idx0[i]=np.sum(datadict[form[i]]['pl']<0)              
            for h in range(H):
                vmat=np.zeros(num_nodes,dtype='complex')
                ppvmat=np.zeros(num_nodes)
                zpvmat=np.zeros(num_nodes)
                pgmat=np.zeros(num_nodes)                
                qgmat=np.zeros(num_nodes) 
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
                    zpvmat[j]=datadict[form[i]][zpvcols[j]][0]
                    if zpvmat[j]==1:
                        ppvmat[j]=datadict[form[i]][ppvcols[j]][0]
                    pdmc[j]=pdm[j]*dci[i][h][cnode[j]]
                    qdmc[j]=qdm[j]*dci[i][h][cnode[j]]                    
                
                infeas1[i]+=np.abs(np.sum(pgmat+(ppvmat*ici[i][h]*pveff)-eqpmat-pdmc))+np.abs(np.sum(qgmat-eqqmat-qdmc))                       
                infeas2[i]+=np.abs(np.sum(pgmat+(ppvmat*ici[i][h]*pveff)-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdmc))+np.abs(np.sum(qgmat-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdmc))
                    
                
        idx=np.logical_or(losses<0,idx0<0)
                
        idxi1=np.logical_or(infeas1>1e-3,infeas1<0)
        idxi2=np.logical_or(infeas2>1e-3,infeas2<0)
        
        mtt=np.multiply(np.abs(times),np.abs(losses))
        mtt=np.multiply(mtt,np.abs(infeas1))
        mtt=np.multiply(mtt,np.abs(infeas2))
        mtt=np.multiply(mtt,np.abs(sigpi))
        mtt=np.multiply(mtt,np.abs(sigpd))
        overview=np.zeros([nform,nov])
        
        resultsdf=pd.DataFrame(index=form,columns=diffs)
        overviewdf=pd.DataFrame(index=form,columns=ov)
        
        idxd=np.argsort(losses+sigpd+sigpi+(100*idx)+(100*idxi1)+(100*idxi2))
        idxl=np.argsort(losses+(100*idx)+(100*idxi1)+(100*idxi2))
        idxt=np.argsort(times+(100*idx)+(100*idxi1)+(100*idxi2))
        idxpx=np.argsort(mtt+(100*idx)+(100*idxi1)+(100*idxi2))
        idxinf1=np.argsort(infeas1+(100*idx)+(100*idxi1)+(100*idxi2))
        idxinf2=np.argsort(infeas2+(100*idx)+(100*idxi1)+(100*idxi2))
        idxpi=np.argsort(sigpi+(100*idx)+(100*idxi1)+(100*idxi2))
        idxpd=np.argsort(sigpd+(100*idx)+(100*idxi1)+(100*idxi2))
        
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
                    ii=np.where(idxl==i)[0][0]
                    if ii==0:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_positional(losses[i], precision=4)+'$'
                    elif ii==1:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_positional(losses[i], precision=4)+'$'
                    elif ii==2:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_positional(losses[i], precision=4)+'$'
                    else:
                        overviewdf[ov[j]][form[i]]='$'+np.format_float_positional(losses[i], precision=4)+'$'
                elif j==1:
                    overview[i][j]=sigpi[i]
                    ii=np.where(idxpi==i)[0][0]
                    if ii==0:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_positional(sigpi[i], precision=4)+'$'
                    elif ii==1:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_positional(sigpi[i], precision=4)+'$'
                    elif ii==2:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_positional(sigpi[i], precision=4)+'$'
                    else:
                        overviewdf[ov[j]][form[i]]='$'+np.format_float_positional(sigpi[i], precision=4)+'$'
                elif j==2:
                    overview[i][j]=sigpd[i]
                    ii=np.where(idxpd==i)[0][0]
                    if ii==0:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{green}$'+np.format_float_positional(sigpd[i], precision=4)+'$'
                    elif ii==1:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{yellow}$'+np.format_float_positional(sigpd[i], precision=4)+'$'
                    elif ii==2:
                        overviewdf[ov[j]][form[i]]='\\cellcolor{red}$'+np.format_float_positional(sigpd[i], precision=4)+'$'
                    else:
                        overviewdf[ov[j]][form[i]]='$'+np.format_float_positional(sigpd[i], precision=4)+'$'
                elif j==3:
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
                elif j==4:
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
                elif j==5:
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
                        
        npv=0 
        for i in range(num_nodes):
            if datadict[form[idxd[0]]][zpvcols[i]][0]==1:
                npv+=1
        
        csol= [['']*(ncluster+1) for i in range(H)]
        
        
        for h in range(H):
            for i in range(ncluster+1):
                if i==0:
                    csol[h][i]='$'+np.format_float_positional(ici[idxd[0]][h],precision=4)+'$'                    
                else:
                    csol[h][i]='$'+np.format_float_positional(dci[idxd[0]][h][i-1],precision=4)+'$'
                    
        bestsolzpv=np.zeros([H,num_nodes])
        bestsolpvc=np.zeros([H,num_nodes])
        bestsolic=np.zeros(H)
        bestsoldc=np.zeros([H,ncluster])
        
        nvarssol=npv+8
        varssol=[['']*nvarssol]
        sumcap=0
        sumnpv=0
        bestsolutions=[['']*nvarssol]
        k=0
        varssol[0][k]='Formulation'
        bestsolutions[0][k]=form[idxd[0]]
        k+=1
        varssol[0][k]='$f_{o}$'
        bestsolutions[0][k]='$'+np.format_float_scientific(losses[idxd[0]]-(1/sigpd[idxd[0]])+sigpi[idxd[0]], precision=4)+'$'
        k+=1
        varssol[0][k]='$P_{loss}$'
        bestsolutions[0][k]='$'+np.format_float_scientific(losses[idxd[0]], precision=4)+'$'
        k+=1
        varssol[0][k]='$-\\Sigma p(DC)$'
        bestsolutions[0][k]='$'+np.format_float_scientific((-1/sigpd[idxd[0]]), precision=4)+'$'
        k+=1
        varssol[0][k]='$\\Sigma p(I_{pv})$'
        bestsolutions[0][k]='$'+np.format_float_scientific(sigpi[idxd[0]], precision=4)+'$'
        k+=1        
        vmax=0
        vmin=2
        for j in range(num_nodes):
            vmax=np.max([vmax,np.max(datadict[form[idxd[0]]][vcols[j]].to_numpy())])
            vmin=np.min([vmin,np.min(datadict[form[idxd[0]]][vcols[j]].to_numpy())])
            if datadict[form[idxd[0]]][zpvcols[j]][0]==1:
                bestsolzpv[0][j]=1
                bestsolpvc[0][j]=datadict[form[idxd[0]]][ppvcols[j]][0]
                bestsolutions[0][k]+='$('+str(j+1)+','+np.format_float_scientific(datadict[form[idxd[0]]][ppvcols[j]][0], precision=4)+')$'
                varssol[0][k]+='$(Loc'+str(k-4)+',Cap'+str(k-4)+')$'                
                sumcap+=datadict[form[idxd[0]]][ppvcols[j]][0]
                sumnpv+=1
                k+=1
        for h in range(H):
            bestsolic[h]=datadict[form[idxd[0]]]['ic'][h]
            for c in range(ncluster):
                bestsoldc[h][c]=datadict[form[idxd[0]]][dccols[c]][h]
        
        varssol[0][k]='$\\Sigma (z_{i}^{pv},CP_{i})$'
        bestsolutions[0][k]='$('+str(sumnpv)+','+np.format_float_scientific(sumcap, precision=4)+')$'
        k+=1
        varssol[0][k]='$V_{max}$'
        bestsolutions[0][k]+='$'+np.format_float_scientific(vmax, precision=4)+'$'
        k+=1
        varssol[0][k]='$V_{min}$'
        bestsolutions[0][k]+='$'+np.format_float_scientific(vmin, precision=4)+'$'
        
        bestsolcol=[]
        
        for i in range(num_nodes):
            bestsolcol.append('zpv'+str(i+1))
        
        for i in range(num_nodes):
            bestsolcol.append('ppv'+str(i+1))
        
        bestsolcol.append('ic')
        
        for c in range(ncluster):
            bestsolcol.append('dc'+str(c+1))
        
        bestsolutiondf=pd.DataFrame(np.transpose(np.vstack((varssol,bestsolutions))),columns=['Variables','Values'])
        
        bestsolout=np.hstack((bestsolzpv,bestsolpvc,np.vstack(bestsolic),bestsoldc))
        bestsoldf=pd.DataFrame(bestsolout,columns=bestsolcol)
        
        bestsoldf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\bestsol.csv')
        
              
        metacols=metadict['Meta'].columns
        mplcol=metacols[0]
        mtcol=metacols[1]
        micol=['']*H
        
        micol1=metacols.drop(metacols[:(npv*2)+2])
        micol1=micol1.drop(metacols[-120:])
        micol1=micol1.to_list()
        
        for h in range(H):
            if 'ic'+str(h+1) in micol1:
                ii=micol1.index('ic'+str(h+1))
                micol[ii+7]=micol1[ii]
            else:
                micol[h]=str(0)
        
        
        mdcol=[['']*H for i in range(ncluster)]
        mdccol=metacols.drop(metacols[:-120])
        mdccol=mdccol.drop(metacols[-96:])
        mdccol=mdccol.to_list()
        mdcol[0]=mdccol
        mdccol=metacols.drop(metacols[:-96])
        mdccol=mdccol.drop(metacols[-72:])
        mdccol=mdccol.to_list()
        mdcol[1]=mdccol
        mdccol=metacols.drop(metacols[:-72])
        mdccol=mdccol.drop(metacols[-48:])
        mdccol=mdccol.to_list()
        mdcol[2]=mdccol
        mdccol=metacols.drop(metacols[:-48])
        mdccol=mdccol.drop(metacols[-24:])
        mdccol=mdccol.to_list()
        mdcol[3]=mdccol
        mdccol=metacols.drop(metacols[:-24])        
        mdccol=mdccol.to_list()
        mdcol[4]=mdccol      
        
        ntmeta=len(metadict['Meta'])
        
        
        mprobi=np.zeros([ntmeta,H])
        mprobd=np.zeros([ntmeta,H,ncluster])
        mlosses=np.zeros(ntmeta)
        
        for i in range(ntmeta):
            mlosses[i]=metadict['Meta'][mplcol][i]
            for h in range(H):
                for c in range(ncluster):
                    distribution=adists[bestfitsd[c][h]]
                    "Exponential"
                    if distribution==adists[0]:
                        mprobd[i][h][c]=1-np.exp(-dparams[c][h][0]*metadict['Meta'][mdcol[c][h]][i])            
                    "Fisk  (Beta is always positive)"
                    if distribution==adists[1]:
                        mprobd[i][h][c]=1/(1+np.power(metadict['Meta'][mdcol[c][h]][i]/dparams[c][h][2],-dparams[c][h][3]))                          
                    "Logistic"                        
                    if distribution==adists[2]:                
                        mprobd[i][h][c]=1/(1+np.exp((-metadict['Meta'][mdcol[c][h]][i]+dparams[c][h][4])/dparams[c][h][5]))
                    "LogNorm"                        
                    if distribution==adists[3]:                
                        mprobd[i][h][c]=ss.norm.cdf(np.log(metadict['Meta'][mdcol[c][h]][i]),dparams[c][h][6],dparams[c][h][7])
                    "Normal"
                    if distribution==adists[4]:                
                        mprobd[i][h][c]=ss.norm.cdf(metadict['Meta'][mdcol[c][h]][i],dparams[c][h][8],dparams[c][h][9])
                    "Rayleigh"
                    if distribution==adists[5]:                
                        mprobd[i][h][c]=1-np.exp(-np.square(metadict['Meta'][mdcol[c][h]][i])/(2*np.square(dparams[c][h][10])))
                    "Weibull"
                    if distribution==adists[6]:                
                        mprobd[i][h][c]=1-np.exp(-np.power(metadict['Meta'][mdcol[c][h]][i]*dparams[c][h][12],dparams[c][h][13]))
                if bestfitsi[h]!=10:
                    distribution=adists[bestfitsi[h]]
                    "Exponential"
                    if distribution==adists[0]:               
                        mprobi[i][h]=1-np.exp(-iparams[h][0]*metadict['Meta'][micol[h]][i])            
                    "Fisk  (Beta is always positive)"
                    if distribution==adists[1]:
                        mprobi[i][h]=1/(1+np.power(metadict['Meta'][micol[h]][i]/iparams[h][2],-iparams[h][3]))                          
                    "Logistic"                        
                    if distribution==adists[2]:                
                        mprobi[i][h]=1/(1+np.exp((-metadict['Meta'][micol[h]][i]+iparams[h][4])/iparams[h][5]))
                    "LogNorm"                        
                    if distribution==adists[3]:                
                        mprobi[i][h]=ss.norm.cdf(np.log(metadict['Meta'][micol[h]][i]),iparams[h][6],iparams[h][7])
                    "Normal"
                    if distribution==adists[4]:                
                        mprobi[i][h]=ss.norm.cdf(metadict['Meta'][micol[h]][i],iparams[h][8],iparams[h][9])
                    "Rayleigh"
                    if distribution==adists[5]:                
                        mprobi[i][h]=1-np.exp(-np.square(metadict['Meta'][micol[h]][i])/(2*np.square(iparams[h][10])))
                    "Weibull"
                    if distribution==adists[6]:                
                        mprobi[i][h]=1-np.exp(-np.power(metadict['Meta'][micol[h]][i]*iparams[h][12],iparams[h][13]))
            
            
        
        for j in range(nmov):
            if j==0:
                metaoverviewdf[metaov[j]][case[ca]]='$'+str(ntmeta)+'$'
            if j==1:
                objsum=mlosses-np.sum(mprobd,axis=(1,2))+np.sum(mprobi,axis=1)
                ii=np.argsort(objsum)[0]
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(objsum[ii], precision=4)+'$'
            if j==2:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.mean(objsum), precision=4)+'$'
            if j==3:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.std(objsum), precision=4)+'$'
            if j==4:
                ii=np.argsort(metadict['Meta']['t'])[0]
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(metadict['Meta']['t'][ii], precision=4)+'$'
            if j==5:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.mean(metadict['Meta']['t'].to_numpy()), precision=4)+'$'
            if j==6:
                metaoverviewdf[metaov[j]][case[ca]]='$'+np.format_float_scientific(np.std(metadict['Meta']['t'].to_numpy()), precision=4)+'$'
        
        for h in range(H):
            bestic[h][ca+1]=ici[idxd[0]][h]
            bestpic[h][ca+1]=probii[idxd[0]][h]
            
            dfi[dicols[ca+1]][hindex[h]]='$'+np.format_float_scientific(ici[idxd[0]][h],precision=2)+'$'
            dfpi[dpicols[ca+1]][hindex[h]]='$'+np.format_float_scientific(probii[idxd[0]][h],precision=2)+'$'
            
            bestdc[h][ca+1]=dci[idxd[0]][h]           
            
            dfd1[ddcols1[ca+1]][hindex[h]]='$'+np.format_float_scientific(dci[idxd[0]][h][0],precision=2)+'$'
            dfd2[ddcols2[ca+1]][hindex[h]]='$'+np.format_float_scientific(dci[idxd[0]][h][1],precision=2)+'$'
            dfd3[ddcols3[ca+1]][hindex[h]]='$'+np.format_float_scientific(dci[idxd[0]][h][2],precision=2)+'$'
            dfd4[ddcols4[ca+1]][hindex[h]]='$'+np.format_float_scientific(dci[idxd[0]][h][3],precision=2)+'$'
            dfd5[ddcols5[ca+1]][hindex[h]]='$'+np.format_float_scientific(dci[idxd[0]][h][4],precision=2)+'$'
            
            bestpdc[h][ca+1]=probdi[idxd[0]][h]
                              
            dfpd1[dpdcols1[ca+1]][hindex[h]]='$'+np.format_float_scientific(probdi[idxd[0]][h][0],precision=2)+'$'
            dfpd2[dpdcols2[ca+1]][hindex[h]]='$'+np.format_float_scientific(probdi[idxd[0]][h][1],precision=2)+'$'
            dfpd3[dpdcols3[ca+1]][hindex[h]]='$'+np.format_float_scientific(probdi[idxd[0]][h][2],precision=2)+'$'
            dfpd4[dpdcols4[ca+1]][hindex[h]]='$'+np.format_float_scientific(probdi[idxd[0]][h][3],precision=2)+'$'
            dfpd5[dpdcols5[ca+1]][hindex[h]]='$'+np.format_float_scientific(probdi[idxd[0]][h][4],precision=2)+'$'
        
        bestform[ca]=int(idxd[0])
        
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
        ov1=['$P_{loss}$','$\\Sigma p(I_{pv})$','$1/\\Sigma p(DC)$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
        resultsdf1=pd.DataFrame(results,index=form,columns=diffs1)
        overviewdf1=pd.DataFrame(overview,index=form,columns=ov1)
        resultsdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\matdifs.csv')
        overviewdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\overview.csv')
        
        latexcase=['\\multirow{'+str(nov)+'}{*}{'+case[ca]+'}']
        
        bestloss=pd.DataFrame([[ov1[0],form[idxl[0]],solvers[idxl[0]]
                                ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxl[0]][0], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxl[0]][1], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxl[0]][2], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxl[0]][3], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxl[0]][4], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxl[0]][5], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxl[0]][6], precision=2)+'$']]
                              ,index=latexcase,columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        bestsigpi=pd.DataFrame([[ov1[1],form[idxpi[0]],solvers[idxpi[0]]
                                 ,'$'+np.format_float_scientific(overview[idxpi[0]][0], precision=2)+'$'
                                 ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxpi[0]][1], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpi[0]][2], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpi[0]][3], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpi[0]][4], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpi[0]][5], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpi[0]][6], precision=2)+'$']]
                               ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        bestsigpd=pd.DataFrame([[ov1[2],form[idxpd[0]],solvers[idxpd[0]]
                                 ,'$'+np.format_float_scientific(overview[idxpd[0]][0], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpd[0]][1], precision=2)+'$'
                                 ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxpd[0]][2], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpd[0]][3], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpd[0]][4], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpd[0]][5], precision=2)+'$'
                                 ,'$'+np.format_float_scientific(overview[idxpd[0]][6], precision=2)+'$']]
                               ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        bestt=pd.DataFrame([[ov1[3],form[idxt[0]],solvers[idxt[0]]
                             ,'$'+np.format_float_scientific(overview[idxt[0]][0], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][1], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][2], precision=2)+'$'
                             ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxt[0]][3], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][4], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][5], precision=2)+'$'
                             ,'$'+np.format_float_scientific(overview[idxt[0]][6], precision=2)+'$']]
                           ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        bestinf1=pd.DataFrame([[ov1[4],form[idxinf1[0]],solvers[idxinf1[0]]
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][0], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][1], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][2], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][3], precision=2)+'$'
                                ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf1[0]][4], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][5], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf1[0]][6], precision=2)+'$']]
                              ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        bestinf2=pd.DataFrame([[ov1[5],form[idxinf2[0]],solvers[idxinf2[0]]
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][0], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][1], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][2], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][3], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][4], precision=2)+'$'
                                ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxinf2[0]][5], precision=2)+'$'
                                ,'$'+np.format_float_scientific(overview[idxinf2[0]][6], precision=2)+'$']]
                              ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        bestpx=pd.DataFrame([[ov1[6],form[idxpx[0]],solvers[idxpx[0]]
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][0], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][1], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][2], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][3], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][4], precision=2)+'$'
                              ,'$'+np.format_float_scientific(overview[idxpx[0]][5], precision=2)+'$'
                              ,'$\\cellcolor{green}'+np.format_float_scientific(overview[idxpx[0]][6], precision=2)+'$']]
                            ,index=[''],columns=['$x$','F','I/S',ov1[0],ov1[1],ov1[2],ov1[3],ov1[4],ov1[5],ov1[6]])
        
        
        bestresult=pd.concat([bestloss,bestsigpi,bestsigpd,bestt,bestinf1,bestinf2,bestpx])
        bestresult.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\best.csv')
            
        
        latexbestsolution = bestsolutiondf.to_latex(index=False, escape=False, column_format='|c c|')        
        
        file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\BestSolution_tab.txt','w')
        file.write(latexbestsolution)
        file.close()
    
    besticcols=dicols
    bestpicols=dpicols
    
    bestdccols1=ddcols1
    bestpdcols1=dpdcols1
    bestdccols2=ddcols2
    bestpdcols2=dpdcols2
    bestdccols3=ddcols3
    bestpdcols3=dpdcols3
    bestdccols4=ddcols4
    bestpdcols4=dpdcols4
    bestdccols5=ddcols5
    bestpdcols5=dpdcols5
    
    for i in range(nca):
        besticcols[i+1]=besticcols[i+1]+(form[bestform[i]],)
        bestpicols[i+1]=bestpicols[i+1]+(form[bestform[i]],)
        
        bestdccols1[i+1]=bestdccols1[i+1]+(form[bestform[i]],)
        bestdccols2[i+1]=bestdccols2[i+1]+(form[bestform[i]],)
        bestdccols3[i+1]=bestdccols3[i+1]+(form[bestform[i]],)
        bestdccols4[i+1]=bestdccols4[i+1]+(form[bestform[i]],)
        bestdccols5[i+1]=bestdccols5[i+1]+(form[bestform[i]],)
        
        bestpdcols1[i+1]=bestpdcols1[i+1]+(form[bestform[i]],)
        bestpdcols2[i+1]=bestpdcols2[i+1]+(form[bestform[i]],)
        bestpdcols3[i+1]=bestpdcols3[i+1]+(form[bestform[i]],)
        bestpdcols4[i+1]=bestpdcols4[i+1]+(form[bestform[i]],)
        bestpdcols5[i+1]=bestpdcols5[i+1]+(form[bestform[i]],)
        
        
    besticdf=pd.DataFrame(bestic,columns=besticcols)
    besticdf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_ic.csv')
    
    bestpidf=pd.DataFrame(bestpic,columns=bestpicols)
    bestpidf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_p_ic.csv')
    
    bestdcdf1=pd.DataFrame(bestdc[:,:,0],columns=bestdccols1)
    bestdcdf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_dc1.csv')
    
    bestdcdf2=pd.DataFrame(bestdc[:,:,1],columns=bestdccols2)
    bestdcdf2.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_dc2.csv')
    
    bestdcdf3=pd.DataFrame(bestdc[:,:,2],columns=bestdccols3)
    bestdcdf3.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_dc3.csv')
    
    bestdcdf4=pd.DataFrame(bestdc[:,:,3],columns=bestdccols4)
    bestdcdf4.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_dc4.csv')
    
    bestdcdf5=pd.DataFrame(bestdc[:,:,4],columns=bestdccols5)
    bestdcdf5.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_dc5.csv')
    
    bestpddf1=pd.DataFrame(bestpdc[:,:,0],columns=bestpdcols1)
    bestpddf1.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_p_dc1.csv')
    
    bestpddf2=pd.DataFrame(bestpdc[:,:,1],columns=bestpdcols2)
    bestpddf2.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_p_dc2.csv')
    
    bestpddf3=pd.DataFrame(bestpdc[:,:,2],columns=bestpdcols3)
    bestpddf3.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_p_dc3.csv')
    
    bestpddf4=pd.DataFrame(bestpdc[:,:,3],columns=bestpdcols4)
    bestpddf4.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_p_dc4.csv')
    
    bestpddf5=pd.DataFrame(bestpdc[:,:,4],columns=bestpdcols5)
    bestpddf5.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\best_p_dc5.csv')   
    
    
    latexic=dfi.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\IC_tab.txt','w')
    file.write(latexic)
    file.close()
    
    latexpic=dfpi.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\P_IC_tab.txt','w')
    file.write(latexpic)
    file.close()
    
    latexdc1=dfd1.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\DC1_tab.txt','w')
    file.write(latexdc1)
    file.close()
    
    latexdc2=dfd2.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\DC2_tab.txt','w')
    file.write(latexdc2)
    file.close()
    
    latexdc3=dfd3.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\DC3_tab.txt','w')
    file.write(latexdc3)
    file.close()
    
    latexdc4=dfd4.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\DC4_tab.txt','w')
    file.write(latexdc4)
    file.close()
    
    latexdc5=dfd5.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\DC5_tab.txt','w')
    file.write(latexdc5)
    file.close()
    
    latexpdc1=dfpd1.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\P_DC1_tab.txt','w')
    file.write(latexpdc1)
    file.close()
    
    latexpdc2=dfpd2.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\P_DC2_tab.txt','w')
    file.write(latexpdc2)
    file.close()
    
    latexpdc3=dfpd3.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\P_DC3_tab.txt','w')
    file.write(latexpdc3)
    file.close()
    
    latexpdc4=dfpd4.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\P_DC4_tab.txt','w')
    file.write(latexpdc4)
    file.close()
    
    latexpdc5=dfpd5.to_latex(index=True, escape=False, column_format='|c| c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\P_DC5_tab.txt','w')
    file.write(latexpdc5)
    file.close()    
             
    latexmetas = metaoverviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c c|')
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\Metas_tab.txt','w')
    file.write(latexmetas)
    file.close()
    
    