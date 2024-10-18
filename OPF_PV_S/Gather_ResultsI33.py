# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:12:04 2024

@author: diego
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
import scipy.stats as ss



city=['Bog','Jam','Pop']
problem='OPF_PV_S'
city1=['BOG','JAM','POP']
adists=['Exponential','Fisk','Logistic','Log-N','Normal','Rayleigh','Weibull']
aclust=['c1','c2','c3','c4','c5']

nct=len(city)
pveff=0.8

datadict={}
bestdict={} 

"----- Read the database -----"
branch = pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Branch.csv")
bus= pd.read_csv("C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\IEEE33Bus.csv")

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
    

probic={}
probdc1={}
probdc2={}
probdc3={}
probdc4={}
probdc5={}
icc={}
dcc1={}
dcc2={}
dcc3={}
dcc4={}
dcc5={}
 
times=np.zeros(nct)
solvers=['*']*nct
losses=np.zeros(nct)
sigpi=np.zeros(nct)
sigpd=np.zeros(nct)

infeas1=np.zeros(nct)
infeas2=np.zeros(nct)
idx0=np.zeros(nct)


ov=['$P_{loss}$','$\\Sigma p(I_{pv})$','$1/\\Sigma p(DC)$','$t[s]$','Inf1','Inf2','$\\Pi |x|$']
nov=len(ov)

overview=np.zeros([nct,nov])
overviewdf=pd.DataFrame(index=city,columns=ov)

for ct in range(nct):    
    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\MeansRAD_'+city1[ct]+'.mat')
    imeans=np.squeeze(mat['means'])

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\ClusterMeans_'+city1[ct]+'.mat')
    dmeans=np.squeeze(mat['clustermeans']).T 
    
    irr = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\'+'ParamTable.csv')
 
    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\NSTparamDem_'+city1[ct]+'.mat')
    dparams=mat['params']
 
    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\NSTparamRAD_'+city1[ct]+'.mat')
    iparams=mat['params']

    mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\IEEE33_'+city[ct]+'_'+'ClusterNode.mat')
    cnode=np.squeeze(mat['clusternode'])
 
    cnode[0]=1
    cnode=cnode-1
    
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
                
    probic.update({(city[ct],'Mean'):probim})
    probdc1.update({(city[ct],'Mean'):probdm[:,0]})
    probdc2.update({(city[ct],'Mean'):probdm[:,1]})
    probdc3.update({(city[ct],'Mean'):probdm[:,2]})
    probdc4.update({(city[ct],'Mean'):probdm[:,3]})
    probdc5.update({(city[ct],'Mean'):probdm[:,4]})
    
    icc.update({(city[ct],'Mean'):imeans})
    dcc1.update({(city[ct],'Mean'):dmeans[:,0]})
    dcc2.update({(city[ct],'Mean'):dmeans[:,1]})
    dcc3.update({(city[ct],'Mean'):dmeans[:,2]})
    dcc4.update({(city[ct],'Mean'):dmeans[:,3]})
    dcc5.update({(city[ct],'Mean'):dmeans[:,4]})
    
   
         
    datadict.update({city[ct]:pd.read_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\IEEE33\\Convex\\BF\\Results.xlsx',index_col=0)})
    probi=np.zeros(H)
    probd=np.zeros([H,ncluster])
    
    probi=datadict[city[ct]]['prob_ic_cal'].to_numpy()
    probd[:,0]=datadict[city[ct]]['prob_dc_cal1'].to_numpy()
    probd[:,1]=datadict[city[ct]]['prob_dc_cal2'].to_numpy()
    probd[:,2]=datadict[city[ct]]['prob_dc_cal3'].to_numpy()
    probd[:,3]=datadict[city[ct]]['prob_dc_cal4'].to_numpy()
    probd[:,4]=datadict[city[ct]]['prob_dc_cal5'].to_numpy()
    
    
    probic.update({(city[ct],'Res'):probi})
    probdc1.update({(city[ct],'Res'):probd[:,0]})
    probdc2.update({(city[ct],'Res'):probd[:,1]})
    probdc3.update({(city[ct],'Res'):probd[:,2]})
    probdc4.update({(city[ct],'Res'):probd[:,3]})
    probdc5.update({(city[ct],'Res'):probd[:,4]})
       
    
    
    "----- Read the results -----" 
    
    cols=datadict[city[0]].columns       
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
    ppvcols=cols.drop(cols[:4*num_nodes+18])
    ppvcols=ppvcols.drop(cols[5*num_nodes+18:])
    ppvcols=ppvcols.to_list()
    zpvcols=cols.drop(cols[:5*num_nodes+18])
    zpvcols=zpvcols.drop(cols[6*num_nodes+18:])
    zpvcols=zpvcols.to_list()
    
    
    
    
    times[ct]=datadict[city[ct]]['t'][0]
    solvers[ct]=datadict[city[ct]]['Solver'][0]
    losses[ct]=np.sum(datadict[city[ct]]['pl'])
    sigpi[ct]=np.sum(probi)
    sigpd[ct]=1/np.sum(probd)             
    idx0[ct]=np.sum(datadict[city[ct]]['pl']<0)
    dcmat=np.zeros([H,ncluster])
    icmat=datadict[city[ct]]['ic'].to_numpy()
              
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
        pgmat[0]=datadict[city[ct]]['pg'][h]
        qgmat[0]=datadict[city[ct]]['qg'][h]
        
        for c in range(ncluster):
            dcmat[h][c]=datadict[city[ct]][dccols[c]][h]
            
        for j in range(num_nodes):
            vmat[j]=np.multiply(datadict[city[ct]][vcols[j]][h],np.cos(np.abs(datadict[city[ct]][phcols[j]][h])))+1j*np.multiply(datadict[city[ct]][vcols[j]][h],np.sin(datadict[city[ct]][phcols[j]][h]))
            eqpmat[j]=datadict[city[ct]][eqpcols[j]][h]
            eqqmat[j]=datadict[city[ct]][eqqcols[j]][h]
            zpvmat[j]=datadict[city[ct]][zpvcols[j]][0]
            if zpvmat[j]==1:
                ppvmat[j]=datadict[city[ct]][ppvcols[j]][0]
            
            pdmc[j]=pdm[j]*dcmat[h][cnode[j]]
            qdmc[j]=qdm[j]*dcmat[h][cnode[j]]                    
        
        infeas1[ct]+=np.abs(np.sum(pgmat+(ppvmat*icmat[h]*pveff)-eqpmat-pdmc))+np.abs(np.sum(qgmat-eqqmat-qdmc))                       
        infeas2[ct]+=np.abs(np.sum(pgmat+(ppvmat*icmat[h]*pveff)-np.real(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-pdmc))+np.abs(np.sum(qgmat-np.imag(np.multiply(vmat,np.matmul(ym.conjugate(),vmat.conjugate())))-qdmc))
    
    
    icc.update({(city[ct],'Res'):icmat})
    dcc1.update({(city[ct],'Res'):dcmat[:,0]})
    dcc2.update({(city[ct],'Res'):dcmat[:,1]})
    dcc3.update({(city[ct],'Res'):dcmat[:,2]})
    dcc4.update({(city[ct],'Res'):dcmat[:,3]})
    dcc5.update({(city[ct],'Res'):dcmat[:,4]})
            
    mtt=np.multiply(np.abs(times),np.abs(losses))
    mtt=np.multiply(mtt,np.abs(infeas1))
    mtt=np.multiply(mtt,np.abs(infeas2))
    mtt=np.multiply(mtt,np.abs(sigpi))
    mtt=np.multiply(mtt,np.abs(sigpd))               
    for j in range(nov):
        if j==0:
            overview[ct][j]=losses[ct]            
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(losses[ct], precision=2)+'$'            
        elif j==1:
            overview[ct][j]=sigpi[ct]
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(sigpi[ct], precision=2)+'$'
            
        elif j==2:
            overview[ct][j]=sigpd[ct]
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(sigpd[ct], precision=2)+'$'
        elif j==3:
            overview[ct][j]=times[ct]
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(times[ct], precision=2)+'$'
        elif j==4:
            overview[ct][j]=infeas1[ct]
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(infeas1[ct], precision=2)+'$'            
        elif j==5:
            overview[ct][j]=infeas2[ct]
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(infeas2[ct], precision=2)+'$'            
        else:
            overview[ct][j]=mtt[ct]                
            overviewdf[ov[j]][city[ct]]='$'+np.format_float_scientific(mtt[ct], precision=2)+'$'
                
    npv=0 
    for i in range(num_nodes):
        if datadict[city[ct]][zpvcols[i]][0]==1:
            npv+=1
    
    
    nvarssol=npv+6
    varssol=[['']*nvarssol]
    sumcap=0
    sumnpv=0
    solutions=[['']*nvarssol]
    k=0
    varssol[0][k]='$f_{o}$'
    solutions[0][k]='$'+np.format_float_scientific(losses[ct]-(1/sigpd[ct])+sigpi[ct], precision=4)+'$'
    k+=1        
    varssol[0][k]='$P_{loss}$'
    solutions[0][k]='$'+np.format_float_scientific(losses[ct], precision=4)+'$'
    k+=1
    varssol[0][k]='$-\\Sigma p(DC)$'
    solutions[0][k]='$'+np.format_float_scientific((-1/sigpd[ct]), precision=4)+'$'
    k+=1
    varssol[0][k]='$\\Sigma p(I_{pv})$'
    solutions[0][k]='$'+np.format_float_scientific(sigpi[ct], precision=4)+'$'
    k+=1
    vmax=0
    vmin=2
    for j in range(num_nodes):
        vmax=np.max([vmax,np.max(datadict[city[ct]][vcols[j]].to_numpy())])
        vmin=np.min([vmin,np.min(datadict[city[ct]][vcols[j]].to_numpy())])
        if datadict[city[ct]][zpvcols[j]][0]==1:                          
            sumcap=datadict[city[ct]][ppvcols[j]][0]
            sumnpv=j            
    
    varssol[0][k]='$\\Sigma (z_{i}^{pv},CP_{i})$'
    solutions[0][k]='$('+str(sumnpv)+','+np.format_float_scientific(sumcap, precision=4)+')$'
    k+=1
    varssol[0][k]='$V_{max}$'
    solutions[0][k]+='$'+np.format_float_scientific(vmax, precision=4)+'$'
    k+=1
    varssol[0][k]='$V_{min}$'
    solutions[0][k]+='$'+np.format_float_scientific(vmin, precision=4)+'$'
    
    solutiondf=pd.DataFrame(np.transpose(np.vstack((varssol,solutions))),columns=['Variables','Values'])
    
    latexsolution = solutiondf.to_latex(index=False, escape=False, column_format='|c c|')        
    
    file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\IEEE33\\SolutionForm_tab.txt','w')
    file.write(latexsolution)
    file.close()


icdf=pd.DataFrame.from_dict(icc)
dc1df=pd.DataFrame.from_dict(dcc1)
dc2df=pd.DataFrame.from_dict(dcc2)
dc3df=pd.DataFrame.from_dict(dcc3)
dc4df=pd.DataFrame.from_dict(dcc4)
dc5df=pd.DataFrame.from_dict(dcc5)

pidf=pd.DataFrame.from_dict(probic)
pd1df=pd.DataFrame.from_dict(probdc1)
pd2df=pd.DataFrame.from_dict(probdc2)
pd3df=pd.DataFrame.from_dict(probdc3)
pd4df=pd.DataFrame.from_dict(probdc4)
pd5df=pd.DataFrame.from_dict(probdc5)

icdf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\ic_form_tab.csv', index=False) 
pidf.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\pi_form_tab.csv', index=False)

dc1df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\dc1_form_tab.csv', index=False) 
pd1df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\pd1_form_tab.csv', index=False)

dc2df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\dc2_form_tab.csv', index=False) 
pd2df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\pd2_form_tab.csv', index=False)  
  
dc3df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\dc3_form_tab.csv', index=False) 
pd3df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\pd3_form_tab.csv', index=False)  

dc4df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\dc4_form_tab.csv', index=False) 
pd4df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\pd4_form_tab.csv', index=False)  

dc5df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\dc5_form_tab.csv', index=False) 
pd5df.to_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\pd5_form_tab.csv', index=False)  

overviewdf.insert(0,'I/S',solvers)
        
latexoverview = overviewdf.to_latex(index=True, escape=False, column_format='|c| c c c c c c|')           
        
file = open('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\OverviewForm_tab.txt','w')
file.write(latexoverview)
file.close()

              
            




     
            
