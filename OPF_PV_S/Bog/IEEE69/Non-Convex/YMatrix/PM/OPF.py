# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:29:05 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL, DataFrame
from scipy.io import loadmat
from scipy.special import factorial
import scipy.stats as ss

case='IEEE69'
city='Bog'
city1='BOG'
problem='OPF_PV'

"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=np.squeeze(mat['clusternode'])

cnode[0]=1


c1 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC1.csv')
c2 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC2.csv')
c3 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC3.csv')
c4 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC4.csv')
c5 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC5.csv')


irr = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\'+'ParamTable.csv')

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\NSTparamDem_'+city1+'.mat')
dparams=np.array(mat['params'])

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\NSTparamRAD_'+city1+'.mat')
iparams=mat['params']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\PWLxyRAD_'+city1+'.mat')
xirrpwl=mat['xpwl']
yirrpwl=mat['ypwl']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\PWLxyDem_'+city1+'.mat')
xdempwl=mat['xpwl']
ydempwl=mat['ypwl']


adists=['Exponential','Fisk','Logistic','Log-N','Normal','Rayleigh','Weibull'];
aclust=['c1','c2','c3','c4','c5']

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
                   bestfitsi[h]=j+1 
                   
    for i in range(ncluster):
        for j in range(ndists):
            if adists[j] in cparameters['bestparams1'][aclust[i],2*h] and cparameters['bestparams1'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd[i][h]=j+1


num_lines = len(branch)
num_nodes=len(bus)
iref=np.where(bus['type']==3)[0][0]

sd=np.zeros(num_nodes,dtype='complex')

for k in range(num_lines):
    sd[branch['j'][k]-1]=(branch['pj'][k]+1j*branch['qj'][k])
    
pdm=np.real(sd)
qdm=np.imag(sd)

ngen=np.sum(bus['type']==2)
pgen=np.zeros(num_nodes)
qgen=np.zeros(num_nodes)
vgen=np.zeros(num_nodes)

vmax=np.array(bus['vmax'])
vmin=np.array(bus['vmin'])

if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]
        
vmax=vmax+0.1
vmin=vmin-0.1 

vmax[iref]=1
vmin[iref]=1

phmin=np.ones(num_nodes)*(-np.pi)
phmax=np.ones(num_nodes)*(np.pi)

phmin[iref]=0
phmax[iref]=0

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]

ym=np.zeros([num_nodes,num_nodes],dtype='complex')

idx1=np.zeros([num_nodes,num_nodes])

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])
    idx1[fr][to]=1
    idx1[to][fr]=1
    
for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    idx1[i][i]=1

idx2=idx1-np.eye(num_nodes)

yr=np.real(ym)
yi=np.imag(ym)


cpdem=np.zeros([num_nodes,ncluster])
cqdem=np.zeros([num_nodes,ncluster])

for i in range(num_nodes):
    for c in range(ncluster):
        if cnode[i]-1==c:
            cpdem[i][c]=pdm[i]
            cqdem[i][c]=qdm[i]



#npv=np.floor(0.1*num_nodes)
npv=3
pvcmax=0.5*np.sum(np.real(sd))

pveff=0.8

dparamsdf =pd.concat([pd.DataFrame(dparams[i][h],index=np.arange(1,(2*ndists)+1)) for i in range(ncluster) for h in range(H)], keys=[(i+1,h+1) for i in range(ncluster) for h in range(H)])

nnorm=20
normden=np.zeros(nnorm)
normnum=np.zeros(nnorm)
for i in range(nnorm):
    normden[i]=factorial(i)*np.power(2,i)*((2*i)+1)
    normnum[i]=np.power(-1,i)
    

mindc=np.zeros([H,ncluster])
maxdc=np.zeros([H,ncluster])

minic=np.zeros(H)
maxic=np.zeros(H)

for h in range(H):
    for c in range(ncluster):
        dist=bestfitsd[c][h]-1
        mindc[h][c]=np.min(xdempwl[h][c][dist])
        maxdc[h][c]=np.max(xdempwl[h][c][dist])
    dist=bestfitsi[h]-1
    if dist!=9:
        minic[h]=np.min(xirrpwl[h][dist])
        maxic[h]=np.max(xirrpwl[h][dist])

"----- Optimization model -----"

ampl = AMPL()

ampl.eval(
    r"""
    param nn;
    param nh;
    param nc;
    param nd;
    param nnorm;       
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nh").set(H)
ampl.get_parameter("nc").set(ncluster)
ampl.get_parameter("nd").set(ndists*2)
ampl.get_parameter("nnorm").set(nnorm)

ampl.eval(
    r"""
    set N :=1..nn;
    set H=1..nh;
    set C=1..nc;
    set D=1..nd;
    set NN=1..nnorm;
    
    param Yr {N,N};
    param Yi {N,N};    
    param vmax{N};
    param vmin{N};
    param phmax{N};
    param phmin{N};
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};
    param idx1{N,N};
    param idx2{N,N};
    param cnode{N};
    param iparams{H,D};        
    param npv;
    param pvcmax;
    param pveff;
    param cpdem{N,C};
    param cqdem{N,C};
    param bestd{H,C};
    param besti{H};
    param pi;    
    param dparams{C,H,D};    
    param normnum{NN};    
    param normden{NN};
    param dcmax{H,C};
    param dcmin{H,C};
    param icmax{H};
    param icmin{H};
              
"""
)

ampl.get_parameter("Yr").set_values(yr)
ampl.get_parameter("Yi").set_values(yi)
ampl.get_parameter("vmax").set_values(vmax)
ampl.get_parameter("vmin").set_values(vmin)
ampl.get_parameter("phmax").set_values(phmax)
ampl.get_parameter("phmin").set_values(phmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
ampl.get_parameter("idx1").set_values(idx1)
ampl.get_parameter("idx2").set_values(idx2) 
ampl.get_parameter("cnode").set_values(cnode)
ampl.get_parameter("iparams").set_values(iparams)
ampl.get_parameter("npv").set(npv)  
ampl.get_parameter("pvcmax").set(pvcmax)
ampl.get_parameter("pveff").set(pveff)
ampl.get_parameter("cpdem").set_values(cpdem)
ampl.get_parameter("cqdem").set_values(cqdem)
ampl.get_parameter("bestd").set_values(np.array(bestfitsd)) 
ampl.get_parameter("besti").set_values(np.array(bestfitsi))
ampl.get_parameter("pi").set(np.pi) 
ampl.get_parameter("dparams").set_values(DataFrame.from_pandas(dparamsdf))
ampl.get_parameter("normnum").set_values(normnum) 
ampl.get_parameter("normden").set_values(normden)
ampl.get_parameter("dcmax").set_values(maxdc)
ampl.get_parameter("dcmin").set_values(mindc)
ampl.get_parameter("icmax").set_values(maxic)
ampl.get_parameter("icmin").set_values(minic)               

ampl.eval(
    r"""
    var  v{h in H,i in N} >= vmin[i], <= vmax[i];
    var  ph{h in H,i in N} >= phmin[i], <= phmax[i];
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    
    var  pv{i in N} >=0, <=pvcmax;
    var  ppv{i in N} >=0, <=pvcmax;
    var  zpv{i in N} binary;
    
    var ic{h in H} >=icmin[h], <=icmax[h];
    var dc{h in H,i in C} >=dcmin[h,i], <=dcmax[h,i]; 
    
    var probd{h in H,i in C} >=0,<=1;
    var probi{h in H} >=0, <=1;
    
    minimize Losses:
       sum{h in H,i in N, j in N} v[h,i]*v[h,j]*Yr[i,j]*cos(ph[h,i]-ph[h,j])
       +sum{h in H,i in N, j in N} v[h,i]*v[h,j]*Yi[i,j]*sin(ph[h,i]-ph[h,j])
       +sum{h in H,i in N, j in N} v[h,i]*v[h,j]*Yr[i,j]*sin(ph[h,i]-ph[h,j])
       -sum{h in H,i in N, j in N} v[h,i]*v[h,j]*Yi[i,j]*cos(ph[h,i]-ph[h,j])
       -sum{h in H, c in C} probd[h,c]
       +sum{h in H} probi[h];
       
    subject to PB {h in H,i in N}: 
       pg[h,i]+(ppv[i]*ic[h]*pveff)-sum{c in C}(cpdem[i,c]*dc[h,c]) = sum {j in N} (v[h,i]*v[h,j]*Yr[i,j]*cos(ph[h,i]-ph[h,j]))+sum {j in N} (v[h,i]*v[h,j]*Yi[i,j]*sin(ph[h,i]-ph[h,j]));
       
    subject to QB {h in H,i in N}: 
       qg[h,i]-sum{c in C}(cqdem[i,c]*dc[h,c]) = sum {j in N} (v[h,i]*v[h,j]*Yr[i,j]*sin(ph[h,i]-ph[h,j]))-sum {j in N} (v[h,i]*v[h,j]*Yi[i,j]*cos(ph[h,i]-ph[h,j]));
    
    subject to PVLIN1 {i in N}:
        ppv[i]<=pvcmax*zpv[i];
    
    subject to PVLIN2 {i in N}:
        ppv[i]<=pv[i];
    
    subject to PVLIN3 {i in N}:
        ppv[i]>=pv[i]-pvcmax*(1-zpv[i]);
    
    subject to MAXPV:
        sum{i in N} ppv[i]=pvcmax;
    
    subject to MAXZPV:
        sum{i in N} zpv[i]=npv;
        
    subject to Probi1{h in H: besti[h]=1}:
        probi[h]=1-exp(-ic[h]*iparams[h,1]);
    subject to Probi2{h in H: besti[h]=2}:
        probi[h]=1/(1+(ic[h]/iparams[h,3])^-iparams[h,4]);
    subject to Probi3{h in H: besti[h]=3}:
        probi[h]=1/(1+exp((-ic[h]+iparams[h,5])/iparams[h,6]));
    subject to Probi4{h in H: besti[h]=4}:
        probi[h]=0.5+sum{k in NN} (normnum[k]*(((log(ic[h])-iparams[h,7])/iparams[h,8])^((2*(k-1)+1))))/(sqrt(2*pi)*normden[k]); 
    subject to Probi5{h in H: besti[h]=5}:
        probi[h]=0.5+sum{k in NN} (normnum[k]*(((ic[h]-iparams[h,9])/iparams[h,10])^((2*(k-1))+1)))/(sqrt(2*pi)*normden[k]);
    subject to Probi6{h in H: besti[h]=6}:
        probi[h]=1-exp(-(ic[h]^2)/(2*(iparams[h,11]^2)));
    subject to Probi7{h in H: besti[h]=7}:
        probi[h]=1-exp(-(iparams[h,13]*ic[h])^iparams[h,14]);
    subject to Probi8{h in H: besti[h]=10}:
        probi[h]=0;
    


    subject to Probd1{h in H,c in C: bestd[h,c]=1}:
        probd[h,c]=1-exp(-dc[h,c]*dparams[c,h,1]);
    subject to Probd2{h in H,c in C: bestd[h,c]=2}:
        probd[h,c]=1/(1+(dc[h,c]/dparams[c,h,3])^-dparams[c,h,4]);
    subject to Probd3{h in H,c in C: bestd[h,c]=3}:
        probd[h,c]=1/(1+exp((-dc[h,c]+dparams[c,h,5])/dparams[c,h,6]));
    subject to Probd4{h in H,c in C: bestd[h,c]=4}:
        probd[h,c]=0.5+sum{k in NN} (normnum[k]*(((log(dc[h,c])-dparams[c,h,7])/dparams[c,h,8])^((2*(k-1))+1)))/(sqrt(2*pi)*normden[k]);
    subject to Probd5{h in H,c in C: bestd[h,c]=5}:
        probd[h,c]=0.5+sum{k in NN} (normnum[k]*(((dc[h,c]-dparams[c,h,9])/dparams[c,h,10])^((2*(k-1))+1)))/(sqrt(2*pi)*normden[k]); 
    subject to Probd6{h in H,c in C: bestd[h,c]=6}:
        probd[h,c]=1-exp(-(dc[h,c]^2)/(2*(dparams[c,h,11]^2)));
    subject to Probd7{h in H,c in C: bestd[h,c]=7}:
        probd[h,c]=1-exp(-(dparams[c,h,13]*dc[h,c])^dparams[c,h,14]);
"""
)

    
"-------Problem/solver Setup--------"

ampl.option["solver"] = "bonmin"
ampl.solve()

"-------------Print solution------------"
pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
vo= np.zeros([H,num_nodes])
pho= np.zeros([H,num_nodes])
pvo= np.zeros(num_nodes)
ppvo= np.zeros(num_nodes)
zpvo= np.zeros(num_nodes)
ico=np.zeros(H)
dco=np.zeros([H,ncluster])
probio=np.zeros(H)
probdo=np.zeros([H,ncluster])

for h in range(H):
    ico[h]=ampl.get_variable('ic')[h+1].value()
    probio[h]=ampl.get_variable('probi')[h+1].value()
    for i in range (num_nodes):
        vo[h][i]=ampl.get_variable('v')[h+1,i+1].value()
        pho[h][i]=ampl.get_variable('ph')[h+1,i+1].value()
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()
    
    for c in range(ncluster):
        dco[h][c]=ampl.get_variable('dc')[h+1,c+1].value()
        probdo[h][c]=ampl.get_variable('probd')[h+1,c+1].value()

for i in range (num_nodes):
    pvo[i]=ampl.get_variable('pv')[i+1].value()
    ppvo[i]=ampl.get_variable('ppv')[i+1].value()
    zpvo[i]=ampl.get_variable('zpv')[i+1].value()

plt.plot(vo)


Equp = [num_nodes*[0] for h in range(H)]
Equq = [num_nodes*[0] for h in range(H)]

for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            Equp[h][i]+=(vo[h][i]*vo[h][j]*yr[i][j]*np.cos(pho[h][i]-pho[h][j]))+(vo[h][i]*vo[h][j]*yi[i][j]*np.sin(pho[h][i]-pho[h][j]))
            Equq[h][i]+=(vo[h][i]*vo[h][j]*yr[i][j]*np.sin(pho[h][i]-pho[h][j]))-(vo[h][i]*vo[h][j]*yi[i][j]*np.cos(pho[h][i]-pho[h][j]))
        
ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])

t=np.zeros(H)
t[0]=ampl.getValue('_solve_elapsed_time')

ppvout=np.zeros([H,num_nodes])
zpvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=pvo
    zpvout[h]=zpvo

probdco= np.zeros([H,ncluster])
probico=  np.zeros(H)

for h in range(H):        
    for i in range(ncluster):            
        distribution=adists[bestfitsd[i][h]-1]
        "Exponential"
        if distribution==adists[0]:                    
            probdco[h][i]=1-np.exp(-dparams[i][h][0]*dco[h][i])            
        "Fisk  (Beta is always positive)"
        if distribution==adists[1]:
            probdco[h][i]=1/(1+np.power(dco[h][i]/dparams[i][h][2],-dparams[i][h][3]))                          
        "Logistic"                        
        if distribution==adists[2]:                
            probdco[h][i]=1/(1+np.exp((-dco[h][i]+dparams[i][h][4])/dparams[i][h][5]))
        "LogNorm"                        
        if distribution==adists[3]:                
            probdco[h][i]=ss.norm.cdf(np.log(dco[h][i]),dparams[i][h][6],dparams[i][h][7])
        "Normal"
        if distribution==adists[4]:                
            probdco[h][i]=ss.norm.cdf(dco[h][i],dparams[i][h][8],dparams[i][h][9])
        "Rayleigh"
        if distribution==adists[5]:                
            probdco[h][i]=1-np.exp(-np.square(dco[h][i])/(2*np.square(dparams[i][h][10])))
        "Weibull"
        if distribution==adists[6]:                
            probdco[h][i]=1-np.exp(-np.power(dco[h][i]*dparams[i][h][12],dparams[i][h][13]))
            
    if bestfitsi[h]!=10:
        distribution=adists[bestfitsi[h]-1]
        "Exponential"
        if distribution==adists[0]:               
            probico[h]=1-np.exp(-iparams[h][0]*ico[h])            
        "Fisk  (Beta is always positive)"
        if distribution==adists[1]:
            probico[h]=1/(1+np.power(ico[h]/iparams[h][2],-iparams[h][3]))                          
        "Logistic"                        
        if distribution==adists[2]:                
            probico[h]=1/(1+np.exp((-ico[h]+iparams[h][4])/iparams[h][5]))
        "LogNorm"                        
        if distribution==adists[3]:                
            probico[h]=ss.norm.cdf(np.log(ico[h]),iparams[h][6],iparams[h][7])
        "Normal"
        if distribution==adists[4]:                
            probico[h]=ss.norm.cdf(ico[h],iparams[h][8],iparams[h][9])
        "Rayleigh"
        if distribution==adists[5]:                
            probico[h]=1-np.exp(-np.square(ico[h])/(2*np.square(iparams[h][10])))
        "Weibull"
        if distribution==adists[6]:                
            probico[h]=1-np.exp(-np.power(ico[h]*iparams[h][12],iparams[h][13]))

    
   
out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,ico,probio,probico)).T
output=np.hstack((vo,pho,Equp,Equq,out1,dco,probdo,probdco,ppvout,zpvout))
df=pd.DataFrame(output)


columns=[]
for i in range(num_nodes):
    columns.append('v'+str(i+1))
for i in range(num_nodes):    
    columns.append('ph'+str(i+1))
for i in range(num_nodes):    
    columns.append('eqp'+str(i+1))
for i in range(num_nodes):    
    columns.append('eqq'+str(i+1))


columns.append('pl')
columns.append('ql')
columns.append('pg')
columns.append('qg')
columns.append('t')
columns.append('ic')
columns.append('prob_ic_opt')
columns.append('prob_ic_cal')
for i in range(ncluster):    
    columns.append('dc_c'+str(i+1))
for i in range(ncluster):    
    columns.append('prob_dc_opt'+str(i+1))
for i in range(ncluster):    
    columns.append('prob_dc_cal'+str(i+1))
for i in range(num_nodes):    
    columns.append('ppv'+str(i+1))
for i in range(num_nodes):    
    columns.append('zpv'+str(i+1))
    
df.columns=columns

solvlist=[0]*H
solvlist[0]='AB'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("Results.xlsx")