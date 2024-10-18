# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:31:54 2024

@author: diego
"""

from amplpy import AMPL, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.special import factorial
import scipy.stats as ss

case='SA'
city='Bog'
city1='BOG'
problem='OPF_PV_S'
case1='SA_J23'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1+'Gen.csv')


mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=mat['clusternode']

c1 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC1.csv')
c2 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC2.csv')
c3 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC3.csv')
c4 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC4.csv')
c5 = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+'ParamTableC5.csv')

irr = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\'+'ParamTable.csv')


mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\PWLxyRAD_'+city1+'.mat')
xirrpwl=mat['xpwl']
yirrpwl=mat['ypwl']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\PWLxyDem_'+city1+'.mat')
xdempwl=mat['xpwl']
ydempwl=mat['ypwl']

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\NSTparamDem_'+city1+'.mat')
dparams=np.array(mat['params'])

mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city+'\\NSTparamRAD_'+city1+'.mat')
iparams=mat['params']

adists=['Exponential','Fisk','Logistic','Log-N','Normal','Rayleigh','Weibull'];
aclust=['c1','c2','c3','c4','c5']

ndists=len(adists)
ncluster=np.size(xdempwl,axis=1)
H=np.size(xdempwl,axis=0)

cparameters=pd.concat([c1,c2,c3,c4,c5],keys=aclust)

bestfitsd=np.zeros([H,ncluster],dtype=int)

bestfitsi=10*np.ones(H,dtype=int)

ihours=np.zeros(H)


for h in range(H):
    if yirrpwl[h][0][9]!=0:
        ihours[h]=1

nihours=int(np.sum(ihours))
    
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
                bestfitsd[h][i]=j+1
            

         
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

vrmax=np.zeros(num_nodes)
vrmin=np.zeros(num_nodes)

if ngen>0:
    for i in range(ngen):
        pgen[bus['i'][i]-1]=gen['pi'][i]
        qgen[bus['i'][i]-1]=gen['qi'][i]        
        vmax[bus['i'][i]-1]=gen['vst'][i]
        vmin[bus['i'][i]-1]=gen['vst'][i]


vmax=vmax+0.2
vmin=vmin-0.2        

vmax[iref]=1
vmin[iref]=1

prefmax=np.zeros(num_nodes)
qrefmax=np.zeros(num_nodes)

prefmin=np.zeros(num_nodes)
qrefmin=np.zeros(num_nodes)

prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
qrefmin[iref]=np.max([0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]])


ym=np.zeros([num_nodes,num_nodes],dtype='complex')

for k in range(num_lines):
    fr=branch['i'][k]-1
    to=branch['j'][k]-1
    ym[fr][to]=-1/(branch['r'][k] + 1j*branch['x'][k])
    ym[to][fr]=-1/(branch['r'][k] + 1j*branch['x'][k])    

for i in range(num_nodes):
    ym[i][i]=-np.sum(ym[i])
    
ymr=np.real(ym)
ymi=np.imag(ym)

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if i==j:
            cnmax[i][i]=vmax[i]*vmax[i]
            cnmin[i][i]=vmin[i]*vmin[i]
        else:
            cnmax[i][j]=vmax[i]*vmax[j]
            cnmin[i][j]=vmin[i]*vmin[j]
            snmax[i][j]=vmax[i]*vmax[j]
            snmin[i][j]=-vmax[i]*vmax[j]

idx1=(np.abs(ym)!=0)*1
idx2=idx1-np.eye(num_nodes)

cnmax=np.zeros([num_nodes,num_nodes])
cnmin=np.zeros([num_nodes,num_nodes])
snmax=np.zeros([num_nodes,num_nodes])
snmin=np.zeros([num_nodes,num_nodes])

for i in range(num_nodes):
    for j in range(num_nodes):
        if idx1[i][j]:
            if i==j:
                cnmax[i][i]=vmax[i]*vmax[i]
                cnmin[i][i]=vmin[i]*vmin[i]
            else:
                cnmax[i][j]=vmax[i]*vmax[j]
                cnmin[i][j]=vmin[i]*vmin[j]
                snmax[i][j]=vmax[i]*vmax[j]
                snmin[i][j]=-vmax[i]*vmax[j]

npv=np.floor(0.1*num_nodes)
pvcmax=0.5*np.sum(np.real(sd))
pveff=0.8

cpdem=np.zeros([num_nodes,ncluster])
cqdem=np.zeros([num_nodes,ncluster])

for i in range(num_nodes):
    for c in range(ncluster):
        if cnode[i]-1==c:
            cpdem[i][c]=pdm[i]
            cqdem[i][c]=qdm[i]

npwl=np.size(xdempwl,axis=3)

mindc=np.zeros([H,ncluster])
maxdc=np.zeros([H,ncluster])

minic=np.zeros(H)
maxic=np.zeros(H)

for h in range(H):
    for c in range(ncluster):
        dist=bestfitsd[h,c]-1
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
    param np;
    param nih;            
"""
)

ampl.get_parameter("nn").set(num_nodes)
ampl.get_parameter("nh").set(H)
ampl.get_parameter("nc").set(ncluster)
ampl.get_parameter("nd").set(ndists)
ampl.get_parameter("np").set(npwl)
ampl.get_parameter("nih").set(nihours)



ampl.eval(
    r"""
    set N=1..nn;
    set H=1..nh;
    set C=1..nc;
    set D=1..nd;
    set P=1..np;
    set NI=1..nih;
    
    
    param yr {N,N};
    param yi {N,N};
    param pd {N};
    param qd {N};
    param cnmax{N,N};
    param cnmin{N,N};
    param snmax{N,N};
    param snmin{N,N};
    param prefmax{N};
    param prefmin{N};
    param qrefmax{N};
    param qrefmin{N};   
    param npv;
    param pvcmax;
    param pveff;
    param idx1{N,N};
    param idx2{N,N};
    param cnode{N};    
    param cpdem{N,C};
    param cqdem{N,C};
    param ihours{H};
    param bestd{H,C};
    param besti{H};
    param xdpwl{H,C,D,P};
    param xipwl{H,D,P};
    param ydpwl{H,C,D,P};
    param yipwl{H,D,P};
    param dcmin{H,C};
    param dcmax{H,C};
    param icmin{H};
    param icmax{H};
        
"""
)

ampl.get_parameter("yr").set_values(ymr)
ampl.get_parameter("yi").set_values(ymi)
ampl.get_parameter("pd").set_values(pdm)
ampl.get_parameter("qd").set_values(qdm)
ampl.get_parameter("cnmax").set_values(cnmax)
ampl.get_parameter("cnmin").set_values(cnmin)
ampl.get_parameter("snmax").set_values(snmax)
ampl.get_parameter("snmin").set_values(snmin)
ampl.get_parameter("prefmax").set_values(prefmax)
ampl.get_parameter("prefmin").set_values(prefmin)
ampl.get_parameter("qrefmax").set_values(qrefmax)
ampl.get_parameter("qrefmin").set_values(qrefmin)
ampl.get_parameter("npv").set(npv)
ampl.get_parameter("pveff").set(pveff)    
ampl.get_parameter("pvcmax").set(pvcmax)
ampl.get_parameter("idx1").set_values(idx1)
ampl.get_parameter("idx2").set_values(idx2) 
ampl.get_parameter("cnode").set_values(cnode)
ampl.get_parameter("cpdem").set_values(cpdem)
ampl.get_parameter("cqdem").set_values(cqdem)
ampl.get_parameter("ihours").set_values(ihours)
ampl.get_parameter("bestd").set_values(bestfitsd) 
ampl.get_parameter("besti").set_values(bestfitsi)

xdpwldf =pd.concat([pd.DataFrame(xdempwl[h][c][d],index=np.arange(1,npwl+1)) for h in range(H) for c in range(ncluster) for d in range(ndists)], keys=[(h+1,c+1,d+1) for h in range(H) for c in range(ncluster) for d in range(ndists)])
xipwldf =pd.concat([pd.DataFrame(xirrpwl[h][d],index=np.arange(1,npwl+1)) for h in range(H) for d in range(ndists)], keys=[(h+1,d+1) for h in range(H) for d in range(ndists)])

ydpwldf= pd.concat([pd.DataFrame(ydempwl[h][c][d],index=np.arange(1,npwl+1)) for h in range(H) for c in range(ncluster) for d in range(ndists)], keys=[(h+1,c+1,d+1) for h in range(H) for c in range(ncluster) for d in range(ndists)])
yipwldf= pd.concat([pd.DataFrame(yirrpwl[h][d],index=np.arange(1,npwl+1)) for h in range(H) for d in range(ndists)], keys=[(h+1,d+1) for h in range(H) for d in range(ndists)])

ampl.get_parameter("xdpwl").set_values(DataFrame.from_pandas(xdpwldf))
ampl.get_parameter("xipwl").set_values(DataFrame.from_pandas(xipwldf))

ampl.get_parameter("ydpwl").set_values(DataFrame.from_pandas(ydpwldf))
ampl.get_parameter("yipwl").set_values(DataFrame.from_pandas(yipwldf))

ampl.get_parameter("dcmax").set_values(maxdc)
ampl.get_parameter("dcmin").set_values(mindc)
ampl.get_parameter("icmax").set_values(maxic)
ampl.get_parameter("icmin").set_values(minic)


"-------Constraint Construction-------- "

ampl.eval(
    r"""
       
    var  cn{h in H,i in N, j in N} >= cnmin[i,j], <= cnmax[i,j];
    var  sn{h in H,i in N, j in N} >= snmin[i,j], <= snmax[i,j];
    var  pg{h in H,i in N} >= prefmin[i], <= prefmax[i];
    var  qg{h in H,i in N} >= qrefmin[i], <= qrefmax[i];
    
    var  pv{i in N} >=0, <=pvcmax;
    var  ppv{i in N} >=0, <=pvcmax;
    var  zpv{i in N} binary;
    
    var ic{h in H} >=icmin[h], <=icmax[h];
    var dc{h in H,c in C} >=dcmin[h,c], <=dcmax[h,c];
    
    var probd{h in H,c in C} >=0, <=1;
    var probi{h in H} >=0, <=1;
    
    var zprobd{h in H, c in C, p in P} binary;
    var zprobi{h in H, p in P} binary;
    
    var lambdad{h in H, c in C, p in P} >=0, <=1;
    var lambdai{h in H, p in P} >=0, <=1;
    
    var lambdad1{h in H, c in C, p in P} >=0, <=1;
    var lambdai1{h in H, p in P} >=0, <=1;
    
     
    minimize Losses:
         sum{h in H,i in N, j in N: idx1[i,j]=1} cn[h,i,j]*yr[i,j]
       + sum{h in H,i in N, j in N: idx1[i,j]=1} sn[h,i,j]*yi[i,j] 
       + sum{h in H,i in N, j in N: idx1[i,j]=1} sn[h,i,j]*yr[i,j]
       - sum{h in H,i in N, j in N: idx1[i,j]=1} cn[h,i,j]*yi[i,j]
       - sum{h in H,c in C} probd[h,c]
       + sum{h in H} probi[h];
       

    subject to PB {h in H,i in N}: 
        pg[h,i]+(ppv[i]*ic[h]*pveff)-sum{c in C}(cpdem[i,c]*dc[h,c]) = sum {j in N: idx1[i,j]=1} cn[h,i,j]*yr[i,j]
        + sum {j in N: idx1[i,j]=1} sn[h,i,j]*yi[i,j];  
        
    subject to QB {h in H,i in N}: 
        qg[h,i]-sum{c in C}(cqdem[i,c]*dc[h,c]) = sum {j in N: idx1[i,j]=1} sn[h,i,j]*yr[i,j]
        - sum {j in N: idx1[i,j]=1} cn[h,i,j]*yi[i,j];
        
    subject to Hermit1 {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        cn[h,i,j]=cn[h,j,i];
        
    subject to Hermit2 {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        sn[h,i,j]=-sn[h,j,i];

    
    subject to SOC {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        (cn[h,i,j]^2)+(sn[h,i,j]^2)=cn[h,i,i]*cn[h,j,j];

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
        
    subject to Probd1 {h in H,c in C, d in D: bestd[h,c]=d}:
        probd[h,c]=sum{p in P} lambdad[h,c,p]*ydpwl[h,c,d,p];
    
    subject to DC1 {h in H,c in C,d in D: bestd[h,c]=d}:
        dc[h,c]=sum{p in P} lambdad[h,c,p]*xdpwl[h,c,d,p];
        
    subject to Lambdad1 {h in H,c in C,p in P}:
        lambdad[h,c,p]<=zprobd[h,c,p];
    subject to Lambdad2 {h in H,c in C,p in P}:
        lambdad[h,c,p]<=lambdad1[h,c,p];
    subject to Lambdad3 {h in H,c in C,p in P}:
        lambdad[h,c,p]>=lambdad1[h,c,p]-1+zprobd[h,c,p];
    
    subject to Zprobd1 {h in H,c in C}:
        sum{p in P} zprobd[h,c,p]=1;
    
    subject to Probi1 {h in H,d in D: besti[h]=d}:
        probi[h]=sum{p in P} lambdai[h,p]*yipwl[h,d,p];
    
    subject to IC1 {h in H, d in D: besti[h]=d}:
        ic[h]=sum{p in P} lambdai[h,p]*xipwl[h,d,p];
    
    subject to Zprobi1 {h in H: ihours[h]=1}:
        sum{p in P} zprobi[h,p]=1;
    
    subject to Lambdai1 {h in H,p in P: ihours[h]=1}:
        lambdai[h,p]<=zprobi[h,p];
    subject to Lambdai2 {h in H,p in P: ihours[h]=1}:
        lambdai[h,p]<=lambdai1[h,p];
    subject to Lambdai3 {h in H,p in P: ihours[h]=1}:
        lambdai[h,p]>=lambdai1[h,p]-1+zprobi[h,p];
    
    subject to probi0{h in H: ihours[h]=0}:
        probi[h]=0;
    subject to zprobi0{h in H, p in P: ihours[h]=0}:
        zprobi[h,p]=0;
    subject to labdai0{h in H, p in P: ihours[h]=0}:
        lambdai[h,p]=0;
    subject to labdai10{h in H, p in P: ihours[h]=0}:
        lambdai1[h,p]=0;
        
"""
)
                                 


"-------Problem/solver Setup--------"

#ampl.option["solver"] = "Couenne"
#ampl.option["couenne_options"] = "use_quadratic=yes"
# ampl.option["solver"] = "kestrel"
# ampl.option["kestrel_options"] = "solver=knitro priority=long"
#ampl.option["gurobi_options"] = "nonconvex=2"
# ampl.option["email"] = "dimendozao@unal.edu.co"
#ampl.option["solver"] = "scip"
#ampl.option["scip_options"] = "outlev=1"
#ampl.option["show_stats"] = 1
#ampl.option["presolve"] = 0
#ampl.option["ipopt_options"] = "max_cpu_time=480"
#ampl.solve()

#ampl.option["pl_linearize"] = 0
ampl.solve(solver='bonmin',bonmin_options='outlev=1 bonmin.nlp_log_level 2')
#ampl.solve(solver='bonmin',bonmin_options='outlev=1 bonmin.oa_decomposition yes bonmin.algorithm B-OA bonmin.nlp_failure_behavior fathom bonmin.nlp_log_level 2')
#ampl.solve(solver="scip", scip_options='outlev=1')
#ampl.solve(solver="highs", highs_options='outlev=1 miploglev=2 socp=0')

"-------------Print solution------------"
pgo=np.zeros([H,num_nodes])
qgo=np.zeros([H,num_nodes])
cno= np.zeros([H,num_nodes,num_nodes])
sno= np.zeros([H,num_nodes,num_nodes])
pvo= np.zeros(num_nodes)
zpvo= np.zeros(num_nodes)
ico=np.zeros(H)
dco=np.zeros([H,ncluster])
probio=np.zeros(H)
probdo=np.zeros([H,ncluster])


for h in range(H):
    ico[h]=ampl.get_variable('ic')[h+1].value()
    probio[h]=ampl.get_variable('probi')[h+1].value()
    for i in range (num_nodes):
        pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
        qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()
        for j in range(num_nodes):
            if idx1[i][j]==1:
                cno[h][i][j]=ampl.get_variable('cn')[h+1,i+1,j+1].value()
                sno[h][i][j]=ampl.get_variable('sn')[h+1,i+1,j+1].value()
    for c in range(ncluster):
        dco[h][c]=ampl.get_variable('dc')[h+1,c+1].value()
        probdo[h][c]=ampl.get_variable('probd')[h+1,c+1].value()

for i in range(num_nodes):
    pvo[i]= ampl.get_variable('pv')[i+1].value()
    zpvo[i]= ampl.get_variable('zpv')[i+1].value()
    
vo=np.zeros([H,num_nodes])       

for h in range(H):        
    vo[h]=np.sqrt(np.diag(cno[h]))

plt.plot(vo)

Equp = [[0] * num_nodes for h in range(H)]
Equq = [[0] * num_nodes for h in range(H)]


for h in range(H):
    for i in range(num_nodes):
        for j in range(num_nodes):
            Equp[h][i]+=(cno[h][i][j]*ymr[i][j])+(sno[h][i][j]*ymi[i][j])
            Equq[h][i]+=(sno[h][i][j]*ymr[i][j])-(cno[h][i][j]*ymi[i][j])
    

ploss=np.zeros(H)
qloss=np.zeros(H)

for h in range(H):
    ploss[h]=np.sum(Equp[h])
    qloss[h]=np.sum(Equq[h])

pho=np.zeros([H,num_nodes])

for h in range(H):
    for k in range(num_lines):
        fr=branch['i'][k]-1
        to=branch['j'][k]-1
        pho[h][to]=pho[h][fr]-np.angle(cno[h][fr][to]+1j*sno[h][fr][to])

t=np.zeros(H)

t[0]=ampl.getValue('_solve_elapsed_time')

npwlout=np.zeros(H)
npwlout[0]=npwl

ppvout=np.zeros([H,num_nodes])
zpvout=np.zeros([H,num_nodes])

for h in range(H):
    ppvout[h]=pvo
    zpvout[h]=zpvo

probdco= np.zeros([H,ncluster])
probico=  np.zeros(H)

for h in range(H):        
    for i in range(ncluster):            
        distribution=adists[bestfitsd[h][i]-1]
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

    
   
out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,npwlout,ico,probio,probico)).T
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
columns.append('npwl_prob')
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
solvlist[0]='AB-PWL'


df.insert(len(df.columns),'Solver',solvlist)
df.to_excel("ResultsPWL.xlsx")