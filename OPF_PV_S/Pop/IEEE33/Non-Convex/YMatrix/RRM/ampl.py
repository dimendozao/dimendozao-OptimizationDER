# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:24:22 2024

@author: diego
"""

from amplpy import AMPL, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.special import factorial
import scipy.stats as ss


case='IEEE33'
city='Jam'
city1='JAM'
problem='OPF_PV_S'


"----- Read the database -----"
branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Branch.csv')
bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Bus.csv')
gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case+'Gen.csv')


mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city+'\\'+case+'_'+city+'_'+'ClusterNode.mat')
cnode=mat['clusternode']

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

bestfitsd=np.zeros([H,ncluster],dtype=int)


bestfitsi=10*np.ones(H,dtype=int)


bestfitsd1=np.zeros([H,ncluster],dtype=int)
bestfitsd2=np.zeros([H,ncluster],dtype=int)
bestfitsd3=np.zeros([H,ncluster],dtype=int)


bestfitsi1=10*np.ones(H,dtype=int)
bestfitsi2=10*np.ones(H,dtype=int)
bestfitsi3=10*np.ones(H,dtype=int)
 
nihours=int(len(irr)/2)
ihours=np.zeros(H)


for h in range(H):
    for hh in range(nihours):
        if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
            ihours[h]=1
    
okfitsdem=[1,1,1,1,1,1,1]
okfitsirr=[1,1,1,1,1,1,1]     
    
for h in range(H):
    for hh in range(nihours):
        if 'h_{'+str(h+1)+'}' in irr['Hour'][2*hh]:
            ihours[h]=1
            for j in range(ndists):
                if adists[j] in irr['bestparams1'][2*hh] and irr['bestparams1'][2*hh].find(adists[j])==0:
                   bestfitsi1[h]=j+1
                if adists[j] in irr['bestparams2'][2*hh] and irr['bestparams2'][2*hh].find(adists[j])==0:
                   bestfitsi2[h]=j+1
                if adists[j] in irr['bestparams3'][2*hh] and irr['bestparams3'][2*hh].find(adists[j])==0:
                   bestfitsi3[h]=j+1
                   
    for i in range(ncluster):
        for j in range(ndists):
            if adists[j] in cparameters['bestparams1'][aclust[i],2*h] and cparameters['bestparams1'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd1[h][i]=j+1
            if adists[j] in cparameters['bestparams2'][aclust[i],2*h] and cparameters['bestparams2'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd2[h][i]=j+1
            if adists[j] in cparameters['bestparams3'][aclust[i],2*h] and cparameters['bestparams3'][aclust[i],2*h].find(adists[j])==0:
                bestfitsd3[h][i]=j+1

nten=np.sum(bestfitsi1==10)                

for h in range(H):
    for i in range(ncluster):
        if okfitsdem[bestfitsd1[h][i]-1]==1:          
            bestfitsd[h][i]=bestfitsd1[h][i]
        elif okfitsdem[bestfitsd2[h][i]-1]==1:
            bestfitsd[h][i]=bestfitsd2[h][i]
        elif okfitsdem[bestfitsd3[h][i]-1]==1:
            bestfitsd[h][i]=bestfitsd3[h][i]
        else:
            bestfitsd[h][i]=10
    if bestfitsi1[h]!=10:
        if okfitsirr[bestfitsi1[h]-1]==1:
            bestfitsi[h]=bestfitsi1[h]            
        elif okfitsirr[bestfitsi2[h]-1]==1:
            bestfitsi[h]=bestfitsi2[h]
        elif okfitsirr[bestfitsi3[h]-1]==1:
            bestfitsi[h]=bestfitsi3[h]
        else:
            bestfitsi[h]=10


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

dp1=dparams[0]
dp2=dparams[1]
dp3=dparams[2]
dp4=dparams[3]
dp5=dparams[4]

cpdem=np.zeros([num_nodes,ncluster])
cqdem=np.zeros([num_nodes,ncluster])

for i in range(num_nodes):
    for c in range(ncluster):
        if cnode[i]-1==c:
            cpdem[i][c]=pdm[i]
            cqdem[i][c]=qdm[i]

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
    set N=1..nn;
    set H=1..nh;
    set C=1..nc;
    set D=1..nd;
    set NN=1..nnorm;
    
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
    param idx{N,N};
    param npv;
    param pvcmax;
    param pveff;
    param idx1{N,N};
    param idx2{N,N};
    param cnode{N};
    param dparams{C,H,D};
    param iparams{H,D};
    param cpdem{N,C};
    param cqdem{N,C};
    param ihours{H};
    param bestd{H,C};
    param besti{H};
    param pi;
    param normden{NN};
    param normnum{NN};
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
ampl.get_parameter("iparams").set_values(iparams)
ampl.get_parameter("cpdem").set_values(cpdem)
ampl.get_parameter("cqdem").set_values(cqdem)
ampl.get_parameter("ihours").set_values(ihours)
ampl.get_parameter("bestd").set_values(bestfitsd) 
ampl.get_parameter("besti").set_values(bestfitsi)
ampl.get_parameter("pi").set(np.pi) 
ampl.get_parameter("dparams").set_values(DataFrame.from_pandas(dparamsdf))
ampl.get_parameter("normnum").set_values(normnum) 
ampl.get_parameter("normden").set_values(normden)
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
    var dc{h in H,i in C} >=dcmin[h,i], <=dcmax[h,i]; 
    
    var probd{h in H,i in C} >=0,<=1;
    var probi{h in H} >=0, <=1;

    
    minimize Losses:
         sum{h in H,i in N, j in N: idx1[i,j]=1} cn[h,i,j]*yr[i,j]
       + sum{h in H,i in N, j in N: idx1[i,j]=1} sn[h,i,j]*yi[i,j] 
       + sum{h in H,i in N, j in N: idx1[i,j]=1} sn[h,i,j]*yr[i,j]
       - sum{h in H,i in N, j in N: idx1[i,j]=1} cn[h,i,j]*yi[i,j]
       - sum{h in H, c in C} probd[h,c]
       + sum{h in H} probi[h];
       
       
    subject to PB {h in H,i in N}: 
        pg[h,i]+(ppv[i]*ic[h]*pveff)-sum{c in C}(cpdem[i,c]*dc[h,c]) = sum {j in N: idx1[i,j]=1} cn[h,i,j]*yr[i,j]
        + sum {j in N: idx1[i,j]=1} sn[h,i,j]*yi[i,j];  
        
    subject to QB {h in H,i in N}: 
        qg[h,i]-sum{c in C}(cqdem[i,c]*dc[h,c]) = sum {j in N: idx1[i,j]=1} sn[h,i,j]*yr[i,j]
        - sum {j in N: idx1[i,j]=1} cn[h,i,j]*yi[i,j]; 
    
    subject to SOC {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        (cn[h,i,j]*cn[h,i,j])+(sn[h,i,j]*sn[h,i,j])=cn[h,i,i]*cn[h,j,j];
    
    subject to Hermit1 {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        cn[h,i,j]=cn[h,j,i];
        
    subject to Hermit2 {h in H,i in N, j in N: idx2[i,j]=1 and j>i}:
        sn[h,i,j]=-sn[h,j,i];


    
           
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


ampl.solve(solver='bonmin',bonmin_options='outlev=1 bonmin.algorithm B-BB')
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