# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:40:38 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from amplpy import AMPL, DataFrame
from scipy.io import loadmat
import scipy.stats as ss
import re

form='YMatrix'
conv='Non-Convex'
fold='PM'
case=['IEEE33','IEEE69','SA','CA141']
city=['Bog','Jam','Pop']
problem='OPF_BESS_ES'
prob_sol='OPF_PV_S'
city1=['BOG','JAM','POP']
case1=['IEEE33','IEEE69','SA_J23','CA141']

nct=len(city)
nca=len(case)

for ct in range(nct):
    for ca in range(nca):
        "----- Read the database -----"
        branch = pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1[ca]+'Branch.csv')
        bus= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1[ca]+'Bus.csv')
        gen= pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Systems\\'+case1[ca]+'Gen.csv')
        
        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Radiacion\\'+city[ct]+'\\MeansRAD_'+city1[ct]+'.mat')
        imeans=np.squeeze(mat['means'])
        
        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\ClusterMeans_'+city1[ct]+'.mat')
        dmeans=np.squeeze(mat['clustermeans']).T
        
        mat= loadmat('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Data\\Demanda\\'+city[ct]+'\\'+case[ca]+'_'+city[ct]+'_'+'ClusterNode.mat')
        cnode=np.squeeze(mat['clusternode'])
        
        pvall=pd.read_csv('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+prob_sol+'\\'+city[ct]+'\\'+case[ca]+'\\bestsol.csv')
        
               
        cnode[0]=1
        
        H=len(imeans)
        num_lines = len(branch)
        num_nodes=len(bus)
        ncluster=np.size(dmeans,axis=1)
        iref=np.where(bus['type']==3)[0][0]
        
        sd=np.zeros(num_nodes,dtype='complex')
        
        for k in range(num_lines):
            sd[branch['j'][k]-1]=branch['pj'][k]+1j*branch['qj'][k]
       
        ppv=np.zeros(num_nodes)
        zpv=np.zeros(num_nodes)
        dc=np.zeros([H,ncluster])
        
        for i in range(num_nodes):
            ppv[i]=pvall['ppv'+str(i+1)][0]
        
        ic=pvall['ic'].to_numpy()
        for c in range(ncluster):
            dc[:,c]=pvall['dc'+str(c+1)].to_numpy()
            
        pdm=np.real(sd)
        qdm=np.imag(sd)
        
        cpdem=np.zeros([num_nodes,ncluster])
        cqdem=np.zeros([num_nodes,ncluster])
        
        for i in range(num_nodes):
            for c in range(ncluster):
                if cnode[i]-1==c:
                    cpdem[i][c]=pdm[i]
                    cqdem[i][c]=qdm[i]
        
        
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
        
        vmax=np.tile(vmax,(H,1))
        vmin=np.tile(vmin,(H,1))
        
        phmax=np.ones(num_nodes)*np.pi
        phmin=np.ones(num_nodes)*-np.pi
        
        phmax[iref]=0
        phmin[iref]=0
        
        phmax=np.tile(phmax,(H,1))
        phmin=np.tile(phmin,(H,1))
        
        prefmax=np.zeros(num_nodes)
        qrefmax=np.zeros(num_nodes)
        
        prefmin=np.zeros(num_nodes)
        qrefmin=np.zeros(num_nodes)
        
        prefmax[iref]=gen['pmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
        prefmin[iref]=gen['pmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]
        qrefmax[iref]=gen['qmax'][np.where(np.array(gen['i'])==iref+1)[0][0]]
        qrefmin[iref]=np.max([0,gen['qmin'][np.where(np.array(gen['i'])==iref+1)[0][0]]])
        
        prefmax=np.tile(prefmax,(H,1))
        prefmin=np.tile(prefmin,(H,1))
        
        qrefmax=np.tile(qrefmax,(H,1))
        qrefmin=np.tile(qrefmin,(H,1))
        
        y=np.zeros([num_nodes,num_nodes],dtype='complex')
        
        fr=np.zeros(num_lines,dtype='int')
        to=np.zeros(num_lines,dtype='int')
        
        idx2=np.zeros([num_nodes,num_nodes],dtype='int')
        
        for k in range(num_lines):
            fr[k]=branch['i'][k]-1
            to[k]=branch['j'][k]-1
            y[fr[k]][to[k]]=-1/(branch['r'][k] + 1j*branch['x'][k])
            y[to[k]][fr[k]]=-1/(branch['r'][k] + 1j*branch['x'][k])  
            idx2[fr[k]][to[k]]=1
            idx2[to[k]][fr[k]]=1
        for i in range(num_nodes):
            y[i][i]=-np.sum(y[i])
        
        idx1=idx2+np.eye(num_nodes)
        
        yr=np.real(y)
        yi=np.imag(y)
        
        pveff=0.8
        
        pbcmax=np.sum(pdm)*1      
        
        "----- Optimization model -----"
        
        ampl = AMPL()
        
        ampl.eval(
            r"""
            param nn;
            param nh;
            param nc;
            
        """
        )
        
        ampl.get_parameter("nn").set(num_nodes)
        ampl.get_parameter("nh").set(H)
        ampl.get_parameter("nc").set(ncluster)
        
        
        
        ampl.eval(
            r"""
            set N=1..nn;
            set H=1..nh;
            set C=1..nc;
            
            
            
            
            param Yr {N,N};
            param Yi {N,N};
            param cpdem{N,C};
            param cqdem{N,C};    
            param vmax{H,N};
            param vmin{H,N};
            param phmax{H,N};
            param phmin{H,N};
            param prefmax{H,N};
            param prefmin{H,N};
            param qrefmax{H,N};
            param qrefmin{H,N};
            param idx1{N,N};
            param idx2{N,N};
            param pveff;
            param pbcmax;
            param ppv{N};
            param ic{H};
            param dc{H,C};
                        
                      
        """
        )
        
        ampl.get_parameter("Yr").set_values(yr)
        ampl.get_parameter("Yi").set_values(yi)
        ampl.get_parameter("cpdem").set_values(cpdem)
        ampl.get_parameter("cqdem").set_values(cqdem)
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
        ampl.get_parameter("ppv").set_values(ppv)
        ampl.get_parameter("ic").set_values(ic)
        ampl.get_parameter("dc").set_values(dc)
        ampl.get_parameter("pveff").set(pveff)
        ampl.get_parameter("pbcmax").set(pbcmax)
        
        
        
                     
        
        ampl.eval(
            r"""
            var  v{h in H,i in N} >= vmin[h,i], <= vmax[h,i];
            var  ph{h in H,i in N} >= phmin[h,i], <= phmax[h,i];
            var  pg{h in H,i in N} >= prefmin[h,i], <= prefmax[h,i];
            var  qg{h in H,i in N} >= qrefmin[h,i], <= qrefmax[h,i];
            
            var pbc >= 0, <= pbcmax;
            var pb{h in H} >= -pbcmax, <=pbcmax;
            var pbe{h in H} >= -2*pbcmax, <=2*pbcmax;
            var pbr{h in H} >=-1,<=1;
            var pbre{h in H} >=0,<=2;
            var zb{h in H, i in N} binary;
            var pbz{h in H, i in N} >=-2*pbcmax, <=2*pbcmax;
            var soc{h in H} >=0, <=1;
            var soc0 >=0,<=1;
            
            
            
            
            minimize Losses:
               sum{h in H,i in N, j in N:idx1[i,j]==1} v[h,i]*v[h,j]*Yr[i,j]*cos(ph[h,i]-ph[h,j])
               +sum{h in H,i in N, j in N:idx1[i,j]==1} v[h,i]*v[h,j]*Yi[i,j]*sin(ph[h,i]-ph[h,j])
               +sum{h in H,i in N, j in N:idx1[i,j]==1} v[h,i]*v[h,j]*Yr[i,j]*sin(ph[h,i]-ph[h,j])
               -sum{h in H,i in N, j in N:idx1[i,j]==1} v[h,i]*v[h,j]*Yi[i,j]*cos(ph[h,i]-ph[h,j]);
               
               
            subject to PB {h in H,i in N}: 
               pg[h,i]+(ppv[i]*ic[h]*pveff)+pbz[h,i]-sum{c in C}(cpdem[i,c]*dc[h,c]) = sum {j in N:idx1[i,j]==1} (v[h,i]*v[h,j]*Yr[i,j]*cos(ph[h,i]-ph[h,j]))+sum {j in N:idx1[i,j]==1} (v[h,i]*v[h,j]*Yi[i,j]*sin(ph[h,i]-ph[h,j]));
        
            subject to QB {h in H,i in N}: 
               qg[h,i]-sum{c in C}(cqdem[i,c]*dc[h,c]) = sum {j in N:idx1[i,j]==1} (v[h,i]*v[h,j]*Yr[i,j]*sin(ph[h,i]-ph[h,j]))-sum {j in N:idx1[i,j]==1} (v[h,i]*v[h,j]*Yi[i,j]*cos(ph[h,i]-ph[h,j]));
            
            subject to ZPB {h in H}:
                sum{i in N} zb[h,i]==1;
            
            subject to PBZ{h in H, i in N}:
                pbz[h,i]=pbe[h]*zb[h,i];
            
            subject to PBER{h in H}:
                pbre[h]= (if pbr[h]<=0 then 1/(1.5-(0.5*sqrt(1-pbr[h]))) else 0.5+0.5*sqrt(1-pbr[h]));
        
            subject to SOC0{h in H: h==1}:
                pbr[h]=soc0-soc[h];
            subject to SOC024{h in H: h==nh}:
                soc[h]=soc0;
            subject to SOCH{h in H: h<nh}:
                pbr[h+1]=soc[h]-soc[h+1];
            subject to PBSOC{h in H}:
                pb[h]=pbr[h]*pbc;            
            subject to PBESOC{h in H}:
                pbe[h]=pbre[h]*pb[h];
                
        """
        )
            
            
        
            
        "-------Problem/solver Setup--------"
        
        #ampl.option["solver"] = "highs"
        #ampl.option["highs_options"] = "return_mipgap=1 timelim=21600 outlev=1 plapproxdomain=1"
        #ampl.option["solver"] = "scip"
        #ampl.option["scip_options"] = "return_mipgap=1 timelim=21600 outlev=1 plapproxdomain=1 "
        #ampl.option["solver"] = "cbc"
        #ampl.option["cbc_options"] = "timelim=21600 outlev=4 plapproxdomain=1"
        ampl.option["solver"] = "bonmin"
        ampl.option["bonmin_options"] = "bonmin.time_limit 21600 bonmin.milp_log_level 4 bonmin.file_solution yes"
        #ampl.cd('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city+'\\'+case+'\\Non-Convex\\Ymatrix\\PM\\')
        #ampl.option["presolve"] = 11
        #ampl.option["presolve"] = 11
        #ampl.option["presolve_warnings"] = 20
        output1 = ampl.solve(return_output=True)
        "-------------Print solution------------"
        pgo=np.zeros([H,num_nodes])
        qgo=np.zeros([H,num_nodes])
        vo= np.zeros([H,num_nodes])
        pho= np.zeros([H,num_nodes])
        pbo= np.zeros(H)
        pbeo= np.zeros(H)
        pbro= np.zeros(H)
        pbreo= np.zeros(H)
        soco= np.zeros(H)
        zbo= np.zeros([H,num_nodes])
        
        pbco=ampl.get_variable('pbc').value()
        soc0o=ampl.get_variable('soc0').value()
        
        
        for h in range(H):
            pbo[h]=ampl.get_variable('pb')[h+1].value()
            pbeo[h]=ampl.get_variable('pbe')[h+1].value()
            pbro[h]=ampl.get_variable('pbr')[h+1].value()
            pbreo[h]=ampl.get_variable('pbre')[h+1].value()
            soco[h]=ampl.get_variable('soc')[h+1].value()    
            for i in range (num_nodes):
                vo[h][i]=ampl.get_variable('v')[h+1,i+1].value()
                pho[h][i]=ampl.get_variable('ph')[h+1,i+1].value()
                pgo[h][i]=ampl.get_variable('pg')[h+1,i+1].value()
                qgo[h][i]=ampl.get_variable('qg')[h+1,i+1].value()        
                zbo[h][i]=ampl.get_variable('zb')[h+1,i+1].value()
            
            
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
        
        gap=np.zeros(H)
        #gap[0]=ampl.getValue('Losses.relmipgap')
        gapstr1=output1.find('Partial search - best objective')
        gapstr2=output1.find('took')
        
        if gapstr1>0 and  gapstr1<gapstr2:
            gapn1=float(re.findall(r"[-+]?(?:\d*\.*\d+)", output1[gapstr1:gapstr2])[0])
            gapn2=float(re.findall(r"[-+]?(?:\d*\.*\d+)", output1[gapstr1:gapstr2])[1])
            gap[0]=(np.max([gapn1,gapn2])-np.min([gapn1,gapn2]))/np.max([gapn1,gapn2])
        
        
        
        ppvout=np.zeros([H,num_nodes])
        
        
        for h in range(H):
            ppvout[h]=ppv
            
        
        pbcout=np.zeros(H)
        pbcout[0]=pbco
        
        soc0out=np.zeros(H)
        soc0out[0]=soc0o
        
        out1=np.vstack((ploss,qloss,pgo[:,iref],qgo[:,iref],t,gap,ic,pbcout,pbo,pbeo,pbreo,soc0out,soco)).T
        output=np.hstack((vo,pho,Equp,Equq,out1,zbo,dc,ppvout))
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
        columns.append('Gap')
        columns.append('ic')
        columns.append('pbc')
        columns.append('pb')
        columns.append('pbe')
        columns.append('pbre')
        columns.append('soc0')
        columns.append('soc')
        for i in range(num_nodes):    
            columns.append('zb'+str(i+1))
        for i in range(ncluster):    
            columns.append('dc_c'+str(i+1))
        for i in range(num_nodes):    
            columns.append('ppv'+str(i+1))
        
            
        df.columns=columns
        
        solvlist=[0]*H
        solvlist[0]='AB'
        
        
        df.insert(len(df.columns),'Solver',solvlist)
        df.to_excel('C:\\Users\\diego\\OneDrive\\Desktop\\aLL\\PhD\\Tesis\\Python\\'+problem+'\\'+city[ct]+'\\'+case[ca]+'\\'+conv+'\\'+form+'\\'+fold+'\\Results.xlsx')