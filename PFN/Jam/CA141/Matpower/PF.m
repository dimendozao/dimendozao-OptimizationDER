clc
clear

casen='CA141';
city='Jam';
city1='JAM';
problem='PFN';

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\ClusterMeans_',city1,'.mat'))

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\',casen,'_',city,'_ClusterNode.mat'));

clusternode(1)=1;

mpc=loadcase(case141);
mpc=order_radial(mpc); 


N=numel(mpc.bus(:,1));
dem=mean(clustermeans,2);
idem=zeros(N,1);
for i=1:N
    idem(i)=dem(clusternode(i));
end

mpc.bus(:,3)=mpc.bus(:,3).*idem;
mpc.bus(:,4)=mpc.bus(:,4).*idem;

opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
results=runpf(mpc,opt);
v=results.bus(:,8);
ph=deg2rad(results.bus(:,9));
pg=zeros(N,1);
qg=zeros(N,1);
pl=zeros(N,1);
ql=zeros(N,1);
t=zeros(N,1);

pl(1)=sum(results.branch(:,14))+sum(results.branch(:,16));
ql(1)=sum(results.branch(:,15))+sum(results.branch(:,17));

pg(1)=results.gen(1,2);
qg(1)=results.gen(1,3);

t(1)=results.et;
path=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city,'\',casen,'\Matpower\Results.csv');

tab=table(v,ph,pl,ql,pg,qg,t,'VariableNames',{'v','ph','pl','ql','pg','qg','t'}); 

writetable(tab,path);

