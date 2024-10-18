clc
clear

load("C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\Bog\ClusterMeans_BOG.mat")
load("C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\Bog\SA_Bog_ClusterNode.mat")

mpc=loadcase(caseSA_J23);

mpc.bus(:,13)=mpc.bus(:,13)-0.1;
mpc.bus(:,12)=mpc.bus(:,12)+0.1;


N=numel(mpc.bus(:,1));
H=numel(clustermeans(1,:));

mpc=repmat(mpc,H,1);

mpc=mod_mpc_cluster_h(mpc,clustermeans,clusternode);

[pg,qg,pl,ql,v,ph,t1]=pfh(mpc);
t=zeros(H,1);
t(1)=sum(t1);
vars={};

for i=1:N
    vars{end+1}=strcat('v',num2str(i));
end

for i=1:N
    vars{end+1}=strcat('ph',num2str(i));
end
vars{end+1}='pl';
vars{end+1}='ql';
vars{end+1}='pg';
vars{end+1}='qg';
vars{end+1}='t';

path='C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\MPF\Bog\SA\Matpower\Results.csv';

carr=cat(2,v,ph,pl,ql,pg,qg,t);


tab=array2table(carr,"VariableNames",vars);

writetable(tab,path);

