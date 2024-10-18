clc
clear

casen='IEEE69';
city='Pop';
city1='POP';
problem='OPF_PV';

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city,'\MeansRAD_',city1,'.mat'));

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\ClusterMeans_',city1,'.mat'))

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\',casen,'_',city,'_ClusterNode.mat'));

clusternode(1)=1;

NP=100; % Population size
NI=100; % Maximum numbef of iterations
NT=20; % Maximum numbef of tests

%name='CA141';
% name='IEEE33';
% name='IEEE69';
% name='SA';

if strcmp(casen,'IEEE33')
    mpc=loadcase(case33_DM);
elseif strcmp(casen,'IEEE69')
    mpc=loadcase(case69_DM);
elseif strcmp(casen,'CA141')
    mpc=loadcase(case141);
    mpc=order_radial(mpc); 
elseif strcmp(casen,'SA')
    mpc=loadcase(caseSA_J23);
end
N=size(mpc.bus,1);
rad=mean(means);
dem=mean(clustermeans,2);
idem=zeros(N,1);
for i=1:N
    idem(i)=dem(clusternode(i));
end


npv=floor(0.1*N);
pvc=0.5*sum(mpc.bus(:,3));
pveff=0.8;
nvar=2*npv;

lb=zeros(1,2*npv);
ub=ones(1,2*npv);

lb(1:npv)=2;
ub(1:npv)=N;
ae=0;

mpc.bus(:,13)=mpc.bus(:,13)-0.1;
mpc.bus(:,12)=mpc.bus(:,12)+0.1;

f=@(x)fo(mpc,ae,pvc,rad,pveff,idem,x);
tr=zeros(NT,1);
Best_pos=zeros(NT,nvar);
Best_score=zeros(NT,1);


for i=1:NT
    disp(['Progress: ',num2str(i),' of ',num2str(NT)])
    popini=initialization(NP,nvar,ub,lb);
    opts = optimoptions('ga', 'PopulationSize',NP, 'InitialPopulationMatrix',popini, 'MaxGenerations',NI,'MaxStallGenerations',NI,'UseParallel',true);
    t1 = tic;
    [Best_pos(i,:),Best_score(i)]=ga(f,nvar,[],[],[],[],lb,ub,[],1:npv,opts);
    tr(i) = toc(t1);    
end
bestpl=zeros(NT,1);

for i=1:NT
    mpc1=mod_mpc_dg(mpc,Best_pos(i,:),pvc,rad,pveff,idem);
    [~,~,~,~,pl,~,~,~]=pf(mpc1);
    bestpl(i)=pl(1);
end

[~,idx]=min(bestpl);
mpc1=mod_mpc_dg(mpc,Best_pos(idx,:),pvc,rad,pveff,idem);

[v,ph,eqp,eqq,pl,ql,pg,qg]=pf(mpc1);

t=zeros(N,1);
t(1)=tr(idx);
zpv=zeros(N,1);
ppv=zeros(N,1);
for i=1:npv
    zpv(round(Best_pos(idx,i)))=1;
    ppv(round(Best_pos(idx,i)))=Best_pos(idx,i+npv)*pvc;
end
pv=ppv;
tab=table(v,ph,eqp,eqq,pl,ql,pg,qg,pv,ppv,zpv,t);
tab.Solver=string(zeros(N,1));
tab.Solver(1)='GA_PAR';
writetable(tab,'Results.xlsx')

types=cell(nvar+2,1);
types(:)={'double'};
names=cell(nvar+2,1);
for i=1:npv+2
    if i==1
        names(i)={'obj'};
    elseif i==2
        names(i)={'t'};
    else
        names(i)={strcat('zpv',num2str(i-2))};
        names(i+npv)={strcat('ppv',num2str(i-2))};
    end
end
tab1 = table('Size',[NT nvar+2],'VariableTypes',types,'VariableNames',names);

for i=1:nvar+2
    if i==1
        tab1.(names{i})=bestpl;
    elseif i==2
        tab1.(names{i})=tr;
    else
        if i<=npv+2
            tab1.(names{i})=round(Best_pos(:,i-2));
        else
            tab1.(names{i})=Best_pos(:,i-2)*pvc;
        end
    end
end


writetable(tab1,'MetaResults.xlsx') 




        



