%___________________________________________________________________%
%  Grey Wolf Optimizer (GWO) source codes version 1.0               %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili, S. M. Mirjalili, A. Lewis             %
%               Grey Wolf Optimizer, Advances in Engineering        %
%               Software , in press,                                %
%               DOI: 10.1016/j.advengsoft.2013.12.007               %
%                                                                   %
%___________________________________________________________________%

% You can simply define your cost in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run GWO: [Best_score,Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%__________________________________________

clc
clear

casen='IEEE33';
city='Bog';
city1='BOG';
problem='OPF_BESS_IS';

prob_sol='OPF_PV_S';

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city,'\MeansRAD_',city1,'.mat'));

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\ClusterMeans_',city1,'.mat'));

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\',casen,'_',city,'_ClusterNode.mat'));
clusternode(1)=1;

ppvtab=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',prob_sol,'\',city,'\',casen,'\bestsol.csv'));

NW=100; % Number of search agents
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

H=size(means,2);
N=size(mpc.bus,1);
C=size(clustermeans,1);

ic=ppvtab.ic;
dc=zeros(C,H);
pvc=zeros(N,1);

dc(1,:)=ppvtab.dc1;
dc(2,:)=ppvtab.dc2;
dc(3,:)=ppvtab.dc3;
dc(4,:)=ppvtab.dc4;
dc(5,:)=ppvtab.dc5;

for i=1:N
    pvc(i)=ppvtab{1,N+i+1};
end

nb=1;
pbc=1*sum(mpc.bus(:,3));
nvar=((2*H)+1)*nb; % 24 Zb + 1 Cap + 1 SOC0 + 23 SOC (SOC(24)=SOC0)
pveff=0.8;

mpc.bus(:,13)=mpc.bus(:,13)-0.1;
mpc.bus(:,12)=mpc.bus(:,12)+0.1;

mpc=repmat(mpc,H,1);

lb=zeros(1,nvar);
ub=ones(1,nvar);

lb(1:H)=1;
ub(1:H)=N;
ub(H+1)=pbc;

ae=0;

f=@(x)fo(mpc,ae,pvc,ic,pveff,x,clusternode,dc);

tr=zeros(NT,1);
Best_pos=zeros(NT,nvar);
Best_score=zeros(NT,1);

for i=1:NT
    disp(['Progress: ',num2str(i),' of ',num2str(NT)])
    t1 = tic;
    [Best_score(i),Best_pos(i,:),~]=GWO(NW,NI,lb,ub,nvar,f);
    tr(i) = toc(t1);
end

bestpl=zeros(NT,1);

for i=1:NT
    mpc1=mod_mpc_dg(mpc,Best_pos(i,:),pvc,pveff,ic,clusternode,dc);
    [~,~,~,~,plh,~,~,~,~]=pfh(mpc1);
    bestpl(i)=sum(plh);
end

[~,idx]=min(bestpl);

mpc1=mod_mpc_dg(mpc,Best_pos(idx,:),pvc,pveff,ic,clusternode,dc);
[v,ph,eqp,eqq,plh,qlh,pg,qg,~]=pfh(mpc1);

t=zeros(H,1);
t(1)=tr(idx);

zb=zeros(H,N);
pb=zeros(H,1);
pbco=zeros(H,1);
soc=zeros(H,1);
soc0=zeros(H,1);
gap=zeros(H,1);
ppvo=zeros(H,N);

for h=1:H
    for i=1:N
        if round(Best_pos(idx,h))==i
            zb(h,i)=1;
        end
    end
end

pbco(1)=Best_pos(idx,H+1);
soc0(1)=Best_pos(idx,H+2);
soc(1:H-1)=Best_pos(idx,H+3:end);
soc(H)=Best_pos(idx,H+2);

for h=1:H
    if h==1
        pb(h)=(soc0(1)-soc(h))*pbco(1);
    elseif h<H
        pb(h)=(soc(h-1)-soc(h))*pbco(1);
    else
        pb(h)=(soc(h-1)-soc0(1))*pbco(1);
    end
end

for i=1:N
    ppvo(:,i)=pvc(i);
end


ico=ic;
dco=dc';


tab=table(v,ph,eqp,eqq,plh,qlh,pg,qg,t,gap,ico,pbco,pb,soc0,soc,zb,dco,ppvo);
tab.Solver=string(zeros(H,1));
tab.Solver(1)='GWO_PAR';
writetable(tab,'Results.xlsx')

types=cell(nvar+2,1);
types(:)={'double'};
names=cell(nvar+2,1);
for i=1:nvar+2
    if i==1
        names(i)={'obj'};
    elseif i==2
        names(i)={'t'};
    elseif i<=H+2
        names(i)={strcat('zbh',num2str(i-2))};
    elseif i==H+3
        names(i)={'pbc'};
    elseif i==H+4
        names(i)={'soc0'};
    else 
        names(i)={strcat('soc',num2str(i-H-4))};    
    end    
end
tab1 = table('Size',[NT nvar+2],'VariableTypes',types,'VariableNames',names);


for i=1:nvar+2
    if i==1
        tab1.(names{i})=bestpl;
    elseif i==2
        tab1.(names{i})=tr;
    elseif i<=H+2        
        tab1.(names{i})=round(Best_pos(:,i-2));    
    else
        tab1.(names{i})=Best_pos(:,i-2);
    end
end

writetable(tab1,'MetaResults.xlsx')



