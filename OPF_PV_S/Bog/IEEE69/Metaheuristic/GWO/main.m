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

casen='IEEE69';
city='Bog';
city1='BOG';
problem='OPF_PV_S';

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\',casen,'_',city,'_','ClusterNode.mat'));
c1 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\','ParamTableC1.csv'));
c2 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\','ParamTableC2.csv'));
c3 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\','ParamTableC3.csv'));
c4 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\','ParamTableC4.csv'));
c5 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\','ParamTableC5.csv'));

irr = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city,'\','ParamTable.csv'));

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\NSTparamDem_',city1,'.mat'));
dparams=params;

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city,'\NSTparamRAD_',city1,'.mat'));
iparams=params;

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city,'\PWLxyRAD_',city1,'.mat'));

xirrpwl=xpwl;
yirrpwl=ypwl;

load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city,'\PWLxyDem_',city1,'.mat'));

xdempwl=xpwl;
ydempwl=ypwl;

adists={'Exponential','Fisk','Logistic','Log-Normal','Normal','Rayleigh','Weibull'};
adists1={'Exponential','LogLogistic','Logistic','LogNormal','Normal','Rayleigh','Weibull'};
nddparams=size(c1,1);


ndists=numel(adists);
ncluster=size(dparams,1);
H=size(dparams,2);

cparameters=[c1;c2;c3;c4;c5];

bestfitsd=zeros(H,ncluster);


bestfitsi=10*ones(H,1);
 
ihours=iparams(:,1)~=0;
nihours=sum(ihours);

for h=1:H
    if ihours(h)~=0
        for j=1:ndists
            if startsWith(irr.bestparams1((2*(h-7))-1),adists{j})
               bestfitsi(h)=j;
            end
        end        
    end
    for i=1:ncluster
        for j=1:ndists
            if startsWith(cparameters.bestparams1(((i-1)*48)+((2*h)-1)),adists{j})
                bestfitsd(h,i)=j;
            end
        end
    end
end

mindc=zeros(H*ncluster,1);
maxdc=zeros(H*ncluster,1);

minic=zeros(nihours,1);
maxic=zeros(nihours,1);
k=1;

for c=1:ncluster
    for h=1:H
        dist=bestfitsd(h,c);
        mindc(k)=min(xdempwl(h,c,dist,:));
        maxdc(k)=max(xdempwl(h,c,dist,:));
        k=k+1;
    end
end
k=1;
for h=1:H
    dist=bestfitsi(h);
    if dist~=10
        minic(k)=min(xirrpwl(h,dist,:));
        maxic(k)=max(xirrpwl(h,dist,:));
        k=k+1;
    end
end

NW=100; % Number of search agents
NI=100; % Maximum numbef of iterations
NT=20; % Maximum numbef of tests

%name='CA141';
% name='IEEE33';
% name='IEEE69';
% name='SA';

if strcmp(casen,'IEEE33')
    mpc=loadcase(case33_DM);
    N=size(mpc.bus,1);
    npv=floor(0.1*N);
elseif strcmp(casen,'IEEE69')
    mpc=loadcase(case69_DM);
    N=size(mpc.bus,1);
    npv=3;
elseif strcmp(casen,'CA141')
    mpc=loadcase(case141);
    mpc=order_radial(mpc);
    N=size(mpc.bus,1);
    npv=5;
elseif strcmp(casen,'SA')
    mpc=loadcase(caseSA_J23);
    N=size(mpc.bus,1);
    npv=floor(0.1*N);
end

pvc=0.5*sum(mpc.bus(:,3));
pveff=0.8;

nvar=(2*npv)+nihours+(H*ncluster); % Npv locations, Npv capacities, nih (ic), H*C (dc)

mpc.bus(:,13)=mpc.bus(:,13)-0.1;
mpc.bus(:,12)=mpc.bus(:,12)+0.1;

mpc=repmat(mpc,H,1);

lb=zeros(1,nvar);
ub=ones(1,nvar);

lb(1:npv)=2;
ub(1:npv)=N;
ub((2*npv)+1:(2*npv)+nihours)=maxic;
ub((2*npv)+nihours+1:end)=maxdc;
lb((2*npv)+1:(2*npv)+nihours)=minic;
lb((2*npv)+nihours+1:end)=mindc;

ae=0;

f=@(x)fo(mpc,ae,pvc,pveff,x,bestfitsi,bestfitsd,iparams,dparams,clusternode,adists1);

tr=zeros(NT,1);
Best_pos=zeros(NT,nvar);
Best_score=zeros(NT,1);

for i=1:NT
    disp(['Progress: ',num2str(i),' of ',num2str(NT)])
    t1 = tic;
    [Best_score(i),Best_pos(i,:),~]=GWO(NW,NI,lb,ub,nvar,npv,f);
    tr(i) = toc(t1);
end
%% 

bestpl=zeros(NT,1);

for i=1:NT
    mpc1=mod_mpc_dg(mpc,Best_pos(i,:),pvc,pveff,clusternode,ihours);
    ico=zeros(H,1);
    dco=zeros(H,ncluster);
    k=1;
    for c=1:ncluster
        for h=1:H
            dist=bestfitsd(h,c);
            dco(h,c)=Best_pos(i,(2*npv)+nihours+k);         
            k=k+1;
        end
    end    
    k=1;
    for h=1:H
        dist=bestfitsi(h);
        if dist~=10
            ico(h)=Best_pos(i,(2*npv)+k);        
            k=k+1;
        end
    end
    [~,~,~,~,plh,~,~,~,~,~,~]=pfh(mpc1,bestfitsi,bestfitsd,iparams,dparams,adists1,ico,dco);
    bestpl(i)=sum(plh);
end

[~,idx]=min(bestpl);

mpc1=mod_mpc_dg(mpc,Best_pos(idx,:),pvc,pveff,clusternode,ihours);
ico=zeros(H,1);
dco=zeros(H,ncluster);

k=1;
for c=1:ncluster
    for h=1:H
        dist=bestfitsd(h,c);
        dco(h,c)=Best_pos(idx,(2*npv)+nihours+k);         
        k=k+1;
    end
end

k=1;
for h=1:H
    dist=bestfitsi(h);
    if dist~=10
        ico(h)=Best_pos(idx,(2*npv)+k);        
        k=k+1;
    end
end

[v,ph,eqp,eqq,plh,qlh,pg,qg,probi,probd,~]=pfh(mpc1,bestfitsi,bestfitsd,iparams,dparams,adists1,ico,dco);

t=zeros(H,1);
t(1)=tr(idx);

zpv=zeros(N,1);
ppv=zeros(N,1);
for i=1:npv
    zpv(round(Best_pos(idx,i)))=1;
    ppv(round(Best_pos(idx,i)))=Best_pos(idx,i+npv)*pvc;
end
pv=ppv;

pvo=zeros(H,N);
ppvo=zeros(H,N);
zpvo=zeros(H,N);

for i=1:N
    pvo(1,i)=pv(i);
    ppvo(1,i)=ppv(i);
    zpvo(1,i)=zpv(i);
end



tab=table(v,ph,eqp,eqq,plh,qlh,pg,qg,t,ico,dco,ppvo,zpvo);
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
    elseif i<=npv+2
        names(i)={strcat('zpv',num2str(i-2))};
    elseif i<=(2*npv)+2
        names(i)={strcat('ppv',num2str(i-npv-2))};
    elseif i<=(2*npv)+nihours+2
        names(i)={strcat('ic',num2str(i-(2*npv)-2+7))};        
    else
        ii=i-(2*npv)-nihours-2;
        cc=ceil(ii/H);
        names(i)={strcat('dc_c',num2str(cc),'h',num2str((ii-((cc-1)*24))))};
    end    
end
tab1 = table('Size',[NT nvar+2],'VariableTypes',types,'VariableNames',names);

for i=1:nvar+2
    if i==1
        tab1.(names{i})=bestpl;
    elseif i==2
        tab1.(names{i})=tr;
    elseif i<=npv+2        
        tab1.(names{i})=round(Best_pos(:,i-2));
    elseif i<=(2*npv)+2 
        tab1.(names{i})=Best_pos(:,i-2)*pvc;
    else
        tab1.(names{i})=Best_pos(:,i-2);    
    end
end

writetable(tab1,'MetaResults.xlsx')


