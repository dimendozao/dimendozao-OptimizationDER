clc
clear


mpc=loadcase(case33_DM);

N=numel(mpc.bus(:,1));
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
path='C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\PF\IEEE33\Matpower\Results.csv';

tab=table(v,ph,pl,ql,pg,qg,t,'VariableNames',{'v','ph','pl','ql','pg','qg','t'}); 

writetable(tab,path);

