clc
clear


branch=readtable('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Systems\CA141Branch.csv');
bus=readtable('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Systems\CA141Bus.csv');
gen=readtable('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Systems\CA141Gen.csv');

num_lines= size(branch,1);
num_nodes= size(bus,1);
iref=find(bus.type==3);


ngen=sum(numel(find(bus.type==2)));
sgen=zeros(num_nodes,1);
vgen=zeros(num_nodes,1);

vmax=bus.vmax;
vmin=bus.vmin;

sd=zeros(num_nodes,1);

for k =1:num_lines
    sd(branch.j(k))=complex(branch.pj(k),branch.qj(k));
    
end

pd=real(sd);
qd=imag(sd);



if ngen>0
    for i=1:ngen
        sgen(bus.i(i))=gen.pi(i);
        vmax(bus.i(i))=gen.vst(i);
        vmin(bus.i(i))=gen.vst(i);
    end
end

vmax(iref)=1;
vmin(iref)=1;

prefmax=zeros(num_nodes,1);
qrefmax=zeros(num_nodes,1);

prefmin=zeros(num_nodes,1);
qrefmin=zeros(num_nodes,1);

prefmax(iref)=gen.pmax(find(gen.i==iref));
prefmin(iref)=gen.pmin(find(gen.i==iref));
qrefmax(iref)=gen.qmax(find(gen.i==iref));
qrefmin(iref)=max(0,gen.qmin(find(gen.i==iref)));


ym=zeros(num_lines);
fr=zeros(num_lines);
to=zeros(num_lines);

for k=1:num_lines
    fr(k)=branch.i(k);
    to(k)=branch.j(k);
    ym(k)=1/complex(branch.r(k),branch.x(k));    
end

yrl=real(ym);
yil=imag(ym);

umax=vmax.^2;
umin=vmin.^2;

wrmax=zeros(num_lines,1);
wrmin=zeros(num_lines,1);

wimax=zeros(num_lines,1);
wimin=zeros(num_lines,1);

for k=1:num_lines       
    wrmax(k)=vmax(fr(k))*vmax(to(k));
    wrmin(k)=vmin(fr(k))*vmin(to(k));
    wimax(k)=vmax(fr(k))*vmax(to(k));
    wimin(k)=-vmax(fr(k))*vmax(to(k));
end

upmin=ones(num_lines,1).*min(umin)*2;
upmax=ones(num_lines,1).*max(umax)*2;
ummin=-ones(num_lines,1).*max(umax);
ummax=ones(num_lines,1).*max(umax);

% c   is the vector of coefficients for linear objective c*x
% a   is the matrix of coefficients for linear constraints a*x
% blc is the vector of the lower bound rhs of the linear constraints ax>=lbc
% buc is the vector of the upper bound rhs of the linear constraints ax<=ubc
%     for any given constraint, if blc=buc, then the constraint is an equality
%     otherwise the constraint is an inequality, and should have an infinite (inf
%     or -inf) value on blc or buc.
% blx is the vector of lower bounds for x
% bux is the vector of upper bounds for x



% variables pg(1:N), qg(N+1:2N), u(2N+1:3N), wr(3N+1:3N+L),
% wi(3N+L+1:3N+2L),up(3N+2L+1:3N+3L),um(3N+3L+1:3N+4L)
% nvar=3N+4L

%variable definition
nvar=(3*num_nodes)+(4*num_lines);
ncon=(2*num_nodes)+(2*num_lines);
%initialization of mosek vectors
blx=zeros(nvar,1);
bux=zeros(nvar,1);
a=zeros(ncon,nvar);
blc=zeros(ncon,1);
buc=zeros(ncon,1);

% variable lower bounds
blx(1:num_nodes)=prefmin;
blx(num_nodes+1:2*num_nodes)=qrefmin;
blx(2*num_nodes+1:3*num_nodes)=umin;
blx(3*num_nodes+1:(3*num_nodes)+(num_lines))=wrmin;
blx((3*num_nodes)+(num_lines)+1:(3*num_nodes)+(2*num_lines))=wimin;
blx((3*num_nodes)+(2*num_lines)+1:(3*num_nodes)+(3*num_lines))=upmin;
blx((3*num_nodes)+(3*num_lines)+1:(3*num_nodes)+(4*num_lines))=ummin;

% variable upper bounds
bux(1:num_nodes)=prefmax;
bux(num_nodes+1:2*num_nodes)=qrefmax;
bux(2*num_nodes+1:3*num_nodes)=umax;
bux(3*num_nodes+1:(3*num_nodes)+(num_lines))=wrmax;
bux((3*num_nodes)+(num_lines)+1:(3*num_nodes)+(2*num_lines))=wimax;
bux((3*num_nodes)+(2*num_lines)+1:(3*num_nodes)+(3*num_lines))=upmax;
bux((3*num_nodes)+(3*num_lines)+1:(3*num_nodes)+(4*num_lines))=ummax;

% linear constraints bounds
blc(1:num_nodes)=-pd;
blc(num_nodes+1:2*num_nodes)=-qd;
buc(1:num_nodes)=-pd;
buc(num_nodes+1:2*num_nodes)=-qd;

% linear constraints coefficient matrix

for i=1:num_nodes
    %Pg(1:N) variable in (i) constraint (active power flow)
    a(i,i)=-1;
    %Qg(N+1:2N) variable (i+N) constraint (reactive power flow)
    a(i+num_nodes,i+num_nodes)=-1;
    for k=1:num_lines
        if i==fr(k)
            %U(2N+1:3N) variable in (i) constraint (active power flow)
            a(i,i+(2*num_nodes))=yrl(k);            
            %Wr(3N+1:3N+L) variable in (i) constraint (active power flow)
            a(i,(3*num_nodes)+k)=-yrl(k);
            %Wi(3N+L+1:3N+2L) variable in (i) constraint (active power flow)
            a(i,(3*num_nodes)+(num_lines)+k)=-yil(k);
            %U(2N+1:3N) variable in (i+N) constraint (reactive power flow)
            a(num_nodes+i,i+(2*num_nodes))=-yil(k);
            %Wr(3N+1:3N+L) variable in (i+N) constraint (reactive power flow)
            a(num_nodes+i,(3*num_nodes)+k)=yil(k);
            %Wi(3N+L+1:3N+2L) variable in (i+N) constraint (reactive power flow)
            a(num_nodes+i,(3*num_nodes)+(num_lines)+k)=-yrl(k);
        end
        if i==to(k)
            %U(2N+1:3N) variable in (i) constraint (active power flow)
            a(i,i+(2*num_nodes))=yrl(k);
            %Wr(3N+1:3N+L) variable in (i) constraint (active power flow)
            a(i,(3*num_nodes)+k)=-yrl(k);
            %Wi(3N+L+1:3N+2L) variable in (i) constraint (active power flow)
            a(i,(3*num_nodes)+(num_lines)+k)=yil(k);
            %U(2N+1:3N) variable in (i+N) constraint (reactive power flow)
            a(num_nodes+i,i+(2*num_nodes))=-yil(k);
            %Wr(3N+1:3N+L) variable in (i+N) constraint (reactive power flow)
            a(num_nodes+i,(3*num_nodes)+k)=yil(k);
            %Wi(3N+L+1:3N+2L) variable in (i+N) constraint (reactive power flow)
            a(num_nodes+i,(3*num_nodes)+(num_lines)+k)=yrl(k);
        end
    end
end

for k=1:num_lines
    % Up-U(fr)-U(to)=0 constraints (blc and buc are defined zero for these
    % constraints)
    % Up
    a((2*num_nodes)+k,(3*num_nodes)+(2*num_lines)+k)=1;
    % -U(fr)
    a((2*num_nodes)+k,(2*num_nodes)+fr(k))=-1;
    % -U(to)
    a((2*num_nodes)+k,(2*num_nodes)+to(k))=-1;
    % Um-U(fr)+U(to)=0 constraints
    % Um
    a((2*num_nodes)+k,(3*num_nodes)+(3*num_lines)+k)=1;
    % -U(fr)
    a((2*num_nodes)+k,(2*num_nodes)+fr(k))=-1;
    % U(to)
    a((2*num_nodes)+k,(2*num_nodes)+to(k))=1;
end

%linear objective
c=zeros(nvar,1);
for i=1:num_nodes    
    for k=1:num_lines
        if i==fr(k)
            %U(2N+1:3N) variable in (i) constraint (active power flow)
            c(i+(2*num_nodes))=c(i+(2*num_nodes))+yrl(k);            
            %Wr(3N+1:3N+L) variable in (i) constraint (active power flow)
            c((3*num_nodes)+k)=c((3*num_nodes)+k)-yrl(k);
            %Wi(3N+L+1:3N+2L) variable in (i) constraint (active power flow)
            c((3*num_nodes)+(num_lines)+k)=c((3*num_nodes)+(num_lines)+k)-yil(k);
            %U(2N+1:3N) variable in (i+N) constraint (reactive power flow)
            c(i+(2*num_nodes))=c(i+(2*num_nodes))-yil(k);
            %Wr(3N+1:3N+L) variable in (i+N) constraint (reactive power flow)
            c((3*num_nodes)+k)=c((3*num_nodes)+k)+yil(k);
            %Wi(3N+L+1:3N+2L) variable in (i+N) constraint (reactive power flow)
            c((3*num_nodes)+(num_lines)+k)=c((3*num_nodes)+(num_lines)+k)-yrl(k);
        end
        if i==to(k)
            %U(2N+1:3N) variable in (i) constraint (active power flow)
            c(i+(2*num_nodes))=c(i+(2*num_nodes))+yrl(k);            
            %Wr(3N+1:3N+L) variable in (i) constraint (active power flow)
            c((3*num_nodes)+k)=c((3*num_nodes)+k)-yrl(k);
            %Wi(3N+L+1:3N+2L) variable in (i) constraint (active power flow)
            c((3*num_nodes)+(num_lines)+k)=c((3*num_nodes)+(num_lines)+k)+yil(k);
            %U(2N+1:3N) variable in (i+N) constraint (reactive power flow)
            c(i+(2*num_nodes))=c(i+(2*num_nodes))-yil(k);
            %Wr(3N+1:3N+L) variable in (i+N) constraint (reactive power flow)
            c((3*num_nodes)+k)=c((3*num_nodes)+k)+yil(k);
            %Wi(3N+L+1:3N+2L) variable in (i+N) constraint (reactive power flow)
            c((3*num_nodes)+(num_lines)+k)=c((3*num_nodes)+(num_lines)+k)+yrl(k);
        end
    end
end

% conic constraints 
% the vector accs defines the type and size of the cone
% accs=[res.symbcon.MSK_DOMAIN_QUADRATIC_CONE 3]
% the matrix f defines the variables involved in the cone 
% for example, if the cone is quadratic and size 3:
% f=[1 0 0 0 0;
%    0 1 0 0 0;
%    0 0 0 1 0]  means x1>=sqrt(x2^2 + x4^2)
% the matrix f has size (nc*szc) X nvar
% where nc is the number of cones and szc is the size of each cone

%cones up>=sqrt(2wr^2+2wi^2+um^2)--> nc= L, szc=4
f=zeros((num_lines*4),nvar);
[r, res] = mosekopt('symbcon');

accs=[res.symbcon.MSK_DOMAIN_QUADRATIC_CONE 4];
for k=1:num_lines
    %Up
    f((4*(k-1))+1,(3*num_nodes)+(2*num_lines)+k)=1;
    %2Wr
    f((4*(k-1))+2,(3*num_nodes)+k)=2;
    %2Wr
    f((4*(k-1))+3,(3*num_nodes)+(num_lines)+k)=2;
    %Um
    f((4*(k-1))+4,(3*num_nodes)+(3*num_lines)+k)=1;
    if k>1
        accs=cat(2,accs,[res.symbcon.MSK_DOMAIN_QUADRATIC_CONE 4]);
    end
end

%prob.c=zeros(nvar,1);
prob.c   = c;
prob.a   = a;
prob.blc = blc;
prob.buc = buc;
prob.blx = blx;
prob.bux = bux;
prob.accs = accs;
prob.f = sparse(f);
param.MSK_IPAR_INFEAS_REPORT_AUTO = 'MSK_ON';
param.MSK_IPAR_PRESOLVE_USE = 'MSK_PRESOLVE_MODE_ON';
probname='Power Flow';
objname='Losses';

con=cell(ncon,1);
var=cell(nvar,1);
acc=cell(num_lines,1);

for i=1:num_nodes
    con{i}=strcat('pf',num2str(i));
    con{i+num_nodes}=strcat('qf',num2str(i));
    var{i}=strcat('pg',num2str(i));
    var{i+num_nodes}=strcat('qg',num2str(i));
    var{i+(2*num_nodes)}=strcat('u',num2str(i));
end
for k=1:num_lines
    con{k+(2*num_nodes)}=strcat('upc',num2str(k));
    con{k+num_lines+(2*num_nodes)}=strcat('umc',num2str(k));
    var{k+(3*num_nodes)}=strcat('wr',num2str(k));
    var{k+num_lines+(3*num_nodes)}=strcat('wi',num2str(k));
    var{k+(2*num_lines)+(3*num_nodes)}=strcat('up',num2str(k));
    var{k+(3*num_lines)+(3*num_nodes)}=strcat('um',num2str(k));
    acc{k}=strcat('SOC',num2str(k));
end
names=struct();
names.name=probname;
names.obj=objname;
names.con=con;
names.var=var;
names.acc=acc;
prob.names=names;

[r,res]=mosekopt('minimize',prob,param);

% eps = 1e-7;
% disp("Variable bounds important for infeasibility: ");
% analyzeCertificate(res.sol.itr.slx, res.sol.itr.sux, eps);
% disp("Constraint bounds important for infeasibility: ")
% analyzeCertificate(res.sol.itr.slc, res.sol.itr.suc, eps);








function analyzeCertificate(sl, su, eps)
    n = size(sl);
    for i=1:n 
        if abs(sl(i)) > eps 
            disp(sprintf("#%d: lower, dual = %e", i, sl(i))); 
        end
        if abs(su(i)) > eps 
            disp(sprintf("#%d: upper, dual = %e", i, su(i)));
        end
    end
end