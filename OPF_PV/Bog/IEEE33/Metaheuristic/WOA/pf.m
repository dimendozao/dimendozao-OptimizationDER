function [v,ph,eqp,eqq,pl,ql,pg,qg]=pf(mpc)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    results=runpf(mpc,opt);
    N=size(mpc.bus,1);
    L=size(mpc.branch,1);
    v=results.bus(:,8);
    ph=deg2rad(results.bus(:,9));
    pl=zeros(N,1);
    ql=zeros(N,1);    
    pl(1)=sum(results.branch(:,14)+results.branch(:,16));
    ql(1)=sum(results.branch(:,15)+results.branch(:,17));
    pg=zeros(N,1);
    qg=zeros(N,1);
    pg(1)=results.gen(1,2);
    qg(1)=results.gen(1,3);
    eqp=zeros(N,1);
    eqq=zeros(N,1);
    ym=zeros(N,N);
    for k=1:L
        ym(mpc.branch(k,1),mpc.branch(k,2))=-1/complex(mpc.branch(k,3),mpc.branch(k,4));
        ym(mpc.branch(k,2),mpc.branch(k,1))=-1/complex(mpc.branch(k,3),mpc.branch(k,4));
    end
    for i=1:N
        ym(i,i)=-sum(ym(i,:));
    end
    for i=1:N
        for j=1:N
            eqp(i)=eqp(i)+(v(i)*v(j)*abs(ym(i,j))*cos(ph(i)-ph(j)-angle(ym(i,j))));
            eqq(i)=eqq(i)+(v(i)*v(j)*abs(ym(i,j))*sin(ph(i)-ph(j)-angle(ym(i,j))));
        end
    end


    
    

    

    
    