function [v,ph,eqp,eqq,plh,qlh,pg,qg,t]=pfh(mpc)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    H=numel(mpc);
    N=numel(mpc(1).bus(:,1));
    L=numel(mpc(1).branch(:,1));
    v=zeros(H,N);
    ph=zeros(H,N);
    pg=zeros(H,1);
    qg=zeros(H,1);
    plh=zeros(H,1);
    qlh=zeros(H,1);    
    t=zeros(H,1);
    eqp=zeros(H,N);
    eqq=zeros(H,N);
    ym=zeros(N,N);
    mpc1=mpc;
    for k=1:L
        ym(mpc1(1).branch(k,1),mpc1(1).branch(k,2))=-1/complex(mpc1(1).branch(k,3),mpc1(1).branch(k,4));
        ym(mpc1(1).branch(k,2),mpc1(1).branch(k,1))=-1/complex(mpc1(1).branch(k,3),mpc1(1).branch(k,4));
    end
    for i=1:N
        ym(i,i)=-sum(ym(i,:));
    end
    for h=1:H
        results=runpf(mpc1(h),opt);
        v(h,:)=results.bus(:,8);
        ph(h,:)=deg2rad(results.bus(:,9));
        plh(h)=sum(results.branch(:,14))+sum(results.branch(:,16));
        qlh(h)=sum(results.branch(:,15))+sum(results.branch(:,17));
        pg(h)=results.gen(1,2);
        qg(h)=results.gen(1,3);
        t(h)=results.et;        
        for i=1:N
            for j=1:N
                eqp(h,i)=eqp(h,i)+(v(h,i)*v(h,j)*abs(ym(i,j))*cos(ph(h,i)-ph(h,j)-angle(ym(i,j))));
                eqq(h,i)=eqq(h,i)+(v(h,i)*v(h,j)*abs(ym(i,j))*sin(ph(h,i)-ph(h,j)-angle(ym(i,j))));
            end
        end
    end