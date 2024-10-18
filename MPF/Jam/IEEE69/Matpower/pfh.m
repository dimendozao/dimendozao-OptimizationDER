function [pg,qg,plh,qlh,v,ph,t]=pfh(mpc)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    H=numel(mpc);
    NN=numel(mpc(1).bus(:,1));
    v=zeros(H,NN);
    ph=zeros(H,NN);
    pg=zeros(H,1);
    qg=zeros(H,1);
    plh=zeros(H,1);
    qlh=zeros(H,1);
    t=zeros(H,1);
    for h=1:H
        results=runpf(mpc(h),opt);
        v(h,:)=results.bus(:,8);
        ph(h,:)=deg2rad(results.bus(:,9));
        plh(h)=sum(results.branch(:,14))+sum(results.branch(:,16));
        qlh(h)=sum(results.branch(:,15))+sum(results.branch(:,17));
        pg(h)=results.gen(1,2);
        qg(h)=results.gen(1,3);
        t(h)=results.et;
    end
    
    