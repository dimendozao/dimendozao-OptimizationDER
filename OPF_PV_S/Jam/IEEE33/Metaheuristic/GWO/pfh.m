function [v,ph,eqp,eqq,plh,qlh,pg,qg,probi,probd,t]=pfh(mpc,bestfitsi,bestfitsd,iparams,dparams,dists,ic,dc)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    H=numel(mpc);
    C=size(dparams,1);
    NN=numel(mpc(1).bus(:,1));
    L=numel(mpc(1).branch(:,1));
    v=zeros(H,NN);
    ph=zeros(H,NN);
    pg=zeros(H,1);
    qg=zeros(H,1);
    plh=zeros(H,1);
    qlh=zeros(H,1);
    probi=zeros(H,1);
    probd=zeros(H,C);
    eqp=zeros(H,NN);
    eqq=zeros(H,NN);
    ym=zeros(NN,NN);
    mpc1=mpc;
    for k=1:L
        ym(mpc1(1).branch(k,1),mpc1(1).branch(k,2))=-1/complex(mpc1(1).branch(k,3),mpc1(1).branch(k,4));
        ym(mpc1(1).branch(k,2),mpc1(1).branch(k,1))=-1/complex(mpc1(1).branch(k,3),mpc1(1).branch(k,4));
    end
    for i=1:NN
        ym(i,i)=-sum(ym(i,:));
    end
    t=zeros(H,1);
    for h=1:H
        results=runpf(mpc1(h),opt);
        v(h,:)=results.bus(:,8);
        ph(h,:)=deg2rad(results.bus(:,9));
        plh(h)=sum(results.branch(:,14))+sum(results.branch(:,16));
        qlh(h)=sum(results.branch(:,15))+sum(results.branch(:,17));
        pg(h)=results.gen(1,2);
        qg(h)=results.gen(1,3);
        t(h)=results.et;
        for i=1:NN
            for j=1:NN
                eqp(h,i)=eqp(h,i)+(v(h,i)*v(h,j)*abs(ym(i,j))*cos(ph(h,i)-ph(h,j)-angle(ym(i,j))));
                eqq(h,i)=eqq(h,i)+(v(h,i)*v(h,j)*abs(ym(i,j))*sin(ph(h,i)-ph(h,j)-angle(ym(i,j))));
            end
        end
        if bestfitsi(h)~=10
            dist=bestfitsi(h);               
            if dist==1
                probi(h)=cdf(dists{dist},ic(h),1/iparams(h,(2*dist)-1));
            elseif dist==2
                probi(h)=cdf(dists{dist},ic(h),log(iparams(h,(2*dist)-1)),1/iparams(h,2*dist)); 
            elseif dist==6
                probi(h)=cdf(dists{dist},ic(h),iparams(h,(2*dist)-1));
            elseif dist==7
                probi(h)=cdf(dists{dist},ic(h),1/iparams(h,(2*dist)-1),iparams(h,2*dist)); 
            elseif dist~=10
                probi(h)=cdf(dists{dist},ic(h),iparams(h,(2*dist)-1),iparams(h,2*dist));            
            end            
        else
            probi(h)=0;
        end        
        for c=1:C
            dist=bestfitsd(h,c);
            if dist==1
                probd(h,c)=cdf(dists{dist},dc(h,c),1/dparams(c,h,(2*dist)-1));
            elseif dist==2
                probd(h,c)=cdf(dists{dist},dc(h,c),log(dparams(c,h,(2*dist)-1)),1/dparams(c,h,2*dist)); 
            elseif dist==6
                probd(h,c)=cdf(dists{dist},dc(h,c),dparams(c,h,(2*dist)-1));
            elseif dist==7
                probd(h,c)=cdf(dists{dist},dc(h,c),1/dparams(c,h,(2*dist)-1),dparams(c,h,2*dist));
            else 
                probd(h,c)=cdf(dists{dist},dc(h,c),dparams(c,h,(2*dist)-1),dparams(c,h,2*dist));            
            end
        end
    end


    
    

    

    
    