function sl=fo(mpc,ae,pvc,pveff,x,bestfitsi,bestfitsd,iparams,dparams,clusternode,dists)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    H=numel(bestfitsi);
    C=size(dparams,1);
    N=numel(clusternode);
    nih=sum(bestfitsi~=10);    
    nvar=numel(x);
    npv=(nvar-(H*C)-nih)/2;
    zpv=x(1:npv);
    pvcap=x(npv+1:2*npv).*pvc;
    ic=x((2*npv)+1:(2*npv)+nih);
    dc=x((2*npv)+nih+1:end);
    mpc1=mpc;
    sl=0;
    k=1;
    for h=1:H
        for i=2:N
            cnode=clusternode(i);
            mpc1(h).bus(i,3)=mpc1(h).bus(i,3)*dc((24*(cnode-1))+h);
            mpc1(h).bus(i,4)=mpc1(h).bus(i,4)*dc((24*(cnode-1))+h);
        end
        if bestfitsi(h)~=10
            for i=1:npv            
                mpc1(h).bus(round(zpv(i)),3)=mpc1(h).bus(round(zpv(i)),3)-(pvcap(i)*ic(k)*pveff);
            end
        end
        results=runpf(mpc1(h),opt);
        v=results.bus(:,8);
        pl=sum(results.branch(:,14)+results.branch(:,16));
        ql=sum(results.branch(:,15)+results.branch(:,17));
        pg=results.gen(1,2);
        qg=results.gen(1,3);
        if bestfitsi(h)~=10
            dist=bestfitsi(h);               
            if dist==1
                probi=cdf(dists{dist},ic(k),1/iparams(h,(2*dist)-1));
            elseif dist==2
                probi=cdf(dists{dist},ic(k),log(iparams(h,(2*dist)-1)),1/iparams(h,2*dist)); 
            elseif dist==6
                probi=cdf(dists{dist},ic(k),iparams(h,(2*dist)-1));
            elseif dist==7
                probi=cdf(dists{dist},ic(k),1/iparams(h,(2*dist)-1),iparams(h,2*dist)); 
            elseif dist~=10
                probi=cdf(dists{dist},ic(k),iparams(h,(2*dist)-1),iparams(h,2*dist));            
            end
            k=k+1;
        else
            probi=0;
        end
        probd=zeros(C,1);        
        for c=1:C
            dist=bestfitsd(h,c);
            if dist==1
                probd(c)=cdf(dists{dist},dc(((c-1)*24)+h),1/dparams(c,h,(2*dist)-1));
            elseif dist==2
                probd(c)=cdf(dists{dist},dc(((c-1)*24)+h),log(dparams(c,h,(2*dist)-1)),1/dparams(c,h,2*dist));
            elseif dist==6
                probd(c)=cdf(dists{dist},dc(((c-1)*24)+h),dparams(c,h,(2*dist)-1));
            elseif dist==7
                probd(c)=cdf(dists{dist},dc(((c-1)*24)+h),1/dparams(c,h,(2*dist)-1),dparams(c,h,2*dist));
            else 
                probd(c)=cdf(dists{dist},dc(((c-1)*24)+h),dparams(c,h,(2*dist)-1),dparams(c,h,2*dist));            
            end
        end
        sl=sl+pl+ql-sum(probd)+probi;        
        if (ae==0 && (pg<0 || qg<0)) 
         sl=sl+10;
        end
        if sum(v<mpc1(h).bus(:,13))>0
         sl=sl+sum(v<mpc1(h).bus(:,13));
        end
        if sum(v>mpc1(h).bus(:,12))>0
          sl=sl+sum(v>mpc1(h).bus(:,12));
        end
    end
    if sum(x(npv+1:2*npv))>1
        sl=sl+(sum(x(npv+1:2*npv))-1)*10;
    end
    if sum(x(npv+1:2*npv))<1
        sl=sl-(sum(x(npv+1:2*npv))+1)*10;
    end
    if numel(unique(round(zpv)))<npv
        sl=sl+numel(unique(round(zpv)))*10;
    end
    