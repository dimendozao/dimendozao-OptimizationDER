function sl=fo(mpc,ae,pvc,imeans,pveff,x,cnode,dmeans)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    H=numel(imeans);
    N=size(mpc(1).bus,1);
    npv=numel(x)/2;
    ppv=zeros(H,N);
    mpc1=mpc;
    demcoeff=zeros(H,N);    
    for h=1:H        
        for i=1:npv
            ppv(h,round(x(i)))=x(npv+i)*imeans(h)*pveff*pvc;
        end
        for i=1:N
            demcoeff(h,i)=dmeans(cnode(i),h);
        end
        mpc1(h).bus(:,3)=(mpc1(h).bus(:,3).*demcoeff(h,:)')-ppv(h,:)';
        mpc1(h).bus(:,4)=(mpc1(h).bus(:,4).*demcoeff(h,:)');
        results=runpf(mpc1(h),opt);
        v=results.bus(:,8);
        pl=sum(results.branch(:,14)+results.branch(:,16));
        ql=sum(results.branch(:,15)+results.branch(:,17));
        pg=results.gen(1,2);
        qg=results.gen(1,3);
        sl=pl+ql;        
        if (ae==0 && (pg<0 || qg<0)) 
         sl=sl+1;
        end
        if sum(v<mpc1(h).bus(:,13))>0
         sl=sl+sum(v<mpc1(h).bus(:,13));
        end
        if sum(v>mpc1(h).bus(:,12))>0
          sl=sl+sum(v>mpc1(h).bus(:,12));
        end
        if sum(x(npv+1:end))>1
            sl=sl+sum(x(npv+1:end))-1;
        end
        if sum(x(npv+1:end))<1
            sl=sl-sum(x(npv+1:end))+1;
        end
        if numel(unique(round(x(1:npv))))<npv
            sl=sl+numel(unique(round(x(1:npv))));
        end
    end
        
    
    
    