function mpc1=mod_mpc_dg(mpc,x,pvc,pveff,imeans,cnode,dmeans)
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
    end
    
        