function mpc1=mod_mpc_dg(mpc,x,pvc,pveff,clusternode,ih)
    H=numel(mpc);
    N=numel(clusternode);
    C=max(clusternode);
    nih=sum(ih);
    nvar=numel(x);
    npv=(nvar-(H*C)-nih)/2;
    zpv=x(1:npv);
    pvcap=x(npv+1:2*npv).*pvc;
    ic=x((2*npv)+1:(2*npv)+nih);
    dc=x((2*npv)+nih+1:end);
    mpc1=mpc;
    k=1;
    for h=1:H
        for i=2:N
            cnode=clusternode(i);
            mpc1(h).bus(i,3)=mpc1(h).bus(i,3)*dc((24*(cnode-1))+h);
            mpc1(h).bus(i,4)=mpc1(h).bus(i,4)*dc((24*(cnode-1))+h);
        end
        if ih(h)~=0
            for i=1:npv
                mpc1(h).bus(round(zpv(i)),3)=mpc1(h).bus(round(zpv(i)),3)-(pvcap(i)*ic(k)*pveff);                
            end
            k=k+1;
        end        
    end
    
        