function mpc1=mod_mpc_dg(mpc,x,pvc,rad,pveff,idem)
    Ndg=numel(x)/2;
    x1=x;
    x1(Ndg+1:end)=(x1(Ndg+1:end)*pvc)*rad*pveff;
    mpc1=mpc;
    mpc1.bus(:,3)=mpc1.bus(:,3).*idem;
    mpc1.bus(:,4)=mpc1.bus(:,4).*idem;
    for i=1:Ndg
        mpc1.bus(round(x(i)),3)=mpc1.bus(round(x(i)),3)-x1(i+Ndg);
    end
    
        