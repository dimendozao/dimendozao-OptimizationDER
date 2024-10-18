function sl=fo(mpc,ae,pvc,rad,pveff,idem,x)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    npv=numel(x)/2;
    x1=x;
    x1(npv+1:end)=(x1(npv+1:end)*pvc)*rad*pveff;
    mpc1=mpc;
    mpc1.bus(:,3)=mpc1.bus(:,3).*idem;
    mpc1.bus(:,4)=mpc1.bus(:,4).*idem;
    for i=1:npv
        mpc1.bus(round(x1(i)),3)=mpc1.bus(round(x1(i)),3)-x1(i+npv);
    end
    results=runpf(mpc1,opt);
    v=results.bus(:,8);
    pl=sum(results.branch(:,14)+results.branch(:,16));
    ql=sum(results.branch(:,15)+results.branch(:,17));
    pg=results.gen(1,2);
    qg=results.gen(1,3);
    sl=pl+ql;
    npv=numel(x)/2;
    if (ae==0 && (pg<0 || qg<0)) 
     sl=sl+1;
    end
    if sum(v<mpc1.bus(:,13))>0
     sl=sl+sum(v<mpc1.bus(:,13));
    end
    if sum(v>mpc1.bus(:,12))>0
      sl=sl+sum(v>mpc1.bus(:,12));
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