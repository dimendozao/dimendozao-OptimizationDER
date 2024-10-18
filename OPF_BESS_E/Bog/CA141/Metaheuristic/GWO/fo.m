function sl=fo(mpc,ae,pvc,imeans,pveff,x,cnode,dmeans)
    opt = mpoption('pf.alg','NR','verbose',0, 'out.all',0); % set option of MATPOWER
    H=numel(imeans);
    N=size(mpc(1).bus,1);    
    ppv=zeros(H,N);
    mpc1=mpc;
    demcoeff=zeros(H,N);
    pbh=zeros(H,N);
    
    for h=1:H        
        ppv(h,:)=pvc.*imeans(h)*pveff;        
        for i=1:N
            demcoeff(h,i)=dmeans(cnode(i),h);
        end
        if h==1
            if (x(H+2)-x(H+2+h))<=0
                pbh(h,round(x(h)))=((x(H+2)-x(H+2+h))*x(H+1))/((3/2)-(0.5*sqrt(1-(x(H+2)-x(H+2+h)))));
            else
                pbh(h,round(x(h)))=((x(H+2)-x(H+2+h))*x(H+1))*((1/2)+(0.5*sqrt(1-(x(H+2)-x(H+2+h)))));
            end
        elseif h<H
            if (x(H+1+h)-x(H+2+h))<=0
                pbh(h,round(x(h)))=((x(H+1+h)-x(H+2+h))*x(H+1))/((3/2)-(0.5*sqrt(1-(x(H+1+h)-x(H+2+h)))));
            else
                pbh(h,round(x(h)))=((x(H+1+h)-x(H+2+h))*x(H+1))*((1/2)+(0.5*sqrt(1-(x(H+1+h)-x(H+2+h)))));
            end
        else
            if (x(H+1+h)-x(H+2))<=0
                pbh(h,round(x(h)))=((x(H+1+h)-x(H+2))*x(H+1))/((3/2)-(0.5*sqrt(1-(x(H+1+h)-x(H+2)))));
            else
                pbh(h,round(x(h)))=((x(H+1+h)-x(H+2))*x(H+1))*((1/2)+(0.5*sqrt(1-(x(H+1+h)-x(H+2)))));
            end
        end        
        mpc1(h).bus(:,3)=(mpc1(h).bus(:,3).*demcoeff(h,:)')-ppv(h,:)'-pbh(h,:)';
        mpc1(h).bus(:,4)=(mpc1(h).bus(:,4).*demcoeff(h,:)');
        results=runpf(mpc1(h),opt);
        v=results.bus(:,8);
        pl=sum(results.branch(:,14)+results.branch(:,16));
        ql=sum(results.branch(:,15)+results.branch(:,17));
        pg=results.gen(1,2);
        qg=results.gen(1,3);
        sl=pl+ql;        
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
        
    
    
    