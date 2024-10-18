function mpc1=mod_mpc_dg(mpc,x,pvc,pveff,imeans,cnode,dmeans)
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
    end
    
        