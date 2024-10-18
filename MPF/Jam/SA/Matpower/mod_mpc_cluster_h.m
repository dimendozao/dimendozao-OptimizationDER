function mpc1=mod_mpc_cluster_h(mpc,cluster,nodes)
    H=numel(mpc);
    N=numel(nodes);
    mpc1=mpc;
    for h=1:H
        for i=1:N
            if nodes(i)~=0
                mpc1(h).bus(i,3)=mpc1(h).bus(i,3)*cluster(nodes(i),h);
                mpc1(h).bus(i,4)=mpc1(h).bus(i,4)*cluster(nodes(i),h);
            end
        end        
    end