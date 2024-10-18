%% 

clc
clear
clf

city={'Bog','Jam','Pop'};
city1={'BOG','JAM','POP'};
casen={'IEEE33','IEEE69','SA','CA141'};
problem='OPF_BESS_IS';
prob_sol='OPF_PV_S';

ncity=numel(city);
ncase=numel(casen);

for ct=1:ncity
    for ca=1:ncase    
        load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city{ct},'\MeansRAD_',city1{ct},'.mat'));
        load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\ClusterMeans_',city1{ct},'.mat'));    
        load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\',casen{ca},'_',city{ct},'_ClusterNode.mat'));
        
        clusternode(1)=1;        
        ncluster=size(clustermeans,1);
        H=size(clustermeans,2);
        
        ppvtab=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',prob_sol,'\',city{ct},'\',casen{ca},'\bestsol.csv'));
        
        ic=ppvtab.ic;
        dc=zeros(ncluster,H);
        
        dc(1,:)=ppvtab.dc1;
        dc(2,:)=ppvtab.dc2;
        dc(3,:)=ppvtab.dc3;
        dc(4,:)=ppvtab.dc4;
        dc(5,:)=ppvtab.dc5;
        
        pveff=0.8;

        besstab=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\',casen{ca},'\bestsol.csv'));

        if strcmp(casen{ca},'IEEE33')
            mpc=loadcase(case33_DM);
        elseif strcmp(casen{ca},'IEEE69')
            mpc=loadcase(case69_DM);
        elseif strcmp(casen{ca},'CA141')
            mpc=loadcase(case141);
            mpc=order_radial(mpc); 
        elseif strcmp(casen{ca},'SA')
            mpc=loadcase(caseSA_J23);
        end

        N=size(mpc.bus,1);

        pvc=zeros(N,1);
        for i=1:N
            pvc(i)=ppvtab{1,N+i+1};
        end

        pd=zeros(H,1);        
        pv=zeros(H,1);
        pbc=zeros(H,1);
        pbd=zeros(H,1);

        locs=cell(1+H,1);
        locs{1}='';       
        
        for h=1:H
            locs{h+1}=strcat('(',num2str(besstab.ZB(h)),')');
            for i=1:N
                pd(h)=pd(h)-(mpc.bus(i,3)*dc(clusternode(i),h));
                pv(h)=pv(h)+(pvc(i)*pveff*ic(h));
            end
            if besstab.PB(h)<=0
                pbc(h)=besstab.PB(h);
            else
                pbd(h)=besstab.PB(h);
            end
        end
        
        figures=figure('visible','off');
        figures.Position=[50,50,1800,700];
        
        subplot(1,2,1);
        bar(1:24,[pd,pv,pbc,pbd],'stacked')
        grid on
        fontsize(gca, 17,'points')        
        legends={'$P^{d}$','$PPV$','$PB^{-}$','$PB^{+}$'};
        legend(legends,'Location','northoutside','Orientation','horizontal','FontSize',18,'Interpreter','Latex');        
        xlabel('Hour','FontSize',20,'Interpreter','latex'); 
        ylabel('Energy [$MWh$]','FontSize',20,'Interpreter','latex');
        subplot(1,2,2);
        soc=zeros(H+1,1);
        soc(1)=besstab.SOC0(1);
        soc(2:end)=besstab.SOC;
        bar(0:24,soc);
        grid on
        fontsize(gca, 17,'points')                
        xlabel('Hour','FontSize',20,'Interpreter','latex'); 
        ylabel('SOC','FontSize',20,'Interpreter','latex');        
        text(0:24, soc, locs, 'HorizontalAlignment','center', 'VerticalAlignment','bottom','FontSize',12)
        ylim([0 1.05]);
        
        file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\',casen{ca},'\','Profiles');
        saveas(gcf,file,'epsc')   
    end
end