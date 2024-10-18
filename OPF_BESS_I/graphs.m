%% 

clc
clear
clf

city={'Bog','Jam','Pop'};
city1={'BOG','JAM','POP'};
casen={'IEEE33','IEEE69','SA','CA141'};
problem='OPF_BESS_I';
prob_sol='OPF_PV_D';

ncity=numel(city);
ncase=numel(casen);

npwlb=12;
        
socx=zeros(npwlb,1);
socy=zeros(npwlb,1);

for i=1:npwlb
    if i==1
        socx(i)=-1;
        socy(i)=1/((3/2)-((1/2)*sqrt(1-socx(i))));
        socx(i+1)=0.3;
        socy(i+1)=(1/2)+((1/2)*sqrt(1-socx(i+1)));    
    elseif i>2 && i<=8
        socx(i)=0.3+((i-2)*((0.9-0.3)/6));
        socy(i)=((1/2)+((1/2)*sqrt(1-socx(i))));
    elseif i>=9
        socx(i)=0.9+((i-8)*((1-0.9)/4));
        socy(i)=((1/2)+((1/2)*sqrt(1-socx(i))));
    end
end

socx1=linspace(-1,1,1000);
socy1=zeros(1000,1);
for i=1:1000
    if socx1(i)<=0
        socy1(i)=1/((3/2)-(0.5*sqrt(1-socx1(i))));
    else
        socy1(i)=(1/2)+(0.5*sqrt(1-socx1(i)));
    end
end
        

figures=figure('visible','off');
figures.Position=[50,50,900,700];
plot(socx1,socy1,'LineWidth',1.5,"LineStyle","-.")
hold on
plot(socx,socy,'LineWidth',1.5,"LineStyle","--","Marker","o",'MarkerSize',10)
%scatter(socx,socy,200,'*')   
    
xlim([-1.1 1.1])
ylim([0.4 1.3])

grid on
fontsize(gca, 17,'points')

legends={'$P^{f}$','PWLA'};
legend(legends,'FontSize',18,'Interpreter','Latex');

xlabel('Rated Power Fraction $p^{f}$','FontSize',20,'Interpreter','latex'); 
ylabel('Effective Power Fraction $P^{f}$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\PWLA_SOC');
saveas(gcf,file,'epsc')




%% 

for ct=1:ncity
    for ca=1:ncase
    
        load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city{ct},'\MeansRAD_',city1{ct},'.mat'));
        load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\ClusterMeans_',city1{ct},'.mat'));    
        load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\',casen{ca},'_',city{ct},'_ClusterNode.mat'));
        
        clusternode(1)=1;        
        ncluster=size(clustermeans,1);
        H=size(clustermeans,2);
        
        ppvtab=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',prob_sol,'\',city{ct},'\',casen{ca},'\bestsol.csv'));
        
        pvc=ppvtab.ppv;
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

        pd=zeros(H,1);        
        pv=zeros(H,1);
        pbc=zeros(H,1);
        pbd=zeros(H,1);

        locs=cell(1+H,1);
        locs{1}='';       
        
        for h=1:H
            locs{h+1}=strcat('(',num2str(besstab.ZB(h)),')');
            for i=1:N
                pd(h)=pd(h)-(mpc.bus(i,3)*clustermeans(clusternode(i),h));
                pv(h)=pv(h)+(pvc(i)*pveff*means(h));
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