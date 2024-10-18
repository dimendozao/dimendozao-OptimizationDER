clc
clear
clf

city={'Bog','Jam','Pop'};
city1={'BOG','JAM','POP'};
problem='OPF_PV_S';

ncity=numel(city);
all_marks = {'*','x','s','d','o'};
colors={'k','r','c','b','m'};

for ct=1:ncity
    c1 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\','ParamTableC1.csv'));
    c2 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\','ParamTableC2.csv'));
    c3 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\','ParamTableC3.csv'));
    c4 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\','ParamTableC4.csv'));
    c5 = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\','ParamTableC5.csv'));
    
    irr = readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city{ct},'\','ParamTable.csv'));
    
    load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\NSTparamDem_',city1{ct},'.mat'));
    dparams=params;
    
    load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city{ct},'\NSTparamRAD_',city1{ct},'.mat'));
    iparams=params;
    
    load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Radiacion\',city{ct},'\PWLxyRAD_',city1{ct},'.mat'));
    
    xirrpwl=xpwl;
    yirrpwl=ypwl;
    
    load(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Data\Demanda\',city{ct},'\PWLxyDem_',city1{ct},'.mat'));
    
    xdempwl=xpwl;
    ydempwl=ypwl;
    
    adists={'Exponential','Fisk','Logistic','Log-Normal','Normal','Rayleigh','Weibull'};
    adists1={'Exponential','LogLogistic','Logistic','LogNormal','Normal','Rayleigh','Weibull'};
    nddparams=size(c1,1);
    
    
    ndists=numel(adists);
    ncluster=size(dparams,1);
    H=size(dparams,2);
    
    cparameters=[c1;c2;c3;c4;c5];
    
    bestfitsd=zeros(H,ncluster);   
    
    bestfitsi=10*ones(H,1);
     
    ihours=iparams(:,1)~=0;
    nihours=sum(ihours);
    for h=1:H
        if ihours(h)~=0
            for j=1:ndists
                if startsWith(irr.bestparams1((2*(h-7))-1),adists{j})
                   bestfitsi(h)=j;
                end
            end        
        end
        for i=1:ncluster
            for j=1:ndists
                if startsWith(cparameters.bestparams1(((i-1)*48)+((2*h)-1)),adists{j})
                    bestfitsd(h,i)=j;
                end
            end
        end
    end

    mindc=zeros(ncluster,H);
    maxdc=zeros(ncluster,H);
    
    minic=zeros(1,H);
    maxic=zeros(1,H);    
    
    for c=1:ncluster
        for h=1:H
            dist=bestfitsd(h,c);
            mindc(c,h)=min(xdempwl(h,c,dist,:));
            maxdc(c,h)=max(xdempwl(h,c,dist,:));
        end
    end
    
    for h=1:H
        dist=bestfitsi(h);
        if dist~=10
            minic(h)=min(xirrpwl(h,dist,:));
            maxic(h)=max(xirrpwl(h,dist,:)); 
        else
            minic(h)=min(xirrpwl(h,1,:));
            maxic(h)=max(xirrpwl(h,1,:));
        end
    end
%% 

    Tic=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_ic.csv'),'VariableNamingRule','preserve');
    Tpic=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_p_ic.csv'),'VariableNamingRule','preserve');

    Tdc1=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_dc1.csv'),'VariableNamingRule','preserve');
    Tpdc1=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_p_dc1.csv'),'VariableNamingRule','preserve');

    Tdc2=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_dc2.csv'),'VariableNamingRule','preserve');
    Tpdc2=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_p_dc2.csv'),'VariableNamingRule','preserve');

    Tdc3=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_dc3.csv'),'VariableNamingRule','preserve');
    Tpdc3=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_p_dc3.csv'),'VariableNamingRule','preserve');

    Tdc4=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_dc4.csv'),'VariableNamingRule','preserve');
    Tpdc4=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_p_dc4.csv'),'VariableNamingRule','preserve');

    Tdc5=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_dc5.csv'),'VariableNamingRule','preserve');
    Tpdc5=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\best_p_dc5.csv'),'VariableNamingRule','preserve');
%%    
    nvar=7;
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tic,2)-1
        if i==1
            plot(Tic{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tic{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([min(minic)-0.1 max(maxic)+0.05])
    fill([1:H,fliplr(1:H)],[minic,fliplr(maxic)],'r',FaceAlpha=0.1,EdgeColor='none')
    grid on
    fontsize(gca, 17,'points')
    legends={Tic.Properties.VariableNames{2:end}};    

    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$I_{pv}$, ','');
    end
    legends{1}='Mean';
    legends{end+1}='Max-Min';

    legend(legends,'Location','northwest','Orientation','vertical','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Coefficient $I_{pv}$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_IC');
    saveas(gcf,file,'epsc')
%% 

    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tpic,2)-1
        if i==1
            plot(Tpic{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tpic{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([-0.1 1.1])
    grid on
    fontsize(gca, 17,'points')
    legends={Tpic.Properties.VariableNames{2:end}};
    legends{1}='Mean';
    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$p(I_{pv})$, ','');
    end

    legend(legends,'Location','northwest','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Probability $p(I_{pv})$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_P_IC');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tdc1,2)-1
        if i==1
            plot(Tdc1{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tdc1{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    fill([1:H,fliplr(1:H)],[mindc(1,:),fliplr(maxdc(1,:))],'r',FaceAlpha=0.1,EdgeColor='none')
    grid on
    fontsize(gca, 17,'points')
    legends={Tdc1.Properties.VariableNames{2:end}};

    xlim([0 25])
    ylim([min(mindc(1,:))-0.3 max(maxdc(1,:))+0.05])

    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$DC_{1}$, ','');
    end
    legends{1}='Mean';
    legends{end+1}='Max-Min';

    legend(legends,'Location','southeast','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_DC1');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tdc2,2)-1
        if i==1
            plot(Tdc2{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tdc2{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([min(mindc(2,:))-0.3 max(maxdc(2,:))+0.05])
    fill([1:H,fliplr(1:H)],[mindc(2,:),fliplr(maxdc(2,:))],'r',FaceAlpha=0.1,EdgeColor='none')
    grid on
    fontsize(gca, 17,'points')
    legends={Tdc2.Properties.VariableNames{2:end}};    

    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$DC_{2}$, ','');
    end
    legends{1}='Mean';
    legends{end+1}='Max-Min';

    legend(legends,'Location','south','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_DC2');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tdc3,2)-1
        if i==1
            plot(Tdc3{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tdc3{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([min(mindc(3,:))-0.3 max(maxdc(3,:))+0.05])
    fill([1:H,fliplr(1:H)],[mindc(3,:),fliplr(maxdc(3,:))],'r',FaceAlpha=0.1,EdgeColor='none')
    grid on
    fontsize(gca, 17,'points')
    legends={Tdc3.Properties.VariableNames{2:end}};    

    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$DC_{3}$, ','');
    end
    legends{1}='Mean';
    legends{end+1}='Max-Min';

    legend(legends,'Location','southeast','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_DC3');
    saveas(gcf,file,'epsc')

%% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tdc4,2)-1
        if i==1
            plot(Tdc4{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tdc4{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([min(mindc(4,:))-0.3 max(maxdc(4,:))+0.05])
    fill([1:H,fliplr(1:H)],[mindc(4,:),fliplr(maxdc(4,:))],'r',FaceAlpha=0.1,EdgeColor='none')
    grid on
    fontsize(gca, 17,'points')
    legends={Tdc4.Properties.VariableNames{2:end}};    

    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$DC_{4}$, ','');
    end
    legends{1}='Mean';
    legends{end+1}='Max-Min';

    legend(legends,'Location','southeast','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_DC4');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tdc5,2)-1
        if i==1
            plot(Tdc5{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else
            plot(Tdc5{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([min(mindc(5,:))-0.3 max(maxdc(5,:))+0.05])
    fill([1:H,fliplr(1:H)],[mindc(5,:),fliplr(maxdc(5,:))],'r',FaceAlpha=0.1,EdgeColor='none')
    grid on
    fontsize(gca, 17,'points')
    legends={Tdc5.Properties.VariableNames{2:end}};    

    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$DC_{5}$, ','');
    end
    legends{1}='Mean';
    legends{end+1}='Max-Min';

    legend(legends,'Location','southeast','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_DC5');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
        
    for i=1:size(Tpdc1,2)-1
        if i==1
            plot(Tpdc1{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else           
            plot(Tpdc1{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([-0.1 1.1])
    grid on
    fontsize(gca, 17,'points')
    legends={Tpdc1.Properties.VariableNames{2:end}};
    
    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$p(DC_{1})$, ','');
    end
    legends{1}='Mean';

    legend(legends,'Location','southwest','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_P_DC1');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tpdc2,2)-1
        if i==1
            plot(Tpdc2{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else           
            plot(Tpdc2{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([-0.1 1.1])
    grid on
    fontsize(gca, 17,'points')
    legends={Tpdc2.Properties.VariableNames{2:end}};
    
    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$p(DC_{2})$, ','');
    end
    legends{1}='Mean';

    legend(legends,'Location','southwest','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_P_DC2');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tpdc3,2)-1
        if i==1
            plot(Tpdc3{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else           
            plot(Tpdc3{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([-0.1 1.1])
    grid on
    fontsize(gca, 17,'points')
    legends={Tpdc3.Properties.VariableNames{2:end}};
    
    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$p(DC_{3})$, ','');
    end
    legends{1}='Mean';

    legend(legends,'Location','southwest','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_P_DC3');
    saveas(gcf,file,'epsc')

%% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tpdc4,2)-1
        if i==1
            plot(Tpdc4{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else           
            plot(Tpdc4{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([-0.1 1.1])
    grid on
    fontsize(gca, 17,'points')
    legends={Tpdc4.Properties.VariableNames{2:end}};
    
    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$p(DC_{4})$, ','');
    end
    legends{1}='Mean';

    legend(legends,'Location','southwest','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_P_DC4');
    saveas(gcf,file,'epsc')

    %% 
    figures=figure('visible','off');
    figures.Position=[50,50,900,700];
    
    for i=1:size(Tpdc5,2)-1
        if i==1
            plot(Tpdc5{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})
            hold on
        else           
            plot(Tpdc5{:,i+1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{i})            
        end
    end
    xlim([0 25])
    ylim([-0.1 1.1])
    grid on
    fontsize(gca, 17,'points')
    legends={Tpdc5.Properties.VariableNames{2:end}};
    
    for i=1:numel(legends)
        legends{i}=regexprep(legends{i}, '''', '');
        legends{i}=strrep(legends{i},'$p(DC_{5})$, ','');
    end
    legends{1}='Mean';

    legend(legends,'Location','southwest','FontSize',18,'Interpreter','Latex');
    xlabel('Hour','FontSize',20,'Interpreter','latex'); 
    ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

    file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\',city{ct},'\Best_P_DC5');
    saveas(gcf,file,'epsc')
end