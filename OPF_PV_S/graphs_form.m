clc
clear
clf
close

city={'Bog','Jam','Pop'};
city1={'BOG','JAM','POP'};
problem='OPF_PV_S';

ncity=numel(city);
all_marks = {'s','^','o'};
colors={'k','r','k','c','k','b'};
colorsmm={'r','b','g'};

maxsic=cell(ncity,1);
minsic=cell(ncity,1);
maxsdc=cell(ncity,1);
minsdc=cell(ncity,1);

Tic=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\ic_form_tab.csv'),'VariableNamingRule','preserve');
Tpic=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\pi_form_tab.csv'),'VariableNamingRule','preserve');

Tdc1=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\dc1_form_tab.csv'),'VariableNamingRule','preserve');
Tpdc1=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\pd1_form_tab.csv'),'VariableNamingRule','preserve');

Tdc2=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\dc2_form_tab.csv'),'VariableNamingRule','preserve');
Tpdc2=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\pd2_form_tab.csv'),'VariableNamingRule','preserve');

Tdc3=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\dc3_form_tab.csv'),'VariableNamingRule','preserve');
Tpdc3=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\pd3_form_tab.csv'),'VariableNamingRule','preserve');

Tdc4=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\dc4_form_tab.csv'),'VariableNamingRule','preserve');
Tpdc4=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\pd4_form_tab.csv'),'VariableNamingRule','preserve');

Tdc5=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\dc5_form_tab.csv'),'VariableNamingRule','preserve');
Tpdc5=readtable(strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\pd5_form_tab.csv'),'VariableNamingRule','preserve');


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
    maxsic{ct}=maxic;
    minsic{ct}=minic;
    maxsdc{ct}=maxdc;    
    minsdc{ct}=mindc;
end

%%    
nvar=7;
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tic.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*3)-2}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*3)-1}=strcat(legends{(ct*2)-1},' Solution');
    legends1{ct*3}=strcat(legends{(ct*2)-1},' Max-Min');
end

for i=1:ncity
    plot(Tic{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tic{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
    fill([1:H,fliplr(1:H)],[minsic{i},fliplr(maxsic{i})],colorsmm{i},FaceAlpha=0.1,EdgeColor=colors{(i*2)},LineStyle="--")  
end
mii=1;
mai=0;
for ct=1:ncity
    mii=min(mii,min(minsic{ct}));
    mai=max(mai,max(maxsic{ct}));
end
xlim([0 25])
ylim([mii-0.1 mai+0.05])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Coefficient $I_{pv}$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\IC_form');
saveas(gcf,file,'epsc')
%% 

figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tpic.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*2)-1}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*2)}=strcat(legends{(ct*2)-1},' Solution');    
end

for i=1:ncity
    plot(Tpic{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tpic{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
end
xlim([0 25])
ylim([-0.1 0.7])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Probability $p(I_{pv})$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\P_IC_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tdc1.Properties.VariableNames;
nlegends=numel(legends);

legends1={};
for ct=1:ncity
    legends1{(ct*3)-2}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*3)-1}=strcat(legends{(ct*2)-1},' Solution');
    legends1{ct*3}=strcat(legends{(ct*2)-1},' Max-Min');
end

for i=1:ncity
    plot(Tdc1{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tdc1{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
    fill([1:H,fliplr(1:H)],[minsdc{i}(1,:),fliplr(maxsdc{i}(1,:))],colorsmm{i},FaceAlpha=0.1,EdgeColor=colors{(i*2)},LineStyle="--")
end
mii=5;
mai=0;
for ct=1:ncity
    mii=min(mii,min(minsdc{ct}(1,:)));
    mai=max(mai,max(maxsdc{ct}(1,:)));
end
xlim([0 25])
ylim([mii-0.1 mai+0.05])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\DC1_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tdc2.Properties.VariableNames;
nlegends=numel(legends);

legends1={};
for ct=1:ncity
    legends1{(ct*3)-2}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*3)-1}=strcat(legends{(ct*2)-1},' Solution');
    legends1{ct*3}=strcat(legends{(ct*2)-1},' Max-Min');
end

for i=1:ncity
    plot(Tdc2{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tdc2{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
    fill([1:H,fliplr(1:H)],[minsdc{i}(2,:),fliplr(maxsdc{i}(2,:))],colorsmm{i},FaceAlpha=0.1,EdgeColor=colors{(i*2)},LineStyle="--")
end
mii=5;
mai=0;
for ct=1:ncity
    mii=min(mii,min(minsdc{ct}(2,:)));
    mai=max(mai,max(maxsdc{ct}(2,:)));
end
xlim([0 25])
ylim([mii-0.1 mai+0.05])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\DC2_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tdc3.Properties.VariableNames;
nlegends=numel(legends);

legends1={};
for ct=1:ncity
    legends1{(ct*3)-2}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*3)-1}=strcat(legends{(ct*2)-1},' Solution');
    legends1{ct*3}=strcat(legends{(ct*2)-1},' Max-Min');
end

for i=1:ncity
    plot(Tdc3{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tdc3{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
    fill([1:H,fliplr(1:H)],[minsdc{i}(3,:),fliplr(maxsdc{i}(3,:))],colorsmm{i},FaceAlpha=0.1,EdgeColor=colors{(i*2)},LineStyle="--")
end
mii=5;
mai=0;
for ct=1:ncity
    mii=min(mii,min(minsdc{ct}(3,:)));
    mai=max(mai,max(maxsdc{ct}(3,:)));
end
xlim([0 25])
ylim([mii-0.1 mai+0.05])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\DC3_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tdc4.Properties.VariableNames;
nlegends=numel(legends);

legends1={};
for ct=1:ncity
    legends1{(ct*3)-2}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*3)-1}=strcat(legends{(ct*2)-1},' Solution');
    legends1{ct*3}=strcat(legends{(ct*2)-1},' Max-Min');
end

for i=1:ncity
    plot(Tdc4{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tdc4{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
    fill([1:H,fliplr(1:H)],[minsdc{i}(4,:),fliplr(maxsdc{i}(4,:))],colorsmm{i},FaceAlpha=0.1,EdgeColor=colors{(i*2)},LineStyle="--")
end
mii=5;
mai=0;
for ct=1:ncity
    mii=min(mii,min(minsdc{ct}(4,:)));
    mai=max(mai,max(maxsdc{ct}(4,:)));
end
xlim([0 25])
ylim([mii-0.1 mai+0.05])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\DC4_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tdc5.Properties.VariableNames;
nlegends=numel(legends);

legends1={};
for ct=1:ncity
    legends1{(ct*3)-2}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*3)-1}=strcat(legends{(ct*2)-1},' Solution');
    legends1{ct*3}=strcat(legends{(ct*2)-1},' Max-Min');
end

for i=1:ncity
    plot(Tdc5{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tdc5{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
    fill([1:H,fliplr(1:H)],[minsdc{i}(5,:),fliplr(maxsdc{i}(5,:))],colorsmm{i},FaceAlpha=0.1,EdgeColor=colors{(i*2)},LineStyle="--")
end
mii=5;
mai=0;
for ct=1:ncity
    mii=min(mii,min(minsdc{ct}(5,:)));
    mai=max(mai,max(maxsdc{ct}(5,:)));
end
xlim([0 25])
ylim([mii-0.1 mai+0.05])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Coefficient $DC$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\DC5_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tpdc1.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*2)-1}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*2)}=strcat(legends{(ct*2)-1},' Solution');    
end

for i=1:ncity
    plot(Tpdc1{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tpdc1{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
end
xlim([0 25])
ylim([-0.1 1.1])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\P_DC1_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tpdc2.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*2)-1}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*2)}=strcat(legends{(ct*2)-1},' Solution');    
end

for i=1:ncity
    plot(Tpdc2{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tpdc2{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
end
xlim([0 25])
ylim([-0.1 1.1])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\P_DC2_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tpdc3.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*2)-1}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*2)}=strcat(legends{(ct*2)-1},' Solution');    
end

for i=1:ncity
    plot(Tpdc3{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tpdc3{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
end
xlim([0 25])
ylim([-0.1 1.1])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\P_DC3_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tpdc4.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*2)-1}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*2)}=strcat(legends{(ct*2)-1},' Solution');    
end

for i=1:ncity
    plot(Tpdc4{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tpdc4{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
end
xlim([0 25])
ylim([-0.1 1.1])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\P_DC4_form');
saveas(gcf,file,'epsc')

%% 
figures=figure('visible','off');
figures.Position=[50,50,900,700];

legends=Tpdc5.Properties.VariableNames;
nlegends=numel(legends);
legends1={};
for ct=1:ncity
    legends1{(ct*2)-1}=strcat(legends{(ct*2)-1},' Mean');
    legends1{(ct*2)}=strcat(legends{(ct*2)-1},' Solution');    
end

for i=1:ncity
    plot(Tpdc5{:,(2*i)-1},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{(2*i)-1})
    if i==1        
        hold on
    end
    plot(Tpdc5{:,(2*i)},'LineWidth',1,'LineStyle','-','Marker',all_marks{i},'MarkerSize',10,'Color',colors{2*i})
end
xlim([0 25])
ylim([-0.1 1.1])
grid on
fontsize(gca, 17,'points')

legend(legends1,'Location','northoutside','Orientation','vertical','NumColumns',3,'FontSize',15,'Interpreter','Latex');
xlabel('Hour','FontSize',20,'Interpreter','latex'); 
ylabel('Probability $p(DC)$','FontSize',20,'Interpreter','latex');

file=strcat('C:\Users\diego\OneDrive\Desktop\aLL\PhD\Tesis\Python\',problem,'\P_DC5_form');
saveas(gcf,file,'epsc')
