% a1T10:此程序的目的是画出不同的惯性权重减速下降曲线图（将公式进行了修改）（此程序与a1T9相同）

%% 初始化
clear ;
close all;
clc;
% warning off

%% 开始计算时间点0
tStart0 = tic;

%% 设置参数变量并且赋值
a = 1.06;                                                        % 常量 a 为底数,通常情况下，a大于0   svmpredict   fitcsvm
b = 20;

x = 1:1:50;                                                      % x 为指数的值

wmax = 0.9;
wmin = 0.4;
T = 200;                                                       % 最大的迭代次数
ti = 1:1:T;                                                     % 每次的迭代次数

k = ti / T;

u1 = k .^ 2;                                                     % y 为公式计算得到的值 log
u2 = k .^ 3;
u3 = k .^ 4;
u4 = k .^ 5;

y0 = wmax - k .* (wmax - wmin);

y11 = wmax - u1 .* (wmax - wmin);

y12 = wmax - u2 .* (wmax - wmin);

y13 = wmax - u3 .* (wmax - wmin);

y14 = wmax - u4 .* (wmax - wmin);

v1 = (T / (4 + T)) .^ (ti);
v2 = (T / (6 + T)) .^ (ti);
v3 = (T / (8 + T)) .^ (ti);
v4 = (T / (10 + T)) .^ (ti);

y21 = wmin + v1 .* (wmax - wmin);
y22 = wmin + v2 .* (wmax - wmin);
y23 = wmin + v3 .* (wmax - wmin);
y24 = wmin + v4 .* (wmax - wmin);



%% 画出加速惯性权重因子变化的图形

%{
f1 = figure;
f1.Name  = '惯性权重加速下降曲线';                        % 图像的标题名称
hold on;
% f1p0 = plot(ti,y0,'-r','Linewidth',1.8);                           % 画出线条L1

f1p11 = plot(ti,y11);                                                % 画出线条L1
f1p11.LineStyle = '-';
f1p11.Color = 'blue';
f1p11.LineWidth = 3;

f1p12 = plot(ti,y12);                           % 画出线条L2
f1p12.LineStyle = '--';
f1p12.Color = 'magenta';
f1p12.LineWidth = 3;

f1p13 = plot(ti,y13);                           % 画出线条L3
f1p13.LineStyle = ':';
f1p13.Color = 'black';
f1p13.LineWidth = 3;

f1p14 = plot(ti,y14);                           % 画出线条L4
f1p14.LineStyle = '-.';
f1p14.Color = 'red';
f1p14.LineWidth = 3;

axis padded;    

set(gca,'FontName','Times New Roman','FontSize',20);

xla1 = xlabel('迭代次数');
xla1.FontSize = 20;
xla1.FontWeight = 'bold';
xla1.FontName = '仿宋';
xla1.Color = 'black';

yla1 = ylabel('权值大小');
yla1.FontSize = 20;
yla1.FontWeight = 'bold';
yla1.FontName = '仿宋';
yla1.Color = 'black';

leg1 = legend('m=2','m=3','m=4','m=5');
leg1.FontSize = 20;
leg1.TextColor = 'black';
leg1.Box = 'off';
leg1.FontName = 'Times New Roman';
leg1.FontWeight = 'bold';
leg1.Position = [0.73274 0.65817 0.17393 0.27214];

set(gca,'xtick',[0 40 80 120 160 200]); 
set(gca,'ytick',[0.4 0.5 0.6 0.7 0.8 0.9 ]); 

% set(gca,'FontName','Times New Roman','FontSize',14)  

% set(gca,'FontName','Times New Roman','FontSize',8,'LineWidth',2);

%}

%% 画出减速惯性权重因子变化的图形

f2 = figure;
f2.Name  = '惯性权重减速下降曲线';                        % 图像的标题名称
hold on;
% f2p0 = plot(ti,y0,'-r','Linewidth',1.8);                             % 画出线条L1

f2p11 = plot(ti,y21);                                               % 画出线条L1
f2p11.LineStyle = '-';
f2p11.Color = 'blue';
f2p11.LineWidth = 3;

f2p12 = plot(ti,y22,'-m','Linewidth',1.8);                           % 画出线条L2
f2p12.LineStyle = '--';
f2p12.Color = 'magenta';
f2p12.LineWidth = 3;

f2p13 = plot(ti,y23,'-k','Linewidth',1.8);                           % 画出线条L3
f2p13.LineStyle = ':';
f2p13.Color = 'black';
f2p13.LineWidth = 3;

f2p14 = plot(ti,y24,'-c','Linewidth',1.8);                           % 画出线条L4
f2p14.LineStyle = '-.';
f2p14.Color = 'red';
f2p14.LineWidth = 3;

axis padded;   
box on;

set(gca,'FontName','Times New Roman','FontSize',20);

xla2 = xlabel('迭代次数');
xla2.FontSize = 20;
xla2.FontWeight = 'bold';
xla2.FontName = '仿宋';
xla2.Color = 'black';

yla2 = ylabel('惯性权重');
yla2.FontSize = 20;
yla2.FontWeight = 'bold';
yla2.FontName = '仿宋';
yla2.Color = 'black';

leg2 = legend('n=4','n=6','n=8','n=10');
leg2.FontSize = 20;
leg2.TextColor = 'black';
leg2.Box = 'off';
leg2.FontName = 'Times New Roman';
leg2.FontWeight = 'bold';

axis padded;  

%% 设置结束计算时间点0                                                % text
tEnd0 = toc(tStart0); 

%% 计时结果输出
disp(' ');
disp('************************************************************');
disp('计时结果输出：');
fprintf('总共用时 ：%d 分 , %f 秒', floor(tEnd0/60), rem(tEnd0,60));                    % 程序的总用时间
disp(' ');
