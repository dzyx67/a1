% a1T8:此程序的目的是画出已经训练好的模型的回归效果图

%% 初始化
clear ;
close all;
clc;

%% 程序开始运行标时的日期与时间输出
fprintf('************************************************************\n');

fprintf('程序运行开始\n');

t_time1 = datetime;
fprintf('程序开始运行时的日期与时间为：      %s\n',t_time1);

fprintf('************************************************************\n');

%% 开始计算时间点0
tStart0 = tic;

%% 读取数据文件
load('a1D3.mat','uz1','uz2','ra1','n_use','scaA','list_trainA','list_testA',...
    'input_trainA','input_testA','output_trainA','output_testA','output_trainB','output_testB',...
    'popsize','maxgen','c1','c2','uxv','xmaxA','wmax','wmin',...
    'net1','output_trainE1','output_testE1','output_trainF1','output_testF1','Best_posA','Convergence_curveA',...
    'net2','output_trainE2','output_testE2','output_trainF2','output_testF2','Best_posB','Convergence_curveB',...
    'net3','output_trainE3','output_testE3','output_trainF3','output_testF3','Best_posC','Convergence_curveC');
% 读取保存在 a1D3.mat 数据文件里面的数据

%% 参数的计算
xA = length(output_testB);                                                    % 计算出测试集样本个数
xB = 1:1:xA;                                                                  % 列出横坐标

a = output_testB;                                                            % 列出 y1

b1 = output_testE1;                                                           % 列出 y2
b2 = output_testE2;                                                           % 列出 y3
b3 = output_testE3;                                                           % 列出 y4

c1 = min(a);
c2 = floor(c1);
c3 = max(a);
c4 = floor(c3);
c5 = c2 - 5;
c6 = c4 + 5;
c7 = c5:1:c6;



y3 = b1;
y1 = b2;
y2 = b3;

%% 画出预测值图像

% 画出第 1 副图
fig1 = figure;                                                                % 定义要画的图像编号 fig1
fig1.Name  = '回归效果图';                                                    % 图像的标题名称 '回归效果图'

subplot(1,1,1);

hold on;

f1s1 = scatter(a,y1);                                                         % 画出真实值的散点图 f1s1
f1s1.Marker = 'v';                                                             % 散点类型
f1s1.LineWidth = 0.5;                                                          % 散点线宽
f1s1.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s1.MarkerFaceColor = 'magenta';                                              % 散点填充颜色
f1s1.SizeData = 20;                                                            % 散点大小

f1s2 = scatter(a,y2);                                                         % 画出真实值的散点图 f1s1
f1s2.Marker = '^';                                                             % 散点类型
f1s2.LineWidth = 0.5;                                                          % 散点线宽
f1s2.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s2.MarkerFaceColor = 'green';                                                % 散点填充颜色
f1s2.SizeData = 20;                                                            % 散点大小

f1s3 = scatter(a,y3);                                                         % 画出真实值的散点图 f1s1
f1s3.Marker = 'square';                                                             % 散点类型
f1s3.LineWidth = 0.5;                                                          % 散点线宽
f1s3.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s3.MarkerFaceColor = 'blue';                                                 % 散点填充颜色
f1s3.SizeData = 20;                                                            % 散点大小


f1d1 = plot(c7,c7);
f1d1.Color = 'red';
f1d1.LineStyle = '--';
f1d1.LineWidth = 1;
% f1d1.Alpha = 1;                                                                 % 默认为 0.7

% axis padded;
axis([83 125 83 125]);
set(gca,'FontName','Times New Roman','FontSize',20);
box on;

xla1 = xlabel('真实值/kN');
xla1.FontSize = 20;
xla1.FontWeight = 'bold';
xla1.FontName = '仿宋';
xla1.Color = 'black';

yla1 = ylabel('预测值/kN');
yla1.FontSize = 20;
yla1.FontWeight = 'bold';
yla1.FontName = '仿宋';
yla1.Color = 'black';

leg1 = legend('LPSO-ANN','APSO-ANN','DPSO-ANN');
leg1.FontSize = 14;
leg1.TextColor = 'black';
leg1.Box = 'off';
leg1.FontName = 'Times New Roman';
leg1.FontWeight = 'bold';
leg1.Location = 'northwest';

%% 设置结束计算时间点0           text
tEnd0 = toc(tStart0);

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：%d 分 , %.4f 秒\n',...
    floor(tEnd0/60), rem(tEnd0,60));                                              % 程序的总用时间

fprintf('************************************************************\n');
