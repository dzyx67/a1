% a1T3:此程序的目的是画出已经训练好的预测值与真实值的对比图，添加了点与点之间的连线

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

y1 = output_testB;                                                            % 列出 y1
y2 = output_testE1;                                                           % 列出 y2
y3 = output_testE2;                                                           % 列出 y3
y4 = output_testE3;                                                           % 列出 y4

%% 画出预测值图像

% 画出第 1 副图
fig1 = figure;                                                                % 定义要画的图像编号 fig1
fig1.Name  = '预测值比较';                                                     % 图像的标题名称 '预测值比较'

subplot(1,1,1);

hold on;

f1s1 = plot(xB,y1);                                                         % 画出真实值的散点图 f1s1
f1s1.Marker = 'o';                                                             % 散点类型
f1s1.LineWidth = 0.5;                                                          % 散点线宽
f1s1.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s1.MarkerFaceColor = 'black';                                                % 散点填充颜色
% f1s1.SizeData = 30;                                                            % 散点大小

f1s2 = plot(xB,y2);                                                         % 画出真实值的散点图 f1s1
f1s2.Marker = 'v';                                                             % 散点类型
f1s2.LineWidth = 0.5;                                                          % 散点线宽
f1s2.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s2.MarkerFaceColor = 'magenta';                                              % 散点填充颜色
% f1s2.SizeData = 30;                                                            % 散点大小

f1s3 = plot(xB,y3);                                                         % 画出真实值的散点图 f1s1
f1s3.Marker = '^';                                                             % 散点类型
f1s3.LineWidth = 0.5;                                                          % 散点线宽
f1s3.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s3.MarkerFaceColor = 'green';                                              % 散点填充颜色
% f1s3.SizeData = 30;                                                            % 散点大小

f1s4 = plot(xB,y4);                                                         % 画出真实值的散点图 f1s1
f1s4.Marker = 'square';                                                             % 散点类型
f1s4.LineWidth = 0.5;                                                          % 散点线宽
f1s4.MarkerEdgeColor = 'black';                                                % 散点边缘颜色
f1s4.MarkerFaceColor = 'blue';                                              % 散点填充颜色
% f1s4.SizeData = 30;                                                            % 散点大小

% axis padded;
axis([-10 160 80 150]);
set(gca,'FontName','Times New Roman','FontSize',20);

xla1 = xlabel('样本数');
xla1.FontSize = 20;
xla1.FontWeight = 'bold';
xla1.FontName = '仿宋';
xla1.Color = 'black';

yla2 = ylabel('水平力/KN');
yla2.FontSize = 20;
yla2.FontWeight = 'bold';
yla2.FontName = '仿宋';
yla2.Color = 'black';

leg2 = legend('true','DPSO-ANN','LPSO-ANN','APSO-ANN');
leg2.FontSize = 14;
leg2.TextColor = 'black';
leg2.Box = 'off';
leg2.FontName = 'Times New Roman';
leg2.FontWeight = 'bold';

%% 设置结束计算时间点0           text
tEnd0 = toc(tStart0);

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：%d 分 , %.4f 秒\n',...
    floor(tEnd0/60), rem(tEnd0,60));                                              % 程序的总用时间

fprintf('************************************************************\n');
