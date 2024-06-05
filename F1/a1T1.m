% a1T1:此程序的目的是画出预测值与真实值的对比图

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
load('a1D2.mat','net1','uz1','uz2','ra1','n_use',...
    'output_trainA','output_testA','output_trainF','output_testF');           % 将读取的变量保存在 d1D2.mat 数据文件里面

%% 计算数值
                                            
% 设置x轴
c1 = length(output_testA);                                                    % x轴的长度
c2 = c1 / 2;                                                                  % x轴长度的一半
c3 = floor(c2);                                                               % x轴长度的一半向下取整

d1 = 1;                                                                       % 第 1 副图片x轴的起点
d2 = c3;                                                                      % 第 1 副图片x轴的终点
d3 = c3+1;                                                                    % 第 2 副图片x轴的起点
d4 = c1;                                                                      % 第 2 副图片x轴的终点

e1 = d1:1:d2;                                                                 % 第 1 副图片的x轴                       
e2 = d3:1:d4;                                                                 % 第 2 副图片的x轴
e3 = d1:1:d4;

% 设置y轴
g1 = output_testA(d1:1:d2);                                                   % 第 1 副图片的 y1
g2 = output_testA(d3:1:d4);                                                   % 第 2 副图片的 y1
g3 = output_testA(d1:1:d4);

h1 = output_testF(d1:1:d2);                                                   % 第 1 副图片的 y2
h2 = output_testF(d3:1:d4);                                                   % 第 2 副图片的 y2
h3 = output_testF(d1:1:d4);

%% 画出预测值图像

% 画出第 1 副图
f1 = figure;                                                                  % 定义要画的图像编号 1 
f1.Name  = '部分曲线的比较';                                                   % 图像的标题名称

subplot(1,1,1);     

hold on;

pf11 = plot(e1,g1,'-',...                                                     % 画出第 2 副图片的第 1 条线           
    'color','k',...
    'linewidth',1,...
    'Marker','o',...
    'MarkerFaceColor','k',...
    'MarkerEdgeColor','k',...
    'MarkerIndices',1:1:length(g1));



pf12 = plot(e1,h1,'-',...                                                     % 画出第 2 副图片的第 1 条线           
    'color','b',...
    'linewidth',1,...
    'Marker','s',...
    'MarkerFaceColor','b',...
    'MarkerEdgeColor','b',...
    'MarkerIndices',1:1:length(h1));

set(gca,'FontSize',14);

legend({'\fontname{宋体}真实值\fontname{Times New Roman}',...
        '\fontname{Times New Roman}ANN\fontname{宋体}预测值',...
        '\fontname{Times New Roman}GA-ANN\fontname{宋体}预测值'},...
        'FontSize',14)

xlabel('\fontname{宋体}测试样本编号\fontname{Times New Roman}1-80','FontSize',18)
ylabel('\fontname{宋体}水平力\fontname{Times New Roman}(kN)','FontSize',18)
title('测试集预测值比较','FontSize',18)
set(get(gca,'XLabel'),'FontSize',18);

f2 = figure;                                                                 % 定义要画的图像编号 2

subplot(1,1,1);     
hold on;
pf21 = plot(e2,g2,'-',...                                                    % 画出第 2 副图片的第 1 条线           
    'color','k',...
    'linewidth',1,...
    'Marker','o',...
    'MarkerFaceColor','k',...
    'MarkerEdgeColor','k',...
    'MarkerIndices',1:1:length(g2));


pf22 = plot(e2,h2,'-',...                                                    % 画出第 2 副图片的第 1 条线           
    'color','b',...
    'linewidth',1,...
    'Marker','s',...
    'MarkerFaceColor','b',...
    'MarkerEdgeColor','b',...
    'MarkerIndices',1:1:length(h2));

hold off;
set(gca, 'FontSize',14)                                                      % 设置坐标轴字体是 8
legend({'\fontname{宋体}真实值\fontname{Times New Roman}',...
        '\fontname{Times New Roman}ANN\fontname{宋体}预测值',...
        '\fontname{Times New Roman}GA-ANN\fontname{宋体}预测值'},...
        'FontSize',14)
xlabel('\fontname{宋体}测试样本编号\fontname{Times New Roman}81-160','FontSize',18)

ylabel('\fontname{宋体}水平力\fontname{Times New Roman}(kN)','FontSize',18)

title('测试集预测值比较','FontSize',18)

f3 = figure;                                                        % 定义要画的图像编号 2
subplot(1,1,1);     

hold on;

pf31 = plot(e3,g3,'-',...                                                  % 画出第 2 副图片的第 1 条线           
    'color','k',...
    'linewidth',1,...
    'Marker','o',...
    'MarkerFaceColor','k',...
    'MarkerEdgeColor','k',...
    'MarkerIndices',1:1:length(g3));

ph32 = plot(e3,h3,'-',...                                                  % 画出第 2 副图片的第 1 条线           
    'color','b',...
    'linewidth',1,...
    'Marker','s',...
    'MarkerFaceColor','b',...
    'MarkerEdgeColor','b',...
    'MarkerIndices',1:1:length(h3));

hold off;

legend('真实值','ANN预测值','GA-ANN预测值')
xlabel('测试样本编号1-160')
ylabel('预测值')
title('测试集预测值比较')

%% 设置结束计算时间点0           text
tEnd0 = toc(tStart0); 

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：%d 分 , %.4f 秒\n',...
floor(tEnd0/60), rem(tEnd0,60));                                              % 程序的总用时间

fprintf('************************************************************\n');
