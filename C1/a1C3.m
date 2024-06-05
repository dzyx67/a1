% a1C3:此程序的目的是为了查看数据的分布

%% 初始化
clear ;
close all;
clc;

%% 开始计算时间点0
tStart0 = tic;

%% 数据读取
% u1 = xlsread('a1E1.xlsx','a3','A2:A40002');                                      % x1 为读取的横坐标序列
% v1 = xlsread('a1E1.xlsx','a3','B2:B40002');                                      % y1 为读取的水平力数值

load('a1D1.mat','u1a','u2a','u3a','u4a');                                      % 将读取保存在 a1D1.mat 数据文件里面的数据  

ra1 = 1:1:1;
ra2 = 2:1:2;
u1 = u1a(:,ra1);
v1 = u1a(:,ra2);

%% 参数的设定
n1 = length(u1);                                                              % n1 为样本点的个数
m1 = 1;                                                                       % 选取样本点的间隔
i1 = 1:m1:n1;                                                                 % 抽选间隔的样本点
r1 = length(i1);                                                              % 找出抽取的样本的长度
x1 = 1:1:r1;                                                                  % 列出图像的 x 轴的数值
y1 = v1(i1);                                                                  % 列出图像的 y 轴的数值
a1u = min(x1);                                                                % a1 散点图的横坐标的最小值
a1d = max(x1);                                                                % a2 散点图的横坐标的最大值
b1u = min(y1);                                                                % b1 散点图的纵坐标的最小值
b1d = max(y1);                                                                % b2 散点图的纵坐标的最大值
c1x = n1/10;                                                                  % c1 为图像 x 坐标轴的前后的间隔
c1y = 10;                                                                     % c2 为图像 y 坐标轴的前后的间隔

%% 画出图形
f1 = figure;
k1 = 10;
k2 = 20;
k3 = 40;
k4 = 80;

subplot(2,2,1);
h1 = histogram(y1,k1);
title('分为10组','color','b');                                  % 图像的标题

subplot(2,2,2);
h2 = histogram(y1,k2);
title('分为20组','color','b');                                    % 图像的标题

subplot(2,2,3);
h3 = histogram(y1,k3);
title('分为40组','color','b');                                    % 图像的标题

subplot(2,2,4);
h4 = histogram(y1,k4);
title('分为80组','color','b');                                    % 图像的标题

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
disp(' ');
disp('************************************************************');
disp('计时结果输出：');
fprintf('总共用时 ：%d 分 , %f 秒', floor(tEnd0/60), rem(tEnd0,60));            % 程序的总用时间
