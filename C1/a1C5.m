% a1C5:此程序的目的是为了对数据进行滤波处理

%% 初始化
clear ;
close all;
clc;

%% 开始计算时间点0
tStart0 = tic;

%% 数据读取
load('C4d1.mat','u1','v1','g1');                                              % 读取储存的变量

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

%% 对数据进行滤波处理
% % 采用的方法为滑动平均法
q1 = 201;                                                                     % 滑动平均法的窗口长度为 m1                  
h1 = movmean(g1,q1);                                                          % 滑动平均法处理后的曲线为 k2

% 采用的方法为小波去噪法
h2 = wdenoise(g1);


%% 画出图形
% 画出消除离群值后的曲线
f1 = figure;
subplot(2,1,1);
p1a = plot(x1,h1,'-b','Linewidth',1);                                         % 画出原始图形的线条 p1a
axis([a1u-c1x,a1d+c1x,b1u-c1y,b1d+c1y]);                                      % 图形的坐标轴范围
title('滑动平均法滤波后的曲线','color','b');                                   % 图像的标题  

subplot(2,1,2);                                                  
p1b = plot(x1,h2,'-k','Linewidth',1);                                         % 画出离群值处理后的线条 p1b
axis([a1u-c1x,a1d+c1x,b1u-c1y,b1d+c1y]);                                      % 图形的坐标轴范围
title('小波去噪法后的曲线','color','b');                                       % 图像的标题  

%% 保存数据
save('C5d1.mat','u1','v1','h1','h2');                                              % 将滤波后的数据保存

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
disp(' ');
disp('************************************************************');
disp('计时结果输出：');
fprintf('总共用时 ：%d 分 , %f 秒', floor(tEnd0/60), rem(tEnd0,60));           % 程序的总用时间