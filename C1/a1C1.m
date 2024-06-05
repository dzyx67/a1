% a1C1:此程序的目的是画出全部的曲线

%% 初始化
clear ;
close all;
clc;

%% 开始计算时间点0
tStart0 = tic;

%% 数据读取
% u1 = xlsread('a1E1.xlsx','a1','A2:A126366');                                    % x1 为读取的横坐标序列
% v1 = xlsread('a1E1.xlsx','a1','B2:B126366');                                    % y1 为读取的水平力数值
u1a = readmatrix('a1E2.xlsx','Sheet','a2','Range','B4:BD40003');
u2a = readcell('a1E2.xlsx','Sheet','a2','Range','A1:BD40003');
u3a = readtable('a1E2.xlsx','Sheet','a2','Range','A1:BD40003');

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
f1 = figure;                                                                  % 图像的名称
subplot(1,1,1);                                                               % 标明作图
p1a = plot(x1,y1,'-b','Linewidth',1);                                         % 画出原始图形的线条 L1
% axis([a1u-c1x,a1d+c1x,b1u-c1y,b1d+c1y]);                                      % 图像的坐标范围
axis padded;                                                                % 设置图像的坐标范围
title('全部曲线');                                                            % 图像的标题

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
disp(' ');
disp('************************************************************');
disp('计时结果输出：');
fprintf('总共用时 ：%d 分 , %f 秒', floor(tEnd0/60), rem(tEnd0,60));            % 程序的总用时间
