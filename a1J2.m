% a1J2:此程序的目的是为了计算出参数的特征

%% 初始化
clear ;
close all;
clc;

%% 程序开始运行标时的日期与时间输出
fprintf('************************************************************\n');

fprintf('程序运行开始\n');

t_time1 = datetime;
fprintf('程序开始运行时的日期与时间为：       %s\n',t_time1);

fprintf('************************************************************\n');

%% 开始计算时间点0
tStart0 = tic;

%% 数据读取
load('a1D2.mat','net1','uz1','uz2','ra1','n_use',...
    'output_trainA','output_testA','output_trainF','output_testF');              % 将读取的变量保存在 d1Z1.mat 数据文件里面

%% 数据的预处理
a1 = uz2(:,3:1:4);                                                               % 2-带钢宽度（入口与出口）
a2 = uz2(:,5:1:12);                                                              % 8-张力（F0-F7）
a3 = uz2(:,13:1:19);                                                             % 7-带钢速度（F1-F7）
a4 = uz2(:,20:1:26);                                                             % 7-弯辊力矩（F1-F7）
a5 = uz2(:,27:1:33);                                                             % 7-轧制力（F1-F7）
a6 = uz2(:,34:1:40);                                                             % 7-弯辊力（F1-F7）
a7 = uz2(:,41:1:47);                                                             % 7-辊缝（F1-F7）
a8 = uz2(:,48:1:53);                                                             % 6-带钢厚度（F0-F5）
a9 = uz2(:,54:1:55);                                                             % 2-卷径（出口与入口）

%% 数据的运算
b1 = numel(a1);                                                                  % 求出参数变量的总数
c1 = [b1,1];                                                                     % 调整格式样式
d1 = reshape(a1,c1);                                                             % 重新构建矩阵
e1 = max(d1);                                                                    % 求出变量的最大值
f1 = min(d1);                                                                    % 求出变量的最小值
g1 = 1:1:b1;
h1 = transpose(d1);

b2 = numel(a2);                                                                  % 求出参数变量的总数
c2 = [b2,1];                                                                     % 调整格式样式
d2 = reshape(a2,c2);                                                             % 重新构建矩阵
e2 = max(d2);                                                                    % 求出变量的最大值
f2 = min(d2);                                                                    % 求出变量的最小值
g2 = 1:1:b2;
h2 = transpose(d2);

b3 = numel(a3);                                                                  % 求出参数变量的总数
c3 = [b3,1];                                                                     % 调整格式样式
d3 = reshape(a3,c3);                                                             % 重新构建矩阵
e3 = max(d3);                                                                    % 求出变量的最大值
f3 = min(d3);                                                                    % 求出变量的最小值
g3 = 1:1:b3;
h3 = transpose(d3);

b4 = numel(a4);                                                                  % 求出参数变量的总数
c4 = [b4,1];                                                                     % 调整格式样式
d4 = reshape(a4,c4);                                                             % 重新构建矩阵
e4 = max(d4);                                                                    % 求出变量的最大值
f4 = min(d4);                                                                    % 求出变量的最小值
g4 = 1:1:b4;
h4 = transpose(d4);

b5 = numel(a5);                                                                  % 求出参数变量的总数
c5 = [b5,1];                                                                     % 调整格式样式
d5 = reshape(a5,c5);                                                             % 重新构建矩阵
e5 = max(d5);                                                                    % 求出变量的最大值
f5 = min(d5);                                                                    % 求出变量的最小值
g5 = 1:1:b5;
h5 = transpose(d5);

b6 = numel(a6);                                                                  % 求出参数变量的总数
c6 = [b6,1];                                                                     % 调整格式样式
d6 = reshape(a6,c6);                                                             % 重新构建矩阵
e6 = max(d6);                                                                    % 求出变量的最大值
f6 = min(d6);                                                                    % 求出变量的最小值
g6 = 1:1:b6;
h6 = transpose(d6);

b7 = numel(a7);                                                                  % 求出参数变量的总数
c7 = [b7,1];                                                                     % 调整格式样式
d7 = reshape(a7,c7);                                                             % 重新构建矩阵
e7 = max(d7);                                                                    % 求出变量的最大值
f7 = min(d7);                                                                    % 求出变量的最小值
g7 = 1:1:b7;
h7 = transpose(d7);

b8 = numel(a8);                                                                  % 求出参数变量的总数
c8 = [b8,1];                                                                     % 调整格式样式
d8 = reshape(a8,c8);                                                             % 重新构建矩阵
e8 = max(d8);                                                                    % 求出变量的最大值
f8 = min(d8);                                                                    % 求出变量的最小值
g8 = 1:1:b8;
h8 = transpose(d8);

b9 = numel(a9);                                                                  % 求出参数变量的总数
c9 = [b9,1];                                                                     % 调整格式样式
d9 = reshape(a9,c9);                                                             % 重新构建矩阵
e9 = max(d9);                                                                    % 求出变量的最大值
f9 = min(d9);                                                                    % 求出变量的最小值
g9 = 1:1:b9;
h9 = transpose(d9);

%% 画出图像
fig1 = figure;
fig1.Name  = '带宽（入口与出口）';                                                % 图像的标题名称
f1p1 = scatter(g1,h1);                                                              % 画出线条L1
f1p1.Marker = '.';

t1 = title('带宽（入口与出口）');
axis padded;   


fig2 = figure;
fig2.Name  = '张力（F0-F7）  ';                                                % 图像的标题名称
f2p1 = scatter(g2,h2);                                                              % 画出线条L1
f2p1.Marker = '.';

t2 = title('张力（F0-F7） ');
axis padded;   


fig3 = figure;
fig3.Name  = '带钢速度（F1-F7）';                                                % 图像的标题名称
f3p1 = scatter(g3,h3);                                                              % 画出线条L1
f3p1.Marker = '.';

t3 = title('带钢速度（F1-F7）');
axis padded;   


fig4 = figure;
fig4.Name  = '弯辊力矩（F1-F7）';                                                % 图像的标题名称
f4p1 = scatter(g4,h4);                                                              % 画出线条L1
f4p1.Marker = '.';

t4 = title('弯辊力矩（F1-F7）');
axis padded;   


fig5 = figure;
fig5.Name  = '轧制力（F1-F7）';                                                % 图像的标题名称
f5p1 = scatter(g5,h5);                                                              % 画出线条L1
f5p1.Marker = '.';

t5 = title('轧制力（F1-F7）');
axis padded;   

fig6 = figure;
fig6.Name  = '弯辊力（F1-F7）';                                                % 图像的标题名称
f6p1 = scatter(g6,h6);                                                              % 画出线条L1
f6p1.Marker = '.';

t6 = title('弯辊力（F1-F7）');
axis padded;   

fig7 = figure;
fig7.Name  = '辊缝（F1-F7）';                                                % 图像的标题名称
f7p1 = scatter(g7,h7);                                                              % 画出线条L1
f7p1.Marker = '.';

t7 = title('辊缝（F1-F7）');
axis padded;   

fig8 = figure;
fig8.Name  = '带钢厚度（F0-F5）';                                                % 图像的标题名称
f8p1 = scatter(g8,h8);                                                              % 画出线条L1
f8p1.Marker = '.';

t8 = title('带钢厚度（F0-F5）');
axis padded;   

fig9 = figure;
fig9.Name  = '卷径（出口与入口）';                                                % 图像的标题名称
f9p1 = scatter(g9,h9);                                                              % 画出线条L1
f9p1.Marker = '.';

t9 = title('卷径（出口与入口）');
axis padded;   

%% 结果的输出
fprintf('\n************************************************************\n');
fprintf('参数名称                     最大值                       最小值 \n');
fprintf('带宽（入口与出口）            %.2f                         %.2f\n',e1,f1);
fprintf('张力（F0-F7）                 %.2f                         %.2f\n',e2,f2);
fprintf('带钢速度（F1-F7）             %.2f                         %.2f\n',e3,f3);
fprintf('弯辊力矩（F1-F7）             %.2f                         %.2f\n',e4,f4);
fprintf('轧制力（F1-F7）               %.2f                         %.2f\n',e5,f5);
fprintf('弯辊力（F1-F7）               %.2f                         %.2f\n',e6,f6);
fprintf('辊缝（F1-F7）                 %.2f                         %.2f\n',e7,f7);
fprintf('带钢厚度（F0-F5）             %.2f                         %.2f\n',e8,f8);
fprintf('卷径（出口与入口）            %.2f                         %.2f\n',e9,f9);

%% 设置结束计算时间点0
tEnd0 = toc(tStart0);

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：%d 分 , %.4f 秒\n',...
    floor(tEnd0/60), rem(tEnd0,60));                                               % 程序的总用时间

fprintf('************************************************************\n');
