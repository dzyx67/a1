% a1C7:此程序的目的是为了读取并且保存样本数据

%% 初始化
clear ;
close all;
clc;

%% 开始计算时间点0
tStart0 = tic;

%% 数据读取
u1a = readmatrix('a1E2.xlsx','Sheet','a2','Range','B4:BD40003');             % 使用 readmatrix 函数读取工作表数据并且保存为 u1a
u2a = readcell('a1E2.xlsx','Sheet','a2','Range','A1:BD40003');               % 使用 readcell 函数读取工作表数据并且保存为 u2a
u3a = readtable('a1E2.xlsx','Sheet','a2','Range','A1:BD40003');              % 使用 readtable 函数读取工作表数据并且保存为 u3a
filename_a = 'a1E2.xlsx';
sheet_a = 'a2';
xlRange_a = 'B4:BD40003';
u4a = xlsread(filename_a,sheet_a,xlRange_a);

%% 数据的保存
save('d1C7.mat','u1a','u2a','u3a','u4a');                                    % 将读取的变量保存在 d1C7.mat 数据文件里面                                      

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
disp(' ');
disp('************************************************************');
disp('计时结果输出：');
fprintf('总共用时 ：%d 分 , %10.2f 秒\n',...
floor(tEnd0/60), rem(tEnd0,60));                                            % 程序的总用时间
disp('************************************************************');
disp(' ');
