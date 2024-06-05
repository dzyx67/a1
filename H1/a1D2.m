% a1D2:此部分文件是为了将 a1Z1 程序文件中保存下来的数据文件 d1Z1.mat 进行转换为 a1D2.mat

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

%% 数据的读取
load('d1Z1.mat','net1','uz1','uz2','ra1','n_use',...
    'output_trainA','output_testA','output_trainF','output_testF');           % 将读取保存在 d1Z1.mat 数据文件里面的数据

%% 数据的保存
save('a1D2.mat','net1','uz1','uz2','ra1','n_use',...
    'output_trainA','output_testA','output_trainF','output_testF');           % 将读取的变量保存在 d1D2.mat 数据文件里面

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：                       %d 分 , %.4f 秒\n',...
floor(tEnd0/60), rem(tEnd0,60));                                               % 程序的总用时间

fprintf('************************************************************\n');
