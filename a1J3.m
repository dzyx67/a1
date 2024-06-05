% a1J3:此部分文件是为了计算已经训练好的数据的误差指标

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
load('a1D3.mat','uz1','uz2','ra1','n_use','scaA','list_trainA','list_testA',...
    'input_trainA','input_testA','output_trainA','output_testA','output_trainB','output_testB',...
    'popsize','maxgen','c1','c2','uxv','xmaxA','wmax','wmin',...
    'net1','output_trainE1','output_testE1','output_trainF1','output_testF1','Best_posA','Convergence_curveA',...
    'net2','output_trainE2','output_testE2','output_trainF2','output_testF2','Best_posB','Convergence_curveB',...
    'net3','output_trainE3','output_testE3','output_trainF3','output_testF3','Best_posC','Convergence_curveC');                               
 % 读取保存在 a1D3.mat 数据文件里面的数据      
 
%% 误差指标的计算
[rmse_test1,R2_test1] = a1F1a(output_testA,output_testF1,output_trainA,output_trainF1);

[rmse_test2,R2_test2] = a1F1a(output_testA,output_testF2,output_trainA,output_trainF2);

[rmse_test3,R2_test3] = a1F1a(output_testA,output_testF3,output_trainA,output_trainF3);

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：                     %d 分 , %.4f 秒\n',...
floor(tEnd0/60), rem(tEnd0,60));                                               % 程序的总用时间

fprintf('************************************************************\n');
