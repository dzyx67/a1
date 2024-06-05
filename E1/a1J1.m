% a1J1:此程序的目的是为了计算出预测值的误差指标

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
    'output_trainA','output_testA','output_trainF','output_testF');           % 将读取的变量保存在 d1Z1.mat 数据文件里面

%% 进行变量的替换
out_test = output_testA;
out_train = output_trainA;
sim_test = output_testF;
sim_train = output_trainF;

%% 进行计算
[mae_test,mse_test,rmse_test,mape_test,error_test,errorPercent_test,R2_test,...
    mae_train,mse_train,rmse_train,mape_train,error_train,errorPercent_train,R2_train,...
    mae_all,mse_all,rmse_all,mape_all,error_allA,errorPercent_all,R2_all]...
    =a1F1(out_test,sim_test,out_train,sim_train);

%% 数据的保存

%% 设置结束计算时间点0
tEnd0 = toc(tStart0);

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('总共用时 ：%d 分 , %.4f 秒\n',...
    floor(tEnd0/60), rem(tEnd0,60));                                               % 程序的总用时间

fprintf('************************************************************\n');
