% a1Z1:此程序的目的是为了对参数不经过滤波处理，随机挑选500个样本，进行水平力预测，采用的标准ANN神经网络

%% 初始化
clear ;
close all force;
clc;

%% 程序开始运行标时的日期与时间输出
fprintf('************************************************************\n');

fprintf('程序运行开始\n');

t_time1 = datetime;
fprintf('程序开始运行时的日期与时间为：      %s\n',t_time1);

fprintf('************************************************************\n');

%% 开始计算时间点0
tStart0 = tic;

%% 数据读取
load('a1D1.mat','u1a','u2a','u3a','u4a');                                   % 将读取保存在 a1D1.mat 数据文件里面的数据  

%% 对原始数据进行划分处理
% 对原始数据进行抽样处理
uz1 = u1a;                                                                  % 将选中的数据赋值给 uzA
[sAz1,sAz2] = size(uz1,1,2);                                                % 查看矩阵 uzA 的维度
n_use = 500;                                                                % 定义需要挑选的样本个数为 n_use
ra1 = randperm(sAz1,n_use);                                                 % 生成需要指定个数的无重复元素的序列 ra1
uz2 = u1a(ra1,:);                                                           % 定义挑选指定样本个数后的数据矩阵为 uzB

% 划分数据集的输入变量与输出变量
rb1 = (1:1:1);                                                               % 对列序 r1 赋值，为序列的列序 
rb2 = (3:1:55);                                                              % 对列序 r2 赋值，为自变量的列序
rb3 = (2:1:2);                                                               % 对列序 r3 赋值，为因变量水平力的列序

v1 = uz2(:,rb1);                                                             % 选择数据矩阵 uz 的 ra1 序列的数据为 v1
[s1z1,s1z2] = size(v1,1,2);                                                  % 查看矩阵 v1 维度
x1 = 1:1:s1z1;                                                               % 根据维度设置 x 轴的变量
y1 = v1;                                                                     % 将变量赋值转换

v2 = uz2(:,rb2);                                                             % 选择数据矩阵 uz 的 ra2 序列自变量数据为 v2  
[s2z1,s2z2] = size(v2,1,2);                                                 % 查看矩阵 v2 维度
x2 = 1:1:s2z1;                                                              % 根据维度设置 x 轴的变量
y2 = v2;                                                                    % 自变量矩阵数据为 y2

v3 = uz2(:,rb3);                                                             % 选择数据矩阵 uz 的 ra3 因变量数据为 v3 
[s3z1,s3z2] = size(v3,1,2);                                                 % 查看矩阵 v3 维度
x3 = 1:1:s3z1;                                                              % 根据维度设置 x 轴的变量
y3 = v3;                                                                    % 因变量矩阵数据为 y3

%% 数据的预处理
% 划分数据集
n_all = n_use;                                                               % 定义 n_all 为总的样本的数量
scaB = 0.3;                                                                 % 定义 sca 为测试样本占总样本的比例
n_test1 = n_all * scaB;  
n_test = floor(n_test1);                                                    % 将测试集数量 n_test 向下取整
n_train = n_all - n_test;                                                   % 计算训练样本数目为 n_train

rc_train = (1:1:n_train);                                                   % 训练集的序列数组为  r_train
rc_test = (n_train+1:1:n_all);                                              % 测试集的序列数组为  r_test


input_trainA = y2(rc_train,:);                                                  % 训练集输入数据矩阵为 input_trainA
input_testA = y2(rc_test,:);                                                    % 测试集输入数据矩阵为 input_testA

output_trainA = y3(rc_train,:);                                                 % 训练集输出数据矩阵为 output_trainA
output_testA = y3(rc_test,:);                                                   % 测试集输出数据矩阵为 output_test  B = transpose(A)

% 将数据转置
input_trainB = transpose(input_trainA);                        % 将训练集输入数据矩阵 input_trainA 转置后得到转置后的训练集输入数据矩阵 input_trainB
input_testB = transpose(input_testA);                          % 将测试集输入数据矩阵 input_testA 转置后得到转置后的测试集输入数据矩阵 input_testB

output_trainB = transpose(output_trainA);                      % 将训练集输出数据矩阵 output_trainA 转置后得到转置后的训练集输出数据矩阵 output_trainB   
output_testB = transpose(output_testA);                        % 将测试集输出数据矩阵 output_testA 转置后得到转置后的测试集输出数据矩阵 output_testB  

% 数据的归一化
[input_trainC,inputPS_train]=mapminmax(input_trainB);                         % 将训练集输入数据矩阵 input_trainB 归一化为 input_trainC
[input_testC,inputPS_test]=mapminmax(input_testB);                            % 将测试集输入数据矩阵 input_testB 归一化为 input_testC

[output_trainC,outputPS_train] = mapminmax(output_trainB);                    % 将训练集输出数据矩阵 output_trainB 归一化为 output_trainC
[output_testC,outputPS_test] = mapminmax(output_testB);                       % 将测试集输出数据矩阵 output_testB 归一化为 output_testC

%% 设置开始计时点1
tStart1 = tic;

%% 创建标准BP神经网络并且设置参数
fprintf('标准BP神经网络\n');
hiddennum_best = 8;
net1 = feedforwardnet(hiddennum_best);

% 网络训练函数与传输函数的设置
net1.trainFcn = 'trainbr';                                                    % 网络的训练函数选用trainlm
net1.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig  
net1.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net1.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net1.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net1.trainParam.goal=0.012;                                                   % 训练目标最小误差，这里设置为0.0001
net1.trainParam.show=25;                                                      % 显示频率，这里设置为每训练25次显示一次

% 标准BP神经网络开始训练
net1 = train(net1,input_trainC,output_trainC);                                % 生成神经网络

% 预测
output_trainD = sim(net1,input_trainC);                                       % 仿真结果训练集输出为 output_trainD 
output_testD = sim(net1,input_testC);                                         % 仿真结果测试集输出为 output_testD 

% 仿真结果的反归一化
output_trainE = mapminmax('reverse',output_trainD,outputPS_train);            % 反归一化后的仿真测试集输出为 output_trainE
output_testE = mapminmax('reverse',output_testD,outputPS_test);               % 反归一化后的仿真测试集输出为 output_testE

% 将数据反转置
output_trainF = transpose(output_trainE);                                     % 将反归一化后的仿真测试集输出 output_trainE 反转置为 output_trainF
output_testF = transpose(output_testE);                                       % 将反归一化后的仿真测试集输出 output_testE 反转置为 output_testF

%% 设置结束计时点1
tEnd1 = toc(tStart1); 

%% 数据的保存
save('d1Z1.mat','net1','uz1','uz2','ra1','n_use',...
    'output_trainA','output_testA','output_trainF','output_testF');           % 将读取的变量保存在 d1Z1.mat 数据文件里面   

%% 设置结束计算时间点0                                                         
tEnd0 = toc(tStart0); 

%% 计时结果输出
fprintf('\n************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('\n标准BP神经网络用时 ：     %d 分 , %6.2f 秒\n',...
floor(tEnd1/60), rem(tEnd1,60));                                              % 标准BP神经网络的所用的时间    

fprintf('\n总共用时 ：               %d 分 , %6.2f 秒\n',...
floor(tEnd0/60), rem(tEnd0,60));                                              % 程序的总用时间

fprintf('************************************************************\n');

