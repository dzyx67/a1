% a1Z1:此程序的目的是为了对参数不经过滤波处理，随机挑选500个样本，进行水平力预测，采用标准ANN神经网络与改进的LPSO和APSO与DPSO
%{
1.参数结构:
tStart0
hiddennum_best, inputnum, outputnum, dim

tStart1
net0, w1_0, w1_0a, Best_pos0, 
tEnd1
popsize, maxgen, c1, c2, uxv, xmaxA, wmax, wmin

tStart2
net1, a1F2, Best_posA, Convergence_curveA, w1_1, output_trainD1, output_trainE1
tEnd2

tStart3
net2, a1F4, Best_posB, Convergence_curveB, w1_2, output_trainD2, output_trainE2
tEnd3

tStart4
net3, a1F5, Best_posC, Convergence_curveC, w1_3, output_trainD3, output_trainE3
tEnd4

tStart5
net4, a1F6, Best_posD, Convergence_curveD, w1_4, output_trainD4, output_trainE4
tEnd5

f1x, f1, f1p1, f2x, f2, f2p1, 
tEnd0
%}

%% 初始化
clear ;
close all force;
clc;

%% 程序开始运行标时的日期与时间输出
fprintf('***********************************************************************************************\n');

fprintf('程序运行开始\n');

t_time1 = datetime;
fprintf('程序开始运行时的日期与时间为：      %s\n',t_time1);

fprintf('***********************************************************************************************\n');

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
scaA = 0.3;                                                                 % 定义 sca_A 为测试样本占总样本的比例
n_test1 = n_all * scaA;
n_test = floor(n_test1);                                                    % 将测试集数量 n_test 向下取整
n_train = n_all - n_test;                                                   % 计算训练样本数目为 n_train

rc_train = (1:1:n_train);                                                   % 训练集的序列数组为  rc_train
rc_test = (n_train+1:1:n_all);                                              % 测试集的序列数组为  rc_test

list_trainA = y1(rc_train,:);                                               % 训练集的序列为  list_trainA
list_testA = y1(rc_test,:);                                                 % 测试集的序列为  list_testA

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

%% 参数的设置
hiddennum_best = 8;
inputnum = s2z2;
outputnum = s3z2;
dim = inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum;                             %优化因子的个数

%% 设置开始计时点1
tStart1 = tic;

%% 创建标准BP神经网络并且设置参数
fprintf('\n标准ANN神经网络\n');

net0 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net1

% 网络训练函数与传输函数的设置
net0.trainFcn = 'trainbr';                                                    % 网络的训练函数选用trainlm     newff
net0.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig
net0.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net0.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net0.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net0.trainParam.goal=0.01;                                                    % 训练目标最小误差，这里设置为0.0001
% net1.trainParam.show=25;                                                    % 显示频率，这里设置为每训练25次显示一次
net0.trainParam.showWindow = 0;
% 标准BP神经网络开始训练
net0 = train(net0,input_trainC,output_trainC);                                % 生成神经网络

% 预测
output_trainD0 = sim(net0,input_trainC);                                       % 仿真结果训练集输出为 output_trainD
output_testD0 = sim(net0,input_testC);                                         % 仿真结果测试集输出为 output_testD

% 仿真结果的反归一化
output_trainE0 = mapminmax('reverse',output_trainD0,outputPS_train);            % 反归一化后的仿真测试集输出为 output_trainE
output_testE0 = mapminmax('reverse',output_testD0,outputPS_test);               % 反归一化后的仿真测试集输出为 output_testE

% 将数据反转置
output_trainF0 = transpose(output_trainE0);                                     % 将反归一化后的仿真测试集输出 output_trainE 反转置为 output_trainF
output_testF0 = transpose(output_testE0);                                       % 将反归一化后的仿真测试集输出 output_testE 反转置为 output_testF

w1_0 = net0.iw{1,1};
B1_0 = net0.b{1};
w2_0 = net0.lw{2,1};
B2_0 = net0.b{2};

w1_0a = reshape(w1_0,1,(inputnum * hiddennum_best));
B1_0a = reshape(B1_0,1,hiddennum_best);
w2_0a = reshape(w2_0,1,hiddennum_best * outputnum);
B2_0a = reshape(B2_0,1,outputnum);

Best_pos0 = [w1_0a,B1_0a,w2_0a,B2_0a];     

% 计算误差
[mae_test0,mse_test0,rmse_test0,mape_test0,error_testA0,errorPercent_test0,R2_test0,...
    mae_train0,mse_train0,rmse_train0,mape_train0,error_trainA0,errorPercent_train0,R2_train0,...
    mae_all0,mse_all0,rmse_all0,mape_all0,error_allA0,errorPercent_all0,R2_all0]...
    = a1F1(output_testA,output_testF0,output_trainA,output_trainF0);

%% 设置结束计时点1
tEnd1 = toc(tStart1);

%% 设置粒子群算法的通用参数
popsize = 5;                                                                 % 初始种群规模
maxgen = 30;                                                                 % 最大进化代数
c1 = 2;                                                                       % 每个粒子的个体学习因子，也称为个体加速常数
c2 = 2;                                                                       % 每个粒子的社会学习因子，也称为社会加速常数
uxv = 0.1;                                                                 % 最大速度与上限的比值
xmaxA = 1;                                                                       % 粒子位置的上限
wmax = 0.9;                                                                   % 权重的推荐最大值
wmin = 0.4;                                                                   % 权重的推荐最小值

%% 设置开始计时点2
tStart2 = tic;

%% 创建 ANN 与标准粒子群算法相结合的 PSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与标准的粒子群算法 PSO 结合，PSO-ANN 神经网络\n');

net1 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net1.trainFcn = 'trainbr';                                                    % 网络的训练函数选用trainlm
net1.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig
net1.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net1.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net1.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net1.trainParam.goal=0.01;                                                   % 训练目标最小误差，这里设置为0.0001
% net2.trainParam.show=25;                                                      % 显示频率，这里设置为每训练25次显示一次
net1.trainParam.showWindow = 0;

%% 调用标准的粒子群算法

[Best_posA,Best_scoreA,Convergence_curveA] = a1F2(net1);

%% 将优化好的权值与阈值赋值到神经网络net2中

w1_1 = Best_posA(:,1:1:inputnum * hiddennum_best);                                                         % 输入层到中间层的权值
B1_1 = Best_posA(:,(inputnum * hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best));        % 中间各层神经元阈值
w2_1 = Best_posA(:,(inputnum * hiddennum_best + hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum));
%中间层到输出层的权值
B2_1 = Best_posA(:,(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + 1)...
    :1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum));          % 输出层各神经元阈值
%矩阵重构
net1.iw{1,1} = reshape(w1_1,hiddennum_best,inputnum);
net1.lw{2,1} = reshape(w2_1,outputnum,hiddennum_best);
net1.b{1} = reshape(B1_1,hiddennum_best,1);
net1.b{2} = reshape(B2_1,outputnum,1);

%% 优化后的神经网络训练

net1=train(net1,input_trainC,output_trainC);                                  % 开始训练，其中input_trainC,output_trainC 分别为归一化后的训练集输入与输出样本

%% 优化后的神经网络测试
%预测
output_trainD1 = sim(net1,input_trainC);                                      % 训练结果仿真输出训练集  an2 为训练数据集的仿真输出
output_testD1 = sim(net1,input_testC);                                        % 测试结果仿真输出        bn2为测试数据集的仿真输出

%仿真结果的反归一化
output_trainE1 = mapminmax('reverse',output_trainD1,outputPS_train);
output_testE1 = mapminmax('reverse',output_testD1,outputPS_test);

% 进行转置
output_trainF1 = transpose(output_trainE1);                                   % output_trainF2 为转置后的预测的训练集输出
output_testF1 = transpose(output_testE1);                                     % output_testF2 为转置后的预测的测试集输出

% 计算误差
[mae_test1,mse_test1,rmse_test1,mape_test1,error_testA1,errorPercent_test1,R2_test1,...
    mae_train1,mse_train1,rmse_train1,mape_train1,error_trainA1,errorPercent_train1,R2_train1,...
    mae_all1,mse_all1,rmse_all1,mape_all1,error_allA1,errorPercent_all1,R2_all1]...
    = a1F1(output_testA,output_testF1,output_trainA,output_trainF1);

%% 设置结束计时点2
tEnd2 = toc(tStart2);

%% 设置开始计时点3
tStart3 = tic;

%% 创建 ANN 与改进的标准粒子群算法相结合的 LPSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 LPSO 结合，LPSO-ANN 神经网络\n');

net2 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net2.trainFcn = 'trainbr';                                                    % 网络的训练函数选用trainlm
net2.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig
net2.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net2.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net2.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net2.trainParam.goal=0.01;                                                   % 训练目标最小误差，这里设置为0.0001
% net2.trainParam.show=25;                                                      % 显示频率，这里设置为每训练25次显示一次
net2.trainParam.showWindow = 0;

%% 调用改进的线性权重粒子群 LPSO 算法

[Best_posB,Best_scoreB,Convergence_curveB] = a1F4(net2);

%% 将优化好的权值与阈值赋值到神经网络 net2 中

w1_2 = Best_posB(:,1:1:inputnum * hiddennum_best);                                                         % 输入层到中间层的权值
B1_2 = Best_posB(:,(inputnum * hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best));        % 中间各层神经元阈值
w2_2 = Best_posB(:,(inputnum * hiddennum_best + hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum));
%中间层到输出层的权值
B2_2 = Best_posB(:,(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + 1)...
    :1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum));          % 输出层各神经元阈值
%矩阵重构
net2.iw{1,1} = reshape(w1_2,hiddennum_best,inputnum);
net2.lw{2,1} = reshape(w2_2,outputnum,hiddennum_best);
net2.b{1} = reshape(B1_2,hiddennum_best,1);
net2.b{2} = reshape(B2_2,outputnum,1);

%% 优化后的神经网络训练

net2=train(net2,input_trainC,output_trainC);                                  % 开始训练，其中input_trainC,output_trainC 分别为归一化后的训练集输入与输出样本

%% 优化后的神经网络测试
%预测
output_trainD2 = sim(net2,input_trainC);                                      % 训练结果仿真输出训练集  an2 为训练数据集的仿真输出
output_testD2 = sim(net2,input_testC);                                        % 测试结果仿真输出        bn2为测试数据集的仿真输出

%仿真结果的反归一化
output_trainE2 = mapminmax('reverse',output_trainD2,outputPS_train);
output_testE2 = mapminmax('reverse',output_testD2,outputPS_test);

% 进行转置
output_trainF2 = transpose(output_trainE2);                                   % output_trainF2 为转置后的预测的训练集输出
output_testF2 = transpose(output_testE2);                                     % output_testF2 为转置后的预测的测试集输出

% 计算误差
[mae_test2,mse_test2,rmse_test2,mape_test2,error_testA2,errorPercent_test2,R2_test2,...
    mae_train2,mse_train2,rmse_train2,mape_train2,error_trainA2,errorPercent_train2,R2_train2,...
    mae_all2,mse_all2,rmse_all2,mape_all2,error_allA2,errorPercent_all2,R2_all2]...
    = a1F1(output_testA,output_testF2,output_trainA,output_trainF2);

%% 设置结束计时点3
tEnd3 = toc(tStart3);


%% 设置开始计时点4
tStart4 = tic;

%% 创建 ANN 与改进的标准粒子群算法相结合的 LPSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 APSO 结合，APSO-ANN 神经网络\n');

net3 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net3.trainFcn = 'trainbr';                                                    % 网络的训练函数选用trainlm
net3.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig
net3.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net3.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net3.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net3.trainParam.goal=0.01;                                                   % 训练目标最小误差，这里设置为0.0001
% net2.trainParam.show=25;                                                      % 显示频率，这里设置为每训练25次显示一次
net3.trainParam.showWindow = 0;

%% 调用改进的线性权重粒子群 A1PSO 算法

[Best_posC,Best_scoreC,Convergence_curveC] = a1F5(net3);

%% 将优化好的权值与阈值赋值到神经网络 net3 中

w1_3 = Best_posC(:,1:1:inputnum * hiddennum_best);                                                         % 输入层到中间层的权值
B1_3 = Best_posC(:,(inputnum * hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best));        % 中间各层神经元阈值
w2_3 = Best_posC(:,(inputnum * hiddennum_best + hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum));
%中间层到输出层的权值
B2_3 = Best_posC(:,(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + 1)...
    :1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum));          % 输出层各神经元阈值
%矩阵重构
net3.iw{1,1} = reshape(w1_3,hiddennum_best,inputnum);
net3.lw{2,1} = reshape(w2_3,outputnum,hiddennum_best);
net3.b{1} = reshape(B1_3,hiddennum_best,1);
net3.b{2} = reshape(B2_3,outputnum,1);

%% 优化后的神经网络训练

net3 = train(net3,input_trainC,output_trainC);                                  % 开始训练，其中input_trainC,output_trainC 分别为归一化后的训练集输入与输出样本

%% 优化后的神经网络测试
%预测
output_trainD3 = sim(net3,input_trainC);                                      % 训练结果仿真输出训练集  an2 为训练数据集的仿真输出
output_testD3 = sim(net3,input_testC);                                        % 测试结果仿真输出        bn2为测试数据集的仿真输出

%仿真结果的反归一化
output_trainE3 = mapminmax('reverse',output_trainD3,outputPS_train);
output_testE3 = mapminmax('reverse',output_testD3,outputPS_test);

% 进行转置
output_trainF3 = transpose(output_trainE3);                                   % output_trainF2 为转置后的预测的训练集输出
output_testF3 = transpose(output_testE3);                                     % output_testF2 为转置后的预测的测试集输出

% 计算误差
[mae_test3,mse_test3,rmse_test3,mape_test3,error_testA3,errorPercent_test3,R2_test3,...
    mae_train3,mse_train3,rmse_train3,mape_train3,error_trainA3,errorPercent_train3,R2_train3,...
    mae_all3,mse_all3,rmse_all3,mape_all3,error_allA3,errorPercent_all3,R2_all3]...
    = a1F1(output_testA,output_testF3,output_trainA,output_trainF3);

%% 设置结束计时点4
tEnd4 = toc(tStart4);


%% 设置开始计时点5
tStart5 = tic;

%% 创建 ANN 与改进的标准粒子群算法相结合的 LPSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 APSO 结合，DPSO-ANN 神经网络\n');

net4 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net4.trainFcn = 'trainbr';                                                    % 网络的训练函数选用trainlm
net4.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig
net4.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net4.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net4.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net4.trainParam.goal=0.01;                                                   % 训练目标最小误差，这里设置为0.0001
% net2.trainParam.show=25;                                                      % 显示频率，这里设置为每训练25次显示一次
net4.trainParam.showWindow = 0;

%% 调用改进的线性权重粒子群 A1PSO 算法

[Best_posD,Best_scoreD,Convergence_curveD] = a1F6(net4);

%% 将优化好的权值与阈值赋值到神经网络 net3 中

w1_4 = Best_posC(:,1:1:inputnum * hiddennum_best);                                                         % 输入层到中间层的权值
B1_4 = Best_posC(:,(inputnum * hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best));        % 中间各层神经元阈值
w2_4 = Best_posC(:,(inputnum * hiddennum_best + hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum));
%中间层到输出层的权值
B2_4 = Best_posC(:,(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + 1)...
    :1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum));          % 输出层各神经元阈值
%矩阵重构
net4.iw{1,1} = reshape(w1_3,hiddennum_best,inputnum);
net4.lw{2,1} = reshape(w2_3,outputnum,hiddennum_best);
net4.b{1} = reshape(B1_3,hiddennum_best,1);
net4.b{2} = reshape(B2_3,outputnum,1);

%% 优化后的神经网络训练

net4 = train(net4,input_trainC,output_trainC);                                  % 开始训练，其中input_trainC,output_trainC 分别为归一化后的训练集输入与输出样本

%% 优化后的神经网络测试
%预测
output_trainD4 = sim(net4,input_trainC);                                      % 训练结果仿真输出训练集  an2 为训练数据集的仿真输出
output_testD4 = sim(net4,input_testC);                                        % 测试结果仿真输出        bn2为测试数据集的仿真输出

%仿真结果的反归一化
output_trainE4 = mapminmax('reverse',output_trainD4,outputPS_train);
output_testE4 = mapminmax('reverse',output_testD4,outputPS_test);

% 进行转置
output_trainF4 = transpose(output_trainE4);                                   % output_trainF2 为转置后的预测的训练集输出
output_testF4 = transpose(output_testE4);                                     % output_testF2 为转置后的预测的测试集输出

% 计算误差
[mae_test4,mse_test4,rmse_test4,mape_test4,error_testA4,errorPercent_test4,R2_test4,...
    mae_train4,mse_train4,rmse_train4,mape_train4,error_trainA4,errorPercent_train4,R2_train4,...
    mae_all4,mse_all4,rmse_all4,mape_all4,error_allA4,errorPercent_all4,R2_all4]...
    = a1F1(output_testA,output_testF4,output_trainA,output_trainF4);

%% 设置结束计时点5
tEnd5 = toc(tStart5);

%% 绘制粒子群算法的适应度进化曲线 用图片 f1 表示
% 参数的设定
f1x = 1:1:maxgen;
[f1x1,f1y1] = stairs(f1x,Convergence_curveA);
[f1x2,f1y2] = stairs(f1x,Convergence_curveB);
[f1x3,f1y3] = stairs(f1x,Convergence_curveC);
[f1x4,f1y4] = stairs(f1x,Convergence_curveD);

f1 = figure;
f1.Name  = '不同类型粒子群算法的迭代收敛曲线';                                % 图像的标题名称
subplot(1,1,1);                                                              % 标明作图

hold on;
f1p1 = plot(f1x1,f1y1);                                                      % 画出 PSO-ANN 的迭代收敛曲线
f1p1.Color = 'red';
f1p1.LineStyle = '-';
f1p1.LineWidth = 2;

f1p2 = plot(f1x2,f1y2);                                                      % 画出 LPSO-ANN 的迭代收敛曲线
f1p2.Color = 'blue';
f1p2.LineStyle = '-';
f1p2.LineWidth = 2;

f1p3 = plot(f1x3,f1y3);                                                      % 画出 APSO-ANN 的迭代收敛曲线
f1p3.Color = 'cyan';
f1p3.LineStyle = '-';
f1p3.LineWidth = 2;

f1p4 = plot(f1x4,f1y4);                                                      % 画出 APSO-ANN 的迭代收敛曲线
f1p4.Color = 'magenta';
f1p4.LineStyle = '-';
f1p4.LineWidth = 2;

axis padded;                                                                 % 设置图像的坐标范围
xlabel('迭代次数');
ylabel('适应度值');
legend('PSO','LPSO','APSO','DPSO');
title('不同类型粒子群算法的迭代收敛曲线');

%% 绘制优化因子的散点图
% 参数的设定
f2x = 1:1:dim;

f2y1 = Best_pos0;
f2y2 = Best_posA;
f2y3 = Best_posB;
f2y4 = Best_posC;
f2y5 = Best_posD;


f2 = figure;
f2.Name  = '不同神经网络的优化因子散点值';                                    % 图像的标题名称
subplot(1,1,1);                                                              % 标明作图

hold on;
f2p1 = scatter(f2x,f2y1); 
f2p1.Marker = 'o';
f2p1.LineWidth = 0.6;
f2p1.SizeData = 30;
f2p1.MarkerEdgeColor = 'black';
f2p1.MarkerFaceColor = 'blue';



f2p2 = scatter(f2x,f2y2); 
f2p2.Marker = 'd';
f2p2.LineWidth = 0.6;
f2p2.SizeData = 30;
f2p2.MarkerEdgeColor = 'black';
f2p2.MarkerFaceColor = 'red';



f2p3 = scatter(f2x,f2y3);
f2p3.Marker = 's';
f2p3.LineWidth = 0.6;
f2p3.SizeData = 30;
f2p3.MarkerEdgeColor = 'black';
f2p3.MarkerFaceColor = 'magenta';


f2p4 = scatter(f2x,f2y4);
f2p4.Marker = 'v';
f2p4.LineWidth = 0.6;
f2p4.MarkerEdgeColor = 'black';
f2p4.MarkerFaceColor = 'green';

f2p5 = scatter(f2x,f2y5);
f2p5.Marker = '^';
f2p5.LineWidth = 0.6;
f2p5.MarkerEdgeColor = 'black';
f2p5.MarkerFaceColor = 'cyan';

axis padded;                                                                 % 设置图像的坐标范围
xlabel('优化因子序列');
ylabel('优化因子大小');
legend('ANN','SPSO','LPSO','APSO','DPSO');
title('不同神经网络的优化因子的数字大小');

%% 数据的保存
save('d1Z3.mat','uz1','uz2','ra1','n_use','scaA','list_trainA','list_testA',...
    'input_trainA','input_testA','output_trainA','output_testA','output_trainB','output_testB',...
    'net0','output_trainE0','output_testE0','output_trainF0','output_testF0','Best_pos0',...
    'popsize','maxgen','c1','c2','uxv','xmaxA','wmax','wmin',...
    'net1','output_trainE1','output_testE1','output_trainF1','output_testF1','Best_posA','Convergence_curveA',...
    'net2','output_trainE2','output_testE2','output_trainF2','output_testF2','Best_posB','Convergence_curveB',...
    'net3','output_trainE3','output_testE3','output_trainF3','output_testF3','Best_posC','Convergence_curveC',...
    'net4','output_trainE4','output_testE4','output_trainF4','output_testF4','Best_posD','Convergence_curveD');                                                                               
 
 % 将读取的变量保存在 d1Z2.mat 数据文件里面
                                                                                        % 读取的变量保存在 d1Z2.mat 数据文件里面
%% 设置结束计算时间点0
tEnd0 = toc(tStart0);

%% 计时结果输出
fprintf('****************************************************************************************************\n');

fprintf('程序运行结束\n');
fprintf('计时结果输出：\n');

fprintf('\n标准 ANN 神经网络用时 ：                                                   %d 分 , %6.2f 秒\n',...
    floor(tEnd1/60), rem(tEnd1,60));                                              % 标准BP神经网络的所用的时间

fprintf('\n标准 ANN 神经网络与标准的 PSO 组合神经网络 PSO-ANN 用时 ：                  %d 分 , %6.2f 秒\n',...
    floor(tEnd2/60), rem(tEnd2,60));                                              % PSO-ANN 神经网络的所用的时间

fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 LPSO 结合神经网络 LPSO-ANN 用时 ：     %d 分 , %6.2f 秒\n',...
    floor(tEnd3/60), rem(tEnd3,60));                                              % LPSO-ANN 神经网络的所用的时间

fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 LPSO 结合神经网络 APSO-ANN 用时 ：     %d 分 , %6.2f 秒\n',...
    floor(tEnd4/60), rem(tEnd4,60));                                              % APSO-ANN 神经网络的所用的时间

fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 LPSO 结合神经网络 APSO-ANN 用时 ：     %d 分 , %6.2f 秒\n',...
    floor(tEnd5/60), rem(tEnd5,60));                                              % DPSO-ANN 神经网络的所用的时间

fprintf('\n总共用时 ：                                                                %d 分 , %6.2f 秒\n',...
    floor(tEnd0/60), rem(tEnd0,60));                                              % 程序的总用时间

fprintf('********************************************************************************************************\n');

