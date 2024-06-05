% a1Z1:此程序的目的是为了对参数不经过滤波处理，随机挑选500个样本，进行水平力预测，采用标准ANN神经网络与标准的PSO

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
net0.trainFcn = 'trainlm';                                                    % 网络的训练函数选用trainlm     newff
net0.layers{1}.transferFcn='logsig';                                          % 网络的第一层传递函数选用tansig      tansig    purelin    logsig
net0.layers{2}.transferFcn='purelin';                                         % 网络的第一层传递函数选用tansig
net0.trainParam.epochs=1000;                                                  % 训练次数，这里设置为1000次
net0.trainParam.lr = 0.1;                                                     % 学习速率，这里设置为0.01
net0.trainParam.goal=0.01;                                                   % 训练目标最小误差，这里设置为0.0001
% net1.trainParam.show=25;                                                      % 显示频率，这里设置为每训练25次显示一次
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

%% 设置结束计时点1
tEnd1 = toc(tStart1);

%% 设置粒子群算法的通用参数
popsize = 5;                                                                 % 初始种群规模
maxgen = 30;                                                                 % 最大进化代数
c1 = 2;                                                                       % 每个粒子的个体学习因子，也称为个体加速常数
c2 = 2;                                                                       % 每个粒子的社会学习因子，也称为社会加速常数
uxv = 0.1;                                                                 % 最大速度与上限的比值
xmaxA = 3;                                                                       % 粒子位置的上限
wmax = 0.9;                                                                   % 权重的推荐最大值
wmin = 0.4;                                                                   % 权重的推荐最小值

%% 设置开始计时点2
tStart2 = tic;

%% 创建 ANN 与标准粒子群算法相结合的 PSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与标准的粒子群算法 PSO 结合，PSO-ANN 神经网络\n');

net1 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net1.trainFcn = 'trainlm';                                                    % 网络的训练函数选用trainlm
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

%% 设置结束计时点2
tEnd2 = toc(tStart2);

%% 设置开始计时点3
tStart3 = tic;

%% 创建 ANN 与改进的标准粒子群算法相结合的 LPSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 LPSO 结合，LPSO-ANN 神经网络\n');

net2 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net2.trainFcn = 'trainlm';                                                    % 网络的训练函数选用trainlm
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


%% 设置结束计时点3
tEnd3 = toc(tStart3);


%% 设置开始计时点4
tStart4 = tic;


%% 创建 ANN 与改进的标准粒子群算法相结合的 LPSO-ANN 神经网络，并且设置参数
fprintf('\n标准 ANN 神经网络与线性权值粒子群算法 APSO 结合，APSO-ANN 神经网络\n');

net3 = newff(input_trainC,output_trainC,hiddennum_best);                      % 初步创建人工神经网络模型 net2

% 网络训练函数与传输函数的设置
net3.trainFcn = 'trainlm';                                                    % 网络的训练函数选用trainlm
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




%% 设置结束计时点4
tEnd4 = toc(tStart4);


%% 绘制粒子群算法的适应度进化曲线 用图片 f1 表示
% 参数的设定
f1x = 1:1:maxgen;
[f1x1,f1y1] = stairs(f1x,Convergence_curveA);
[f1x2,f1y2] = stairs(f1x,Convergence_curveB);
[f1x3,f1y3] = stairs(f1x,Convergence_curveC);

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

f1p3 = plot(f1x3,f1y3);                                                      % 画出 LPSO-ANN 的迭代收敛曲线
f1p3.Color = 'cyan';
f1p3.LineStyle = '-';
f1p3.LineWidth = 2;


axis padded;                                                                 % 设置图像的坐标范围
xlabel('迭代次数');
ylabel('适应度值');
legend('PSO','LPSO','A1PSO');
title('不同类型粒子群算法的迭代收敛曲线');



%% 数据的保存
save('d1Z2.mat','uz1','uz2','ra1','n_use',...
    'output_trainA','output_testA',...
    'output_trainB','output_testB',...
    'net1','output_trainE1','output_testE1',...
    'output_trainF1','output_testF1',...
    'net2','output_trainE2','output_testE2',...
    'output_trainF2','output_testF2');                                          % 将读取的变量保存在 d1Z2.mat 数据文件里面

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
    floor(tEnd3/60), rem(tEnd3,60));                                              % APSO-ANN 神经网络的所用的时间

fprintf('\n总共用时 ：                                                                %d 分 , %6.2f 秒\n',...
    floor(tEnd0/60), rem(tEnd0,60));                                              % 程序的总用时间

fprintf('********************************************************************************************************\n');

