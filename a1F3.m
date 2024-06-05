%% a1F3:此程序为调用的函数，为粒子群的适应度函数
%{

1.输入参数说明：
xall：需要优化的权值与阈值的向量
inputnum：输入参数的个数
hiddennum_best：隐含层神经元个数
outputnum：输出层神经元个数
net：人工神经网络模型
output_trainB：转置后的训练集输出
output_testB：转置后的测试集输出
input_trainC：归一化后的训练集输入
input_testC：归一化后的测试集输入
output_trainC：归一化后的训练集输出
outputPS_train：训练集输出归一化设定结构体
outputPS_test：测试集输出归一化设定结构体

2.输出参数说明：
error_pso：适应度函数的损失函数的值

%}

%% 函数主体
function loss_pso = a1F3(xall,net)

%% 加载主函数中保存的变量
hiddennum_best = evalin('base','hiddennum_best');
inputnum = evalin('base','inputnum');
outputnum = evalin('base','outputnum');
output_trainB = evalin('base','output_trainB');
output_testB = evalin('base','output_testB');
input_trainC = evalin('base','input_trainC');
input_testC = evalin('base','input_testC');
output_trainC = evalin('base','output_trainC');
outputPS_train = evalin('base','outputPS_train');
outputPS_test = evalin('base','outputPS_test');

%% 对函数的输入变量进行限制
if nargin == 2                                                               % 确保函数的输入变量的个数为 2 个   
    
    sz1 = size(output_trainB,1,2);
    a1 = min(sz1);                                                           % 定义 a1 为函数输入变量 out_test 的最小维度
    
    sz2 = size(output_testB,1,2);
    a2 = min(sz2);                                                           % 定义 a2 为函数输入变量 sim_test 的最小维度
    
    sz3 = size(output_trainC,1,2);
    a3 = min(sz3);                                                           % 定义 a3 为函数输入变量 out_train 的最小维度
    
    a4 = [a1,a2,a3];
    a5 = max(a4);                                                            % 定义 a6 为所有函数输入变量的最小维度的最大值
    a6 = min(a5);                                                            % 定义 a7 为所有函数输入变量的最小维度的最小值
    
    if a5 == 1 && a6 == 1                                                    % 确保所有函数输入变量的最小维度均为 1
        
        if size(output_trainB,2) == 1
            output_trainB = transpose(output_trainB);                        % 如果输入变量 output_trainB 为列向量，将其转换为行向量
        end
        
        if size(output_testB,2) == 1
            output_testB = transpose(output_testB);                          % 如果输入变量 output_testB 为列向量，将其转换为行向量
        end
        
        if size(output_trainC,2) == 1
            output_trainC = transpose(output_trainC);                        % 如果输入变量 output_trainC 为列向量，将其转换为行向量
        end
               
        %% 粒子群算法的适应度函数
        % 对权值与阈值进行赋值
        w1 = xall(:,1:1:inputnum * hiddennum_best);                                                           % 取到输入层与隐含层连接的权值   randi
        B1 = xall(:,(inputnum * hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best));          % 隐含层神经元阈值
        w2 = xall(:,(inputnum * hiddennum_best+hiddennum_best + 1):1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum));              
        % 取到隐含层与输出层连接的权值
        B2 = xall(:,(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum+1)...
            :1:(inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum));   
        % 输出层神经元阈值
        net.trainParam.showWindow = 0;                                               % 隐藏仿真界面
        
        % 网络权值赋值
        net.iw{1,1} = reshape(w1,hiddennum_best,inputnum);                           % 将 w1 由1行 inputnum * hiddennum 列转为 hiddennum 行 inputnum 列的二维矩阵
        net.lw{2,1} = reshape(w2,outputnum,hiddennum_best);                          % 更改矩阵的保存格式
        net.b{1} = reshape(B1,hiddennum_best,1);                                     % 1 行hiddennum 列，为隐含层的神经元阈值
        net.b{2} = reshape(B2,outputnum,1);
        
        % 网络训练
        net = train(net,input_trainC,output_trainC);
        
        % 预测
        output_trainD2 = sim(net,input_trainC);                                      % output_trainD2 为未反归一化的训练集输出
        output_testD2 = sim(net,input_testC);                                        % output_testD2 为未反归一化的测试集输出
      
        %仿真结果的反归一化
        output_trainE2 = mapminmax('reverse',output_trainD2,outputPS_train);         % output_trainE2 为反归一化的训练集输出
        output_testE2 = mapminmax('reverse',output_testD2,outputPS_test);            % output_testE2 为反归一化的测试集输出
                
        % loss_pso = (mse(output_trainB,output_trainE2) + mse(output_testB,output_testE2)) / 2; 
        
        loss_pso = mse(output_testB,output_testE2); 
        
        %适应度函数选取为训练集与测试集整体的均方误差平均值，适应度函数值越小，表明训练越准确，且兼顾模型的预测精度更好。
        
    else
        fprintf('\n函数调用方法有误，请检查每个输入参数的维度\n');
    end
    
else
    fprintf('\n函数调用方法有误，请检查输入参数的个数\n');
end

end

