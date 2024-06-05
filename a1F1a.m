%% a1F1a:此程序为调用的函数，用于计算误差指标，此程序输入测试集和训练集的真实值和预测值，输出预测的误差指标
%{

1.输入参数说明：
out_test：测试集输出的真实值
sim_test：测试集输出的预测值
out_train：训练集输出的真实值
sim_train：训练集输出的预测值

2.输出参数说明：
rmse_test：测试集的均方误差根（是预测值与实际值偏差的平方和与样本总数的比值的平方根，也就是mse开根号，用来衡量预测值同实际值之间的偏差）
R2_test：测试集的相关系数

%}

%% 函数主体
function [rmse_test,R2_test] = a1F1a(out_test,sim_test,out_train,sim_train)

%% 对函数的输入变量进行限制
if nargin == 4                                                               % 确保函数的输入变量的个数为 4 个
    
    sz1 = size(out_test,1,2);
    a1 = min(sz1);                                                           % 定义 a1 为函数输入变量 out_test 的最小维度
    
    sz2 = size(sim_test,1,2);
    a2 = min(sz2);                                                           % 定义 a2 为函数输入变量 sim_test 的最小维度
    
    sz3 = size(out_train,1,2);
    a3 = min(sz3);                                                           % 定义 a3 为函数输入变量 out_train 的最小维度
    
    sz4 = size(sim_train,1,2);
    a4 = min(sz4);                                                           % 定义 a4 为函数输入变量 sim_train 的最小维度
    
    a5 = [a1,a2,a3,a4];
    a6 = max(a5);                                                            % 定义 a6 为所有函数输入变量的最小维度的最大值
    a7 = min(a5);                                                            % 定义 a7 为所有函数输入变量的最小维度的最小值
    
    if a6 == 1 && a7 == 1                                                    % 确保所有函数输入变量的最小维度均为 1
        
        if size(out_test,2) == 1
            out_test = transpose(out_test);                                  % 如果输入变量 out_test 为列向量，将其转换为行向量
        end
        
        if size(sim_test,2) == 1
            sim_test = transpose(sim_test);                                  % 如果输入变量 sim_test 为列向量，将其转换为行向量
        end
        
        %% 测试集评价指标的计算
                
        num_test = size(out_test,2);                                         % 统计测试样本数量为 num_test
        error_testA = out_test - sim_test;                                   % 计算测试样本输出预测值与输出真实值之间的误差为 error_testA
        mse_test = sum(error_testA .* error_testA) / num_test;               % 计算测试集输出预测值的均方误差为  mse_test
        rmse_test = sqrt(mse_test);                                          % 计算测试集输出预测值的均方误差根为 rmse_test
        
        mean_testA = mean(out_test);                                         % 计算测试集输出真实值的平均值为 mean_testA
        ones_test = ones(size(out_test),'like',out_test);                    % ones_test 为具有和测试集输出真实值相同维度的单位全一向量
        mean_testB = ones_test .* mean_testA;                                % mean_testB 为元素值为测试集输出真实值的平均值且维度与测试集输出真实值相同的向量
        error_testB = out_test - mean_testB;                                 % 测试集输出的真实值与预测值的平均数的误差向量为 error_testB
        se_test = sum(error_testB .* error_testB) / num_test;                % 测试集输出的 se
        R2_test = 1 - (mse_test / se_test);                                  % 测试集输出的 R2
        R2_testA = R2_test * 100;                                            % 将测试集输出的 R2 扩大 100 倍
        
        %% 误差指标结果输出
 
        % 测试集结果输出
        fprintf('\n测试集结果输出：\n');
        fprintf('测试集的均方误差根rmse_test为：--------   %.4f\n',rmse_test);
        fprintf('测试集的测试集的R2 R2_train为：--------   %.4f %%\n',R2_testA);
          
    else
        fprintf('\n函数调用方法有误，请检查每个输入参数的维度\n');
    end
    
else
    fprintf('\n函数调用方法有误，请检查输入参数的个数\n');
end

end

