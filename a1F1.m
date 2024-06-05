%% a1F1:此程序为调用的函数，用于计算误差指标，此程序输入测试集和训练集的真实值和预测值，输出预测的误差指标
%{

1.输入参数说明：
out_test：测试集输出的真实值
sim_test：测试集输出的预测值
out_train：训练集输出的真实值
sim_train：训练集输出的预测值

2.输出参数说明：
mae_test：测试集的绝对误差（是绝对误差的平均值，反映预测值误差的实际情况.）
mse_test：测试集的均方误差（是预测值与实际值偏差的平方和与样本总数的比值）
rmse_test：测试集的均方误差根（是预测值与实际值偏差的平方和与样本总数的比值的平方根，也就是mse开根号，用来衡量预测值同实际值之间的偏差）
mape_test：测试集的平均绝对百分比误差（是预测值与实际值偏差绝对值与实际值的比值，取平均值的结果，可以消除量纲的影响，用于客观的评价偏差）
error_test：测试集的误差
errorPercent_test：测试集的误差百分比
R2_test：测试集的相关系数

mae_train：训练集的绝对误差（是绝对误差的平均值，反映预测值误差的实际情况.）
mse_train：训练集的均方误差（是预测值与实际值偏差的平方和与样本总数的比值）
rmse_train：训练集的均方误差根（是预测值与实际值偏差的平方和与样本总数的比值的平方根，也就是mse开根号，用来衡量预测值同实际值之间的偏差）
mape_train：训练集的平均绝对百分比误差（是预测值与实际值偏差绝对值与实际值的比值，取平均值的结果，可以消除量纲的影响，用于客观的评价偏差）
error_train：训练集的误差
errorPercent_train：训练集的误差百分比
R2_train：训练集的相关系数

%}

%% 函数主体
function [mae_test,mse_test,rmse_test,mape_test,error_testA,errorPercent_test,R2_test,...
    mae_train,mse_train,rmse_train,mape_train,error_trainA,errorPercent_train,R2_train,...
    mae_all,mse_all,rmse_all,mape_all,error_allA,errorPercent_all,R2_all]...
    = a1F1(out_test,sim_test,out_train,sim_train)

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
        
        if size(out_train,2) == 1
            out_train = transpose(out_train);                                % 如果输入变量 out_train 为列向量，将其转换为行向量
        end
        
        if size(sim_train,2) == 1
            sim_train = transpose(sim_train);                                % 如果输入变量 sim_train 为列向量，将其转换为行向量
        end
        
        %% 评价指标的计算
        % 计算测试集的评价指标
        
        num_test = size(out_test,2);                                         % 统计测试样本数量为 num_test
        error_testA = out_test - sim_test;                                   % 计算测试样本输出预测值与输出真实值之间的误差为 error_testA
        errorPercent_test = abs(error_testA) ./ out_test;                    % 计算测试集每个样本的绝对百分比误差为 errorPercent_test
        
        mae_test = sum(abs(error_testA)) / num_test;                         % 计算测试集输出预测值的平均绝对误差为  mae_test
        mse_test = sum(error_testA .* error_testA) / num_test;               % 计算测试集输出预测值的均方误差为  mse_test
        rmse_test = sqrt(mse_test);                                          % 计算测试集输出预测值的均方误差根为 rmse_test
        mape_test = mean(errorPercent_test);                                 % 计算测试集输出预测值的平均绝对百分比误差为 mape_test
        
        mean_testA = mean(out_test);                                         % 计算测试集输出真实值的平均值为 mean_testA
        ones_test = ones(size(out_test),'like',out_test);                    % ones_test 为具有和测试集输出真实值相同维度的单位全一向量
        mean_testB = ones_test .* mean_testA;                                % mean_testB 为元素值为测试集输出真实值的平均值且维度与测试集输出真实值相同的向量
        error_testB = out_test - mean_testB;                                 % 测试集输出的真实值与预测值的平均数的误差向量为 error_testB
        se_test = sum(error_testB .* error_testB) / num_test;                % 测试集输出的se
        R2_test = 1 - (mse_test / se_test);                                  % 测试集输出的R2
        
        % 计算训练集的评价指标
        
        num_train = size(out_train,2);                                       % 统计训练样本数量为 num_train
        error_trainA = out_train - sim_train;                                % 计算训练样本输出预测值与输出真实值之间的误差为  error_trainA
        errorPercent_train = abs(error_trainA) ./ out_train;                 % 计算训练集每个样本的绝对百分比误差为 errorPercent_train
        
        mae_train = sum(abs(error_trainA)) / num_train;                      % 计算训练集输出预测值的平均绝对误差为  mae_train
        mse_train = sum(error_trainA.*error_trainA) / num_train;             % 计算训练集输出预测值的均方误差为  mse_train
        rmse_train = sqrt(mse_train);                                        % 计算训练集输出预测值的均方误差根为 rmse_train
        mape_train = mean(errorPercent_train);                               % 计算训练集输出预测值的平均绝对百分比误差为 mape_train
        
        mean_trainA = mean(out_train);                                       % 计算训练集输出真实值的平均值为 mean_testA
        ones_train = ones(size(out_train),'like',out_train);                 % ones_train 为具有和训练集输出真实值相同维度的单位全一向量
        mean_trainB = ones_train .* mean_trainA;                             % mean_trainB 为元素值为训练集输出真实值的平均值且维度与训练集输出真实值相同的向量
        error_trainB = out_train - mean_trainB;                              % 训练集输出的真实值与预测值的平均数的误差向量为 error_trainB
        se_train = sum(error_trainB .* error_trainB) / num_train;            % 训练集的se
        R2_train = 1 - (mse_train / se_train);                               % 训练集的R2
        
        % 计算数据集的评价指标
        
        num_all = num_test + num_train;                                      % 统计全部样本数量为 num_train
        out_all = [out_train,out_test];                                      % 全部样本的输出向量为 out_all
        error_allA = [error_trainA,error_testA];                             % 计算全部样本输出预测值与输出真实值之间的误差为  error_trainA
        errorPercent_all = abs(error_allA) ./ out_all;                       % 计算全集每个样本的绝对百分比误差为 errorPercent_all
        
        mae_all = sum(abs(error_allA)) / num_all;                            % 计算全集输出预测值的平均绝对误差为  mae_all
        mse_all = sum(error_allA .* error_allA) / num_all;                   % 计算全集输出预测值的均方误差为  mse_all
        rmse_all = sqrt(mse_all);                                            % 计算全集输出预测值的均方误差根为 rmse_all
        mape_all = mean(errorPercent_all);                                   % 计算全集输出预测值的平均绝对百分比误差为  mape_all
        
        mean_allA = mean(out_all);                                           % 计算全集输出的平均值为 mean_testA
        ones_all = ones(size(out_all),'like',out_all);                       % ones_all 为具有和全集输出真实值相同维度的单位全一向量
        mean_allB = ones_all .* mean_allA;                                   % mean_allB 为元素值为全集输出真实值的平均值且维度与训练集输出真实值相同的向量
        error_allB = out_all - mean_allB;                                    % 全集输出的真实值与预测值的平均数的误差向量为 error_allB
        se_all = sum(error_allB .* error_allB) / num_all;                    % 全集的se
        R2_all = 1 - (mse_all / se_all);                                     % 全集的R2
        
        %% 误差指标结果输出
        % 训练集结果输出
        fprintf('\n误差指标结果输出：\n');
        fprintf('\n训练集结果输出：\n');
        fprintf('训练集的平均绝对误差mae_train为：------   %.4f\n',mae_train);
        fprintf('训练集的均方误差mse_train为：----------   %.4f\n',mse_train);
        fprintf('训练集的均方误差根rmse_train为：-------   %.4f\n',rmse_train);
        fprintf('训练集的平均绝对百分比误差mape_train为：  %.4f %%\n',mape_train*100);
        fprintf('训练集的测试集的R2 R2_train为：--------   %.4f\n',R2_train);
        
        % 测试集结果输出
        fprintf('\n测试集结果输出：\n');
        fprintf('训练集的平均绝对误差mae_train为：------   %.4f\n',mae_test);
        fprintf('测试集的均方误差mse_test为：-----------   %.4f\n',mse_test);
        fprintf('测试集的均方误差根rmse_test为：--------   %.4f\n',rmse_test);
        fprintf('测试集的平均绝对百分比误差mape_test为：   %.4f %%\n',mape_test*100);
        fprintf('测试集的测试集的R2 R2_train为：--------   %.4f\n',R2_test);
        
        % 总数据集结果输出
        fprintf('\n数据集结果输出：\n');
        fprintf('数据集的平均绝对误差mae_all为：--------   %.4f\n',mae_all);
        fprintf('数据集的均方误差mse_all为：------------   %.4f\n',mse_all);
        fprintf('数据集的均方误差根rmse_all为：---------   %.4f\n',rmse_all);
        fprintf('数据集的平均绝对百分比误差mape_all为：-   %.4f %%\n',mape_all*100);
        fprintf('数据集的测试集的R2 R2_all为：----------   %.4f\n',R2_all);
        
    else
        fprintf('\n函数调用方法有误，请检查每个输入参数的维度\n');
    end
    
else
    fprintf('\n函数调用方法有误，请检查输入参数的个数\n');
end

end

