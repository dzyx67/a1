%% a1F4:此程序为调用的函数，为改进的线性权重粒子群 DPSO 的算法
%{

1.输入参数说明：
popsize：初始种群规模
maxgen：最大进化代数
inputnum：输入层神经元个数
hiddennum_best：隐含层神经元个数
outputnum：输出层神经元个数
c1：每个粒子的个体学习因子，也称为个体加速常数
c2：每个粒子的社会学习因子，也称为社会加速常数
w：惯性权重
uxv：最大速度与上限的比值
xmaxA：权值的正上限


2.输出参数说明：
Best_pos：全局最优粒子
Best_score：最好的得分
Convergence_curve1：进化曲线
%}

%% 函数主体
function [Best_pos,Best_score,Convergence_curve] = a1F6(net)

%% 加载主函数中保存的变量
popsize = evalin('base','popsize');
maxgen = evalin('base','maxgen');
c1 = evalin('base','c1');
c2 = evalin('base','c2');
uxv = evalin('base','uxv');
wmax = evalin('base','wmax');
wmin = evalin('base','wmin');
xmaxA = evalin('base','xmaxA');
dim = evalin('base','dim');

%% 对函数的输入变量进行限制
if nargin == 1                                                                 % 确保函数的输入变量的个数为 4 个
    
    %% 粒子群算法
    % 中间参数的设定
    
    xminA = -1 * xmaxA;                                                        % 粒子群位置的下限为 xx
    
    xmaxB = repmat(xmaxA,1,dim);                                               % 自变量上限
    xminB = repmat(xminA,1,dim);                                               % 自变量下限
    
    vmaxA = uxv * xmaxA;                                                       % 单个粒子的最大速度
    vminA = uxv * xminA;                                                       % 单个粒子的最小速度
    
    vmaxB = vmaxA .* ones(popsize,dim);                                        % 将粒子的最大速度增加维度
    vminB = vminA .* ones(popsize,dim);
    
    %% 计算第0代PSO的适应度值
    x = zeros(popsize,dim);                                                    % 将粒子群的初始位置设置为原点
    
    for i = 1:1:popsize
        x(i,:) = xminB + (xmaxB - xminB) .* rand(1,dim);                       % 随机初始化粒子群所在的位置在定义域内
    end
    
    v = vminB + (vmaxB - vminB) .* rand(popsize,dim);                          % 随机初始化粒子群的步长（设置为在[vmin,vmax]之间的一个随机数）
    
    % 计算适应度
    lossA = ones(popsize,1);                                                   % 将粒子群的群体适应度归 1 （可以省掉这一行）
    
    for i = 1:1:popsize                                                        % 计算初始化粒子群的群体适应度
        lossA(i,:) = a1F3(x(i,:),net);
    end
    
    pbest = x;                                                                 % 初始化这n个粒子迄今为止找到的最佳位置
    
    lossB = lossA;                                                             % lossB 为粒子群的历史局部最优适应度
    
    
    lossC = min(lossA);                                                        % 定义粒子群全局历史最佳适应度值为 lossC
    
    m = find(lossA == lossC,1);                                                % 找到适应度最小的那个粒子的下标为 k evalin assignin
    
    gbest = x(m,:);                                                            % 定义最佳的粒子为 gbest
    
    
    
    
    %% 开始粒子的更新迭代
    Convergence_curve = lossC .* ones(maxgen,1);                               % 初始化每次迭代得到的最佳的适应度
    h0 = waitbar(0,'1','Name','DPSO-ANN optimization...',...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(h0,'canceling',0);
    for d = 1:1:maxgen                                                         % 开始迭代，一共迭代 maxgen 次
        
        % 对 DPSO 的权重进行更新
        r = 10;
        
        k = ((r + maxgen) / maxgen) .^ (-d);
        
        w = wmin + k .* (wmax - wmin);
        
        for i = 1:1:popsize                                                    % 依次更新第i个粒子的速度与位置
            v(i,:) = w .* v(i,:) + c1 * rand(1) .* (pbest(i,:) - x(i,:)) + c2 * rand(1) .* (gbest - x(i,:));          % 更新第i个粒子的步长
            
            %% 对粒子的步长大小与位置范围进行限制
            
            % 如果粒子的速度超过了最大速度限制，就对其进行调整
            for j = 1:1:dim
                if v(i,j) < vminB(i,j)
                    v(i,j) = vminB(i,j);
                elseif v(i,j) > vmaxB(i,j)
                    v(i,j) = vmaxB(i,j);
                end
            end
            
            x(i,:) = x(i,:) + v(i,:);                                          % 更新第i个粒子的位置
            
            % 如果粒子的位置超出了定义域，就对其进行调整
            for j = 1:1:dim
                if x(i,j) < xminB(:,j)
                    x(i,j) = xminB(:,j);
                elseif x(i,j) > xmaxB(:,j)
                    x(i,j) = xmaxB(:,j);
                end
            end
            
            %% 更新当前适应度、局部最优粒子与全局最优粒子
            
            lossA(i,:) = a1F3(x(i,:),net);                                       % 迭代过程中第 i 个粒子的适应度为 lossA_pso1(i,:)
            
            % 更新局部最优粒子位置
            
            if lossA(i,:) < lossB(i,:)
                
                lossB(i,:) = lossA(i,:);                                          % 更新这一代中局部最优适应度 lossA_pso1(i,:)
                
                pbest(i,:) = x(i,:);                                              % 更新这一代中最佳的粒子位置
                
            end
            
            % 更新全局最优粒子位置
            
            if  lossA(i,:) < lossC
                
                
                lossC = lossA(i,:);
                
                gbest = pbest(i,:);                                              % 那就更新所有粒子迄今为止找到的最佳位置
                
            end
            
        end                                                                      % 第i个粒子更新完毕
        
        Convergence_curve(d,:) = lossC;                                          % 将这一代中的全局最优适应度值赋值给 Convergence_curve1(d,:)
        
        
        if getappdata(h0,'canceling')                                            % 添加取消按钮功能
            break
        end
        
        waitbar(d/maxgen,h0,sprintf('%8.2f %%',d/maxgen*100));
    end
    
    delete(h0);
    Best_pos = gbest;
    Best_score = Convergence_curve(end,:);
    
else
    fprintf('\n函数调用方法有误，请检查输入参数的个数\n');
end

end

