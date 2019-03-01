close all; clear; clc; 
sprintf("Loading mnist-zeroshot datasets...");
opts.dirs.data = '/home/chenshen/Projects/Hash/OnlineHash/data';
opts.unsupervised = 0;
opts.nbits = 64;
normalizeX = 0;
opts.unseen = 1


%% 数据处理
% 根据unseen标志选择对应的数据集
if opts.unseen == 1
    DS = Datasets.places_zs(opts, normalizeX);
else
    DS = Datasets.places(opts, normalizeX);
end

% 训练集
trainCNN = double(DS.Xtrain);
trainLabels = DS.Ytrain;
%train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));   % mapped into a sphere space
train = trainCNN';
trainLabel = trainLabels;
[Ntrain, Dtrain] = size(trainCNN);

% 测试集
testCNN = double(DS.Xtest);
testLabels = DS.Ytest;
%test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  % mapped into a sphere space
test = testCNN';
testLabel = testLabels;
[Ntest, Dtest] = size(testCNN);

% 检索集
if opts.unseen == 1
    retrievalCNN = double(DS.Xretrieval);
    retrievalLabels = DS.Yretrieval;
    %retrieval = retrievalCNN ./ sqrt(sum(retrievalCNN .* retrievalCNN, 2));  % mapped into a sphere space
    retrieval = retrievalCNN';
    retrievalLabel = retrievalLabels;
    [Nretrieval, Dretrieval] = size(retrievalCNN);
end

clear trainCNN testCNN retrievalCNN trainLabels testLabels retrievalLabels


%% 参数初始化
n_t = 5000;
alpha = 1e-2;
sigma = 0.5;  % places 上0.4上下调整, 一般情况下，sigma越大loss越小,反之越大. 
lr = 2;
threshold = 1e-4;
pos = 60;
neg = 10;

W_t = randn(Dtest, opts.nbits);
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1);
training_size = 100000;

results = [];

tic;
for t = 1:n_t:training_size
    % X_t = train(:, 1:n_t);
    % L_t = trainLabel(1:n_t, :);
    X_t = train(:, t:t+n_t-1);
    L_t = trainLabel(t:t+n_t-1, :);

    P = zeros(n_t, n_t);
    if size(L_t, 2) > 1
        num1 = 1./ sqrt(sum(L_t .* L_t, 2));
        num1(isinf(num1) | isnan(num1)) = 1;
        L_T = diag(num1) * L_t;
        
        P = L_T * L_T';   % cosine  for multi-label cases
    else
        P = double(L_t == L_t');
        P = normpdf(P, 1, sigma);
        S = double(L_t == L_t');
        % P(P == 1) = 0.4;
        % P(P == 0) = 0.001;
    end
    [ydata, W_t, obj, Q, P] = minKLD(alpha/n_t, P, opts.nbits, X_t, W_t, S, lr, threshold, pos, neg);
    
end
toc;


%% 评价指标
Htest = single(W_t' * test > 0);
if opts.unseen == 1
    Htrain = single(W_t' * retrieval > 0);
    Aff = affinity([], [], retrievalLabel, testLabel, opts);
else
    Htrain = single(W_t' * train > 0);
    Aff = affinity([], [], trainLabel, testLabel, opts);
end

opts.metric = 'prec_k1';
opts.prec_k = 1;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_n2';
opts.prec_n = 2;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'mAP';
res = evaluate(Htrain', Htest', opts, Aff);

% results = [results; res];
% auc   = arrayfun(@(x) mean(x), results);
% final = arrayfun(@(x) x(end) , results);
% logInfo('');
% logInfo('  AUC mAP: %.3g +/- %.3g', mean(auc), std(auc));
% logInfo('FINAL mAP: %.3g +/- %.3g', mean(final), std(final));

clear;