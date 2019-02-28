close all; clear; clc; 
sprintf("Loading cifar-zeroshot datasets...");
opts.dirs.data = '/home/chenshen/Projects/Hash/OnlineHash/data';
opts.unsupervised = 0;
opts.nbits = 8;
normalizeX = 0;

DS = Datasets.cifar_zs(opts, normalizeX);
trainCNN = DS.Xtrain;
testCNN = DS.Xtest;
retrievalCNN = DS.Xretrieval;
trainLabels = DS.Ytrain;
testLabels = DS.Ytest;
retrievalLabels = DS.Yretrieval;

% mapped into a sphere space
% 训练集
%train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));   %  n x d
train = trainCNN;
trainLabel = trainLabels; % n x 1
% 测试集
%test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  % n x  d
test = testCNN;
testLabel = testLabels;  % n x 1
% 检索集
%retrieval = retrievalCNN ./ sqrt(sum(retrievalCNN .* retrievalCNN, 2));  % n x  d
retrieval = retrievalCNN;
retrievalLabel = retrievalLabels;  % n x 1

clear trainCNN testCNN retrievalCNN trainLabels testLabels retrievalLabels

[Ntrain, Dtrain] = size(train);
[Ntest, Dtest] = size(test);
[Nretrieval, Dretrieval] = size(retrieval);

n_t = 2000;
alpha = 1e-2;
sigma = 0.3;  % places 上0.4上下调整, 一般情况下，sigma越大loss越小,反之越大. 
lr = 0.1;
threshold = 1e-3;
pos = 1;
neg = 1;

train = train';
test = test';
retrieval = retrieval';
W_t = randn(Dtest, opts.nbits);
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1);
training_size = 20000;

ii = 0;
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
%        P(P == 1) = 0.4;
%        P(P == 0) = 0.001;
    end
    [ydata, W_t, obj, Q, P] = minKLD(alpha/n_t, P, opts.nbits, X_t, W_t, S, lr, threshold, pos, neg);
    
    % ii = ii + 1;
    % logInfo('iter: %g', ii);
    % Htrain = single(W_t' * train > 0);
    % Htest = single(W_t' * test > 0);
    % Aff = affinity([], [], trainLabel, testLabel, opts);
    % opts.metric = 'mAP';
    % res = evaluate(Htrain', Htest', opts, Aff);
    % results = [results; res];
end
toc;

Hretrieval = single(W_t' * retrieval > 0);
Htest = single(W_t' * test > 0);
Aff = affinity([], [], retrievalLabel, testLabel, opts);
opts.metric = 'mAP';
res = evaluate(Hretrieval', Htest', opts, Aff);
results = [results; res];

disp(results);
auc   = arrayfun(@(x) mean(x), results);
final = arrayfun(@(x) x(end) , results);
logInfo('');
logInfo('  AUC mAP: %.3g +/- %.3g', mean(auc), std(auc));
logInfo('FINAL mAP: %.3g +/- %.3g', mean(final), std(final));

% opts.metric = 'mAP';
% res = evaluate(Htrain', Htest', opts, Aff);

% opts.metric = 'prec_n2';
% opts.prec_n = 2;
% res = evaluate(Htrain', Htest', opts, Aff);

% opts.metric = 'prec_k1';
% opts.prec_k = 1;
% res = evaluate(Htrain', Htest', opts, Aff);

clear;