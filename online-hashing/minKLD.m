function [trainCode, W_t, obj, Q, P] = minKLD(alpha, P, bit, X_t, W_t, S, lr, threshold, pos, neg)
% minKLD:   minimizing KL-divergence, developed from tsne_p.m of t-SNE by Laurens van der Maaten
% Input: 
%   alpha: model parameter \alpha (1 * 1)
%   P: probability distribution before hashing (n * n, n being training set size)
%   bit: length of hash codes (1 * 1)
%   initC: initial values of to-be-learnt hash code matrix (n * bit)
% Output:
%   trainCode: learnt hash code matrix of training data (n * bit)
%   obj: optimal value of the objective function

    % Initialize some variables
    n = size(P, 1);                                     % number of instances
    max_iter = 1000;                                    % maximum number of iterations
    
    % Make sure P-vals are set properly
    P(1:n + 1:end) = 0;                                 % set diagonal to zero
    P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
    
    const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
    
    % lr = 2; % places205  2上下调整 
    % threshold = 1e-3;
    tt = 1;

    % pos, neg一般情况下设置为1， 如果要调整，需要根据比特位大小，pos越接近比特位，neg越原理比特位大小，loss越低
    % pos=1; % 60
    % neg=1; % 40

    trainCode = tanh(tt*X_t' * W_t);
    
    % Record the minimal objective function and the corresponding optimal
    % hash code matrix of training data
    minCost = realmax;
    minTrainCode = trainCode;
    
    % Run the iterations
    for iter=1:max_iter        
        % Compute joint probability that point i and j are neighbors
        sum_trainCode = sum(trainCode .^ 2, 2);
        % hamming distance --> t-distribution
        dis = 0.25 * bsxfun(@plus, sum_trainCode, bsxfun(@plus, sum_trainCode', -2 * (trainCode * trainCode')));
                
        
        dis(S==1) = dis(S==1)/ pos;
        dis(S==0) = dis(S==0)/ neg;

       
        num = 1 ./ (1 + dis); 
        num(1:n+1:end) = 0;                                                 % set diagonal to zero
        Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
        

        % Compute the gradients (faster implementation)
        L = (P - Q) .* num;
        L(S==1) = L(S==1);
        L(S==0) = L(S==0);
        h_grads = (diag(sum(L, 2)) - L) * trainCode;
        w_grads = X_t * (h_grads .* (1 - trainCode.*trainCode));

        W_t = W_t - lr*w_grads;
        
        cost = const - sum(P(:) .* log(Q(:)));
%        cost = -sum(P(:) .* log(Q(:)));
%        minCost - cost
        if cost <= minCost
            if minCost - cost > threshold
                minCost = cost;
                minTrainCode = trainCode;
                trainCode = tanh(X_t' * W_t);
            else
                disp("break threshold");
                break; 
            end
        else
            disp("break lr");
            break;
        end
        
        % Print out progress, modified
        if ~rem(iter, 1)
            disp(['Iteration ' num2str(iter)  ': error is ' num2str(cost)]);
        end
    end
    
    trainCode = minTrainCode;
    obj = minCost;
	
	
	
