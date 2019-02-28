classdef CVPR


properties
    batchSize
    stepsize
    alpha
    sigma
    threshold
    pos
    neg
end

methods
    function [W, R, obj] = init(obj, R, X, Y, opts)
        obj.batchSize = opts.batchSize;
        obj.stepsize = opts.stepsize;
        obj.alpha = opts.alpha;
        obj.sigma = opts.sigma;
        obj.threshold = opts.threshold;
        obj.pos = opts.pos;
        obj.neg = opts.neg;
        disp(obj);

        d = size(X, 2);
        W = randn(d, opts.nbits);
        W = W ./ repmat(diag(sqrt(W' * W))', d, 1);
    end


    function [W, ind] = train1batch(obj, W, R, X, Y, I, t, opts)

        n_t = obj.batchSize;
        
        % ind = I((t-1)*n_t + (1:n_t));
        ind = (t-1)*n_t + (1:n_t);
        X_t = X(ind, :)';
        L_t = Y(ind, :);

        disp(sum(sum(X_t)));

        P = zeros(n_t, n_t);
        if size(L_t, 2) > 1
            num1 = 1./ sqrt(sum(L_t .* L_t, 2));
            num1(isinf(num1) | isnan(num1)) = 1;
            L_T = diag(num1) * L_t;
            
            P = L_T * L_T';   % cosine  for multi-label cases
        else
            P = double(L_t == L_t');
            P = normpdf(P, 1, obj.sigma);
            S = double(L_t == L_t');
    %        P(P == 1) = 0.4;
    %        P(P == 0) = 0.001;
        end
        [ydata, W, obj, Q, P] = minKLD(obj.alpha/n_t, P, opts.nbits, X_t, W, S, obj.stepsize, obj.threshold, obj.pos, obj.neg);
    end


    function H = encode(obj, W, X, isTest)
        H = single(X * W > 0);
    end

    function P = get_params(obj)
        P = [];
    end

end % methods

end % classdef
