function info = test_online(Dataset, trial, opts)
% Computes the performance by loading and evaluating the "checkpoint" files
% saved during training. 
%
% INPUTS
%       paths (struct)
%               result - (string) final result path
%               diary  - (string) exp log path
%               trials - (cell, string) paths of result for each trial
%	opts  (struct) 
% 
% OUTPUTS
%	none

info = struct(...
    'metric'         , [] , ...
    'train_time'     , [] , ...
    'train_iter'     , [] , ...
    'train_examples' , [] , ...
    'bit_recomp'     , [] );

testX = Dataset.Xtest;
if opts.unseen == 1
    logInfo('evaluating on unseen class datasets.');
    logInfo('');
    retrievalX = Dataset.Xretrieval;
    Aff = affinity(Dataset.Xretrieval, Dataset.Xtest, Dataset.Yretrieval, Dataset.Ytest, opts);
else
    Aff = affinity(Dataset.Xtrain, Dataset.Xtest, Dataset.Ytrain, Dataset.Ytest, opts);
end

prefix = sprintf('%s/trial%d', opts.dirs.exp, trial);
model = load([prefix '.mat']);

for i = 1:length(model.test_iters)
    iter = model.test_iters(i);
    fprintf('Trial %d, Checkpoint %5d/%d, ', trial, iter*opts.batchSize, ...
        opts.numTrain*opts.epoch);

    % determine whether to actually run test or not
    % if there's no HT update since last test, just copy results
    if i == 1
        runtest = true;
    else
        st = model.test_iters(i-1);
        ed = model.test_iters(i);
        runtest = any(model.update_iters>st & model.update_iters<=ed);
    end

    itmd = load(sprintf('%s_iter/%d.mat', prefix, iter));
    P = itmd.params;
    if strcmp(opts.methodID, 'OKH')
        % do kernel mapping for test data
        testX = exp(-0.5*sqdist(Dataset.Xtest', P.Xanchor')/P.sigma^2)';
        testX = [testX; ones(1,size(testX,2))]';
        %TODO do kernel mapping for retrieval data
        if opts.unseen == 1
            retrievalX = exp(-0.5*sqdist(Dataset.Xretrieval', P.Xanchor')/P.sigma^2)';
            retrievalX = [retrievalX; ones(1,size(retrievalX,2))]';
        end
    elseif strcmp(opts.methodID, 'SketchHash')
        % subtract estimated mean
        testX = bsxfun(@minus, Dataset.Xtest, P.instFeatAvePre);
        %TODO subtract estimated mean for retrieval data
        if opts.unseen == 1
            retrievalX = bsxfun(@minus, Dataset.Xretrieval, P.instFeatAvePre);
        end
    end
    if runtest
        % NOTE: for intermediate iters, need to use Wsnapshot (not W!)
        %       to compute Htest, to make sure it's computed using the same
        %       hash mapping as Htrain.
        Htest  = (testX * itmd.Wsnapshot) > 0;
        if opts.unseen == 1
            Htrain  = (retrievalX * itmd.Wsnapshot) > 0;
        else
            Htrain = itmd.H;
        end
        info.metric(i) = evaluate(Htrain, Htest, opts, Aff);
        info.bit_recomp(i) = itmd.bit_recomp;
    else
        info.metric(i) = info.metric(i-1);
        info.bit_recomp(i) = info.bit_recomp(i-1);
        fprintf(' %g\n', info.metric(i));
    end
    info.train_time(i) = itmd.time_train;
    info.train_iter(i) = iter;
    info.train_examples(i) = iter * opts.batchSize;

    % TODO
    % delete .mat file
    delete(sprintf('%s_iter/%d.mat', prefix, iter));
    
    % evaluate prec_k1 and prec_n2
    if i == length(model.test_iters)
        opts.metric = 'prec_k1';
        opts.prec_k = 1;
        evaluate(Htrain, Htest, opts, Aff);

        opts.metric = 'prec_n2';
        opts.prec_n = 2;
        evaluate(Htrain, Htest, opts, Aff);
        
        opts.metric = 'prec_recall';
        res = evaluate(Htrain, Htest, opts, Aff);
        if size(res.recall, 1) > 0
            plot_recall_prec(res.recall, res.precision, opts);
        end
        info.recall = res.recall;
        info.precision = res.precision;
    end
end

end
