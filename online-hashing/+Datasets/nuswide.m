function DS = nuswide(opts, normalizeX)
% Load and prepare CNN features. The data paths must be changed. For all datasets,
% X represents the data matrix. Rows correspond to data instances and columns
% correspond to variables/features.
% Y represents the label matrix where each row corresponds to a label vector of 
% an item, i.e., for multiclass datasets this vector has a single dimension and 
% for multilabel datasets the number of columns of Y equal the number of labels
% in the dataset. Y can be empty for unsupervised datasets.
% 
%
% INPUTS
%	opts   - (struct)  Parameter structure.
% 		
% OUTPUTS: struct DS
% 	Xtrain - (nxd) 	   Training data matrix, each row corresponds to a data
%			   instance.
%	Ytrain - (nxl)     Training data label matrix. l=1 for multiclass datasets.
%			   For unsupervised dataset Ytrain=[], see LabelMe in 
%			   load_gist.m
%	Xtest  - (nxd)     Test data matrix, each row corresponds to a data instance.
%	Ytest  - (nxl)	   Test data label matrix, l=1 for multiclass datasets. 
%			   For unsupervised dataset Ytrain=[], see LabelMe in 
%			   load_gist.m
% 

if nargin < 2, normalizeX = 1; end
        
tic;
load(fullfile(opts.dirs.data, 'NUS-WIDE-split.mat'), ...
    'I_tr', 'I_te', 'L_tr', 'L_te');

X = [I_tr; I_te];
Y = [L_tr; L_te];

if normalizeX && opts.normalize
    X = bsxfun(@minus, X, mean(X,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
else
    logInfo('will NOT pre-normalize data');
end

DS = [];
DS.Xtrain = X(1:size(I_tr,1), :);
DS.Ytrain = Y(1:size(L_tr,1), :);
DS.Xtest  = X(size(I_tr,1)+1:end, :);
DS.Ytest  = Y(size(L_tr,1)+1:end, :);
DS.thr_dist = -Inf;

logInfo('[NUS-WIDE] loaded in %.2f secs', toc);
end

% if nargin < 2, normalizeX = 1; end
% if ~normalizeX, logInfo('will NOT pre-normalize data'); end
    
% tic;
% load(fullfile(opts.dirs.data, 'NUS-WIDE-split.mat'), ...
%     'I_tr', 'I_te', 'L_tr', 'L_te');

% DS = [];
% DS.Xtrain = I_tr;
% DS.Ytrain = L_tr;
% DS.Xtest  = I_te;
% DS.Ytest  = L_te;
% DS.thr_dist = -Inf;

% logInfo('[NUS-WIDE] loaded in %.2f secs', toc);
% end