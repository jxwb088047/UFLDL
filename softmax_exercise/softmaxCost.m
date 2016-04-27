function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


h=zeros(numClasses,numCases);
h=theta*data;
h=bsxfun(@minus,h,max(h,[],1));
h=bsxfun(@rdivide,exp(h),sum(exp(h),1));
size(h);

cost= -1/numCases*sum(sum(groundTruth.*log(h)))+lambda/2*sum(sum(theta.^2));

%% 对于下面矩阵相乘的解释：
%正常情况下，thetagrad(j)=x(i)*delta(j,i)的求和（i：m）(这里忽略系数-1/numCases),
%用矩阵表示为thetagrad(j)=x*delta(j,:)',
%由于thetagrad在维度上和theta是一致的，而正好theta=(theta(1),theta(2),...theta(k))'
%所以要对所求表示做一次转置操作
thetagrad=-1/numCases*(groundTruth-h)*data'+lambda*theta;










% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

