function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


num_activation=size(stack,1)+1;
z=cell(num_activation,1);
activation=cell(num_activation,1);

activation{1}=data;

for i=2:num_activation
    %size(stack{i-1}.w)
    %size(activation{i-1})
    %size(stack{i-1}.b)
    z{i}=bsxfun(@plus,stack{i-1}.w*activation{i-1},stack{i-1}.b);
    activation{i}=sigmoid(z{i});    
end

H=zeros(size(groundTruth));
H=softmaxTheta*activation{num_activation};
H=bsxfun(@minus,H,max(H,[],1));
H=exp(H);
H=bsxfun(@rdivide,H,sum(H,1));

decayTerm1=0;
for d = 1:numel(stack)
    decayTerm1=decayTerm1+sum(sum(stack{d}.w.^2));
end
decayTerm1=lambda/2*decayTerm1;

decayTerm2=lambda/2*sum(sum(softmaxTheta.^2));
cost=-1/M*sum(sum(groundTruth.*log(H)))+decayTerm1+decayTerm2;


%softmaxThetaGrad=softmaxTheta'*(groundTruth-H);
softmaxThetaGrad=-1/M*(groundTruth-H)*activation{num_activation}+lambda*softmaxTheta;

delta=cell(size(num_activation,1));
delta{num_activation}=softmaxTheta'*(groundTruth-H);
delta{num_activation}=-1*delta{num_activation}.*sigmoidGrad(z{num_activation});

for l=num_activation-1:-1:2
    delta{l}=(stack{l}.w'*delta{l+1}).*sigmoidGrad(z{l});
end

for d=numel(stack):-1:1
    size(stackgrad{d}.w);
    stackgrad{d}.w = 1/M*delta{d+1}*activation{d}'+lambda*stack{d}.w;
    size(stackgrad{d}.w);
    size(stackgrad{d}.b);
    stackgrad{d}.b = 1/M*sum(delta{d+1},2);
    size(stackgrad{d}.b);
end









% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function sigmGrad=sigmoidGrad(x)
    sigmGrad=sigmoid(x).*(1-sigmoid(x));
end
