%% 载入数据
%images=loadMNISTImages('train-images.idx3-ubyte');
%labels=loadMNISTLabels('train-labels.idx1-ubyte');


%% 参数设置
visibleSize=28*28
hiddenSize=196
sparsityParam=0.1
lambda=3*10^-3
beta=3

debug=1;
%% 载入数据
patches=sampleIMAGES;


%% 展示部分数据
display_network(patches(:,1:100),8);

if debug==1
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
%options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
                          
%% 可视化                          
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);

W2 = reshape(opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 

m=size(patches,2);

Z2=W1*patches+repmat(b1,1,m);
A2=sigmoid(Z2);

Z3=W2*A2+repmat(b2,1,m);
A3=sigmoid(Z3);

display_network(A3(:,1:16),20);
print -djpeg result.jpg

end
