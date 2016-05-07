function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

tmpConvolvedImage=zeros(poolDim,poolDim);
tmpCIV=zeros(poolDim*poolDim,1);

for imageNum=1:numImages
    for featureNum=1:numFeatures
        for r=1:floor(convolvedDim / poolDim)
            for c=1:floor(convolvedDim / poolDim)
                tmpConvolvedImage=convolvedFeatures(featureNum,imageNum,(r-1)*poolDim+1:(r-1)*poolDim+poolDim,(c-1)*poolDim+1:(c-1)*poolDim+poolDim);
                tmpCIV=tmpConvolvedImage(:);
                pooledFeatures(featureNum,imageNum,r,c)=mean(tmpCIV);
            end
        end
    end
end

