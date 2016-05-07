function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);
imageDim = size(images, 1);
imageChannels = size(images, 3);

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps

rows=imageDim/patchDim
cols=imageDim/patchDim

tmp=zeros(patchDim,patchDim,imageChannels);
tmp_expanded=zeros(patchDim*patchDim*imageChannels,1);


ZCAWhitenedImages=zeros(size(images));
for imageNum=1:numImages
    for r=0:rows-1
        for c=0:cols-1
            tmp=images(r*cols+c+1:r*cols+c+patchDim,...
                    c*patchDim+1:c*patchDim+patchDim,...
                    :,...
                    imageNum);
            tmp_expanded=tmp(:);
            tmp_expanded=tmp_expanded-meanPatch;
            tmp_expanded=ZCAWhite*tmp_expanded;
            ZCAWhitenedImages(r*patchDim+1:r*patchDim+patchDim,...
                            c*patchDim+1:c*patchDim+patchDim,...
                            :,...
                            imageNum)=reshape(tmp_expanded,[patchDim,patchDim,imageChannels]);
        end
    end
end

images=ZCAWhitenedImages;
ZCAWhitenedImages=[];




% --------------------------------------------------------

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      feature = zeros(8,8); % You should replace this
      
      feature=reshape(W(featureNum,1+(channel-1)*patchDim*patchDim:patchDim*patchDim*channel)',8,8);
      
      
      % ------------------------

      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----

      convolvedImage=convolvedImage+bsxfun(@plus, conv2(im,feature,'valid'),b(featureNum));
      %convolvedImage=
      
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----
    %convolvedImage=convolvedImage/3;
%     convolvedImage=...
%         bsxfun(@plus,convolvedImage,b(featureNum));
    
    convolvedImage=sigmoid(convolvedImage);
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

