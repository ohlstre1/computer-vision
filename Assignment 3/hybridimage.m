%% Uncomment the commented code lines and implement missing functions 
% (and anything else that is asked or needed)

%% Load test images.
man=double(imread('man.jpg'));
wolf=double(imread('wolf.jpg'));

% the pixel coordinates of eyes and chin have been manually found 
% from both images in order to enable affine alignment 
man_eyes_chin=[452 461;652 457;554 823];
wolf_eyes_chin=[851 919; 1159 947; 975 1451];
[A,b]=affinefit(man_eyes_chin', wolf_eyes_chin');

[X,Y]=meshgrid(1:size(man,2), 1:size(man,1));
pt=A*([X(:) Y(:)]')+b*ones(1,size(man,1)*size(man,2));
wolft=interp2(wolf,reshape(pt(1,:),size(man)),reshape(pt(2,:),size(man)));

%% Below we simply blend the aligned images using additive superimposition
additive_superimposition=man+wolft;

%% Next we create two different Gaussian kernels for low-pass filtering
sigmaA = 15;%7; 
sigmaB = 5;%2; 
filterA = fspecial('Gaussian', ceil(sigmaA*4+1), sigmaA);
filterB = fspecial('Gaussian', ceil(sigmaB*4+1), sigmaB);

man_lowpass = imfilter(man, filterA, 'replicate');
wolft_lowpass= imfilter(wolft, filterB, 'replicate');

%% Task:
% Your task is to create a hybrid image by combining a low-pass filtered 
% version of the human face with a high-pass filtered wolf face.
%
% HINT: You get a high-pass version by subtracting the low-pass filtered version
% from the original image. Experiment also by trying different values for
% 'sigmaA' and 'sigmaB' above.
%
% Thus, your task is to replace the zero image on the following line
% with a high-pass filtered version of 'wolft'
wolft_highpass=zeros(size(man_lowpass));

%% Replace also the zero image below with the correct hybrid image
hybrid_image=zeros(size(man_lowpass));
%hybrid_image = man_lowpass + wolft_highpass;

%% Visualization and interpretation
% Notice how strongly the interpretation of the hybrid image is affected 
% by the viewing distance
fighybrid=figure;colormap('gray');imagesc((hybrid_image));axis image;
%
% Display input images and both output images.
figure; clf;
set(gcf,'Name','Results of superimposition');

subplot(2,2,1); imagesc(man);
axis image; colormap gray;
title('Input Image A');

subplot(2,2,2); imagesc(wolf);
axis image; colormap gray;
title('Input Image B');

subplot(2,2,3); imagesc(additive_superimposition);
axis image; colormap gray;
title('Additive Superimposition');

subplot(2,2,4); imagesc(hybrid_image);
axis image; colormap gray;
title('Hybrid Image');

%figure(fighybrid);