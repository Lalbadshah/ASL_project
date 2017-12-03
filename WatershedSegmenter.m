% read in the image, for now we will have one image as the read in file 
clear
clc
close all 

filename = '1_P_hgr1_id02_1.bmp';
filename = '2_P_hgr1_id01_2.bmp';

filename = '4_P_hgr1_id09_2.bmp';

filepath = 'hgr1_skin/skin_masks/';
BW = imread([filepath,filename]);

% this is of the binarized and segmented image 
sigmaBlur = 6; % in pixels

figure 
imshow(BW,[])
pause


% black part of image represents hands, white part of image represents
% background 

IMC = ~BW; % inverts the image 
D2E = bwdist(BW); % distance to edge 
% great, now we apply a gaussian blur operator to 
D2E_Blurred = imgaussfilt(D2E,sigmaBlur);
% now we assign non-object pixels to - infinity 



figure 
imshow(D2E_Blurred,[])

figure 
imshow(-D2E_Blurred,[])


D2E_Blurred = -D2E_Blurred;
D2E_Blurred(BW == 1) = inf;

figure 
imshow(D2E_Blurred,[])


% perform watershed on compliment of distance transform 
L = watershed(D2E_Blurred);
L(BW) = 0;

Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
figure
imshow(Lrgb)

%%%%%%%%%%%%%%

B = bwboundaries(BW);
% stupid cell format 
nBoundaryPts = length(B);

Boundary = B{1};
nBoundaryPts = length(Boundary);

plot(Boundary(:,1),Boundary(:,2))

