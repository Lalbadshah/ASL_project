clear all
filepath = 'data2/';
filename = '4A.JPG';

% %%% 
A = imread([filepath,filename]);
sizeIMG = size(A);
xdir =  sizeIMG(1);
ydir =  sizeIMG(2);

length_vec = xdir*ydir;

figure
imshow(A)

A = imgaussfilt(A,2);

[GmagR, Gdir] = imgradient(A(:,:,1),'prewitt');
[GmagG, Gdir] = imgradient(A(:,:,2),'prewitt');
[GmagB, Gdir] = imgradient(A(:,:,3),'prewitt');

Gtotal = GmagR + GmagG + GmagB;
figure 
imshow(Gtotal,[])


R = A(:,:,1);
R = double(R(:));
G = A(:,:,2);
G = double(G(:));
B = A(:,:,3);
B = double(B(:));
VectorizedImage = zeros(length_vec,3);
VectorizedImage(:,1) = R;
VectorizedImage(:,2) = G;
VectorizedImage(:,3) = B;

Sums = sum(VectorizedImage,2);
Sums(Sums == 0) = 1;

R = R ./ Sums;
G = G ./ Sums;
B = B ./ Sums;


% H = acos(double(.5.*(2.*R - G - B)) ./ (sqrt(double((R-G).^2 -(R-B).*(G-B)))) );
% V = max([R,G,B],[],2);
% MinC = min([R,G,B],[],2);
% S = (V - MinC) ./ (V);
%HSV = [H,S,V];

HSV = rgb2hsv([R,G,B]);

Transform = [.279,.504,.098; -.148 , .291 , .439 ; .439 , -.368 , -.071];
shift1 = [16,128,128];
YCbCr = (Transform * VectorizedImage')' + repmat(shift1 ,length_vec,1) ; 

k = 4;
%[idx,C] = kmeans([HSV,YCbCr],k);
%[idx,C] = kmeans([HSV],k,'Distance','correlation');
[idx,C] = kmeans(HSV,k);

% lets bring this bad boy back 
Label_img = reshape(idx,xdir,ydir);

Lrgb = label2rgb(Label_img, 'jet', 'w', 'shuffle');
figure
imshow(Lrgb)


h = figure;
subplot(131);
imshow(A)
title('Original Image');
subplot(132);
%imshow(Gtotal,[])
imagesc(Gtotal); axis image; axis off; caxis([0 200]);
title('Gradient');
subplot(133);
imshow(Lrgb)
title(sprintf('KNN with k=%g',k));

saveas(h,sprintf('%s_KNN_k%g_data%s.png',filename(1:2),k,filepath(5)))





