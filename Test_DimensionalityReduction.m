%% MLSP Final Project

%% Load train images
numtest = 10;   % Starting with 10 images from each letter
folder = [pwd '/A/'];

alpha = char([65:73 75:89]);    % alphabet not including J or Z
alpha_lower = char([97:105 107:121]);
nums = [0:8 10:25];
l=1;

for i = 1:length(alpha_lower)
    for j = 79:79+numtest-1
        filename = fullfile(folder,sprintf('%s/color_%g_%04d.png',alpha_lower(i),nums(i),j+1));
        image = rgb2gray(imread(filename));
        [m, n] = size(image);
        % Reshape the image to be a vector
        Images{l}=imresize(image,[123 126], 'bilinear'); 
        Train(:,l) = double(reshape(Images{l},123*126,1)); l=l+1;
    end
end

%% Load test images
numtest = 2;   % Starting with 10 images from each letter
folder = [pwd '/A/'];

l=1;

for i = 1:length(alpha_lower)
    for j = 99:99+numtest-1
        filename = fullfile(folder,sprintf('%s/color_%g_%04d.png',alpha_lower(i),nums(i),j+1));
        image = rgb2gray(imread(filename));
        [m, n] = size(image);
        % Reshape the image to be a vector
        Images{l}=imresize(image,[123 126], 'bilinear'); 
        Test(:,l) = double(reshape(Images{l},123*126,1)); l=l+1;
    end
end


%% PCA - to reduce dimensions
eigen = 100; % starting with 100 eigenvectors
avgHand = mean(Train,2);
Train = Train - avgHand;

coef = pca(Train');
projTrain = Train'*coef(:,1:eigen);
projTrain = projTrain';
% corr = Train'*Train;
% [U,S,V]=svd(corr);
% eigenValues = diag(S);
% eigenVectors = V;
% eigenSpace = eigenVectors*diag(1./sqrt(eigenValues'));
% imageSpace = Train*eigenSpace;
% projTrain = imageSpace(:,1:eigen);

% Seperate out the train set based on letter
k=10; l=1;
for i = 1:10:size(projTrain,2)
    class{l} = projTrain(:,i:k); k=k+10; l=l+1;
end

%% LDA - to find most relavent features

L = size(projTrain,1);
% Initialize variables needed for covariance
Sk = zeros(size(projTrain,1),size(projTrain,1));
Sw = Sk;
Sb = Sk;
Sb_temp = Sb;
mk = zeros(size(projTrain,1),L);

% Within class covariance
for i=1:length(alpha)
    X = class{i};
    mk(:,i) = mean(X,2); % mean of class
    %Sk = bsxfun(@minus,X,mk(:,i));
    Sk = (X-mk(:,i))*(X-mk(:,i))';
    Sw = Sw + Sk; %*Sk';
end

% Class to class covariance
m = mean(projTrain,2);  % global mean based on PCA
k=1;
for i=1:length(alpha)
    avgClass = mk(:,i);
    Sb_temp = length(alpha)*(avgClass-m)*(avgClass-m)';
    Sb = Sb + Sb_temp;
end

[V,D] = eigs(Sb,Sw,L-1);

%% Linear Gaussian Classifier

% Project test data into PCA then LDA spaces
PCAprojTest = (Test'*coef(:,1:eigen));
projTest = PCAprojTest*V;

for i = 1:length(alpha)
   meanClass{i} = mean(Class{i});
   covClass{i} = cov(Class{i});
end

% Using each covariance matrix for each classification
for i = 1:length(alpha)
    probClass(i,:) = (size(Class{i},2)/size(projTrain,1)) * mvnpdf(projTest,meanClass{i},covClass{i});
end
[~,decision] = max(probClass,[],1);

alpha_decision = alpha(decision);
truth = 'aabbccddeeffgghhiikkllmmnnooppqqrrssttuuvvwwxxyy';

numCorrect(z) = sum((alpha_decision==truth));
