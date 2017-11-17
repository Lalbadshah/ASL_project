%% MLSP Final Project

%% Load images
folder = [pwd '/data/'];
dir = dir([folder '/*.jpg']);
numphotos = length(dir);

for i = 1:numphotos
    image = imread(dir(i).name);
    true = dir(i).name;
    label(i) = true(2); % make array of labels
    
    [m, n] = size(image);
    % Reshape the image to be a vector
    Train(:,i) = double(reshape(image,m*n,1)); 
end

folder = [pwd '/data/Test/'];
dir = dir([folder '/*.jpg']);
numphotos = length(dir);

for i = 1:numphotos
    image = imread(dir(i).name);
    true = dir(i).name;
    test_truth(i) = true(2); % make array of labels
    
    [m, n] = size(image);
    % Reshape the image to be a vector
    Test(:,i) = double(reshape(image,m*n,1)); 
end


%% PCA - to reduce dimensions
eigen = 300; % starting with 300 eigenvectors
avgFace = mean(Train,2);
Train = Train - avgFace;

corr = Train'*Train;
[U,S,V]=svd(corr);
eigenValues = diag(S);
eigenVectors = V;
eigenSpace = eigenVectors*diag(1./sqrt(eigenValues'));
imageSpace = centeredFace*eigenSpace;
projTrain = imageSpace(:,1:eigen);

alpha = char([65:73 75:89]);    % alphabet not including J or Z

% Seperate out the train set based on letter
for i = 1:length(alpha)
    ind = find(label==alpha(i));
    class{i} = projTrain(:,ind);
end

%% LDA - to find most relavent features

L = n;
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
PCAprojTest = Test*eigenSpace(:,1:eigen);
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

numCorrect(z) = sum((alpha_decision==truth));
