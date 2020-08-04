% The following code is the implementation of Table 4 of the following
% Reference:
%  1.SRDA: An Efficient Algorithm for Large Scale Discriminant Analysis. 
%  Deng Cai, Xiaofei He, Jiawei Han. IEEE Transactions on Knowledge and 
%  Data Engineering, vol. 20, no. 1, pp. 1-12, January, 2008. 

clear all;
close all;
load PIE_32x32.mat; %PIE_32x32.mat can be downloaded from http://www.cs.uiuc.edu/homes/dengcai2/Data/PIE/PIE_32x32.mat
tic;
mu=1;% 0.5 is set according to Section 6.1 of "IDR/QR: An Incremental Dimension Reduction Algorithm via QR Decomposition"
rate=zeros(1,50);
%fea = fea/256; %Pre-process the face image by scaling features (pixel values) to [0,1] (divided by 256). 
[nSmp,nFea] = size(fea);
for i=1:nSmp
    a=norm(fea(i,:));
    fea(i,:)=fea(i,:)/a;
end   %Pre-process the data by normalizing each face image vector to unit.
for i=1:50
    filename = strcat('.\10Train\',num2str(i)); %10Train.zip can be downloaded from http://www.cs.uiuc.edu/homes/dengcai2/Data/PIE/10Train.zip
    load (filename);
    fea_Train = fea(trainIdx,:);
    gnd_Train = gnd(trainIdx);
    gnd_Test= gnd(testIdx);
    [eigvector] = IDRQR(fea_Train, gnd_Train,mu);
    newfea = fea*eigvector;
    newfea_Train = newfea(trainIdx,:);
    newfea_Test = newfea(testIdx,:);
    rate(1,i)=KNN(newfea_Train,gnd_Train,newfea_Test,gnd_Test,1);
    i
end
mean(rate)
std(rate)
save PIE_10_rate_mat rate;
toc;