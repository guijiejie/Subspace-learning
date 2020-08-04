function G  = IDRQR(Data, Class,mu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:                                                             %
%     Data:  data matrix (Each row is a data point)                  %
%     Class: class label (class 1, ..., k)                           %
% Output:                                                            %
%      G:    transformation matrix                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference:  
%
%  1.Jieping Ye, Qi Li, Hui Xiong, Haesun Park, Ravi Janardan, and Vipin Kumar. 
%  IDR/QR: An Incremental Dimension Reduction Algorithm via QR Decomposition. 
%  The Tenth ACM SIGKDD International Conference on Knowledge Discovery and 
%  Data Mining (SIGKDD 2004), pp. 364—373.
%
%  This code is the implementation of Algorithm1 of Reference 1,which is written 
%  by Gui Jie(University of science and technology of China,guijiejie@gmail.com) 
%  in the afternoon 2009/05/27.
%  If you have find some bugs in the codes, feel free to contract me

k        = max(Class); % number of classes

%-------------------------------------------------------------------------
[m,n] = size(Data);
cc    = sum(Data)/m;
C=[];
for i = 1:k 
	loc          = find(Class==i);
	[num,o]      = size(loc);
	TMP          = sum(Data(loc,1:n))/num;
    C            = [C,TMP'];
    B(i,1:n)     = sqrt(num)*(TMP - cc);
	B(loc+k,1:n) = Data(loc,1:n) - ones(num,1)*TMP;
end;
[Q,R]=qr(C,0);
Hw = B(k+1:k+m, :)';
Hb = B(1:k, :)';
Z=(Hw)'*Q;
Y=(Hb)'*Q;
W=Z'*Z;    % Reduced within-class scatter matrix
B=Y'*Y;    % Reduced between-class scatter matrix
Temp=inv(W+mu*eye(k,k))*B;
[M,D]=eig(Temp);
G = Q*M;

