function G  = LDAQR(Data, Class)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:                                                             %
%     Data:  data matrix (Each row is a data point)                  %
%     Class: class label (class 1, ..., k)                           %
% Output:                                                            %
%      G:    transformation matrix                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference:  
%
%  1.Jieping Ye and Qi Li. A two-stage linear discriminant analysis via QR
%  decomposition. IEEE Transactions on Pattern Analysis and Machine 
%  Intelligence. Vol. 27, No. 6, pp. 929—941, 2005.
%
%  This code is the implementation of Algorithm2 of Reference 1,which is written 
%  by Gui Jie(University of science and technology of China,guijiejie@gmail.com) 
%  in the morning of 2009/05/28.
%  If you have find some bugs in the codes, feel free to contract me

k        = max(Class); % number of classes

%-------------------------------------------------------------------------
[m,n] = size(Data);
cc    = sum(Data)/m;

for i = 1:k 
	loc          = find(Class==i);
	[num,o]      = size(loc);
	TMP          = sum(Data(loc,1:n))/num;
    B(i,1:n)     = sqrt(num)*(TMP - cc);
	B(loc+k,1:n) = Data(loc,1:n) - ones(num,1)*TMP;
end;
Hw = B(k+1:k+m, :)';
Hb = B(1:k, :)';
t  = rank(Hb);
[Q,R,E]=qr(Hb);   %QR decomposition of Hb with column pivoting
Z=(Hw)'*Q;
Sb=R*R';    % Reduced between-class scatter matrix
Sw=Z'*Z;    % Reduced within-class scatter matrix

Temp=-inv(Sb)*Sw; % Minus is to ensure the step 6 of Algorithm2.
[W,D]=eig(Temp);
G = Q*W;

