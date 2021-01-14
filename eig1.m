function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% A：matrix
% c: number of clusters
% ismax: 
if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
[v d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0 %升序,取前c小
    [d1, idx] = sort(d);
else %降序,取前c大
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);%取前c个
eigval = d(idx1);%取前c个特征值
eigvec = v(:,idx1);%取前c个特征向量

eigval_full = d(idx);