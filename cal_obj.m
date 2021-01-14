function [ obj ] = cal_obj(X,N,k,vN,alpha,gma,para3,para4,Zv,Lv,Fv,Fstar,R,Y)
%CALOBJ Summary of this function goes here
%   Detailed explanation goes here
T=eye(k);% initial T ,t_c=T(c,:)

for i_idx = 1:N
    for c_idx = 1:k
        TFR(i_idx,c_idx) = norm( T(c_idx,:) - (Fstar(i_idx,:)*R),2)^2;% calculate spetral rotation term
    end
end
SpectRota = sum(sum((Y.^gma).*TFR));

obj = 0;
term = zeros(1,4);
for num=1:vN
    term(1) = term(1) + (norm(X{num}-X{num}*Zv{num},'fro'))^2;
    term(2) = term(2) + alpha*(norm(Zv{num},'fro'))^2;
    term(3) = term(3) + para3*trace(Fv{num}'*(diag(diag(Zv{num})) - 0.5*(Zv{num}+Zv{num}'))*Fv{num});
    term(4) = term(4) + para4*norm(Fv{num}-Fstar,'fro')^2;
end
obj=term(1)+term(2)+term(3)+term(4)+SpectRota;
% disp([' T1: ',num2str(term(1)),' T2: ',num2str(term(2)),' T3: ',num2str(term(3)),...
%      ' T4: ',num2str(term(4)),' SpectRota: ',num2str(SpectRota),' --OBJ--',num2str(obj)]);

