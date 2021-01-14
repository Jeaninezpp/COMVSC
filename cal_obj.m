function [ obj ] = cal_obj(X,N,k,vN,alpha,gma,Zv,Lv,Fv,Fstar,R,Y)

T=eye(k);% initial T ,t_c=T(c,:)

for i_idx = 1:N
    for c_idx = 1:k
        TFR(i_idx,c_idx) = norm( T(c_idx,:) - (Fstar(i_idx,:)*R),2)^2; % calculate spetral rotation term
    end
end
SpectRota = sum(sum((Y.^gma).*TFR));

obj = 0;
term = zeros(1,4);
for num=1:vN
    term(1) = term(1) + (norm(X{num}-X{num}*Zv{num},'fro'))^2;
    term(2) = term(2) + alpha*(norm(Zv{num},'fro'))^2;
    term(3) = term(3) + trace(Fv{num}'*(diag(diag(Zv{num})) - 0.5*(Zv{num}+Zv{num}'))*Fv{num});
    term(4) = term(4) + norm(Fv{num}-Fstar,'fro')^2;
end
obj=term(1)+term(2)+term(3)+term(4)+SpectRota;
