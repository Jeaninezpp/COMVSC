function [p] = calculate(F)
%% direct
% [sample,classnumber]= size(F);
% p =zeros(sample,sample);
% for i=1:sample
%     for j=i+1:sample
%         p(i,j) = norm(F(i,:)- F(j,:),2);
%     end
% end
% p = p + p';

%% parfor
[sample,classnumber]= size(F);
p =zeros(sample,sample);
parfor i=1:sample
    tmp = zeros(1,sample);
    for j=1:sample
        tmp(j) = norm(F(i,:)- F(j,:),2);
    end
    p(i,:)=tmp;
end

end

