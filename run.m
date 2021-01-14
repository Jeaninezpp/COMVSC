clc;
clear;
addpath ./measure ./tsne
datasetname = {'WikipediaArticles','bbcsport_seg14of4','Handwritten_numerals',...
         'MSRCV1','Caltech101-7','Caltech101-20'};

for di = 1:5
    path = ['.\dataset\',datasetname{di},'.mat']
    f=load(path);
    data=f.X;
    label=f.Y;
    viewN = length(data);
    k = length(unique(label));
    
    FV = cell(size(data)); 
    WV= cell(size(data));
    LV = cell(size(data));
    
    for i=1:viewN % Normalization
        data{i} = mapstd(data{i}',0,1);
    end

    for i =1:viewN
        W = constructW_PKN(data{i}, 5, 1);  %   X : d*n  / 5 neighbors /  is symmetric
        D = diag(sum(W));
        L = D-W;
        [Fv, ~, ~]=eig1(L,k,0);
        FV{i} = Fv;
        LV{i} = L;
        WV{i} = W;
    end

    lmd = 10.^[1:1:5];
    gma = [1:0.1:2];


    B=cell(size(data)); % B=inv(X'X+alpha*I)^(-1)
    %========for========
    for i=1:length(lmd)
        for vnum=1:viewN % view
            B{vnum}= inv(data{vnum}'*data{vnum}+lmd(i)*eye(size(data{1},2)));
        end
        for j=1:length(gma)
           disp(['alpha:',num2str(lmd(i)),' gamma:',num2str(gma(j))]);
           for it = 1:1
               tic;
               [res,Fv,Fstar,Lv,Wv,Yres,obj]=CLOMV(data,label,B,lmd(i),gma(j),para3(p),para4(q),FV,WV,LV);
               t=toc;
               disp(['ACC: ',num2str(res(end,7)), '  Time: ',num2str(t)]);
           end
        end
    end
end
