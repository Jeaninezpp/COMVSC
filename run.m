clc;
clear;
addpath ./measure ./tsne
datasetname = {'WikipediaArticles','bbcsport_seg14of4','Handwritten_numerals',...
         'MSRCV1','Caltech101-7','Caltech101-20'};

for di = 5:5
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
% %         [data{i},~] = mapminmax(data{i}');
%         data{i} = data{i}'; % data: d*n
%         datanor = (data{i} - min(min(data{i})))/(max(max(data{i})) - min(min(data{i})));
%         data{i} = 2 * datanor - 1;
    end

    for i =1:viewN
        W = constructW_PKN(data{i}, 5, 1);  %   X : d*n  / 5 neighbors /  is symmetric
        D = diag(sum(W));
        L = D-W;
        [Fv, ~, ~]=eig1(L,k,0);% 取前k小个特征值和特征向量
        FV{i} = Fv;
        LV{i} = L;
        WV{i} = W;
    end

    lmd = 10.^[1:1:5];
    gma = [1:0.1:2];
%     para3 = 10.^[-5:2:5];
%     para4 = 10.^[-5:2:5];
    
%    lmd = [10,20,30];
    
%     lmd = 10;
%     gma = 1.2;
      para3=1;
      para4=1;


    B=cell(size(data)); % B=inv(X'X+alpha*I)^(-1)
    %========for========
    for i=1:length(lmd)
        for vnum=1:viewN % view
            B{vnum}= inv(data{vnum}'*data{vnum}+lmd(i)*eye(size(data{1},2)));
        end
        for j=1:length(gma)
            for p = 1:length(para3)
                for q = 1:length(para4)
                    disp(['alpha:',num2str(lmd(i)),' gamma:',num2str(gma(j)),' para3:',num2str(para3(p)),' para4:',num2str(para4(q))]);
                    %[result,FVret,obj,w]=CLOMV(data_nml,label,B,para1(i),para2(j),FV,WV,LV);
                    for it = 1:1
                        tic;
                        [res,Fv,Fstar,Lv,Wv,Yres,obj]=CLOMV(data,label,B,lmd(i),gma(j),para3(p),para4(q),FV,WV,LV);
                        t=toc;
                        disp(['ACC: ',num2str(res(end,7)), '  Time: ',num2str(t)]);
                    
                    folder = ['.\newresult\v1\',datasetname{di},'\'];
                    if exist(folder)==0   
                        mkdir(folder);
                    else
                        dir=1;
                    end
                    filename=[folder,num2str(lmd(i)),'_',num2str(gma(j)),'_',...
                        num2str(para3(p)),'_',num2str(para4(q)),'_',num2str(it),'.mat'];
                    save(filename,'Wv','Fv','Fstar','Yres','obj','res');
                    savetxt = ['.\newresult\v1\',datasetname{di},'-_CLOMV_0929.txt'];
                    dlmwrite(savetxt,[lmd(i) gma(j) para3(p) para4(q) res(end,1) res(end,2) res(end,3) res(end,4) res(end,5) res(end,6) res(end,7) res(end,8) t],'-append','delimiter','\t','newline','pc');
                    end
                end
            end
        end
    end
end
