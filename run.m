clc;
clear;
addpath ./measure ./tsne
path = './'; 
datapath = '../0datasets/';


datasetname = {'bbcsport_seg14of4','yaleA_3view'};
for di = 1  % Datasets
    dataName = datasetname{di};
    load([datapath,'/',dataName,'.mat'],'X','Y');   % Ubuntu
    disp(['\n\n Current dataset  : ',dataName]);
    
    viewN = length(X);
    k = length(unique(Y));
    N = length(Y);
    
    for iv = 1:viewN
        X{iv} = mapstd(X{iv}',0,1); 
        data{iv} = X{iv}';
    end
    % X{i} is d * N
    % data{i} is N *d
    
    FV = cell(size(X));
    WV= cell(size(X));
    LV = cell(size(X));
    for i =1:viewN
        W = constructW_PKN(X{i}, 5, 1);  %   X : d*n  / 5 neighbors /  is symmetric
        D = diag(sum(W));
        L = D-W;
        [Fv, ~, ~]=eig1(L,k,0); 
        FV{i} = Fv;
        LV{i} = L;
        WV{i} = W;
    end


    lmd = 2.^[3:2:13];
    gma = [1.3:0.2:2.7];
    para3=1;
    para4=1; % Fixed

    
    B=cell(size(X)); % B=inv(X'X+alpha*I)^(-1)
    idx = 1; 
    for i=1:length(lmd)
        for vnum=1:viewN 
            % B{vnum}= inv(X{vnum}'*X{vnum}+lmd(i)*eye(size(X{1},2)));
            I_d{vnum} = eye(size(X{vnum},1)); % X : d*N
            I_n{vnum} = eye(size(X{vnum},2));
            B{vnum} = (1/lmd(i))*(I_n{vnum}-(1/lmd(i))*X{vnum}'*inv(I_d{vnum}+(1/lmd(i))*X{vnum}*X{vnum}')*X{vnum});
        end
        for j=1:length(gma)
            tic;
            [ress,Fv,Fstar,Lv,Wv,Yres,obj]=CLOMV_w(X,Y,B,lmd(i),gma(j),para3,para4,FV,WV,LV);
            t(idx)=toc;
            res_comvsc = ress(end,:);
            Result_11COMVSC(idx,:) = [lmd(i) gma(j) res_comvsc];
            disp(['Para1 = ',num2str(lmd(i)),'  Para2 = ',num2str(gma(j)), '  ACC = ', num2str(res_comvsc(:,7)),'  Time = ',num2str(t(idx))]);
            ourFscore(i,j)  = res_comvsc(1); ourPrecision(i,j) = res_comvsc(2); ourRecall(i,j) = res_comvsc(3); 
            ourNmi(i,j) = res_comvsc(4);ourAR(i,j) = res_comvsc(5); ourEntropy(i,j) = res_comvsc(6); 
            ourACC(i,j) = res_comvsc(7); ourPurity(i,j) = res_comvsc(8);
%             dlmwrite(['./', dataName, '.txt'], Result_11COMVSC(idx,:),'-append','delimiter','\t','newline','pc');
            idx = idx+1;
        end
    end
    maxresult = max(Result_11COMVSC,[],1);
    Allresult(1,:) = maxresult(3:10);
    fprintf('- Finish Proposed Method \n ');

    save(['./' , dataName ,'_Our.mat'], 't','Allresult');

end
