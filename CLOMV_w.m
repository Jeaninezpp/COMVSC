function [ress,FV,Fstar, LV,WV,Yres,obj]=CLOMV_w(X,ylabel,B,alpha,gamma,para3,para4,FV,WV,LV)

MaxIter = 100; 
v1=length(X);% # of view
n=length(ylabel);  % # of samples
c=length(unique(ylabel));% # of clusters
R = eye(c);
Fstar = RandOrthhMat(n,c);
% Fstar = rand(n,c);
T = eye(c);

% initial Y of one 1 in each row
Ik = eye(c);
randorder = randperm(size(Ik,1));
numceil = ceil(n/c);
largeY = repmat(Ik(randorder,:),numceil,1);% (size(A,1)*M, size(A,2)*N)
Yres = largeY(1:n,:);% N*k
[~,res_label] = max(Yres,[],2);
%% update Wv, Fv, Fstar, Y, R
for i=1:MaxIter
    resold = res_label;
    % disp([num2str(i),'iter']);
    for num = 1:v1% begin a view
        Fv=FV{num};
        xx=X{num}'*X{num};% n*n dim
        
        P = calculate(Fv);
        %% Zv
% % distance
%         for ij=1:n
%          	h=distance(Fv,n,ij);% 1*n  //    h_ij = norm( f_ij - fj )^2
%         	Zv(:,ij)=B{num}*(xx(:,ij) - 1/4*para3*h') ;% hyper para�? beta
%         end

% % closed-form
        Zv = 0.5 * B{num}*(2*xx-P/2);


        %%-projection
        Zv = Zv - diag(diag(Zv));
        for ii = 1:size(Zv,2)
            idx= 1:size(Zv,2);
            idx(ii) = [];
            Zv(ii,idx) = EProjSimplex_new(Zv(ii,idx));
        end

        
        % nor L
        D = diag(sum(Zv));
        L = D-(Zv+Zv')/2;
        ZV{num} = Zv;
        LV{num} = L;
    end

    %% F*
    SumFv = zeros(size(FV{1}));
    for num = 1:v1
        SumFv = SumFv + FV{num};
    end
    G = Yres.^gamma;
    FGR=2*para4*SumFv+G*R';
    
    [Uf,~,Vf] = svd(FGR,'econ');
    Fstar = Uf*Vf';
    
%     [a,~] = kmeans(Fstar,c);
%     [Fstar_kmresult(i,:)] = Clustering8Measure(ylabel,a); 
%     disp(['Fstar_acc: ', num2str(Fstar_kmresult(i,7))]);
    
%     if i == 1
% %          Fstar_tsne{i} = tsne(Fstar, ylabel);
% %          name = ['iter',num2str(i)];
% %          saveas(gcf,name,'epsc');
%         pause
%     elseif i==25
%         Fstar_tsne = tsne(Fstar, ylabel);
%         pause
%     elseif i==50
%         Fstar_tsne = tsne(Fstar, ylabel);
%         pause
%     elseif i==75
%         Fstar_tsne = tsne(Fstar, ylabel);
%         pause
%     elseif i==100
%         Fstar_tsne = tsne(Fstar, ylabel);
%         pause
%     end

    %% Fv Reweighting
    for vidx = 1:v1
        fv=FV{vidx};
        lv = LV{vidx};
        
        [~,L_lmd,~] = eig1(lv,c);
        L_lmd_max = max(L_lmd); 
        Lmx= L_lmd_max*eye(size(lv)) - lv;
        
        for rep = 1:100
            M = 2 * para4 * Lmx * fv+2 * Fstar;
            [Um,~,Vm] = svd(M,'econ');
            fv = Um*Vm';
            
            fobj(rep+1) = trace( fv' * Lmx * fv ) + 2 * trace( fv' * Fstar );
            if rep>4 && ((fobj(rep)-fobj(rep-1))/fobj(rep)<1e-3)
                break;
            end
        end
        FV{vidx}=fv;
    end

% %% Fv solveF
%     opts.record = 0;
%     opts.mxitr  = 1000;%1000
%     opts.xtol = 1e-5;
%     opts.gtol = 1e-5;
%     opts.ftol = 1e-8;
%     out.tau = 1e-3;
% 
%     for vidx = 1:v1
%         fv=FV{vidx};
%         lv=LV{vidx};
%         [fv,out]=solveF(fv,@fun1,opts,para4/para3,Fstar,T,lv);
%         FV{vidx}=fv;
%     end
%     obj(i,3)=cal_obj(X,n,c,v1,alpha,gamma,para3,para4,ZV,LV,FV,Fstar,R,Yres);

    %% R
    % max tr(R'Fstar'G)
    [Ur,~,Vr] = svd(Fstar'*G,'econ');
    R = Ur*Vr';
  
    %% updata Y
    E=zeros(n,c);
    for ei = 1:n
        for ec = 1:c
            E(ei,ec) = norm( T(ec,:) - Fstar(ei,:) * R , 2)^2;
        end
    end

    if gamma == 1
        for yi = 1:n
            [~,yindex]=min(E(yi,:));
            Yres(yi,yindex)=1;% n*c result
        end
        [~,res_label] = max(Yres,[],2);
    else
        Yup = E.^(1/(1-gamma)); % n × k
        Ydown = sum(Yup,2);% n × 1 //sum of a row
        Yres = Yup ./ repmat(Ydown,1,c); % n × k result
        
        [~, res_label] = max(Yres, [], 2);% 
    end
 
	%disp('After Y');
    obj(i,5)=cal_obj(X,n,c,v1,alpha,gamma,ZV,LV,FV,Fstar,R,Yres);
   
    [ress(i,:)] = Clustering8Measure(ylabel,res_label);

% %     convergence
    objres(i) = norm(res_label - resold)/norm(resold);
    if i>99 && (norm(res_label - resold)/norm(resold)<1e-3)
        break
    end
end%interaction end
end

% %% hi = || fi - fj ||^2
% function [all]=distance(F,n,ij)
%     for ji=1:n
%         all(ji)=(norm(F(ij,:)-F(ji,:)))^2;
%     end
% end
% 
% function [F,G]=fun1(P,alpha,Q,T,L) % Q=F*  P=Fv  T=I
%     G=2 * L * P - 2 * alpha * Q * T;
%     F=trace(P'*L*P)+(norm(Q-P*T,'fro'))^2;
% end
