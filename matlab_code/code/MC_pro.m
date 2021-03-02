function [B,P,pcutoff] = MC_pro(Xr,Mtx)

[I,J]=size(Xr);

for i=1:I
     [m_observed,n_observed]=find(Mtx(i,:)~=2);
    b=glmfit(Xr(i,n_observed)',Mtx(i,n_observed)','binomial','link','logit');
%    p=glmval(b,Xr(i,:)','logit');
    B(i).coe=b;
    P(i,:) = 1./ (1 + exp(-b(1) - b(2) .*Xr(i,:) ));
    pcutoff(i) = -b(1)/b(2);
    
%     P(i,:)=p;
    
end

end

