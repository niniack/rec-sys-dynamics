function [Xr,err,error]=MC_nuclear1(M,M_s,lambda1,lambda2,tol)
% M is the incomplete matrix with nan values
% M_s acts as the mask, if M(i,j)=nan, M_s(i,j)=0, otherwis M_s(i,j)=1
% lambda1 is always set to be small 
% lambda2 is always larger than lambda2
[I,J]=size(M);
err=100;
X_o=zeros(I,J);
% X_o=ones(I,J);
Z=zeros(I,J);
Gamma=zeros(I,J);
iter=1;
% Niter=1000;
error = 100;
while(err>tol)

% for k=1:Niter
    E=Z-Gamma/lambda2-(X_o-M).*M_s/lambda2;
%    E=Z-Gamma/lambda2-(X_o-M.*M_s)/(lambda2);
    [u,s,v]=svd(E+Gamma/lambda2,'econ');
    S=sign(s).*max(abs(s)-(lambda1/lambda2),0);
    Z=u*S*v';
    Z=max(Z,0);% make Z nonnegative
    Z=min(Z,1);
    Gamma=Gamma+lambda2*(E-Z);
    err=max(norm(X_o-Z),norm(X_o.*(ones(I,J)-M_s)-Z.*(ones(I,J)-M_s)));
    error(iter,1)=err;
    X_o=Z;
%     X_o =0.2.*Z + 0.8.*X_o; 
    lambda2=min(1.01*lambda2,100);
    iter=iter+1;
end
%Xr=round(X_o);
Xr = X_o;
end

