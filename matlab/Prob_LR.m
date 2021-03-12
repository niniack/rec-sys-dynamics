function prob = Prob_LR(X,weight)
% Calculates the probalbility of a user to be leftist or rightist
% weight is the column vector specifying the community of items 
%weight(left_item) = -1 and weight(right item) = 1;
% X is user-item matrix of likes and dislikes. Like = 1 and dislike = -1
% *********IF X IS A 0s AND 1s MATRIX, CONVERT 0s TO -1.**********

[Nusr,Nitm] = size(X);
%  beta = zeros(Nusr,1);

 for i = 1:Nusr
    for  j = 1:Nitm
     if X(i,j)==0
       X(i,j)=-1  ;
     elseif X(i,j)==2
       X(i,j)=0;  
         
     end
    end
 end
 
 
beta = (X*weight)./Nitm;

%probablity of beta by binning
% bin size = 0.2
% beta varies from -1 to 1 hence total range is 2
betamin = -1;
betamax = 1;
db = 0.05;
Nbins = ceil(2/db)+1;% 1 is offset to include beta = 0
Nsf = zeros(Nbins,1);
prob = zeros(Nbins,2);
for i = 1:Nusr
    bin = nearest((beta(i)-betamin)/db)+1;
    
    Nsf(bin) = Nsf(bin) + 1;
end
prob(:,2) = Nsf/Nusr;
for bin = 1:Nbins
prob(bin,1) = (bin-1)*db+betamin;
end