% U = 100;
% N= 100;
% NLU = 10;
% NRU = 10;
% deg=4;
% Mtx = MatrixCreation3(U,N,Nextl,Nextr,deg);


function[Bias] = BiasMatrixCreation3(U,N,NLU,NRU,NLI,NRI,deg_bias)
% function to compute the input matrix. deg=k0 is the degree
% U - # of users
% N - # of items
% only (U,N)=(100,100) and (200,200) are available
% beta0 - bias factor
% Next - number of extremists
% k0 - degree of a user; same for all users
% beta0 = 0.01,0.03,0.05 only these options are available.

% (U,N)=(100,100)
Bias = zeros(U,N);


for usr = NLU+1:(U-NRU)/2
    Bias(usr,1:NLI)=-1;     
    Bias(usr,N-NRI+1:N)=1;
if (N-NRI - (NLI+1))>0
 rnditm = randsample(NLI+1:N-NRI,deg_bias);
    for itm = 1:deg_bias
        Bias(usr,rnditm(itm)) = randsample([1 -1],1);
    end
    
 end       
end

for usr = ((U-NRU)/2)+1:U-NRU
    Bias(usr,1:NLI)=1;     
    Bias(usr,N-NRI+1:N)=-1;
 if (N-NRI - (NLI+1))>0
 rnditm = randsample(NLI+1:N-NRI,deg_bias);
    for itm = 1:deg_bias
        Bias(usr,rnditm(itm)) = randsample([1 -1],1);
    end
 end
end

end %function end






