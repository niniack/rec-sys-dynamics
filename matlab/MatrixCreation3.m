% U = 100;
% N= 100;
% NLU = 10;
% NRU = 10;
% deg=4;
% Mtx = MatrixCreation3(U,N,Nextl,Nextr,deg);


function[Mtx] = MatrixCreation3(U,N,NLU,NRU,NLI,NRI,deg)
% function to compute the input matrix. deg=k0 is the degree
% U - # of users
% N - # of items
% only (U,N)=(100,100) and (200,200) are available
% beta0 - bias factor
% Next - number of extremists
% k0 - degree of a user; same for all users
% beta0 = 0.01,0.03,0.05 only these options are available.

% (U,N)=(100,100)
Mtx = 2.*ones(U,N);


for usr = NLU+1:U-NRU
    rnditm = randsample(N,deg);
    for itm = 1:deg
        Mtx(usr,rnditm(itm)) = randsample([1 0],1);
        
    end
    
end

if NLU >0 
 for usr = 1:NLU
 Mtx(usr,1:NLI)=1;     
 Mtx(usr,N-NRI+1:N)=0; 
 if (N-NRI - (NLI+1))>0
 rnditm = randsample(NLI+1:N-NRI,deg);
    for itm = 1:deg
        Mtx(usr,rnditm(itm)) = randsample([1 0],1);
    end
 end  
 end
    
end

if NRU >0  
 for usr = U-NRU+1:U
 Mtx(usr,1:NLI)=0;     
 Mtx(usr,N-NRI+1:N)=1;  
 if (N-NRI - (NLI+1))>0
 rnditm = randsample(NLI+1:N-NRI,deg);
    for itm = 1:deg
        Mtx(usr,rnditm(itm)) = randsample([1 0],1);
    end
    
 end
 end
    
end

end %function end






