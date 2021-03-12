clear

U = 200;
N = 200;
NLU = 60;
NRU = 60;
NLI = 60;
NRI = 60;
deg = 20;

[Mtx] = MatrixCreation3(U,N,NLU,NRU,NLI,NRI,deg);


M_s = zeros(N,N);
Mask = Mtx<2; 
M_s(Mask) = 1;

[Xr,err,error]=MC_nuclear1(Mtx,M_s,1.0,10.0,1e-3);

% [Xr,err,error]=MC_nuclear(Mtx,M_s,Xrtemp,1.0,10.0,1e-3);

[B,P,pcutoff] = MC_pro(Xr,Mtx);

