% Includes Saif's model for dynamics
%  M=readmatrix('M_sparse.csv');
inpcase = '1inp';
dyntype = '5';
p = 0.8;
p_0 = 0.01;
No = 20; %[10 20 30 50 70];

NewItemdeg = 4;
repeat = false;
global U N NLI NRI NLIX NRIX NRU NLU
switch inpcase
    
 case '1inp'
% Use only U=N case     
U = 200;
N= 200;
NLU = 20;
NRU = 20;
NLIX = 60;
NRIX = 60;
NRI = 100;
NLI = 100;
% NLI+ NRI = N always
deg=8;
deg_bias = 10;
Mtx = MatrixCreation3(U,N,NLU,NRU,NLIX,NRIX,deg);
Bias = BiasMatrixCreation3(U,N,NLU,NRU,NLIX,NRIX,deg_bias);
weight(1:NLI,1) =-1;
weight(NLI+1:N,1) = 1;

 case 'repeat'
     
Noprev = 200;
%  repeat = true;
        filename = sprintf('N%dNo%d.mat',N,Noprev);
 load(filename)
 Noprev = 200;
 repeat = true;
 
 
end % switch end for input cases
%%%%%%%%%%%%%%%%%%%%%%%%%%

% for row = 11:Ne-10
%    Mtxtemp(row,:) = Mtx(row,randperm(100)); 
% end

Xrtemp = zeros(N,N);
pcutoff_l = zeros(No,U);



for l = 1:No  % iterations for evolution
tic
M_s = zeros(N,N);
Mask = Mtx<2; 

M_s(Mask) = 1;


[Xr,err,error] = MC_nuclear(Mtx,M_s,Xrtemp,1.0,10.0,1.0e-5);

[Bb,Pp,pcutoff] = MC_pro(Xr,Mtx);

pcutoff_l(l,1:U)=pcutoff;


like = (Xr == 1 & M_s == 0);
dislike = (Xr == 0 & M_s == 0);


switch dyntype
 
    case '1'
% user sees both likes and dislikes and randomly samples between the recommendations
% Rejects the recommendations with prob. p. i.e., flips the
% recommendations with prob. p
for row = 1:N
    To_be_liked = find(like(row,:)==1);
    To_be_disliked = find(dislike(row,:)==1);
    To_be_recommended = [To_be_liked, To_be_disliked];

    if size(To_be_recommended,2) == 1
            Mtx(row, To_be_recommended) = Xr(row, To_be_recommended);
            if rand < p
                Mtx(row, To_be_recommended) = 1 - Xr(row, To_be_recommended);
            end
        elseif size(To_be_recommended,2)>1
            recommend = randsample(To_be_recommended, 1);
            Mtx(row, recommend) = Xr(row, recommend);
            if rand < p
                Mtx(row, recommend) = 1 - Xr(row, recommend);
            end
    end
 

end
    case '2'
 % user sees only likes and randomly samples among them. flips the
% recommendations with prob. p      
    for row = 1:N
    To_be_liked = find(like(row,:)==1);
    To_be_disliked = find(dislike(row,:)==1);
    To_be_recommended = [To_be_liked];

    if size(To_be_recommended,2) == 1
            Mtx(row, To_be_recommended) = Xr(row, To_be_recommended);
            if rand < p
                Mtx(row, To_be_recommended) = 1 - Xr(row, To_be_recommended);
            end
        elseif size(To_be_recommended,2)>1
            recommend = randsample(To_be_recommended, 1);
            Mtx(row, recommend) = Xr(row, recommend);
            if rand < p
                Mtx(row, recommend) = 1 - Xr(row, recommend);
            end
    end
 

    end      
        
    
    case '3'
 % user picks item from recommended likes with prob. p and randomly samples among them. With 1-p the user 
 %explores over all set of items.
    for row = 1:N
    To_be_liked = find(like(row,:)==1);
    To_be_disliked = find(dislike(row,:)==1);
     To_be_recommended = [To_be_liked, To_be_disliked];
%      if size(To_be_recommended,2)<1
%         fprintf('Error:To_be_recommended<1=%d\n',size(To_be_recommended))
% %         exit
%      end
     if size(To_be_recommended,2) == 1
            Mtx(row, To_be_recommended) = Xr(row, To_be_recommended);
     end      
     
     temprand = rand;
     
     if temprand <= p
      if size(To_be_liked,2)>1   
        recommend = randsample(To_be_liked, 1);
        Mtx(row, recommend) = Xr(row, recommend);
      end
     elseif temprand > p 
       if size(To_be_recommended,2)>1    
        recommend = randsample(To_be_recommended, 1);
        Mtx(row, recommend) = Xr(row, recommend);
%           Mtx(row, recommend) = randsample([1 -1],1);
       end
     end
     
    end      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        

    case '4'
 % user picks item from recommended likes with prob. p and randomly samples among them.
 %With 1-p the user 
 %explores over all set of items. Further, an item is replaced after each
 %ten iterations. 
    for row = 1:N
    To_be_liked = find(like(row,:)==1);
    To_be_disliked = find(dislike(row,:)==1);
     To_be_recommended = [To_be_liked, To_be_disliked];
%      if size(To_be_recommended,2)<1
%         fprintf('Error:To_be_recommended<1=%d\n',size(To_be_recommended))
% %         exit
%      end
     if size(To_be_recommended,2) == 1
            Mtx(row, To_be_recommended) = Xr(row, To_be_recommended);
     end      
     
     temprand = rand;
     
     if temprand <= p
      if size(To_be_liked,2)>1   
        recommend = randsample(To_be_liked, 1);
        Mtx(row, recommend) = Xr(row, recommend);
      end
     elseif temprand > p 
       if size(To_be_recommended,2)>1    
%         recommend = randsample(To_be_recommended, 1);
%         Mtx(row, recommend) = Xr(row, recommend);
          Mtx(row, recommend) = randsample([1 -1],1);
       end
     end
    end
     
  % Randomly pick an item. If it is from left community, add rating from
  % availble leftists in the system and so on. 
 
  
if mod(l,1)==0  % item replacement after every ten time steps.
    
     repitem = randsample(N,1);
  Mtx(:,repitem)=2;
  if repitem <= NLI % convention used is 1 to NLI are leftist columns(items)
  u =zeros(N,2);
Nbeta = zeros(N,1);


for i = 1:N
u(i,1) = 1*size(find(Xr(i,1:NLI)==1),2)+...
                (-1)*size(find(Xr(i,1:NLI)==0),2);
% u(i,1) = -1*u(i,1);            % left community weight -1 not included!!
u(i,2) =  1*size(find(Xr(i,NLI+1:NLI+Nrighti)==1),2)+...
                (-1)*size(find(Xr(i,NLI+1:NLI+Nrighti)==0),2); % weight is 1 in this case
            
Nbeta(i) = (-1)*u(i,1) + u(i,2);
end

[Bb,Ii] = mink(Nbeta,NewItemdeg);

Mtx(Ii(:),repitem)=1;

  elseif repitem >NLI    
        u =zeros(N,2);
Nbeta = zeros(N,1);

for i = 1:N
u(i,1) = 1*size(find(Xr(i,1:NLI)==1),2)+...
                (-1)*size(find(Xr(i,1:NLI)==0),2);
% u(i,1) = -1*u(i,1);            % left community weight -1 not included!!
u(i,2) =  1*size(find(Xr(i,NLI+1:NLI+Nrighti)==1),2)+...
                (-1)*size(find(Xr(i,NLI+1:NLI+Nrighti)==0),2); % weight is 1 in this case
            
Nbeta(i) = (-1)*u(i,1) + u(i,2);
end

[Bb,Ii] = maxk(Nbeta,NewItemdeg);

Mtx(Ii(:),repitem)=1;
      
  end     
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case '5'
       
        % Here, the output of the RS i.e. Xr, is a probability matrix
        % Input Mtx is the rating matrix.
        %  A user decides to rate with probability pi = p_0(1-abs(x_ij)) and not
        % to rate with probability 1-pi. An element of Xr will be rounded to 1 with prob x_ij and to 0
        % with probability 1-x_ij if a user decides to rate.
    
   % Biased User
     for i = 1:NLU %Leftist user(s) placed at the top most rows for convenience
      if size(find(Mtx(i,1:N)==2),2)>0
         IBrate =  randsample(find(Mtx(i,1:N)==2),1); %IBrate: item index for bias user to rate
      
      if IBrate <=NLI
          Mtx(i,IBrate) = 1;
      else
          Mtx(i,IBrate) = 0;
      end
     end   
     end
         
      for i = U-NRU+1:U %Rightist user(s) placed at the bottom most rows for convenience
      if size(find(Mtx(i,1:N)==2),2)>0    
      IBrate =  randsample(find(Mtx(i,1:N)==2),1); %IBrate: item index for bias user to rate
      
      if IBrate <=NLI
          Mtx(i,IBrate) = 0;
      else
          Mtx(i,IBrate) = 1;
      end
      end   
     end
   
        
rated_count = 0;        
  %Common User 
    for i = NLU+1:U-NRU
        
        for j = 1:N
     weight = exp(-rated_count/20);
     thresh = pcutoff(i)+weight*Bias(i,j);
     
     if thresh > 1
         thresh = 1;
     elseif thresh < 0
         thresh = 0;
     end
     
     if Mtx(i,j)==2
           piij = p_0;
     else
         piij = 0;
     end
   temprand = rand;
   if temprand <= piij% user decides to rate
       rated_count = rated_count +1;
       temprand2 = rand;
 %      if temprand2 <= Xr(i,j)
        if temprand2 <= thresh
           Mtx(i,j) = 1;%nearest(Xr(i,j));
       %elseif temprand2 > Xr(i,j)
        elseif temprand2 > thresh
           Mtx(i,j) = 0;%1-nearest(Xr(i,j));
       end
   end
        end
   end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
end %end switch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
Xrtemp = Xr;


if repeat == false
if mod(l,10)==0
  probMtx = Prob_LR(Mtx,weight);  
  probXr = Prob_LR(Xr,weight);
filename = sprintf('N%dNo%d.mat',N,l);
save(filename);
end
if l ==1
 probMtx = Prob_LR(Mtx,weight);  
 probXr = Prob_LR(Xr,weight);
filename = sprintf('N%dNo%d.mat',N,l);
save(filename);
end
elseif repeat == true
 if mod(l,10)==0
  probMtx = Prob_LR(Mtx,weight);  
  probXr = Prob_LR(Xr,weight);
filename = sprintf('N%dNo%d.mat',N,l+Noprev);
save(filename);
end
if l ==1
 probMtx = Prob_LR(Mtx,weight);  
 probXr = Prob_LR(Xr,weight);
filename = sprintf('N%dNo%d.mat',N,l+Noprev);
save(filename);
end   
    
    
end

u =zeros(N,2);

% writing video of state space evolution
for i = 1:N
u(i,1) = 1*size(find(Mtx(i,1:NLI)==1),2)+...
                (-1)*size(find(Mtx(i,1:NLI)==0),2);
% u(i,1) = u(i,1);            % weight -1 of community is not taken care of.
u(i,2) =  1*size(find(Mtx(i,N-NRI+1:N)==1),2)+...
                (-1)*size(find(Mtx(i,N-NRI+1:N)==0),2); % weight is 1 in this case
            
            
end
plot(u(:,1),u(:,2),'*')
set(gca, 'XAxisLocation', 'origin')
set(gca, 'YAxisLocation', 'origin')
xlim([-NLI NRI]);
ylim([-NLI NRI]);
figframe(l) = getframe(gcf);
drawnow;

toc
end

writerObj = VideoWriter('StateSpace.mp4');
  writerObj.FrameRate = 10;
  
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(figframe)
    % convert the image to a frame
    frame = figframe(i) ;    
    writeVideo(writerObj, frame);
end


% close the writer object
close(writerObj);
clear figframe;

MAsk = pcutoff_l<0;
pcutoff(MAsk) = 0;

MAsk = pcutoff_l>1;
pcutoff(MAsk) = 1;

save('pcutoff.mat','pcutoff_l');

plot(pcutoff_l);
title('Progression of Probability Cutoff Values for Users')
xlabel('Number of Iterations') 
ylabel('Probability Cutoff') 
saveas(gcf,"pcutoff_plot.png");

pcutoff_avg = mean(pcutoff_l,2);
plot(pcutoff_avg,'*-');
title('Average Probability Cutoff by Iteration - All Users')
xlabel('Number of Iterations') 
ylabel('Average Probability Cutoff over all Users') 
saveas(gcf,"pcutoff_average_all_plot.png");

pcutoff_avg = mean(pcutoff_l(:,NLU+1:U-NRU),2);
plot(pcutoff_avg,'*-');
title('Average Probability Cutoff by Iteration - Unbiased Users')
xlabel('Number of Iterations') 
ylabel('Average Probability Cutoff over Unbiased Users') 
saveas(gcf,"pcutoff_average_unbiased_plot.png");

pcutoff_avg = mean(pcutoff_l(:,1:NLU),2);
plot(pcutoff_avg,'*-');
title('Average Probability Cutoff by Iteration - Left Biased Users')
xlabel('Number of Iterations') 
ylabel('Average Probability Cutoff over Left Biased Users') 
saveas(gcf,"pcutoff_average_left_plot.png");

pcutoff_avg = mean(pcutoff_l(:,U-NRU+1:U),2);
plot(pcutoff_avg,'*-');
title('Average Probability Cutoff by Iteration - Right Biased Users')
xlabel('Number of Iterations') 
ylabel('Average Probability Cutoff over Right Biased Users') 
saveas(gcf,"pcutoff_average_right_plot.png");