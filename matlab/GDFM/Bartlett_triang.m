% function w = Bartlett_triang(N)
% Compute a triangular window (similar Bartlett)
% 
% INPUT
% N number of points in the window
% 
% OUTPUT
% w triangular window of size N

function w = Bartlett_triang(N)

if N<=0
    disp(('ERROR MESSAGE: number of points must be positive'));
end

N_out = 0;
w = [];

if  N == floor(N),
   N_out = N;
else
   N_out = round(N);
   disp(('WARNING MESSAGE: rounding to nearest integer'));
end

if isempty(N_out) || N_out == 0,
   w = zeros(0,1); 
   return
elseif N_out == 1,
   w = 1;   
   return
end

if rem(N_out,2)
    % It's an odd length sequence
    w = 2*(1:(N_out+1)/2)/(N_out+1);
    w = [w w((N_out-1)/2:-1:1)]';
else
    % It's even
    w = (2*(1:(N_out+1)/2)-1)/N_out;
    w = [w w(N_out/2:-1:1)]';
end



    
% [EOF] triang.m
