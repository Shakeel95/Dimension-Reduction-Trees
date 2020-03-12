% example 1 factor with AR loadings only twosided and unrestricted estimation 
% is possible as nstaticfactors = infinite

clear all
clc

%% simulated data
N = 100;
T = 120;
q = 3;
noise = 0.4;     % the variance of idio term over the variance of the panel

randn('state',sum(100*clock));
rand('state',sum(100*clock));

factors = randn(T+50,q);

coeff1 = randn(q,N);
coeff2 = rand(q,N)*1.6-.8;
coeff3 = rand(q,N)*1.6-.8;

for i=1:N,
    a = zeros(T+50,1);
    for j=1:q,
        a = coeff1(j,i)*filter(1,[1 coeff3(j,i)],filter(1,[1 coeff2(j,i)],factors(:,j))) + a;
    end
    common(:,i) = a(51:T+50);
end

common = sqrt(1-noise)*(common- ones(T,1)*mean(common))./(ones(T,1)*std(common));

err = randn(T+20,N+20);

idio = (err(5:T+4,5:N+4)+0.5*err(5:T+4,4:N+3)+0.5*err(5:T+4,6:N+5) +0.3*err(5:T+4,3:N+2)+0.3*err(5:T+4,7:N+6) + ...
    0.5*err(6:T+5,5:N+4)  - 0.3*err(6:T+5,4:N+3));

idio = sqrt(noise)*(idio- ones(T,1)*mean(idio))./(ones(T,1)*std(idio));

x = common + idio;

%% estimation
m = floor(sqrt(T)); 
h = m;
nfactors = q;
k = 1;

[chi_2, xi_2, X] = gdfm_twosided(x, nfactors, m, h, 0);
[chi_3, CL, v, C1] = gdfm_unrestricted(x, nfactors, k, m, 20, 1:q, q+1, 100);

series = 10;
figure
plot(common(:,series))
hold all
plot(chi_2(:,series))    
plot([NaN(k,1); chi_3(:,series)])          
legend('Simulated','FHLR00','FHLZ17')
title('Estimates of common components')

MSE(:,1) = mean(mean((chi_2-common).^2));
MSE(:,2) = mean(mean((chi_3-common(k+1:end,:)).^2));
