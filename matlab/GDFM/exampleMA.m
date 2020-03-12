% example 1 factor with MA loadings and 1 lag, i.e. 2 static factors

clear all
clc

%% simulated data
N = 100;
T = 120;
q = 3;
noise = 0.4;     % the variance of idio term over the variance of the panel

randn('state',sum(100*clock));
rand('state',sum(100*clock));

i2 = 2;                       % number of lags of MA
factors = randn(T+i2,q);
sfactors = factors(1+i2:T+i2,:);

for i1=1:i2
    sfactors = [sfactors factors(1+i2-i1:T+i2-i1,:)];
end

coeff = rand((i2+1)*q,N);
common = sfactors*coeff;

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
nstaticfactors = q*(i2+1);
k = 1;

[chi_1, xi_1, ~, factors, loadings, forecast_chi] = gdfm_onesided(x, nfactors, nstaticfactors, m, h, 1, 0);
[chi_2, xi_2, ~] = gdfm_twosided(x, nfactors, m, h, 0);
[chi_3, CL, v, C1] = gdfm_unrestricted(x, nfactors, k, m, 20, 1:q, q+1, 100);

series = 10;
figure
plot(common(:,series))
hold all
plot(chi_1(:,series))
plot(chi_2(:,series))          
plot([NaN(k,1); chi_3(:,series)])          
legend('Simulated','FHLR05','FHLR00','FHLZ17')
title('Estimates of common components')

MSE(:,1) = mean(mean((chi_1-common).^2));
MSE(:,2) = mean(mean((chi_2-common).^2));
MSE(:,3) = mean(mean((chi_3-common(k+1:end,:)).^2));