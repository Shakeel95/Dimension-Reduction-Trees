% function [chi, xi, X, factors, loadings, forecast_chi, vardec] = gdfm_onesided(panel, q, r, m, h, steps, vardec_opt)
%
% Function to estimate and forecast the factor decomposition according to
% Forni Hallin Lippi Reichlin (2005) "The Generalized Dynamic Factor Model: 
% One-sided Estimation and Forecasting", Journal of the American Statistical
% Association, 100, 830-840
% 
% INPUT:    panel           :   T x n data matrix 
%                               data should be covariance stationary 
%           q               :   number of dynamic factors 
%                               (run numfactors.m to determine q)
%           r               :   number of static factors
%           m               :   covariogram truncation 
%                               (default value: floor(sqrt(T)))
%           h               :   number of points in which the spectral 
%                               density is computed (default value: m)
%           steps           :   number of steps ahead in the forecast
%                               (default value: 1) it must be <= m
%           vardec_opt      :   option for to obtain explained variance 
%                               (yes == 1, no == 0) (default value: 1)
%                             
% OUTPUT:  chi              :   T x n common components of standardized 
%                               data
%          xi               :   T x n idiosyncratic components of 
%                               standardized data
%          X                :   T x n matrix of mean-standardized data
%                               if panel is already mean-standaridzed then
%                               X = panel
%          factors          :   estimate of static factors (not identified)
%          loadings         :   estimate of loadings (not identified)
%          forecast_chi     :   n x 1 forecast of common component 
%                               number of steps ahead defined in inputs
%                               destandardized using std(panel)
%          vardec           :   n x 1 vector with variance explained by 
%                               each factor (only if vardec_option == 1)
 
function [chi, xi, X, factors, loadings, forecast_chi, vardec] = gdfm_onesided(panel, q, r, m, h, steps, vardec_opt)

%% Preliminary settings
[T,n] = size(panel);

if q > n 
    disp('ERROR MESSAGE: Number of factors higher than dimension'); 
    return 
end

if r > n 
    disp('ERROR MESSAGE: Number of static factors higher than dimension'); 
    return 
end

if nargin < 3 
    disp('ERROR MESSAGE: Too few input arguments'); 
    return 
end
 
if nargin == 3 
    m = floor(sqrt(T)); 
    h = m; 
    steps = 1;
    vardec_opt = 1;  
end

if nargin == 4 
    h = m; 
    steps = 1;
    vardec_opt = 1; 
end

if nargin == 5 
    steps = 1;
    vardec_opt = 1;  
end

if nargin == 6 
    vardec_opt = 1;  
    if steps > m
        disp('ERROR MESSAGE: Too many steps ahead in forecasting');
    return
end

end

if nargout == 7 && vardec_opt == 0  
    disp('ERROR MESSAGE: Too many output arguments'); 
    return 
end

if nargout == 6 && vardec_opt == 1  
    disp('ERROR MESSAGE: Too few output arguments'); 
    return 
end

%% Mean-standardize data
m_X = mean(panel);
s_X = std(panel);
X = (panel - ones(T,1)*m_X)./(ones(T,1)*s_X);

%% Spectral analysis
[P_chi, D_chi, Sigma_chi] = spectral(X, q, h, m);                           % compute q largest dynamic eigenvalues    

if vardec_opt == 1
    [P_X, D_X, Sigma_X] = spectral(X, n, h, m);                             % compute all dynamic eigenvalues
    E = [D_X(:,h+1)  D_X(:,h+2:2*h+1)*2]*ones(h+1,1)/(2*h+1);               
    vardec = E./sum(E);
end

%% Estimation
M = 2*m + 1;
H = 2*h + 1;
Gamma_chi = zeros(n,n,M);                                                   

Factor = exp(-sqrt(-1)*(-m:m)'*(-2*pi*h/H:2*pi/H:2*pi*h/H));                % the "e^(i*k*theta)" factor of the integral
for j = 1:n
    Gamma_chi(:,j,:) = real(squeeze(Sigma_chi(:,j,:))*conj(Factor).'/H);    % lagged covariance matrix of the common component
end

Gamma_chi_0 = squeeze(Gamma_chi(:,:,m+1));                                  % contemporaneous covariance matrix of common component 
Gamma_X_0 = (X(1:T,:))'*(X(1:T,:))/(T-1);                                   % contemporaneous covariance matrix of observations
Gamma_xi_0 = Gamma_X_0 - Gamma_chi_0;                                       % contemporaneous covariance matrix of the idiosyncratic component

opt.disp = 0;
[P, D] = eigs(Gamma_chi_0, diag(diag(Gamma_xi_0)),r,'LM',opt);              % compute generalized eigenvectors and eigenvalues

factors = X*P;                                                              % static common factors 
loadings = Gamma_chi_0*P/(P'*Gamma_chi_0*P);                                % factor loadings

chi = (loadings*P'*X')';                                                    % T x n insample estimator of the common component 
xi = X - chi;                                                               % T x n insample estimator of the idiosyncratic component 

%% Forecasting
Gamma_chi_h = squeeze(Gamma_chi(:,:,m+1+steps));                            % steps-lagged covariance matrix of common component
new_loadings = Gamma_chi_h*P/(P'*Gamma_chi_0*P);                            % new factor loadings  
forecast_chi = (new_loadings*P'*X(T,:)')';                                  % 1 x n steps-ahead forecast of the common component
forecast_chi = forecast_chi.*s_X;
