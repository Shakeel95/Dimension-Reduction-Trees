% function [chi, xi, X, vardec] = gdfm_twosided(panel, q, m, h, vardec_opt)
%
% Function to estimate the factor decomposition according to
% Forni Hallin Lippi Reichlin (2000) "The Generalized Dynamic Factor Model: 
% Identification and Estimation", The Review of Economics and Statistics,
% 82, 540-554
% 
% INPUT:    panel           :   T x n data matrix 
%                               data should be covariance stationary 
%           q               :   number of dynamic factors 
%                               (run numfactors.m to determine q)
%           m               :   covariogram truncation
%                               (default value: floor(sqrt(T)))
%           h               :   number of points in which the spectral 
%                               density is computed (default value: m)
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
%          vardec           :   n x 1 vector with variance explained by 
%                               each factor (only if vardec_option == 1)
 
function [chi, xi, X, vardec] = gdfm_twosided(panel, q, m, h, vardec_opt)

%% Preliminary settings
[T,n] = size(panel);

if q > n 
    disp('ERROR MESSAGE: Number of factors higher than dimension'); 
    return 
end

if nargin < 2 
    disp('ERROR MESSAGE: Too few input arguments'); 
    return 
end
 
if nargin == 2 
    m = floor(sqrt(T)); 
    h = m; 
    vardec_opt = 1;  
end

if nargin == 3 
    h = m; 
    vardec_opt = 1; 
end

if nargin == 4 
    vardec_opt = 1; 
end

if nargout == 4 && vardec_opt == 0  
    disp('ERROR MESSAGE: Too many output arguments'); 
    return 
end

if nargout == 3 && vardec_opt == 1  
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
H = 2*h + 1;
K_chi = zeros(n,n,H);                                                       
chi = zeros(T+2*m,n); 
                                                                            
K_chi(:,:,h+1) = P_chi(:,:,h+1)*P_chi(:,:,h+1)';                            % frequency zero

for j=1:h                                                                   % other frequencies
    K_chi(:,:,j) = P_chi(:,:,j)*P_chi(:,:,j)';  
    K_chi(:,:,H+1-j) = conj(K_chi(:,:,j));
end

Factor = exp(-sqrt(-1)*(-m:m)'*(-2*pi*h/H:2*pi/H:2*pi*h/H));                % the "e^(i*k*theta)" factor of the integral
for j = 1:n
    K = real(conj(Factor)*squeeze(K_chi(:,j,:)).'/H);                       % create filter K(L)
    chi = chi + conv2(K,X(:,j));                                            % covolution of K and X   
end

chi = chi(m+1:T+m,:);                                                       % T x n insample estimator of the common component
xi = X - chi;                                                               % T x n insample estimator of the idiosyncratic component 
