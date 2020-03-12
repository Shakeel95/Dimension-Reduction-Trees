% [nfact, v_nfact, cr] = numfactors(panel, q_max, nbck, stp, c_max, penalty, cf, m, h, plot_opt)
% 
% log criterion to determine the number of dynamic factors according to 
% Hallin and Liska (2007) "Determining the Number of Factors in the General 
% Dynamic Factor Model", Journal of the American Statistical Association, 
% 102, 603-617     
%
% INPUT:    panel           :   T x n data matrix 
%                               data should be covariance stationary 
%           q_max           :   upper bound on the number of factors  
%           nbck, stp       :   T x n_j subpanels are used where
%                               n_j = n - nbck : stp: n 
%                               (default value: nbck = floor(n/4), stp = 1)
%           c_max           :   c = [0:cmax] (default value: 3)
%           penalty         :   p1 = ((m/T)^0.5 + m^(-2) + n^(-1))*log(min([(T/m)^0.5;  m^2; n]))  
%                               p2 = (min([(T/m)^0.5;  m^2; n])).^(-1/2)  
%                               p3 = (min([(T/m)^0.5;  m^2; n])).^(-1)*log(min([(T/m)^0.5;  m^2; n]))
%                               (default value: 'p1')
%           cf              :   1/cf is granularity of c 
%                               (default value: 1000)
%           m               :   covariogram truncation 
%                               (default value: floor(sqrt(T)))
%           h               :   number of points in which the spectral 
%                               density is computed (default value: m)
%           plot_opt        :   option to draw the plot 
%                               (yes == 1, no == 0)(default value: 1)
%
% OUTPUT:   nfact           :   number of dynamic factors as function of c
%                               computed for n_j = n
%           v_nfact         :   variance in the number of dynamic factors
%                               as function of c and computed as the 
%                               n_j varies  
%           cr              :   values of c (needed for the plot)

function [nfact, v_nfact, cr] = numfactors(panel, q_max, nbck, stp, c_max, penalty, cf, m, h, plot_opt)

%% Preliminary settings
[T,n] = size(panel);

if nargin < 2 
    disp('ERROR MESSAGE: Too few input arguments'); 
    return 
end

if nargin == 2
    nbck = floor(n/4);
    stp = 1;
    c_max = 3;
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 1;
end

if nargin == 3
    stp = 1;
    c_max = 3;
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 1;
end

if nargin == 4
    c_max = 3;
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 1;
end

if nargin == 5   
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 1;
end

if strcmp(penalty, 'p1') == 0 && strcmp(penalty, 'p2') == 0 && strcmp(penalty, 'p3') == 0
    disp('ERROR MESSAGE : Penalty function can only take 3 values: p1, p2 and p3');
    return
end

if nargin == 6
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 1;
end

if nargin == 7
    m = floor(sqrt(T));
    h = m;
    plot_opt = 1;
end

if nargin == 8
    h = m;
    plot_opt = 1;
end

if nargin == 9
    plot_opt = 1;
end

%% Mean-standardize data
m_X = mean(panel);
s_X = std(panel);
X = (panel - ones(T,1)*m_X)./(ones(T,1)*s_X);

%% Compute the number of dynamic factors
s=0;
for N = n-nbck:stp:n
    disp(sprintf('subsample size %d',N));
    s = s+1;
    [a rv] = sort(rand(n,1));                                                 % select randomly N series
    subpanel = X(1:T,rv(1:N));

    m_subpanel = mean(subpanel);
    s_subpanel = std(subpanel);
    subpanel = (subpanel - ones(T,1)*m_subpanel)./(ones(T,1)*s_subpanel);   % standardize the subpanel

    [P_X, D_X, Sigma_X] = spectral(subpanel, N, h, m);                      % in this case we use spectral with q = N
    E = [D_X(:,h+1)  D_X(:,h+2:2*h+1)*2]*ones(h+1,1)/(2*h+1);               % all the n dynamic eigenvalues
    IC1 = flipud(cumsum(flipud(E)));                                        % compute information criterion
    IC1 = IC1(1:q_max+1,:);
    
    if strcmp(penalty, 'p1') == 1                                   
        p = ((m/T)^0.5 + m^(-2) + N^(-1))*log(min([(T/m)^0.5;  m^2; N]))*ones(q_max+1,1);  
    elseif strcmp(penalty, 'p2') == 1
        p = (min([(T/m)^0.5;  m^2; N])).^(-1/2)*ones(q_max+1,1);  
    elseif strcmp(penalty, 'p3') == 1    
        p = (min([(T/m)^0.5;  m^2; N])).^(-1)*log(min([(T/m)^0.5;  m^2; N]))*ones(q_max+1,1);  
    end

    for c = 1:floor(c_max*cf)
        cc = c/cf;
        IC_log = log(IC1./N) + (0:q_max)'.*p*cc;
        rr = find((IC_log == ones(q_max+1,1)*min(IC_log))==1);              % compute minimum of IC
        o_log(s,c) = rr-1;
    end
end

cr = (1:floor(c_max*cf))'/cf;
nfact = o_log(end,:);                                                       % number of factors when N = n
v_nfact = std(o_log);

%% Plot if needed
if plot_opt == 1
    figure
    plot(cr,5*v_nfact,'b-')
    hold all
    plot(cr,nfact,'r-')
    xlabel('c')
    axis tight
    legend('S_c','q^{*T}_{c;n}')
    title('estimated number of factors - log criterion')
end
