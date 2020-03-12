% function [chi,CL,v,C1] = gdfm_unrestricted(X,q,k,w,nlagsimp,idvar,qq,nrepli)
%
% Function to estimate the factor model according to
% Forni Hallin Lippi Zaffaroni (2017) "Dynamic factor models with
% infinite-dimensional factor space: asymptotic analysis,",
% Journal of Econometrics, 199, 74-92
%
% The model is
%   H(L)*X(t)=R*v(t)+H(L)*xi(t)
%   chi(t)=C(L)v(t)
%   H(L)*chi(t)=R*v(t)
%   Aj(L)*chi(t)=Rj*vj(t) for j=1:m
%
% INPUT:    X               :   T x n data matrix
%                               data should be covariance stationary
%           q               :   number of factors (use numfactors.m)
%           k               :   number of lags in VAR blocks Aj(L)
%           w               :   covariogram truncation
%                               (default value: floor(sqrt(T)))
%           nlagsimp        :   number of lags in MA representation
%                               (default value: 20)
%           idvar           :   index of variables used for identification
%                               should be of size q (default value: [1:q])
%           qq              :   size of the VAR blocks
%                               (default value: q+1)
%           nrepli          :   number of replications to estimate the
%                               model (default value: 100)
%           dummy_std       :   binary = 1 if standardize data, 0 otherwise
%                               (default value: 1)
%
% OUTPUT:  chi              :   T x n common components of standardized data
%          CL               :   n x q x nlagsimp matrix of impulse responses
%          v                :   T x q matrix of common shocks
%          C1               :   n x q matrix of long run impulse responses
%          eta1             :   T x n matrix of innovations R*v(t)

function [chi,CL,v,C1,eta1] = gdfm_unrestricted(X,q,k,w,nlagsimp,idvar,qq,nrepli,dummy_std)

%% preliminary setting
T = size(X,1);

if nargin < 2
    disp('ERROR MESSAGE: Too few input arguments');
    return
end

if nargin == 2
    k = 1;
    w = floor(sqrt(T));
    nlagsimp = 20;
    idvar = [1:q];
    qq = q+1;
    nrepli = 100;
    dummy_std = 1;
end

if nargin == 3
    w = floor(sqrt(T));
    nlagsimp = 20;
    idvar = [1:q];
    qq = q+1;
    nrepli = 100;
    dummy_std = 1;
end

if nargin == 4
    nlagsimp = 20;
    idvar = [1:q];
    qq = q+1;
    nrepli = 100;
    dummy_std = 1;
end

if nargin == 5
    idvar = [1:q];
    qq = q+1;
    nrepli = 100;
    dummy_std = 1;
end

if nargin == 6
    qq = q+1;
    nrepli = 100;
    dummy_std = 1;
end

if nargin == 7
    nrepli = 100;
    dummy_std = 1;
end

if nargin == 8
    dummy_std = 1;
end

%% Mean-standardize data
if dummy_std == 1
    sigma = std(X);
    mu = mean(X);
    z = (X - ones(T,1)*mu)./(ones(T,1)*sigma);
else
    z = X;
end

%% compute cross and autocovariances of common component
covmat = cestimate(z,q,w);                                                  
[n , ~ , h] = size(covmat);
H =(h+1)/2;
m = floor(n/qq);                                                            

%% permutation of variables repeated nrepli times
for h = 1:nrepli
    imp = nan*ones(n,q,nlagsimp);
    imp1 = nan*ones(n,q);
    riord = randperm(n);                                                    % generate permutation of variabiles
    while not(isempty(intersect(idvar ,riord(qq*m+1:end))))
        riord = randperm(n);                                                % check that none of the variables on idvar is excluded from the permutation
    end
    x = z(:,riord);                                                         % permute veriables
    covmatrix = covmat(riord,riord,:);                                      % permutation of the covariance matrix;
    
    %% Estimation of H(L), G(L)=inv(H(L)), w(t)
    [HL, GL, GL1, w] = StaticRepresentation(x,covmatrix,qq,m,nlagsimp,T,k,H);    
    S = diag(std(w)); 
    ww = (w-ones(size(w,1),1)*mean(w))*(S^-1);
    eta(:,:,h) = ww;
    
    %% pca on VAR residuals
    opt.disp = 0;
    [V, MM] = eigs(cov(ww), q,'LM',opt);
    M = diag(sqrt(diag(MM)));                                               
    uu = ww*V*inv(M);                                                       
    
    %% impulse responses C(L)
    for lag = 1:nlagsimp
        BBB(:,:,lag) = GL(:, :, lag)*S*V*M;          
    end;
    imp(riord(1:qq*m),:,:) = BBB;                                           % reorder variabiles
    imp1(riord(1:qq*m),:,:) = GL1*S*V*M;
    [beta(:,:,:,h), u(:,:,h) beta1(:,:,h)] = DfmCholIdent(imp,idvar,uu,imp1);
end

%% average of impulse responses over nrepli identified via Choleski
CL = nanmean(beta,4);
v = nanmean(u,3);
C1 = nanmean(beta1,3);
eta1 = nanmean(eta,3);

vv=[zeros(nlagsimp-1,q); v];
chi=zeros(n,T-k);
for ii=1:T-k;
    for jj=1:nlagsimp;
        chi(:,ii)=chi(:,ii)+CL(:,:,jj)*vv(ii+nlagsimp-jj,:)';
    end;
end;
chi=chi';

%%% =================================================================== %%%
%%% =================================================================== %%%
                        % SUBROUTINES %
%%% =================================================================== %%%
%%% =================================================================== %%%

% cestimate - computes the cross and auto covariances of the common component

function S = cestimate(x,nfactors,w)
%%% define some useful quantities %%%
[T,N] = size(x);
W = 2*w+1;
B = triang(W);
%%% compute covariances %%%
S = zeros(N,N,W);
for k = 1:w+1,
    S(:,:,w+k) = B(w+k)*center(x(k:T,:))'*center(x(1:T+1-k,:))/(T-k);
    S(:,:,w-k+2) = S(:,:,w+k)';
end
%%% compute the spectral matrix in W points (S) %%%
Factor = exp(-sqrt(-1)*(-w:w)'*(0:2*pi/W:4*pi*w/W));
for j = 1:N
    S(j,:,:) = squeeze(S(j,:,:))*Factor;
end
%%% compute the egenvectors  for all points (E) %%%
opt.disp = 0;
[A,D] = eigs(S(:,:,1),nfactors,'LM',opt);
S(:,:,1) = A*D*A';
for j = 2:w+1,
    [A,D] = eigs(S(:,:,j),nfactors,'LM',opt);
    S(:,:,j) = A*D*A';
    S(:,:,W+2-j) = conj(S(:,:,j));
end
for j = 1:N
    S(:,j,:) = real(squeeze(S(:,j,:))*conj(Factor).'/W);
end

%%% =================================================================== %%%
%%% =================================================================== %%%

% StaticRepresentation - estimate the static representation of model

function [HL,GL,GL1,w]=StaticRepresentation(x,covmatrix,qq,m,nlagsimp,T,k,H)

GL = zeros(qq*m,qq*m,nlagsimp); HL = zeros(qq*m,qq*m,k+1); w = zeros(T-k,qq*m);
for mm = 1:m;
    [C, CC, B, u]=VARcov(x,covmatrix,T,k,qq,mm,H,nlagsimp);                 % Estimation of Aj(L) - Gj(L) - vj(t)
    HL((mm-1)*qq+1:qq*mm,(mm-1)*qq+1:qq*mm,:)=CC;                           % Building the H(L) Matrix
    w(:,(mm-1)*qq+1:qq*mm) = u;                                             % Residual w(t)
    GL((mm-1)*qq+1:qq*mm,(mm-1)*qq+1:qq*mm,:)=B;                            % Building G(L)=inv(H(L))
    GL1((mm-1)*qq+1:qq*mm,(mm-1)*qq+1:qq*mm,:)=inv(sum(CC,3));              % long run impact matrix
end

%%% =================================================================== %%%
%%% =================================================================== %%%

% VARcov - estimate a VAR from autocovariance of the variables (YuleWalker)
% C - autoregressive coefficients
% B - Moving average Representation
% u - residuals

function [C,CC,B,u]=VARcov(x,covmatrix,T,k,qq,mm,H,nlagsimp)
A = zeros(qq*k,qq*k); B = zeros(qq*k,qq); xx = zeros(T-k,qq*k);
for j = 1:k
    for i = 1:k;
        A((j-1)*qq+1:qq*j,(i-1)*qq+1:qq*i) = covmatrix((mm-1)*qq+1:mm*qq, (mm-1)*qq+1:mm*qq,H + i - j);
    end
    B((j-1)*qq+1:qq*j,:) = covmatrix((mm-1)*qq+1:mm*qq, (mm-1)*qq+1:mm*qq,H - j);
    xx(:,(j-1)*qq+1:qq*j) =  x(k+1-j:T-j ,(mm-1)*qq+1:mm*qq);
end
C = inv(A)*B;                                                               % Aj(L)
u = x(k+1:T , (mm-1)*qq+1:mm*qq) - xx*C;                                    % Residual wj(t)
CC(:,:,1) = eye(qq); CC(:,:,2:k + 1) = - reshape(C',qq,qq,k);               % Reshaping Aj(L) s.t. it is invertible
B = InvPolMatrix(CC,nlagsimp);                                              % Building Gj(L)= inv(Aj(L))

%%% =================================================================== %%%
%%% =================================================================== %%%

% InvPolMatrix - inversion of a matrix of polynomials in the lag operator

function inverse = InvPolMatrix(poly,nlags)
n = size(poly,1); k = size(poly,3) - 1;
for s = 1:k+1
    newpoly(:,:,s)=inv(poly(:,:,1))*poly(:,:,s);                            % Guarantees that poly(:,:,1)=eye(n)
end            
polynomialmatrix = - newpoly(:,:,2:k+1);
A = zeros(n*k,n*k);
A(n+1:n*k,1:n*(k-1)) = eye(n*(k-1));
for j = 1:k
    A(1:n,(j-1)*n+1:j*n) = polynomialmatrix(:,:,j);
end
inverse = zeros(n,n,nlags);
D = eye(n*k);
for j = 1:nlags
    inverse(:,:,j) = D(1:n,1:n)*inv(poly(:,:,1));
    D = A*D;
end

%%% =================================================================== %%%
%%% =================================================================== %%%

% DfmCholIdent - choleski identification of impulse responses

function [imp, u, imp1, H] = DfmCholIdent(rawimp, idvar, rawu,rawimp1)
B0 = rawimp(idvar,:,1);
C = chol(B0*B0')';
H = inv(B0)*C;
k = size(rawimp,3);
for j =  1:k
    imp(:,:,j) = rawimp(:,:,j)*H; 
end
if nargin>2 
    u = rawu*H; 
end
if nargin>3 
    imp1 = rawimp1*H; 
end

%%% =================================================================== %%%
%%% =================================================================== %%%

